#!/usr/bin/env python3
"""Reference fixture for the VibeVoice acoustic tokenizer encoder + decoder.

Self-contained PyTorch port of the upstream architecture (see
vibevoice/modular/modular_vibevoice_tokenizer.py:TokenizerEncoder /
TokenizerDecoder). Builds a tiny model (channels=1, vae_dim=8, n_filters=4,
ratios=[2,2], depths=[1,1,1]) so the fixture stays small while exercising
every component.

Output: tests/fixtures/acoustic.gguf with weights for both encoder and
decoder, plus expected I/O for both.

Tensor naming convention (so the C++ side can load by string prefix):
  enc.stem.kernel / enc.stem.bias                    (stem SConv1d k=7)
  enc.down_{i}.kernel / enc.down_{i}.bias            (strided SConv1d, i = 1..len(ratios))
  enc.stage_{i}_block_{j}.weight.{norm,ffn_norm,
      mixer_kernel,mixer_bias,gamma,ffn_gamma,
      ffn_linear1,ffn_linear1_bias,
      ffn_linear2,ffn_linear2_bias}
  enc.final_norm
  enc.head.kernel / enc.head.bias                    (head SConv1d k=7)
  dec.stem.kernel / dec.stem.bias                    (stem SConv1d k=7)
  dec.up_{i}.kernel / dec.up_{i}.bias                (SConvTranspose1d)
  dec.stage_{i}_block_{j}.weight.*                   (same as encoder)
  dec.final_norm
  dec.head.kernel / dec.head.bias
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- ops ----------

def get_extra_padding(length: int, kernel: int, stride: int, padding_total: int) -> int:
    n_frames = (length - kernel + padding_total) / stride + 1
    ideal = (math.ceil(n_frames) - 1) * stride + (kernel - padding_total)
    return ideal - length


def causal_conv1d(x, w, b, stride=1, dilation=1, groups=1):
    K   = w.shape[-1]
    pad = (K - 1) * dilation - (stride - 1)
    extra = get_extra_padding(x.shape[-1], K, stride, pad)
    x_p = F.pad(x, (pad, extra), mode="constant", value=0.0)
    return F.conv1d(x_p, w, bias=b, stride=stride, dilation=dilation, groups=groups)


def causal_convtranspose1d(x, w, b, stride):
    K = w.shape[-1]
    pad_total = K - stride          # trim_right_ratio = 1.0 (causal)
    y = F.conv_transpose1d(x, w, bias=b, stride=stride)
    if pad_total > 0:
        y = y[..., : y.shape[-1] - pad_total]
    return y


class ConvRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x)
        x = x * self.weight
        return x.transpose(1, 2)


class FFN(nn.Module):
    def __init__(self, dim, ffn_dim, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(dim, ffn_dim, bias=bias)
        self.linear2 = nn.Linear(ffn_dim, dim,    bias=bias)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


class Block1D(nn.Module):
    def __init__(self, dim, kernel=7, ffn_mult=4, eps=1e-5):
        super().__init__()
        self.norm     = ConvRMSNorm(dim, eps)
        self.ffn_norm = ConvRMSNorm(dim, eps)
        self.mixer_w  = nn.Parameter(torch.randn(dim, 1, kernel) * 0.1)
        self.mixer_b  = nn.Parameter(torch.randn(dim) * 0.05)
        self.ffn      = FFN(dim, ffn_mult * dim, bias=True)
        self.gamma     = nn.Parameter(torch.randn(dim) * 0.4 + 0.6)
        self.ffn_gamma = nn.Parameter(torch.randn(dim) * 0.4 + 0.6)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = causal_conv1d(x, self.mixer_w, self.mixer_b, groups=x.shape[1])
        x = x * self.gamma.unsqueeze(-1)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        x = x * self.ffn_gamma.unsqueeze(-1)
        return residual + x


class StridedConv(nn.Module):
    """Wraps a single SConv1d (no nesting)."""

    def __init__(self, in_ch, out_ch, kernel, stride):
        super().__init__()
        self.kernel_w = nn.Parameter(torch.randn(out_ch, in_ch, kernel) * 0.1)
        self.bias     = nn.Parameter(torch.randn(out_ch) * 0.05)
        self.stride   = stride

    def forward(self, x):
        return causal_conv1d(x, self.kernel_w, self.bias, stride=self.stride)


class TransposedUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride):
        super().__init__()
        # ConvTranspose1d weight shape: [in, out, K]
        self.kernel_w = nn.Parameter(torch.randn(in_ch, out_ch, kernel) * 0.1)
        self.bias     = nn.Parameter(torch.randn(out_ch) * 0.05)
        self.stride   = stride

    def forward(self, x):
        return causal_convtranspose1d(x, self.kernel_w, self.bias, self.stride)


class Encoder(nn.Module):
    def __init__(self, channels, vae_dim, n_filters, ratios, depths,
                 kernel_stem=7, kernel_head=7, ffn_mult=4, eps=1e-5):
        super().__init__()
        assert len(depths) == len(ratios) + 1
        self.depths = depths

        self.stem = StridedConv(channels, n_filters, kernel_stem, stride=1)
        self.downs = nn.ModuleList()
        for i, r in enumerate(ratios):
            in_ch  = n_filters * (2 ** i)
            out_ch = n_filters * (2 ** (i + 1))
            self.downs.append(StridedConv(in_ch, out_ch, kernel=2 * r, stride=r))

        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            ch = n_filters * (2 ** i)
            self.stages.append(nn.ModuleList([
                Block1D(ch, ffn_mult=ffn_mult, eps=eps) for _ in range(depths[i])
            ]))

        last_ch = n_filters * (2 ** (len(depths) - 1))
        self.final_norm = ConvRMSNorm(last_ch, eps=eps)
        self.head = StridedConv(last_ch, vae_dim, kernel_head, stride=1)

    def forward(self, x):
        # i = 0: stem applies; i >= 1: downs[i-1] applies
        x = self.stem(x)
        for block in self.stages[0]:
            x = block(x)
        for i in range(1, len(self.depths)):
            x = self.downs[i - 1](x)
            for block in self.stages[i]:
                x = block(x)
        x = self.final_norm(x)
        x = self.head(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, vae_dim, n_filters, ratios, depths,
                 kernel_stem=7, kernel_head=7, ffn_mult=4, eps=1e-5):
        super().__init__()
        # Mirror the encoder: stem expands vae_dim -> top channels
        assert len(depths) == len(ratios) + 1
        self.depths = depths
        top_ch = n_filters * (2 ** (len(depths) - 1))

        self.stem = StridedConv(vae_dim, top_ch, kernel_stem, stride=1)
        self.ups  = nn.ModuleList()
        for i, r in enumerate(ratios):
            in_ch  = n_filters * (2 ** (len(depths) - 1 - i))
            out_ch = n_filters * (2 ** (len(depths) - 1 - i - 1))
            self.ups.append(TransposedUpsample(in_ch, out_ch, kernel=2 * r, stride=r))

        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            ch = n_filters * (2 ** (len(depths) - 1 - i))
            self.stages.append(nn.ModuleList([
                Block1D(ch, ffn_mult=ffn_mult, eps=eps) for _ in range(depths[i])
            ]))

        self.final_norm = ConvRMSNorm(n_filters, eps=eps)
        self.head = StridedConv(n_filters, channels, kernel_head, stride=1)

    def forward(self, x):
        # symmetric to encoder: stem then stage[0], then for i>=1 ups[i-1] then stage[i]
        x = self.stem(x)
        for block in self.stages[0]:
            x = block(x)
        for i in range(1, len(self.depths)):
            x = self.ups[i - 1](x)
            for block in self.stages[i]:
                x = block(x)
        x = self.final_norm(x)
        x = self.head(x)
        return x


# ---------- tensor dump ----------

def dump_block1d(g, prefix, b: Block1D):
    g.add_tensor(prefix + ".weight.norm",          b.norm.weight.detach().numpy())
    g.add_tensor(prefix + ".weight.ffn_norm",      b.ffn_norm.weight.detach().numpy())
    g.add_tensor(prefix + ".weight.mixer_kernel",  b.mixer_w.detach().numpy())
    g.add_tensor(prefix + ".weight.mixer_bias",    b.mixer_b.detach().numpy())
    g.add_tensor(prefix + ".weight.gamma",         b.gamma.detach().numpy())
    g.add_tensor(prefix + ".weight.ffn_gamma",     b.ffn_gamma.detach().numpy())
    g.add_tensor(prefix + ".weight.ffn_linear1",         b.ffn.linear1.weight.detach().numpy())
    g.add_tensor(prefix + ".weight.ffn_linear1_bias",    b.ffn.linear1.bias.detach().numpy())
    g.add_tensor(prefix + ".weight.ffn_linear2",         b.ffn.linear2.weight.detach().numpy())
    g.add_tensor(prefix + ".weight.ffn_linear2_bias",    b.ffn.linear2.bias.detach().numpy())


def dump_strided(g, prefix, m: StridedConv):
    g.add_tensor(prefix + ".kernel", m.kernel_w.detach().numpy())
    g.add_tensor(prefix + ".bias",   m.bias.detach().numpy())


def dump_transposed(g, prefix, m: TransposedUpsample):
    g.add_tensor(prefix + ".kernel", m.kernel_w.detach().numpy())
    g.add_tensor(prefix + ".bias",   m.bias.detach().numpy())


def dump_encoder(g, prefix: str, enc: Encoder):
    dump_strided(g, prefix + ".stem", enc.stem)
    for i, m in enumerate(enc.downs):
        dump_strided(g, prefix + f".down_{i}", m)
    for si, stage in enumerate(enc.stages):
        for bi, blk in enumerate(stage):
            dump_block1d(g, prefix + f".stage_{si}_block_{bi}", blk)
    g.add_tensor(prefix + ".final_norm", enc.final_norm.weight.detach().numpy())
    dump_strided(g, prefix + ".head", enc.head)


def dump_decoder(g, prefix: str, dec: Decoder):
    dump_strided(g, prefix + ".stem", dec.stem)
    for i, m in enumerate(dec.ups):
        dump_transposed(g, prefix + f".up_{i}", m)
    for si, stage in enumerate(dec.stages):
        for bi, blk in enumerate(stage):
            dump_block1d(g, prefix + f".stage_{si}_block_{bi}", blk)
    g.add_tensor(prefix + ".final_norm", dec.final_norm.weight.detach().numpy())
    dump_strided(g, prefix + ".head", dec.head)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",         required=True)
    ap.add_argument("--channels",    type=int, default=1)
    ap.add_argument("--vae-dim",     type=int, default=8)
    ap.add_argument("--n-filters",   type=int, default=4)
    ap.add_argument("--ratios",      type=int, nargs="+", default=[2, 2])
    ap.add_argument("--depths",      type=int, nargs="+", default=[1, 1, 1])
    ap.add_argument("--T",           type=int, default=32)
    ap.add_argument("--seed",        type=int, default=2024)
    ap.add_argument("--eps",         type=float, default=1e-5)
    args = ap.parse_args()

    try:
        import gguf
    except ImportError:
        sys.stderr.write("error: pip install gguf\n")
        return 1

    if len(args.depths) != len(args.ratios) + 1:
        sys.stderr.write(
            f"depths must have len(ratios)+1 = {len(args.ratios)+1}, got {len(args.depths)}\n"
        )
        return 2

    torch.manual_seed(args.seed)
    enc = Encoder(args.channels, args.vae_dim, args.n_filters, args.ratios, args.depths,
                  eps=args.eps).eval()
    torch.manual_seed(args.seed + 1)
    dec = Decoder(args.channels, args.vae_dim, args.n_filters, args.ratios, args.depths,
                  eps=args.eps).eval()

    # Encoder I/O
    audio = torch.randn(1, args.channels, args.T)
    with torch.no_grad():
        latents = enc(audio)             # [1, vae_dim, T_compressed]

    # Decoder I/O — feed independent random latents (not encoder output)
    # so the decoder test isn't entangled with the encoder test.
    z = torch.randn(1, args.vae_dim, args.T // int(np.prod(args.ratios)))
    with torch.no_grad():
        rec = dec(z)                     # [1, channels, T_full]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    g = gguf.GGUFWriter(str(out), arch="acoustic-test")
    g.add_uint32 ("acoustic.channels",    args.channels)
    g.add_uint32 ("acoustic.vae_dim",     args.vae_dim)
    g.add_uint32 ("acoustic.n_filters",   args.n_filters)
    g.add_array  ("acoustic.ratios",      list(args.ratios))
    g.add_array  ("acoustic.depths",      list(args.depths))
    g.add_uint32 ("acoustic.T",           args.T)
    g.add_uint32 ("acoustic.T_compressed",latents.shape[-1])
    g.add_uint32 ("acoustic.T_dec_in",    z.shape[-1])
    g.add_uint32 ("acoustic.T_dec_out",   rec.shape[-1])
    g.add_float32("acoustic.eps",         float(args.eps))

    g.add_tensor("test.audio",            audio.numpy())
    g.add_tensor("test.encoder_output",   latents.numpy())
    g.add_tensor("test.decoder_input",    z.numpy())
    g.add_tensor("test.decoder_output",   rec.numpy())

    dump_encoder(g, "enc", enc)
    dump_decoder(g, "dec", dec)

    g.write_header_to_file()
    g.write_kv_data_to_file()
    g.write_tensors_to_file()
    g.close()

    sys.stderr.write(
        f"wrote {out}: vae_dim={args.vae_dim} n_filters={args.n_filters} "
        f"ratios={args.ratios} depths={args.depths}\n"
        f"  enc: {tuple(audio.shape)} -> {tuple(latents.shape)}\n"
        f"  dec: {tuple(z.shape)} -> {tuple(rec.shape)}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
