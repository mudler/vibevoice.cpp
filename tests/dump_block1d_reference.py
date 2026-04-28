#!/usr/bin/env python3
"""Reference fixture for VibeVoice's Block1D (depthwise mixer + GELU FFN +
layer-scale γ).

Mirrors vibevoice/modular/modular_vibevoice_tokenizer.py:Block1D (mixer_layer
= 'depthwise_conv', layernorm = 'RMSNorm') without importing the upstream
package. Random-init weights + random input → fixed-seed forward pass.

Output: tests/fixtures/block1d.gguf
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


# ---------- helpers (parity with upstream) ----------

def get_extra_padding(length: int, kernel: int, stride: int, padding_total: int) -> int:
    n_frames = (length - kernel + padding_total) / stride + 1
    ideal = (math.ceil(n_frames) - 1) * stride + (kernel - padding_total)
    return ideal - length


def causal_dwconv1d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Depthwise causal Conv1d (k=7, s=1, d=1, groups=C)."""
    C   = x.shape[1]
    K   = w.shape[-1]
    pad = (K - 1)  # since stride=1, dilation=1: pad_total = K-1
    extra = get_extra_padding(x.shape[-1], K, 1, pad)
    x_p = F.pad(x, (pad, extra), mode="constant", value=0.0)
    return F.conv1d(x_p, w, bias=b, stride=1, dilation=1, groups=C)


class ConvRMSNorm(nn.Module):
    """RMSNorm over channels for [B, C, T] tensors."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # transpose so C is last, normalize, scale, transpose back
        x = x.transpose(1, 2)
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x)
        x = x * self.weight
        return x.transpose(1, 2)


class FFN(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(dim, ffn_dim, bias=bias)
        self.linear2 = nn.Linear(ffn_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class Block1D(nn.Module):
    """Random-init Block1D for fixture purposes — mirror of upstream forward."""

    def __init__(self, dim: int, kernel: int = 7, ffn_expansion: int = 4,
                 eps: float = 1e-5, layer_scale_init_value: float = 1e-6,
                 bias: bool = True):
        super().__init__()
        self.norm     = ConvRMSNorm(dim, eps=eps)
        self.ffn_norm = ConvRMSNorm(dim, eps=eps)

        self.mixer_weight = nn.Parameter(torch.randn(dim, 1, kernel) * 0.1)
        self.mixer_bias   = nn.Parameter(torch.randn(dim) * 0.05)

        self.ffn = FFN(dim, ffn_dim=ffn_expansion * dim, bias=bias)

        # gammas (in the real model these init to layer_scale_init_value, but
        # random values exercise the multiply path more thoroughly)
        self.gamma     = nn.Parameter(torch.randn(dim) * 0.5 + 0.5)
        self.ffn_gamma = nn.Parameter(torch.randn(dim) * 0.5 + 0.5)

        # placate the constant
        self.layer_scale_init_value = layer_scale_init_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x
        x = self.norm(x)
        x = causal_dwconv1d(x, self.mixer_weight, self.mixer_bias)
        x = x * self.gamma.unsqueeze(-1)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)             # [B, T, C]
        x = self.ffn(x)
        x = x.permute(0, 2, 1)             # [B, C, T]
        x = x * self.ffn_gamma.unsqueeze(-1)
        x = residual + x
        return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",      required=True)
    ap.add_argument("--dim",      type=int, default=16)
    ap.add_argument("--T",        type=int, default=24)
    ap.add_argument("--kernel",   type=int, default=7)
    ap.add_argument("--ffn-mult", type=int, default=4)
    ap.add_argument("--eps",      type=float, default=1e-5)
    ap.add_argument("--seed",     type=int, default=99)
    args = ap.parse_args()

    try:
        import gguf
    except ImportError:
        sys.stderr.write("error: pip install gguf\n")
        return 1

    torch.manual_seed(args.seed)
    block = Block1D(args.dim, kernel=args.kernel, ffn_expansion=args.ffn_mult,
                    eps=args.eps, bias=True).eval()

    x = torch.randn(1, args.dim, args.T)
    with torch.no_grad():
        y = block(x)

    sd = block.state_dict()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    g = gguf.GGUFWriter(str(out), arch="block1d-test")
    g.add_uint32("block.dim",        args.dim)
    g.add_uint32("block.T",          args.T)
    g.add_uint32("block.kernel",     args.kernel)
    g.add_uint32("block.ffn_mult",   args.ffn_mult)
    g.add_float32("block.eps",       float(args.eps))

    g.add_tensor("test.input",            x.numpy())
    g.add_tensor("test.expected_output",  y.numpy())

    g.add_tensor("weight.norm",          sd["norm.weight"].numpy())
    g.add_tensor("weight.ffn_norm",      sd["ffn_norm.weight"].numpy())
    g.add_tensor("weight.mixer_kernel",  sd["mixer_weight"].numpy())
    g.add_tensor("weight.mixer_bias",    sd["mixer_bias"].numpy())
    g.add_tensor("weight.ffn_linear1",   sd["ffn.linear1.weight"].numpy())
    g.add_tensor("weight.ffn_linear1_bias", sd["ffn.linear1.bias"].numpy())
    g.add_tensor("weight.ffn_linear2",   sd["ffn.linear2.weight"].numpy())
    g.add_tensor("weight.ffn_linear2_bias", sd["ffn.linear2.bias"].numpy())
    g.add_tensor("weight.gamma",         sd["gamma"].numpy())
    g.add_tensor("weight.ffn_gamma",     sd["ffn_gamma"].numpy())

    g.write_header_to_file()
    g.write_kv_data_to_file()
    g.write_tensors_to_file()
    g.close()
    sys.stderr.write(
        f"wrote {out}: dim={args.dim} T={args.T} kernel={args.kernel} "
        f"ffn_mult={args.ffn_mult}\n"
        f"  out: shape={tuple(y.shape)} mean={float(y.mean()):.4f} "
        f"std={float(y.std()):.4f}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
