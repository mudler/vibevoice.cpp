#!/usr/bin/env python3
"""Dump a fixture that replicates VibeVoice's SConv1d (causal Conv1d with
zero left-padding + extra alignment padding for stride). The C++ test loads
this and verifies our ggml implementation matches PyTorch exactly.

Cases produced (one fixture each):
  sconv1d_basic       : k=7, s=1, no stride alignment
  sconv1d_strided     : k=4, s=2, exercise stride alignment padding
  sconv1d_dw          : depthwise (groups=in=out), k=7, s=1
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def get_extra_padding(length: int, kernel: int, stride: int, padding_total: int) -> int:
    n_frames = (length - kernel + padding_total) / stride + 1
    ideal = (math.ceil(n_frames) - 1) * stride + (kernel - padding_total)
    return ideal - length


def causal_conv1d_forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None,
                          stride: int, dilation: int, groups: int) -> torch.Tensor:
    kernel = w.shape[-1]
    padding_total = (kernel - 1) * dilation - (stride - 1)
    extra = get_extra_padding(x.shape[-1], kernel, stride, padding_total)
    # left padding for causal + right extra padding for stride alignment, both zeros
    x_p = F.pad(x, (padding_total, extra), mode="constant", value=0.0)
    return F.conv1d(x_p, w, bias=b, stride=stride, dilation=dilation, groups=groups)


def causal_convtranspose1d_forward(x: torch.Tensor, w: torch.Tensor,
                                   b: torch.Tensor | None,
                                   kernel: int, stride: int) -> torch.Tensor:
    # PyTorch ConvTranspose1d weight shape: [in, out, K]
    y = F.conv_transpose1d(x, w, bias=b, stride=stride)
    # Causal trim: trim_right_ratio=1.0 → trim ALL of (kernel - stride) on right.
    pad_total = kernel - stride
    if pad_total > 0:
        y = y[..., : y.shape[-1] - pad_total]
    return y


def write_convt_case(out: Path, name: str, *, in_ch: int, out_ch: int,
                     kernel: int, stride: int, T: int, seed: int) -> None:
    import gguf
    rng = torch.Generator().manual_seed(seed)
    x  = torch.randn(1, in_ch, T,                      generator=rng, dtype=torch.float32)
    w  = torch.randn(in_ch, out_ch, kernel,            generator=rng, dtype=torch.float32) * 0.1
    bi = torch.randn(out_ch,                           generator=rng, dtype=torch.float32) * 0.1
    y  = causal_convtranspose1d_forward(x, w, bi, kernel=kernel, stride=stride)

    f = out / f"{name}.gguf"
    g = gguf.GGUFWriter(str(f), arch="sconvt1d-test")
    g.add_uint32("convt.in_ch",  in_ch)
    g.add_uint32("convt.out_ch", out_ch)
    g.add_uint32("convt.kernel", kernel)
    g.add_uint32("convt.stride", stride)
    g.add_uint32("convt.T",      T)
    g.add_tensor("test.input",            x.numpy())
    g.add_tensor("test.expected_output",  y.detach().numpy())
    g.add_tensor("weight.kernel", w.numpy())
    g.add_tensor("weight.bias",   bi.numpy())
    g.write_header_to_file()
    g.write_kv_data_to_file()
    g.write_tensors_to_file()
    g.close()
    sys.stderr.write(
        f"wrote {f}: in={in_ch} out={out_ch} k={kernel} s={stride} T={T} -> Tout={y.shape[-1]}\n"
    )


def write_case(out: Path, name: str, *, in_ch: int, out_ch: int, kernel: int,
               stride: int, dilation: int, groups: int, T: int, seed: int) -> None:
    import gguf

    rng = torch.Generator().manual_seed(seed)
    x  = torch.randn(1, in_ch, T,                generator=rng, dtype=torch.float32)
    w  = torch.randn(out_ch, in_ch // groups, kernel,
                     generator=rng, dtype=torch.float32) * 0.1
    bi = torch.randn(out_ch,                     generator=rng, dtype=torch.float32) * 0.1

    y = causal_conv1d_forward(x, w, bi, stride=stride, dilation=dilation, groups=groups)

    f = out / f"{name}.gguf"
    f.parent.mkdir(parents=True, exist_ok=True)
    g = gguf.GGUFWriter(str(f), arch="sconv1d-test")
    g.add_uint32("sconv.in_ch",   in_ch)
    g.add_uint32("sconv.out_ch",  out_ch)
    g.add_uint32("sconv.kernel",  kernel)
    g.add_uint32("sconv.stride",  stride)
    g.add_uint32("sconv.dilation",dilation)
    g.add_uint32("sconv.groups",  groups)
    g.add_uint32("sconv.T",       T)

    # gguf reverses numpy shape to ggml convention.
    # PyTorch x: [B, C, T] -> numpy [B, C, T] -> ggml [T, C, B]
    g.add_tensor("test.input",            x.numpy())
    g.add_tensor("test.expected_output",  y.detach().numpy())
    # PyTorch weight: [out, in/groups, K] -> numpy [out, in/groups, K] -> ggml [K, in/groups, out]
    g.add_tensor("weight.kernel", w.numpy())
    g.add_tensor("weight.bias",   bi.numpy())

    g.write_header_to_file()
    g.write_kv_data_to_file()
    g.write_tensors_to_file()
    g.close()
    sys.stderr.write(
        f"wrote {f}: in={in_ch} out={out_ch} k={kernel} s={stride} d={dilation} "
        f"groups={groups} T={T} -> Tout={y.shape[-1]}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    write_case(out_dir, "sconv1d_basic",
               in_ch=4, out_ch=8, kernel=7, stride=1, dilation=1, groups=1, T=32, seed=11)
    write_case(out_dir, "sconv1d_strided",
               in_ch=4, out_ch=8, kernel=4, stride=2, dilation=1, groups=1, T=33, seed=22)
    write_case(out_dir, "sconv1d_dw",
               in_ch=8, out_ch=8, kernel=7, stride=1, dilation=1, groups=8, T=32, seed=33)
    write_convt_case(out_dir, "sconvt1d_basic",
                     in_ch=8, out_ch=4, kernel=4, stride=2, T=16, seed=44)
    write_convt_case(out_dir, "sconvt1d_long",
                     in_ch=4, out_ch=2, kernel=10, stride=5, T=8, seed=55)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
