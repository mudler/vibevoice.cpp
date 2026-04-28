#!/usr/bin/env python3
"""Dump a RoPE-NeoX reference fixture for the C++ RoPE numerics test.

Replicates the Qwen2 RoPE math:
  inv_freq = 1.0 / theta^(2k/dim) for k in [0, dim/2)
  cos[p, k] = cos(p * inv_freq[k])     duplicated to dim
  sin[p, k] = sin(p * inv_freq[k])     duplicated to dim
  rotate_half(x) = concat([-x[..., dim/2:], x[..., :dim/2]], dim=-1)
  rope(x) = x * cos + rotate_half(x) * sin

Writes a .gguf with: input x, position ids, expected output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def rope_neox(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    # x: [n_tokens, n_heads, dim]
    dim = x.shape[-1]
    half = dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / dim))
    angles = positions[:, None].astype(np.float64) * inv_freq[None, :]   # [T, half]
    cos = np.cos(angles)
    sin = np.sin(angles)
    cos = np.concatenate([cos, cos], axis=-1)                            # [T, dim]
    sin = np.concatenate([sin, sin], axis=-1)
    cos = cos[:, None, :]                                                # [T, 1, dim]
    sin = sin[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    rot = np.concatenate([-x2, x1], axis=-1)
    return (x.astype(np.float64) * cos + rot.astype(np.float64) * sin).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",      required=True)
    ap.add_argument("--dim",      type=int, default=64)
    ap.add_argument("--n-heads",  type=int, default=4)
    ap.add_argument("--seq",      type=int, default=128)
    ap.add_argument("--theta",    type=float, default=1.0e6)
    ap.add_argument("--seed",     type=int,   default=2025)
    args = ap.parse_args()

    try:
        import gguf
    except ImportError:
        sys.stderr.write("error: pip install gguf\n")
        return 1

    rng = np.random.default_rng(args.seed)
    x   = rng.standard_normal((args.seq, args.n_heads, args.dim)).astype(np.float32)
    pos = np.arange(args.seq).astype(np.int32)
    y   = rope_neox(x, pos, args.theta)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(out), arch="rope-test")
    w.add_uint32("rope.dim",     args.dim)
    w.add_uint32("rope.n_heads", args.n_heads)
    w.add_uint32("rope.seq",     args.seq)
    w.add_float32("rope.theta",  float(args.theta))

    # gguf reverses numpy shape (numpy: last dim fastest; ggml: first dim
    # fastest). With numpy shape [seq, n_heads, dim] the resulting ggml tensor
    # is ne=[dim, n_heads, seq] — exactly what ggml_rope_ext expects.
    w.add_tensor("test.input",            np.ascontiguousarray(x))
    w.add_tensor("test.expected_output",  np.ascontiguousarray(y))
    w.add_tensor("test.position_ids",     pos)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    sys.stderr.write(
        f"wrote {out}: dim={args.dim} n_heads={args.n_heads} seq={args.seq} theta={args.theta}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
