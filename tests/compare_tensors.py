#!/usr/bin/env python3
"""Compare two .npy / .gguf-tensor files and report max-abs / cosine-sim.

This is a placeholder used by later milestones (Qwen2 block, acoustic
tokenizer, diffusion head). M1 doesn't use it directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a",   required=True)
    ap.add_argument("--b",   required=True)
    ap.add_argument("--tol", type=float, default=1e-3, help="max-abs tolerance")
    args = ap.parse_args()

    try:
        import numpy as np
    except ImportError:
        sys.stderr.write("error: pip install numpy\n")
        return 1

    def load(p: str):
        path = Path(p)
        if path.suffix == ".npy":
            return np.load(path)
        sys.stderr.write(f"unsupported format: {path.suffix} (only .npy implemented in M1)\n")
        sys.exit(1)

    a = load(args.a)
    b = load(args.b)
    if a.shape != b.shape:
        sys.stderr.write(f"shape mismatch: {a.shape} vs {b.shape}\n")
        return 2
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    a_f = a.astype(np.float64).ravel()
    b_f = b.astype(np.float64).ravel()
    denom = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
    cos = float((a_f * b_f).sum() / denom) if denom > 0 else 1.0

    sys.stdout.write(
        f"max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  cos={cos:.6f}  shape={a.shape}\n"
    )
    return 0 if max_abs <= args.tol else 3


if __name__ == "__main__":
    raise SystemExit(main())
