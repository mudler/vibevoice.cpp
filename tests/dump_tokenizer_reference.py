#!/usr/bin/env python3
"""Encode a fixture corpus with the Hugging Face tokenizer and dump (text, ids)
pairs so the C++ tokenizer can be checked for exact id-level parity.

Output format: JSON Lines, one record per prompt:
  {"text": "...", "ids": [int, ...]}

Usage:
  python tests/dump_tokenizer_reference.py \
      --tokenizer /tmp/vibevoice-tok \
      --corpus tests/fixtures/tokenizer_corpus.txt \
      --out tests/fixtures/tokenizer_reference.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer dir or tokenizer.json")
    ap.add_argument("--corpus",    required=True, help="prompts file, one per line")
    ap.add_argument("--out",       required=True, help="output .jsonl")
    args = ap.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.stderr.write("error: pip install tokenizers\n")
        return 1

    p = Path(args.tokenizer)
    if p.is_dir():
        p = p / "tokenizer.json"
    tok = Tokenizer.from_file(str(p))

    in_path = Path(args.corpus)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.rstrip("\n")
            if not line and not raw:
                continue  # truly empty trailing lines
            ids = tok.encode(line, add_special_tokens=False).ids
            fout.write(json.dumps({"text": line, "ids": ids}, ensure_ascii=False) + "\n")
            n += 1

    sys.stderr.write(f"wrote {out_path} ({n} prompts)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
