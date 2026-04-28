#!/usr/bin/env python3
"""Convert a Qwen2-family `tokenizer.json` to a vibevoice.cpp `tokenizer.gguf`.

The output gguf carries:
  metadata:
    tokenizer.model = "qwen2-bbpe"
    tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id (i32; -1 if unset)
    tokenizer.add_bos, tokenizer.add_eos (bool)
    tokenizer.pretokenizer_regex (str) — informational only
  arrays:
    tokenizer.tokens   : [str]   index = token id
    tokenizer.scores   : [f32]   reserved (zeros for BPE)
    tokenizer.token_type : [i32] (0=normal, 1=control/special, 3=user_defined)
    tokenizer.merges   : [str]   "a b" lines, in order (rank = index)
    tokenizer.special_tokens_ids : [i32]
    tokenizer.special_tokens_text: [str]

This format is used directly by `src/tokenizer.cpp`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import gguf  # type: ignore
except ImportError:
    sys.stderr.write(
        "error: need `gguf` python package. install with: pip install gguf\n"
    )
    sys.exit(1)


TOK_TYPE_NORMAL  = 0
TOK_TYPE_CONTROL = 1  # also used for "special" added tokens
TOK_TYPE_USER    = 3


def load_tokenizer_json(src: Path) -> dict:
    if src.is_dir():
        src = src / "tokenizer.json"
    with src.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_special_tokens_config(src_dir: Path) -> dict:
    p = src_dir / "special_tokens_map.json"
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def find_special_id(vocab_by_str: dict, name: str | None) -> int:
    if not name:
        return -1
    return int(vocab_by_str.get(name, -1))


def get_pretokenizer_regex(tj: dict) -> str:
    pre = tj.get("pre_tokenizer")
    if not pre:
        return ""
    if pre.get("type") == "Sequence":
        for sub in pre.get("pretokenizers", []):
            if sub.get("type") == "Split":
                pat = sub.get("pattern", {})
                return pat.get("Regex", "") or pat.get("String", "")
    elif pre.get("type") == "Split":
        pat = pre.get("pattern", {})
        return pat.get("Regex", "") or pat.get("String", "")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to model dir or tokenizer.json")
    ap.add_argument("--out", required=True, help="output tokenizer.gguf")
    args = ap.parse_args()

    src = Path(args.src)
    src_dir = src if src.is_dir() else src.parent
    tj = load_tokenizer_json(src)

    model = tj.get("model", {})
    if model.get("type") != "BPE":
        sys.stderr.write(f"warning: tokenizer model type is {model.get('type')}, expected BPE\n")

    vocab_by_str: dict[str, int] = dict(model.get("vocab", {}))
    if not vocab_by_str:
        sys.stderr.write("error: empty vocab\n")
        return 2
    n_vocab = max(vocab_by_str.values()) + 1

    # added_tokens may extend the vocab beyond model.vocab
    added = tj.get("added_tokens", []) or []
    for a in added:
        tok_id = int(a["id"])
        content = a["content"]
        if tok_id >= n_vocab:
            n_vocab = tok_id + 1
        vocab_by_str[content] = tok_id

    tokens: list[str] = ["" for _ in range(n_vocab)]
    token_type: list[int] = [TOK_TYPE_NORMAL] * n_vocab

    for tok_str, tok_id in vocab_by_str.items():
        if 0 <= tok_id < n_vocab:
            tokens[tok_id] = tok_str
    for a in added:
        tid = int(a["id"])
        if 0 <= tid < n_vocab:
            tokens[tid] = a["content"]
            token_type[tid] = TOK_TYPE_CONTROL if a.get("special") else TOK_TYPE_USER

    # warn on holes
    n_holes = sum(1 for i, s in enumerate(tokens) if s == "" and i not in vocab_by_str.values())
    if n_holes:
        sys.stderr.write(f"warning: {n_holes} hole(s) in vocab id space\n")

    merges_raw = model.get("merges", []) or []
    merges: list[str] = []
    for m in merges_raw:
        if isinstance(m, list):
            if len(m) == 2:
                merges.append(f"{m[0]} {m[1]}")
        elif isinstance(m, str):
            merges.append(m)

    # special token ids
    stc = extract_special_tokens_config(src_dir)
    bos_name = (stc.get("bos_token") or {}).get("content") if isinstance(stc.get("bos_token"), dict) else stc.get("bos_token")
    eos_name = (stc.get("eos_token") or {}).get("content") if isinstance(stc.get("eos_token"), dict) else stc.get("eos_token")
    pad_name = (stc.get("pad_token") or {}).get("content") if isinstance(stc.get("pad_token"), dict) else stc.get("pad_token")

    # Qwen2 commonly uses <|endoftext|> as eos and no bos.
    if not eos_name and "<|endoftext|>" in vocab_by_str:
        eos_name = "<|endoftext|>"

    bos_id = find_special_id(vocab_by_str, bos_name)
    eos_id = find_special_id(vocab_by_str, eos_name)
    pad_id = find_special_id(vocab_by_str, pad_name)

    special_ids: list[int] = []
    special_text: list[str] = []
    for a in added:
        if a.get("special"):
            special_ids.append(int(a["id"]))
            special_text.append(a["content"])

    pre_re = get_pretokenizer_regex(tj)

    # ---------- write gguf ----------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(out), arch="vibevoice-tokenizer")

    w.add_string("tokenizer.model", "qwen2-bbpe")
    w.add_int32("tokenizer.bos_id", int(bos_id))
    w.add_int32("tokenizer.eos_id", int(eos_id))
    w.add_int32("tokenizer.pad_id", int(pad_id))
    w.add_bool("tokenizer.add_bos", False)
    w.add_bool("tokenizer.add_eos", False)
    if pre_re:
        w.add_string("tokenizer.pretokenizer_regex", pre_re)

    w.add_array("tokenizer.tokens", tokens)
    w.add_array("tokenizer.token_type", token_type)
    w.add_array("tokenizer.merges", merges)
    if special_ids:
        w.add_array("tokenizer.special_tokens_ids", special_ids)
        w.add_array("tokenizer.special_tokens_text", special_text)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sys.stderr.write(
        f"wrote {out}: {n_vocab} tokens, {len(merges)} merges, "
        f"bos={bos_id} eos={eos_id} pad={pad_id}, special={len(special_ids)}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
