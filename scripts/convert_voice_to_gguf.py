#!/usr/bin/env python3
"""Convert a VibeVoice voice cache (`demo/voices/streaming_model/<name>.pt`)
to a `.gguf` that the C++ runtime can load directly.

The .pt holds, for each LM stack:
  last_hidden_state    : [1, seq, hidden] bf16
  past_key_values      : DynamicCache with key_cache / value_cache lists,
                         each [1, n_kv_heads, seq, head_dim] bf16

We dump:
  metadata
    voice.lm.seq_len      / voice.tts_lm.seq_len
    voice.lm.n_layers     / voice.tts_lm.n_layers
    voice.hidden          / voice.head_dim       / voice.n_kv_heads
  tensors  (all fp32 for ggml element-wise op compatibility)
    voice.lm.last_hidden                 [hidden, seq_lm]
    voice.lm.k.{i}                       [head_dim, n_kv, seq_lm]
    voice.lm.v.{i}                       [head_dim, n_kv, seq_lm]
    voice.tts_lm.last_hidden             [hidden, seq_tlm]
    voice.tts_lm.k.{i}                   [head_dim, n_kv, seq_tlm]
    voice.tts_lm.v.{i}                   [head_dim, n_kv, seq_tlm]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def get_kv(pkv) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pull (key_cache, value_cache) out of a DynamicCache or legacy tuple."""
    if hasattr(pkv, "key_cache"):
        return list(pkv.key_cache), list(pkv.value_cache)
    if hasattr(pkv, "to_legacy_cache"):
        legacy = pkv.to_legacy_cache()
        return [t[0] for t in legacy], [t[1] for t in legacy]
    if isinstance(pkv, (list, tuple)):
        return [t[0] for t in pkv], [t[1] for t in pkv]
    raise RuntimeError(f"Unknown cache type: {type(pkv).__name__}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help=".pt voice cache from upstream")
    ap.add_argument("--out", required=True, help="output .gguf path")
    args = ap.parse_args()

    try:
        import gguf
    except ImportError:
        sys.stderr.write("error: pip install gguf\n")
        return 1

    src = Path(args.src)
    if not src.exists():
        sys.stderr.write(f"error: missing {src}\n")
        return 2

    state = torch.load(str(src), map_location="cpu", weights_only=False)
    if not (isinstance(state, dict) and "lm" in state and "tts_lm" in state):
        sys.stderr.write("error: expected dict with keys 'lm' and 'tts_lm'\n")
        return 3

    lm     = state["lm"]
    tlm    = state["tts_lm"]
    n_lm   = state.get("neg_lm")
    n_tlm  = state.get("neg_tts_lm")

    lm_h   = lm.last_hidden_state.float().squeeze(0)   # [seq, hidden]
    tlm_h  = tlm.last_hidden_state.float().squeeze(0)
    lm_k, lm_v   = get_kv(lm.past_key_values)
    tlm_k, tlm_v = get_kv(tlm.past_key_values)

    hidden    = int(lm_h.shape[-1])
    n_kv      = int(lm_k[0].shape[1])
    head_dim  = int(lm_k[0].shape[-1])
    seq_lm    = int(lm_h.shape[0])
    seq_tlm   = int(tlm_h.shape[0])

    have_neg = n_lm is not None and n_tlm is not None
    if have_neg:
        nlm_h  = n_lm.last_hidden_state.float().squeeze(0)
        ntlm_h = n_tlm.last_hidden_state.float().squeeze(0)
        nlm_k,  nlm_v  = get_kv(n_lm.past_key_values)
        ntlm_k, ntlm_v = get_kv(n_tlm.past_key_values)
        seq_neg_lm  = int(nlm_h.shape[0])
        seq_neg_tlm = int(ntlm_h.shape[0])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(out), arch="vibevoice-voice")
    w.add_string ("voice.source",      str(src.name))
    w.add_uint32 ("voice.hidden",      hidden)
    w.add_uint32 ("voice.head_dim",    head_dim)
    w.add_uint32 ("voice.n_kv_heads",  n_kv)
    w.add_uint32 ("voice.lm.seq_len",  seq_lm)
    w.add_uint32 ("voice.tts_lm.seq_len", seq_tlm)
    w.add_uint32 ("voice.lm.n_layers", len(lm_k))
    w.add_uint32 ("voice.tts_lm.n_layers", len(tlm_k))
    w.add_bool   ("voice.has_neg",           have_neg)
    if have_neg:
        w.add_uint32("voice.neg_lm.seq_len",  seq_neg_lm)
        w.add_uint32("voice.neg_tts_lm.seq_len", seq_neg_tlm)

    def add(name: str, arr: np.ndarray):
        w.add_tensor(name, np.ascontiguousarray(arr.astype(np.float32)))

    # last_hidden: [seq, hidden] (numpy) -> ggml [hidden, seq]
    add("voice.lm.last_hidden",     lm_h.numpy())
    add("voice.tts_lm.last_hidden", tlm_h.numpy())

    # K/V: PyTorch [1, n_kv, seq, head_dim] -> drop batch -> permute to
    # ggml-friendly memory: numpy [seq, n_kv, head_dim], which gguf's
    # shape-reverse turns into ggml [head_dim, n_kv, seq] (= what
    # qwen2_layer_forward expects for k_past / v_past).
    def pack_kv(t: torch.Tensor) -> np.ndarray:
        # [1, n_kv, seq, head_dim] -> [seq, n_kv, head_dim]
        return t.squeeze(0).permute(1, 0, 2).contiguous().float().numpy()

    for i, (k, v) in enumerate(zip(lm_k, lm_v)):
        add(f"voice.lm.k.{i}", pack_kv(k))
        add(f"voice.lm.v.{i}", pack_kv(v))
    for i, (k, v) in enumerate(zip(tlm_k, tlm_v)):
        add(f"voice.tts_lm.k.{i}", pack_kv(k))
        add(f"voice.tts_lm.v.{i}", pack_kv(v))
    if have_neg:
        add("voice.neg_lm.last_hidden",     nlm_h.numpy())
        add("voice.neg_tts_lm.last_hidden", ntlm_h.numpy())
        for i, (k, v) in enumerate(zip(nlm_k, nlm_v)):
            add(f"voice.neg_lm.k.{i}", pack_kv(k))
            add(f"voice.neg_lm.v.{i}", pack_kv(v))
        for i, (k, v) in enumerate(zip(ntlm_k, ntlm_v)):
            add(f"voice.neg_tts_lm.k.{i}", pack_kv(k))
            add(f"voice.neg_tts_lm.v.{i}", pack_kv(v))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    extra = (f"  neg_lm={len(nlm_k)}x{seq_neg_lm}  neg_tlm={len(ntlm_k)}x{seq_neg_tlm}"
             if have_neg else "")
    sys.stderr.write(
        f"wrote {out}: hidden={hidden} head_dim={head_dim} n_kv={n_kv} "
        f"lm={len(lm_k)}x{seq_lm}  tlm={len(tlm_k)}x{seq_tlm}{extra}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
