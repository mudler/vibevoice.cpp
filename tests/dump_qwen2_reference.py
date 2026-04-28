#!/usr/bin/env python3
"""Dump a Qwen2DecoderLayer reference fixture for the C++ parity test.

Builds a tiny Qwen2 layer (hidden=128, 4 heads, 2 KV heads, head_dim=32) with
seeded random weights, runs a fixed-seed input through it, and writes a single
.gguf containing weights, input, expected output, and metadata.

Usage:
  python tests/dump_qwen2_reference.py --out tests/fixtures/qwen2_layer0.gguf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",         required=True)
    ap.add_argument("--hidden",      type=int, default=128)
    ap.add_argument("--n-heads",     type=int, default=4)
    ap.add_argument("--n-kv-heads",  type=int, default=2)
    ap.add_argument("--ffn-mult",    type=int, default=3, help="intermediate = hidden * mult")
    ap.add_argument("--seq",         type=int, default=64)
    ap.add_argument("--rope-theta",  type=float, default=1.0e6)
    ap.add_argument("--rms-eps",     type=float, default=1e-6)
    ap.add_argument("--seed",        type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        import gguf
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2DecoderLayer,
            Qwen2RotaryEmbedding,
        )
    except ImportError as e:
        sys.stderr.write(f"error: {e}\n")
        sys.stderr.write("install: pip install gguf transformers torch\n")
        return 1

    head_dim     = args.hidden // args.n_heads
    intermediate = args.hidden * args.ffn_mult

    cfg = Qwen2Config(
        vocab_size=32000,
        hidden_size=args.hidden,
        num_hidden_layers=1,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads,
        intermediate_size=intermediate,
        max_position_embeddings=args.seq * 2,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_eps,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        torch_dtype=torch.float32,
    )

    torch.set_grad_enabled(False)
    layer       = Qwen2DecoderLayer(cfg, layer_idx=0).eval().to(torch.float32)
    rotary_emb  = Qwen2RotaryEmbedding(config=cfg).eval().to(torch.float32)

    # Random input.
    B, T = 1, args.seq
    x   = torch.randn(B, T, args.hidden, dtype=torch.float32)
    pos = torch.arange(T)[None, :]  # [1, T]

    # Causal mask: HF expects shape [B, 1, T, T] with 0 / -inf.
    mask = torch.full((T, T), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    mask = mask[None, None, :, :]  # [1,1,T,T]

    cos, sin = rotary_emb(x, pos)  # each [1, T, head_dim]
    pos_emb  = (cos, sin)

    # Forward; pass position_embeddings explicitly (transformers ≥ 4.43).
    out = layer(
        x,
        attention_mask=mask,
        position_ids=pos,
        position_embeddings=pos_emb,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )
    y = out[0] if isinstance(out, tuple) else out  # [B, T, hidden]

    # Collect weights — use the standard llama.cpp-style names.
    sd  = layer.state_dict()
    def t(n):
        return sd[n].detach().cpu().numpy().astype(np.float32)

    weights = {
        "weight.attn_norm":    t("input_layernorm.weight"),
        "weight.attn_q":       t("self_attn.q_proj.weight"),
        "weight.attn_q_bias":  t("self_attn.q_proj.bias"),
        "weight.attn_k":       t("self_attn.k_proj.weight"),
        "weight.attn_k_bias":  t("self_attn.k_proj.bias"),
        "weight.attn_v":       t("self_attn.v_proj.weight"),
        "weight.attn_v_bias":  t("self_attn.v_proj.bias"),
        "weight.attn_o":       t("self_attn.o_proj.weight"),
        "weight.ffn_norm":     t("post_attention_layernorm.weight"),
        "weight.ffn_gate":     t("mlp.gate_proj.weight"),
        "weight.ffn_up":       t("mlp.up_proj.weight"),
        "weight.ffn_down":     t("mlp.down_proj.weight"),
    }

    # I/O tensors.
    test_io = {
        "test.input":           x.cpu().numpy().astype(np.float32),
        "test.expected_output": y.cpu().numpy().astype(np.float32),
    }
    pos_ids = pos[0].cpu().numpy().astype(np.int32)

    # ---- write gguf ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(out_path), arch="qwen2-test")

    w.add_uint32("qwen2.hidden_size",       args.hidden)
    w.add_uint32("qwen2.n_heads",           args.n_heads)
    w.add_uint32("qwen2.n_kv_heads",        args.n_kv_heads)
    w.add_uint32("qwen2.head_dim",          head_dim)
    w.add_uint32("qwen2.intermediate_size", intermediate)
    w.add_float32("qwen2.rope_theta",       float(args.rope_theta))
    w.add_float32("qwen2.rms_norm_eps",     float(args.rms_eps))
    w.add_uint32("qwen2.test.batch",        B)
    w.add_uint32("qwen2.test.seq_len",      T)
    w.add_uint32("qwen2.test.seed",         int(args.seed))

    for name, arr in weights.items():
        w.add_tensor(name, arr)
    for name, arr in test_io.items():
        w.add_tensor(name, arr)
    w.add_tensor("test.position_ids", pos_ids)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sys.stderr.write(
        f"wrote {out_path}: hidden={args.hidden} heads={args.n_heads}/{args.n_kv_heads} "
        f"head_dim={head_dim} ffn={intermediate} seq={T} seed={args.seed}\n"
        f"  out: shape={tuple(y.shape)} mean={float(y.mean()):.4f} std={float(y.std()):.4f}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
