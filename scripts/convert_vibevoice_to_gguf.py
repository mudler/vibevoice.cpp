#!/usr/bin/env python3
"""Convert a VibeVoice safetensors checkpoint to a single .gguf for vibevoice.cpp.

Currently supports `microsoft/VibeVoice-Realtime-0.5B` (the streaming TTS
variant — single safetensors, hidden=896, 24-layer Qwen2 split into 4 lower
+ 20 upper, acoustic decoder only, diffusion head). The 1.5B podcast variant
will be added later.

Output gguf carries:
  metadata: vibevoice.variant, .hidden, .n_layers_lm, .n_layers_tlm,
            .n_heads, .n_kv_heads, .head_dim, .vocab_size, .rope_theta,
            .rms_norm_eps, .acoustic.vae_dim, .acoustic.ratios,
            .acoustic.depths_dec, .diffusion.head_layers,
            .diffusion.ffn_ratio, .diffusion.latent, .speech.scaling, .speech.bias
  tensors:  remapped per the table at the bottom of this file
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import gguf
    from safetensors import safe_open
except ImportError as e:
    sys.stderr.write(f"error: pip install gguf safetensors\n  {e}\n")
    sys.exit(1)


# ---------- key rewrite table ----------
# Each entry: (regex, replacement template). Order matters; the first match wins.
# Use (?P<i>\d+) etc for named groups referenced in the replacement.

REWRITES: list[tuple[re.Pattern, str]] = [
    # ---- ASR variant: top-level lm_head ----
    (re.compile(r"^lm_head\.weight$"),
     "lm_head.weight"),

    # ---- base LM (4 lower layers in realtime, 28 in ASR/1.5B) ----
    (re.compile(r"^model\.language_model\.embed_tokens\.weight$"),
     "lm.tok_embd.weight"),

    (re.compile(r"^model\.language_model\.layers\.(?P<i>\d+)\.self_attn\.(?P<p>[qkv])_proj\.(?P<x>weight|bias)$"),
     r"lm.blk.\g<i>.attn_\g<p>.\g<x>"),
    (re.compile(r"^model\.language_model\.layers\.(?P<i>\d+)\.self_attn\.o_proj\.weight$"),
     r"lm.blk.\g<i>.attn_o.weight"),
    (re.compile(r"^model\.language_model\.layers\.(?P<i>\d+)\.input_layernorm\.weight$"),
     r"lm.blk.\g<i>.attn_norm.weight"),
    (re.compile(r"^model\.language_model\.layers\.(?P<i>\d+)\.post_attention_layernorm\.weight$"),
     r"lm.blk.\g<i>.ffn_norm.weight"),
    (re.compile(r"^model\.language_model\.layers\.(?P<i>\d+)\.mlp\.(?P<p>gate|up|down)_proj\.weight$"),
     r"lm.blk.\g<i>.ffn_\g<p>.weight"),
    (re.compile(r"^model\.language_model\.norm\.weight$"),
     "lm.output_norm.weight"),

    # ---- TTS LM (20 upper layers) ----
    (re.compile(r"^model\.tts_language_model\.embed_tokens\.weight$"),
     "tlm.tok_embd.weight"),

    (re.compile(r"^model\.tts_language_model\.layers\.(?P<i>\d+)\.self_attn\.(?P<p>[qkv])_proj\.(?P<x>weight|bias)$"),
     r"tlm.blk.\g<i>.attn_\g<p>.\g<x>"),
    (re.compile(r"^model\.tts_language_model\.layers\.(?P<i>\d+)\.self_attn\.o_proj\.weight$"),
     r"tlm.blk.\g<i>.attn_o.weight"),
    (re.compile(r"^model\.tts_language_model\.layers\.(?P<i>\d+)\.input_layernorm\.weight$"),
     r"tlm.blk.\g<i>.attn_norm.weight"),
    (re.compile(r"^model\.tts_language_model\.layers\.(?P<i>\d+)\.post_attention_layernorm\.weight$"),
     r"tlm.blk.\g<i>.ffn_norm.weight"),
    (re.compile(r"^model\.tts_language_model\.layers\.(?P<i>\d+)\.mlp\.(?P<p>gate|up|down)_proj\.weight$"),
     r"tlm.blk.\g<i>.ffn_\g<p>.weight"),
    (re.compile(r"^model\.tts_language_model\.norm\.weight$"),
     "tlm.output_norm.weight"),

    # ---- TTS input-type embedding ----
    (re.compile(r"^model\.tts_input_types\.weight$"),
     "tts.input_types.weight"),

    # ---- speech scaling buffers ----
    (re.compile(r"^model\.speech_scaling_factor$"),
     "speech.scaling"),
    (re.compile(r"^model\.speech_bias_factor$"),
     "speech.bias"),

    # ---- acoustic connector ----
    (re.compile(r"^model\.acoustic_connector\.fc1\.(?P<x>weight|bias)$"),
     r"ac.fc1.\g<x>"),
    (re.compile(r"^model\.acoustic_connector\.norm\.weight$"),
     r"ac.norm.weight"),
    (re.compile(r"^model\.acoustic_connector\.fc2\.(?P<x>weight|bias)$"),
     r"ac.fc2.\g<x>"),

    # ---- prediction head (diffusion) ----
    (re.compile(r"^model\.prediction_head\.noisy_images_proj\.weight$"),
     "dh.noisy_proj"),
    (re.compile(r"^model\.prediction_head\.cond_proj\.weight$"),
     "dh.cond_proj"),
    (re.compile(r"^model\.prediction_head\.t_embedder\.mlp\.0\.weight$"),
     "dh.t_embed_lin1"),
    (re.compile(r"^model\.prediction_head\.t_embedder\.mlp\.2\.weight$"),
     "dh.t_embed_lin2"),
    (re.compile(r"^model\.prediction_head\.layers\.(?P<i>\d+)\.norm\.weight$"),
     r"dh.layer_\g<i>.norm"),
    (re.compile(r"^model\.prediction_head\.layers\.(?P<i>\d+)\.adaLN_modulation\.1\.weight$"),
     r"dh.layer_\g<i>.adaln"),
    (re.compile(r"^model\.prediction_head\.layers\.(?P<i>\d+)\.ffn\.(?P<p>gate|up|down)_proj\.weight$"),
     r"dh.layer_\g<i>.ffn_\g<p>"),
    (re.compile(r"^model\.prediction_head\.final_layer\.linear\.weight$"),
     "dh.final.proj"),
    (re.compile(r"^model\.prediction_head\.final_layer\.adaLN_modulation\.1\.weight$"),
     "dh.final.adaln"),

    # ---- acoustic encoder (ASR / 1.5B variants) ----
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.enc.stem.\g<x>"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.downsample_layers\.(?P<i>\d+)\.0\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.enc.down_\g<i>.\g<x>"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.head\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.enc.head.\g<x>"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.norm\.weight$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.norm"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn_norm\.weight$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.ffn_norm"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.gamma$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.gamma"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn_gamma$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.ffn_gamma"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.mixer\.conv\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.mixer_\g<x>"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn\.linear1\.(?P<x>weight|bias)$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.ffn_linear1__SUFFIX__"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn\.linear2\.(?P<x>weight|bias)$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.ffn_linear2__SUFFIX__"),

    # ---- semantic encoder (ASR / 1.5B variants) ----
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.conv\.(?P<x>weight|bias)$"),
     r"st.enc.stem.\g<x>"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.downsample_layers\.(?P<i>\d+)\.0\.conv\.conv\.(?P<x>weight|bias)$"),
     r"st.enc.down_\g<i>.\g<x>"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.head\.conv\.conv\.(?P<x>weight|bias)$"),
     r"st.enc.head.\g<x>"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.norm\.weight$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.norm"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn_norm\.weight$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.ffn_norm"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.gamma$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.gamma"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn_gamma$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.ffn_gamma"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.mixer\.conv\.conv\.conv\.(?P<x>weight|bias)$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.mixer_\g<x>"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn\.linear1\.(?P<x>weight|bias)$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.ffn_linear1__SUFFIX__"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn\.linear2\.(?P<x>weight|bias)$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.ffn_linear2__SUFFIX__"),

    # ---- semantic connector (ASR / 1.5B variants) ----
    (re.compile(r"^model\.semantic_connector\.fc1\.(?P<x>weight|bias)$"),
     r"sc.fc1.\g<x>"),
    (re.compile(r"^model\.semantic_connector\.norm\.weight$"),
     r"sc.norm.weight"),
    (re.compile(r"^model\.semantic_connector\.fc2\.(?P<x>weight|bias)$"),
     r"sc.fc2.\g<x>"),

    # ---- acoustic decoder: stem (upsample 0) ----
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.upsample_layers\.0\.0\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.dec.stem.\g<x>"),
    # ---- acoustic decoder: transposed upsamples 1..6 ----
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.upsample_layers\.(?P<i>\d+)\.0\.convtr\.convtr\.(?P<x>weight|bias)$"),
     r"at.dec.up_\g<i>.\g<x>"),
    # ---- acoustic decoder: head ----
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.head\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.dec.head.\g<x>"),
    # ---- acoustic decoder: stages ----
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.norm\.weight$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.norm"),
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn_norm\.weight$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.ffn_norm"),
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.gamma$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.gamma"),
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn_gamma$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.ffn_gamma"),
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.mixer\.conv\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.mixer_\g<x>"),
    # Block1D.ffn.linear[12] -> ffn_linearN (weight) or ffn_linearN_bias.
    # Use sentinel `__SUFFIX__` rewritten in remap() below.
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn\.linear1\.(?P<x>weight|bias)$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.ffn_linear1__SUFFIX__"),
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn\.linear2\.(?P<x>weight|bias)$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.ffn_linear2__SUFFIX__"),

    # ---- EOS classifier ----
    (re.compile(r"^tts_eos_classifier\.fc1\.(?P<x>weight|bias)$"),
     r"eos.fc1.\g<x>"),
    (re.compile(r"^tts_eos_classifier\.fc2\.(?P<x>weight|bias)$"),
     r"eos.fc2.\g<x>"),
]


def remap(name: str) -> str | None:
    """Return the new gguf tensor name, or None if no rule matches."""
    for pat, repl in REWRITES:
        m = pat.match(name)
        if not m:
            continue
        out = pat.sub(repl, name)
        if "__SUFFIX__" in out:
            # `weight` → "" (suffix dropped), `bias` → "_bias"
            x = m.group("x")
            out = out.replace("__SUFFIX__", "" if x == "weight" else "_bias")
        return out
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",   required=True, help="model dir with config.json + model.safetensors")
    ap.add_argument("--out",   required=True)
    ap.add_argument("--strict", action="store_true",
                    help="fail on any unmapped source key")
    # fp32 is the default for now: ggml's CPU path doesn't auto-cast in
    # element-wise ops, so a uniform fp32 tensor set keeps the pipeline
    # robust. fp16 doubles the on-disk savings but will crash on first
    # element-wise mul in the orchestrator until the loader pre-casts
    # norm/bias tensors.
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = ap.parse_args()

    src = Path(args.src)
    cfg = json.load(open(src / "config.json"))

    # ------- collect tensors -------
    tensors: list[tuple[str, np.ndarray]] = []
    unmapped: list[str] = []

    def to_dtype(arr: np.ndarray) -> np.ndarray:
        # safetensors gives us the source dtype (often bf16). Convert to fp32
        # first, then optionally cast to fp16 for storage.
        if arr.dtype == np.dtype("bfloat16") if hasattr(np, "bfloat16") else False:
            arr = arr.astype(np.float32)
        elif str(arr.dtype) in ("torch.bfloat16", "bfloat16"):
            arr = arr.astype(np.float32)
        if arr.dtype != np.float32 and arr.dtype != np.float16:
            arr = arr.astype(np.float32)
        if args.dtype == "fp16" and arr.dtype == np.float32 and arr.size > 1:
            arr = arr.astype(np.float16)
        return np.ascontiguousarray(arr)

    files = sorted(src.glob("model*.safetensors"))
    if not files:
        sys.stderr.write(f"error: no model*.safetensors under {src}\n")
        return 1

    for f in files:
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                t = fh.get_tensor(k)
                # bf16 → fp32
                if t.dtype == np.dtype("bfloat16") if hasattr(np.dtype, "bfloat16") else False:
                    t = t.float()
                arr = t.cpu().to(__import__("torch").float32).numpy()
                arr = to_dtype(arr)

                new = remap(k)
                if new is None:
                    unmapped.append(k)
                    continue
                tensors.append((new, arr))

    if unmapped:
        msg = (f"warning: {len(unmapped)} unmapped keys (first 10):\n"
               + "\n".join(f"  {k}" for k in unmapped[:10]))
        sys.stderr.write(msg + "\n")
        if args.strict:
            return 2

    # ------- write gguf -------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(out), arch="vibevoice")

    dec = cfg["decoder_config"]
    ac  = cfg["acoustic_tokenizer_config"]
    sm  = cfg.get("semantic_tokenizer_config") or {}
    dh  = cfg.get("diffusion_head_config") or {}

    # Variant detection from architectures + tensor presence.
    # ASR-7B and 1.5B share the `VibeVoice*ForConditionalGeneration` family
    # but differ in what they ship: ASR has lm_head + no decoder/diffusion;
    # 1.5B has decoder + prediction_head and ties lm_head to embed_tokens
    # (so it does not appear as a separate key in safetensors).
    archs = cfg.get("architectures") or []
    arch  = archs[0] if archs else ""
    tnames = [t[0] for t in tensors]
    has_lm_head        = any(k.startswith("lm_head") for k in tnames)
    has_prediction_hd  = any(k.startswith("dh.") for k in tnames)
    has_at_decoder     = any(k.startswith("at.dec.") for k in tnames)
    if "Streaming" in arch or cfg.get("tts_backbone_num_hidden_layers"):
        variant = "realtime-0.5b"
    elif "ASR" in arch:
        variant = "asr-7b"
    elif has_prediction_hd and has_at_decoder:
        variant = "1.5b"
    elif has_lm_head:
        variant = "asr-7b"
    else:
        variant = cfg.get("model_type", "vibevoice")

    # 1.5B ties lm_head to embed_tokens (cfg.decoder_config.tie_word_embeddings).
    # The C++ loader expects an explicit `lm_head.weight` entry, so we materialise
    # one by aliasing the token-embedding tensor.
    if (variant == "1.5b"
        and dec.get("tie_word_embeddings", False)
        and not has_lm_head):
        embd = next((arr for n, arr in tensors if n == "lm.tok_embd.weight"), None)
        if embd is None:
            sys.stderr.write("error: 1.5b variant missing lm.tok_embd.weight; "
                             "cannot synthesise tied lm_head\n")
            return 4
        tensors.append(("lm_head.weight", embd))

    n_total      = dec["num_hidden_layers"]
    n_tts_layers = cfg.get("tts_backbone_num_hidden_layers", 0) or 0
    n_lm_layers  = n_total - n_tts_layers if n_tts_layers > 0 else n_total

    w.add_string ("vibevoice.variant",       variant)
    w.add_uint32 ("vibevoice.hidden",        dec["hidden_size"])
    w.add_uint32 ("vibevoice.n_layers_lm",   n_lm_layers)
    w.add_uint32 ("vibevoice.n_layers_tlm",  n_tts_layers)
    w.add_uint32 ("vibevoice.n_heads",       dec["num_attention_heads"])
    w.add_uint32 ("vibevoice.n_kv_heads",    dec["num_key_value_heads"])
    head_dim = dec["hidden_size"] // dec["num_attention_heads"]
    w.add_uint32 ("vibevoice.head_dim",      head_dim)
    w.add_uint32 ("vibevoice.vocab_size",    dec["vocab_size"])
    w.add_float32("vibevoice.rope_theta",    float(dec["rope_theta"]))
    w.add_float32("vibevoice.rms_norm_eps",  float(dec["rms_norm_eps"]))
    w.add_uint32 ("vibevoice.acoustic.vae_dim", ac["vae_dim"])
    w.add_array  ("vibevoice.acoustic.encoder_ratios", list(ac["encoder_ratios"]))
    enc_d = ac["encoder_depths"]
    enc_d_list = [int(x) for x in enc_d.split("-")] if isinstance(enc_d, str) else list(enc_d)
    w.add_array  ("vibevoice.acoustic.encoder_depths", enc_d_list)
    dec_d = ac.get("decoder_depths") or list(reversed(enc_d_list))
    if isinstance(dec_d, str):
        dec_d = [int(x) for x in dec_d.split("-")]
    w.add_array  ("vibevoice.acoustic.decoder_depths", list(dec_d))
    if sm:
        w.add_uint32 ("vibevoice.semantic.vae_dim", sm.get("vae_dim", cfg.get("semantic_vae_dim", 128)))
        w.add_array  ("vibevoice.semantic.encoder_ratios", list(sm["encoder_ratios"]))
        sem_d = sm["encoder_depths"]
        sem_d_list = [int(x) for x in sem_d.split("-")] if isinstance(sem_d, str) else list(sem_d)
        w.add_array  ("vibevoice.semantic.encoder_depths", sem_d_list)
    if dh:
        w.add_uint32 ("vibevoice.diffusion.head_layers", dh.get("head_layers", 4))
        w.add_float32("vibevoice.diffusion.ffn_ratio",   float(dh.get("head_ffn_ratio", 3.0)))
        w.add_uint32 ("vibevoice.diffusion.latent",      dh.get("latent_size", 64))
    w.add_uint32 ("vibevoice.sample_rate",           24000)

    # speech scaling factors (also written as tensors above; surface as floats too)
    for n, arr in tensors:
        w.add_tensor(n, arr)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sys.stderr.write(
        f"wrote {out}: {len(tensors)} tensors  (unmapped={len(unmapped)})  "
        f"hidden={dec['hidden_size']} lm_layers={n_lm_layers}+{n_tts_layers} "
        f"vocab={dec['vocab_size']}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
