#!/usr/bin/env python3
"""Quantize a vibevoice gguf in place.

Walks the source gguf and rewrites it with a new dtype for the heavy LM
matmul tensors (attention projs + FFN). Everything else (conv1d kernels,
norms, biases, layer-scale gammas, embeddings, scalars) is preserved as-is
so the rest of the orchestrator keeps working without code changes.

Why selective: ggml_mul_mat handles quantized weights natively (the
Q*_K, Q8_0 row formats are designed for it). But our conv1d wrapper
casts kernels to fp16 inline (`ggml_cast(kernel, GGML_TYPE_F16)`) — it
doesn't dequantize on the fly — so quantizing those would produce
wrong results.

Tensor-name allowlist for quantization (regex):

  lm.blk.*.attn_(q|k|v|o)\\.weight
  lm.blk.*.ffn_(gate|up|down)\\.weight
  tlm.blk.*.attn_(q|k|v|o)\\.weight
  tlm.blk.*.ffn_(gate|up|down)\\.weight

Everything else is rewritten unchanged. Output dtype options: Q8_0
(~50% smaller, near-lossless), Q4_K (~75% smaller, slight quality loss),
Q5_K (~70% smaller, between).

Example:

  scripts/quantize_gguf.py \\
      --src models/vibevoice-asr.gguf \\
      --out models/vibevoice-asr-q4_k.gguf \\
      --type q4_k

After quantizing, you should still pass the closed-loop test:

  build/tests/test_closed_loop  # see tests/test_closed_loop.cpp
"""

import argparse
import re
import sys
from pathlib import Path

import gguf
import numpy as np

QUANT_MAP = {
    "q4_k": gguf.GGMLQuantizationType.Q4_K,
    "q5_k": gguf.GGMLQuantizationType.Q5_K,
    "q6_k": gguf.GGMLQuantizationType.Q6_K,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "f16":  gguf.GGMLQuantizationType.F16,
}

# Tensor names that should be quantized when --type is set. Everything else
# keeps its source dtype. The patterns mirror our gguf naming convention
# (see scripts/convert_vibevoice_to_gguf.py).
QUANTIZABLE = [
    re.compile(r"^(lm|tlm)\.blk\.\d+\.attn_[qkvo]\.weight$"),
    re.compile(r"^(lm|tlm)\.blk\.\d+\.ffn_(gate|up|down)\.weight$"),
    re.compile(r"^lm_head\.weight$"),
]


def should_quantize(name: str) -> bool:
    return any(p.match(name) for p in QUANTIZABLE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src",  required=True, type=Path,
                    help="source gguf (any dtype)")
    ap.add_argument("--out",  required=True, type=Path,
                    help="destination gguf (will be overwritten)")
    ap.add_argument("--type", default="q4_k", choices=sorted(QUANT_MAP),
                    help="quantization for matmul weights (default: q4_k)")
    ap.add_argument("--keep-embed", action="store_true",
                    help="keep tok_embd / lm_head as f16/f32 source dtype "
                         "(some ASR setups prefer this)")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"error: {args.src} not found", file=sys.stderr); return 1

    target_qt = QUANT_MAP[args.type]
    keep_embed_names = {"lm.tok_embd.weight", "lm_head.weight"} if args.keep_embed else set()

    reader = gguf.GGUFReader(str(args.src))
    arch = "vibevoice"
    for f in reader.fields.values():
        if f.name == "general.architecture":
            try:    arch = f.contents()
            except Exception: pass
            break

    writer = gguf.GGUFWriter(str(args.out), arch=arch)

    # ---- copy KV metadata ---------------------------------------------------
    for f in reader.fields.values():
        if f.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue  # written by GGUFWriter
        try:
            value = f.contents()
        except Exception:
            continue
        if value is None: continue
        ft = f.types[0] if f.types else None
        if ft is None: continue
        gv = gguf.GGUFValueType
        try:
            if   ft == gv.STRING: writer.add_string(f.name, value)
            elif ft == gv.UINT8:  writer.add_uint8(f.name, int(value))
            elif ft == gv.INT8:   writer.add_int8(f.name, int(value))
            elif ft == gv.UINT16: writer.add_uint16(f.name, int(value))
            elif ft == gv.INT16:  writer.add_int16(f.name, int(value))
            elif ft == gv.UINT32: writer.add_uint32(f.name, int(value))
            elif ft == gv.INT32:  writer.add_int32(f.name, int(value))
            elif ft == gv.UINT64: writer.add_uint64(f.name, int(value))
            elif ft == gv.INT64:  writer.add_int64(f.name, int(value))
            elif ft == gv.FLOAT32:writer.add_float32(f.name, float(value))
            elif ft == gv.FLOAT64:writer.add_float64(f.name, float(value))
            elif ft == gv.BOOL:   writer.add_bool(f.name, bool(value))
            elif ft == gv.ARRAY:  writer.add_array(f.name, list(value))
            else: pass
        except Exception as e:
            print(f"warning: could not copy KV {f.name} ({ft}): {e}", file=sys.stderr)

    # ---- rewrite tensors ----------------------------------------------------
    n_quantized = 0
    bytes_in = 0
    bytes_out = 0
    for t in reader.tensors:
        src_dtype = t.tensor_type
        # gguf-py stores tensor data row-major in the file; t.data is a 1-D
        # ndarray view. Reshape to logical shape (note: t.shape is gguf
        # convention, fastest-varying first → reverse for numpy).
        np_shape = list(reversed([int(d) for d in t.shape]))
        do_quant = should_quantize(t.name) and t.name not in keep_embed_names
        if do_quant:
            # Dequantize / convert source to f32, reshape to logical, then
            # quantize. gguf.quants.quantize returns the byte representation
            # (uint8 ndarray); GGUFWriter recovers the logical shape from it
            # via quant_shape_from_byte_shape when raw_dtype is set.
            arr = gguf.quants.dequantize(t.data, src_dtype).astype(np.float32)
            arr = arr.reshape(np_shape)
            qbytes = gguf.quants.quantize(arr, target_qt)
            writer.add_tensor(t.name, qbytes, raw_dtype=target_qt)
            n_quantized += 1
            bytes_in  += t.data.nbytes
            bytes_out += qbytes.nbytes
        else:
            # Pass through as-is. Source stored as bytes (raw_dtype set);
            # writer recovers shape from byte-shape via quant_shape_from_byte_shape.
            writer.add_tensor(t.name, np.asarray(t.data), raw_dtype=src_dtype)
            bytes_in  += t.data.nbytes
            bytes_out += t.data.nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    saved = (bytes_in - bytes_out) / 1024**3
    print(f"wrote {args.out}: quantized {n_quantized} tensors → {args.type.upper()}, "
          f"saved {saved:.2f} GB ({100 * (1 - bytes_out/bytes_in):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
