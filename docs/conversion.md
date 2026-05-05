# Converting VibeVoice models to GGUF

`vibevoice.cpp` ships three Python tools that turn upstream Microsoft VibeVoice
artifacts into the GGUF format the C++ runtime consumes.

## Pipeline overview

```
  Hugging Face
  ┌─────────────────────────────┐
  │ Qwen/Qwen2.5-0.5B            │── tokenizer.json ─► convert_tokenizer.py ─► tokenizer.gguf
  │ microsoft/VibeVoice-…        │── safetensors    ─► convert_vibevoice_to_gguf.py ─► …gguf
  │ microsoft/VibeVoice          │── voices/*.pt    ─► convert_voice_to_gguf.py ─► voice.gguf
  └─────────────────────────────┘
                                                                 │
                                              (optional)          │
                                                                 ▼
                                                        quantize_gguf.py ─► …-q8_0.gguf
```

## `scripts/convert_tokenizer.py`

Wraps a Hugging Face tokenizer (`tokenizer.json`) into a GGUF that
`src/tokenizer.cpp` can load directly. The tokenizer is required by every
TTS / ASR run; you only need to convert it once and reuse it across model
checkpoints.

```bash
hf download Qwen/Qwen2.5-0.5B --local-dir models/qwen2.5
python scripts/convert_tokenizer.py \
    --src models/qwen2.5/tokenizer.json \
    --out models/tokenizer.gguf
```

VibeVoice ASR re-uses three Qwen2.5 vision-token IDs to mark speech regions
in the prompt:

| ID     | Token                  | Role        |
| ------ | ---------------------- | ----------- |
| 151646 | `<|object_ref_start|>` | speech_start |
| 151647 | `<|object_ref_end|>`   | speech_end   |
| 151648 | `<|box_start|>`        | speech_pad   |

These are surfaced through the standard tokenizer (no patching required).

## `scripts/convert_vibevoice_to_gguf.py`

Walks a VibeVoice checkpoint directory (`config.json` + sharded
`model*.safetensors`), maps PyTorch tensor names to our GGUF naming
convention via an ordered set of regex rewrites, and writes a single GGUF
with all required metadata.

```bash
hf download microsoft/VibeVoice-Realtime-0.5B --local-dir models/vv-rt-0.5b
python scripts/convert_vibevoice_to_gguf.py \
    --src models/vv-rt-0.5b \
    --out models/vibevoice-realtime-0.5B.gguf
```

Variants are auto-detected from the `architectures` field in `config.json`:

| Variant         | Source repo                                                                       | Notes                                                                |
| --------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `realtime-0.5b` | `microsoft/VibeVoice-Realtime-0.5B`                                               | TTS, `vibevoice-cli tts --voice <voice.gguf>`                        |
| `1.5b`          | `microsoft/VibeVoice-1.5B`                                                        | TTS with runtime voice cloning, `vibevoice-cli tts --ref-audio <wav>`|
| `asr-7b`        | `microsoft/VibeVoice-ASR`                                                         | ASR, `vibevoice-cli asr`                                             |

### GGUF tensor naming

Both variants share a common naming scheme that mirrors the upstream
PyTorch module hierarchy. The notable patterns:

| HF / safetensors name (excerpt)                                                              | gguf tensor name                                |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `model.language_model.embed_tokens.weight`                                                   | `lm.tok_embd.weight`                            |
| `model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}`                     | `lm.blk.{i}.attn_{q,k,v,o}.{weight,bias}`       |
| `model.language_model.layers.{i}.{input,post_attention}_layernorm.weight`                    | `lm.blk.{i}.{attn,ffn}_norm.weight`             |
| `model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight`                             | `lm.blk.{i}.ffn_{gate,up,down}.weight`          |
| `model.tts_language_model.layers.{j}.…`  (realtime only — split LM)                          | `tlm.blk.{j}.…`                                 |
| `model.tts_language_model.norm.weight`   (realtime only)                                     | `tlm.output_norm.weight`                        |
| `model.tts_input_types.weight`            (realtime only)                                    | `tts.input_types.weight`                        |
| `model.acoustic_tokenizer.encoder.…`                                                          | `at.enc.…`                                      |
| `model.acoustic_tokenizer.decoder.…`     (TTS variants — realtime + 1.5b)                    | `at.dec.…`                                      |
| `model.semantic_tokenizer.…`             (ASR + 1.5b)                                        | `st.…`                                          |
| `model.acoustic_connector.{linear1,norm,linear2}.…`                                          | `ac.{fc1,norm,fc2}.…`                           |
| `model.semantic_connector.…`             (ASR + 1.5b)                                        | `sc.…`                                          |
| `model.prediction_head.…`                (TTS variants — realtime + 1.5b)                    | `dh.…`                                          |
| `model.tts_eos_classifier.…`             (realtime only)                                     | `eos.…`                                         |
| `lm_head.weight`                          (ASR + 1.5b; for 1.5b synthesised from tied embed) | `lm_head.weight`                                |

Pass `--strict` to fail the conversion if any source tensor key is left
unmapped — useful when validating against a new upstream release.

### Encoder ratio quirk

The acoustic / semantic encoders apply downsample strides in **reversed**
order vs the decoder, mirroring upstream
`vibevoice/modular/modular_vibevoice_tokenizer.py:713`:

```python
self.ratios = list(reversed(config.ratios))   # encoder
self.ratios = config.ratios                   # decoder
```

That means `down_1` has stride `ratios[N-1]`, not `ratios[0]`. The C++
loader handles this in `src/acoustic_tokenizer.cpp::load_encoder`.

## `scripts/convert_voice_to_gguf.py`

VibeVoice TTS conditions on a precomputed KV cache for a known speaker
(the "voice prompt"). Upstream ships these as `.pt` files under
`microsoft/VibeVoice/demo/voices/streaming_model/`. The converter reads a
voice `.pt` and writes a small GGUF (~5–10 MB) containing per-layer K/V
tensors for both LM stacks plus the negative caches used by classifier-free
guidance.

```bash
curl -sLO https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model/en-Carter_man.pt
python scripts/convert_voice_to_gguf.py \
    --src en-Carter_man.pt \
    --out models/voice-en-Carter_man.gguf
```

Voice GGUFs are model-agnostic in shape but specific to the realtime
TTS architecture (`realtime-0.5b`); they are not used by ASR.

## `scripts/quantize_gguf.py`

Optional. Takes any vibevoice GGUF and produces a quantized variant by
selectively quantizing only the LM matmul weights — attention q/k/v/o,
FFN gate/up/down, and `lm_head` — to a target dtype.  Everything else
(conv1d kernels, RMSNorm scales, biases, layer-scale gammas, embeddings,
small scalars) passes through unchanged.

Why selective: `ggml_mul_mat` handles quantized weights natively, but
the conv1d wrapper in `src/conv1d.cpp` casts kernels to fp16 inline
(`ggml_cast(kernel, GGML_TYPE_F16)`) — it does not dequantize on the
fly, so quantizing those would silently corrupt the convolution outputs.

```bash
python scripts/quantize_gguf.py \
    --src models/vibevoice-asr.gguf \
    --out models/vibevoice-asr-q8_0.gguf \
    --type q8_0
```

Available dtypes (limited by what `gguf-py` can encode in pure Python):

| `--type` | Bits/elem (matmul) | Notes                                              |
| -------- | ------------------ | -------------------------------------------------- |
| `q8_0`   | 8.5                | ~50–60 % size reduction; near-lossless             |
| `q5_k`*  | 5.5                | Not implemented in gguf-py (needs libggml.so)      |
| `q6_k`*  | 6.5                | Not implemented in gguf-py (needs libggml.so)      |
| `q4_k`*  | 4.5                | Not implemented in gguf-py (needs libggml.so)      |
| `f16`    | 16                 | No quantization — just dtype cast                  |

`*` selectable in the script signature; will raise `NotImplementedError`
until we add a libggml-backed backend.

## Closed-loop verification

After conversion (and optional quantization), validate the bundle with:

```bash
cmake -B build -DVIBEVOICE_BUILD_TESTS=ON -DVIBEVOICE_TEST_LARGE=ON \
    && cmake --build build -j

VIBEVOICE_TTS_MODEL=$PWD/models/vibevoice-realtime-0.5B.gguf \
VIBEVOICE_VOICE=$PWD/models/voice-en-Carter_man.gguf \
VIBEVOICE_ASR_MODEL=$PWD/models/vibevoice-asr.gguf \
VIBEVOICE_TOKENIZER=$PWD/models/tokenizer.gguf \
VIBEVOICE_CLI=$PWD/build/bin/vibevoice-cli \
ctest --test-dir build -R test_closed_loop --output-on-failure
```

The test synthesizes a known phrase with TTS, transcribes it with ASR, and
asserts ≥80 % source-word recall in the recovered transcript. With our Q8_0
ggufs it consistently lands at 100 %.
