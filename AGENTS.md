# Maintainer's guide — vibevoice.cpp

A pragmatic guide for whoever is poking at this repo next. Concise on
purpose; the README covers what users see.

## What this project is

A C++/ggml port of Microsoft VibeVoice. One binary (`vibevoice-cli`)
does both **TTS** (text → 24 kHz WAV with voice cloning) and **ASR**
(WAV → JSON transcript). Built on stock ggml (no fork). Public C API in
[`include/vibevoice.h`](include/vibevoice.h) so other projects can
embed it via dlopen / purego / cgo.

Reference impls we trust, in order:
1. **`microsoft/VibeVoice`** (`vibevoice/modular/`) — the one that was
   actually trained. Single source of truth on weights + math.
2. **`Blaizzy/mlx-audio`** (`mlx_audio/{stt,tts}/models/vibevoice…/`) —
   closest analog to what we're writing. Useful when upstream PyTorch
   does something tricky and we want a non-PyTorch confirmation.
3. **`transformers/models/vibevoice_*/`** — a refactored re-port. Differs
   from upstream in subtle places (sampling formula, processor API).
   Cross-check before trusting.

When you suspect a numerical bug in our code, dump the matching tensor
from the chosen reference and diff. There's a template at
`/tmp/asr_ref_compare.py` (not committed; lives only on the dev box) that
shows how to load just the encoder + connector shards from the 7B ASR
checkpoint and run them standalone.

## Layout

```
include/vibevoice.h          # public C API (purego / dlopen target)
src/
  vibevoice.cpp              # public-API impl
  vibevoice_tts.{hpp,cpp}    # TTS orchestrator (M5)
  vibevoice_asr.{hpp,cpp}    # ASR orchestrator (M6)
  qwen2.{hpp,cpp}            # Qwen2 transformer block + GQA + KV cache
  acoustic_tokenizer.{hpp,cpp}# VAE encoder + decoder
  diffusion_head.{hpp,cpp}   # DiffusionHead + DPM-Solver
  conv1d.{hpp,cpp}           # SConv1d / SConvTranspose1d
  rms_norm.hpp               # ConvRMSNorm
  model_loader.{hpp,cpp}     # gguf reader (mmap → name → ggml_tensor)
  tokenizer.{hpp,cpp}        # vendored Qwen2 byte-level BPE
  audio_io.{hpp,cpp}         # dr_wav wrap + linear resampler
scripts/
  convert_tokenizer.py
  convert_vibevoice_to_gguf.py
  convert_voice_to_gguf.py
  quantize_gguf.py
tests/
  test_*.cpp                 # ~14 ctests; SKIP_RETURN_CODE=77 = skip
  fixtures/tokenizer.gguf    # tiny tokenizer fixture (committed)
docs/
  conversion.md              # tensor naming + quant notes
.github/workflows/ci.yml     # build+test (Linux+macOS) + closed-loop on dispatch
third_party/ggml             # pinned submodule
```

## Build

```bash
git clone --recursive <repo> && cd vibevoice.cpp
cmake -B build -DVIBEVOICE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CMake options:
- `VIBEVOICE_BUILD_TESTS` — register ctests
- `VIBEVOICE_TEST_LARGE` — enable model-dependent tests (closed-loop, long-form). They still skip 77 if env vars aren't set, so this is safe to leave on.
- `VIBEVOICE_BUILD_EXAMPLES` (default ON) — `vibevoice-cli`
- `VIBEVOICE_GGML_CUDA` / `VIBEVOICE_GGML_METAL` — pass through to the ggml submodule.

## Tests at a glance

| File                            | What it does                                            | Needs models? |
| ------------------------------- | ------------------------------------------------------- | :-----------: |
| `test_smoke`                    | Lib loads, version string is non-empty                  | no            |
| `test_audio_io`                 | dr_wav round-trip                                       | no            |
| `test_tokenizer`                | Qwen2 BPE id-level parity vs HF on a fixture            | no            |
| `test_rope`                     | RoPE cos/sin tables vs PyTorch                          | no            |
| `test_qwen2_block`              | Qwen2 forward pass numerics                             | no            |
| `test_sconv1d`                  | Causal conv1d / convtranspose1d numerics                | no            |
| `test_block1d`                  | ConvNeXt Block1D forward                                | no            |
| `test_acoustic`                 | Encoder + decoder forward on tiny random weights        | no            |
| `test_diffusion_head`           | TimestepEmbedder + DiffusionHead forward                | no            |
| `test_dpm_solver`               | DPM-Solver++ multistep schedule                         | no            |
| `test_load_realtime`            | Real 0.5B gguf opens cleanly                            | yes (env)     |
| `test_e2e_tts`                  | Real TTS produces non-silent / non-clipped audio        | yes (env)     |
| `test_e2e_asr`                  | Real ASR encoder runs + tone smoke                      | yes (env)     |
| `test_closed_loop`              | TTS → ASR roundtrip; ≥80 % source-word recall           | **yes**       |
| `test_long_form_asr`            | 65 s audio (TTS×N) round-trips with multi-segment match | **yes**       |

Env vars for the model-dependent tests (set whichever ones you need):

```
VIBEVOICE_MODEL        # alias used by older tests; .gguf path
VIBEVOICE_TTS_MODEL    # closed-loop / long-form TTS path
VIBEVOICE_ASR_MODEL    # closed-loop / long-form ASR path
VIBEVOICE_VOICE        # voice-en-Carter_man.gguf or similar
VIBEVOICE_TOKENIZER    # tokenizer.gguf
VIBEVOICE_CLI          # absolute path to build/bin/vibevoice-cli
```

A ready-to-use bundle is published at
[`mudler/vibevoice.cpp-models`](https://huggingface.co/mudler/vibevoice.cpp-models)
(Q8_0 ggufs + voices + tokenizer, ~15 GB). The CI workflow's `closed-loop`
job pulls from there on `workflow_dispatch`.

## Naming + conventions

### gguf tensor names (mirror upstream PyTorch hierarchy)

| HF / safetensors prefix              | gguf prefix         |
| ------------------------------------ | ------------------- |
| `model.language_model.…`             | `lm.…`              |
| `model.tts_language_model.…`         | `tlm.…`             |
| `model.acoustic_tokenizer.encoder.…` | `at.enc.…`          |
| `model.acoustic_tokenizer.decoder.…` | `at.dec.…`          |
| `model.semantic_tokenizer.encoder.…` | `st.…`              |
| `model.acoustic_connector.…`         | `ac.…`              |
| `model.semantic_connector.…`         | `sc.…`              |
| `model.prediction_head.…`            | `dh.…`              |
| `model.tts_eos_classifier.…`         | `eos.…`             |
| `lm_head.weight`                     | `lm_head.weight`    |

Layer-internal naming follows `<prefix>.blk.<i>.attn_{q,k,v,o}.{weight,bias}`,
`<prefix>.blk.<i>.{attn,ffn}_norm.weight`, `<prefix>.blk.<i>.ffn_{gate,up,down}.weight`.

The full mapping with regex is in `scripts/convert_vibevoice_to_gguf.py`.

### Special token IDs (Qwen2.5 vision tokens repurposed for speech)

| ID     | Token                  | Role         |
| ------ | ---------------------- | ------------ |
| 151646 | `<|object_ref_start|>` | speech_start |
| 151647 | `<|object_ref_end|>`   | speech_end   |
| 151648 | `<|box_start|>`        | speech_pad   |

The ASR prompt template puts a `speech_pad` for every 3200-sample window
of input audio. Speech features are spliced into the input embeddings at
those positions before the LM prefill.

## Gotchas / past-bug archaeology

These bit us before. They will probably bite again.

1. **Encoder ratios are reversed vs decoder ratios.** Upstream
   `vibevoice/modular/modular_vibevoice_tokenizer.py:713`:
   ```python
   self.ratios = list(reversed(config.ratios))   # encoder
   self.ratios = config.ratios                   # decoder
   ```
   Our `acoustic_tokenizer.cpp::load_encoder` reverses; `load_decoder`
   does not. Mismatching this gives `[Noise]` transcripts because the
   encoder ends up running K=4 convs with stride=8 → negative pad_total
   → garbled latents. Magnitudes look ~right (std ~1.5 at the connector,
   matching the reference) which makes it nasty to debug. The reference
   PyTorch encoder produces the same magnitude; magnitude is *not* a
   useful signal for this class of bug.

2. **Speech features have ~100× the magnitude of text-token embeddings**
   and that's *correct*. Qwen2.5 token embeddings have std≈0.011, our
   acoustic+semantic connector sum has std≈1.5. The model was trained
   that way. Don't normalize. (We did, it broke things.)

3. **fp16 gguf load needs a separate ggml_context for promotions.** The
   gguf-owned ctx is sized to the data exactly, no slack. If you write
   anything to it (e.g., promote small fp16 → fp32 norm scales),
   `ggml_new_object` aborts.
   [`ModelLoader::promote_small_f16_to_f32`](src/model_loader.cpp)
   allocates a sibling `promote_ctx_` for that.

4. **ggml `mem_size` for compute pools must scale with sequence /
   sample length.** Hardcoded values silently work on small inputs and
   abort on big ones. The encoder uses ~64 KB / sample; the LM prefill
   uses ~32 MB / token. Both are scaled in `vibevoice_asr.cpp`.

5. **TTS is non-deterministic without `--seed`.** The closed-loop test
   pins `--seed=12345` for that reason.

6. **The conv1d wrapper inline-casts kernels to fp16** (see
   `src/conv1d.cpp::sconv1d_causal`). That means quantized conv kernels
   would silently produce wrong output, so `scripts/quantize_gguf.py`
   only quantizes LM matmul weights. If you want to quantize convs,
   teach the wrapper to dequantize first.

7. **`pop3sen` `ggml_conv_1d_dw` reshapes to ne[2]=1 unconditionally,**
   so depthwise conv only works for batch size 1. Our codepath only
   ever runs B=1 so this is fine, but keep it in mind.

8. **Acoustic encoder's `disable_last_norm: true`** in the official
   config means there's no final RMSNorm between the last conv stage
   and the head. Our loader sets `w.final_norm = nullptr` if the tensor
   is absent; the forward pass skips it.

## Adding a new test

```bash
cp tests/test_smoke.cpp tests/test_my_thing.cpp
# edit it, return 0 on pass, 77 to mean "skipped"
echo 'vv_add_test(test_my_thing)' >> tests/CMakeLists.txt
# if it depends on env vars, add it to the SKIP_RETURN_CODE block.
```

For tests that need real model weights, follow the
`tests/test_closed_loop.cpp` pattern: shell out to `vibevoice-cli` via
`system()` so each model gets its own short-lived process. Loading
multiple gguf models in one ggml context will fail because of the
per-context memory pool sizing.

## Adding a new converter

The converter pipeline is:

```
HF safetensors → scripts/convert_vibevoice_to_gguf.py → vibevoice-*.gguf
                                                       ↓ (optional)
                                               scripts/quantize_gguf.py
                                                       ↓
                                               vibevoice-*-q8_0.gguf
```

Always run with `--strict` to catch unmapped source keys. New regex
mappings go in `KEY_REWRITES` near the top of
`convert_vibevoice_to_gguf.py`.

## Releasing model weights

1. Convert from upstream HF safetensors with
   `scripts/convert_vibevoice_to_gguf.py`.
2. Quantize with `scripts/quantize_gguf.py --type q8_0`.
3. Run the closed-loop test against the quantized output to confirm
   no quality regression.
4. `hf upload-large-folder mudler/vibevoice.cpp-models <staging-dir>` —
   atomic commit at end, resumable via `.cache/huggingface/`.

## Useful third-party reference repos

Clone these once and keep them around — most "this should have worked"
debugging starts by diffing our impl against one of them.

- [`microsoft/VibeVoice`](https://github.com/microsoft/VibeVoice) — the
  trained-against PyTorch modeling code and its processor. **Single
  source of truth on shapes, ordering, and prompt format.**
- [`Blaizzy/mlx-audio`](https://github.com/Blaizzy/mlx-audio) — closest
  non-PyTorch port. Useful when upstream does something tricky and you
  want a confirmation in a different framework.
- [`huggingface/transformers`](https://github.com/huggingface/transformers)
  models `vibevoice_asr` / `vibevoice_acoustic_tokenizer` — refactored
  re-port; differs from upstream in subtle places (sampling formula,
  processor API). Cross-check before trusting.
- The upstream HF model checkpoints
  ([`microsoft/VibeVoice-Realtime-0.5B`](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B),
  [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR))
  — for tensor-by-tensor numerical comparisons.

## Style

- C++17, no exceptions in the public API (return `vv_status` codes).
- One translation unit per logical component; keep `vibevoice.cpp`
  thin — it's the C-API shim and should mostly forward into the
  `vv::` C++ namespace.
- Don't add comments for the *what*, only for non-obvious *why* — the
  list above is what the *why* category looks like.
