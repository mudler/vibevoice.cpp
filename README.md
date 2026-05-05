# vibevoice.cpp

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team** - the creators of LocalAI, the open-source AI engine that runs any model - LLMs, vision, voice, image, video - on any hardware. No GPU required.

[![Models on HF](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/mudler/vibevoice.cpp-models)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![LocalAI](https://img.shields.io/badge/LocalAI-Run_Locally-orange)](https://github.com/mudler/LocalAI)

A C++ inference engine for Microsoft [VibeVoice](https://github.com/microsoft/VibeVoice), built on
[ggml](https://github.com/ggml-org/ggml). Supports both **TTS** (text-to-speech with voice
cloning) and **ASR** (long-form transcription with diarization).

> Status: TTS + ASR pipelines run end-to-end on real model weights, with
> classifier-free guidance for TTS and a working closed-loop self-test
> (TTS → ASR produces real English transcripts of the synthesized audio).

## Quickstart - prebuilt models

We publish quantized GGUFs at [`mudler/vibevoice.cpp-models`](https://huggingface.co/mudler/vibevoice.cpp-models).
Pull them and you're running in two commands:

```bash
# `--recursive` is mandatory - third_party/ggml is a submodule; without
# it cmake configure fails with 'add_subdirectory called with empty
# directory'. If you've already cloned without it:
#     git submodule update --init --recursive
git clone --recursive https://github.com/mudler/vibevoice.cpp && cd vibevoice.cpp

cmake -B build -DVIBEVOICE_BUILD_TESTS=ON && cmake --build build -j

mkdir -p models && hf download mudler/vibevoice.cpp-models --local-dir models

# TTS
./build/bin/vibevoice-cli tts \
  --model     models/vibevoice-realtime-0.5B-q8_0.gguf \
  --tokenizer models/tokenizer.gguf \
  --voice     models/voice-en-Carter_man.gguf \
  --text "Hello from vibevoice cpp." --out hello.wav

# ASR
./build/bin/vibevoice-cli asr \
  --model     models/vibevoice-asr-q8_0.gguf \
  --tokenizer models/tokenizer.gguf \
  --audio     hello.wav
```

## Quickstart - convert from upstream

If you want to roll your own (different quant, different voice, etc.):

```bash
# Tokenizer
hf download Qwen/Qwen2.5-0.5B --local-dir models/qwen2.5
python scripts/convert_tokenizer.py --src models/qwen2.5 --out models/tokenizer.gguf

# Realtime TTS model (~1.9 GB → ~3.8 GB fp32 gguf)
hf download microsoft/VibeVoice-Realtime-0.5B --local-dir models/vibevoice-realtime-0.5B
python scripts/convert_vibevoice_to_gguf.py \
  --src models/vibevoice-realtime-0.5B \
  --out models/vibevoice-realtime-0.5B.gguf

# (optional) quantize to Q8_0 - ~50% smaller, no quality loss in the closed-loop test
python scripts/quantize_gguf.py \
  --src models/vibevoice-realtime-0.5B.gguf \
  --out models/vibevoice-realtime-0.5B-q8_0.gguf \
  --type q8_0

# Voice prompt (one of the en-*/de-*/fr-* etc. files in the upstream demo)
curl -sL -o /tmp/voice.pt \
  https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model/en-Carter_man.pt
python scripts/convert_voice_to_gguf.py --src /tmp/voice.pt --out models/voice.gguf

./build/bin/vibevoice-cli tts \
  --model models/vibevoice-realtime-0.5B-q8_0.gguf \
  --tokenizer models/tokenizer.gguf \
  --voice models/voice.gguf \
  --text "Hello from vibevoice cpp." \
  --out hello.wav \
  --cfg 3.0 --steps 20 --max-frames 40 --verbose
```

## Closed-loop sanity (TTS → ASR)

```bash
# 1. synthesize
./build/bin/vibevoice-cli tts \
    --model models/vibevoice-realtime-0.5B-q8_0.gguf \
    --voice models/voice-en-Carter_man.gguf \
    --tokenizer models/tokenizer.gguf \
    --text "Hello world this is a test of the synthesis system." \
    --seed 12345 --out say.wav

# 2. transcribe back
./build/bin/vibevoice-cli asr \
    --model models/vibevoice-asr-q8_0.gguf \
    --tokenizer models/tokenizer.gguf \
    --audio say.wav
# -> [{"Start":0,"End":2.8,"Speaker":0,"Content":"Hello world, this is a test of the synthesis system."}]
```

This is the same roundtrip codified as `tests/test_closed_loop.cpp` - see
[`docs/conversion.md`](docs/conversion.md) for how to wire it into ctest.

## Quickstart - voice cloning (1.5B)

The `microsoft/VibeVoice-1.5B` model conditions on a raw reference WAV
at synthesis time — no separate voice gguf needed. Hand it ~5 s of any
speaker and it'll synthesize new text in that voice.

```bash
hf download microsoft/VibeVoice-1.5B --local-dir models/vibevoice-1.5B
python scripts/convert_vibevoice_to_gguf.py \
  --src models/vibevoice-1.5B \
  --out models/vibevoice-1.5B.gguf
# shrink the gguf. Two recommended profiles for the 1.5B path:
#
#   1) Q8_0 across the board: 11 GB -> 6.8 GB, no measurable recall
#      hit on the closed-loop benchmark.
./build/bin/vibevoice-quantize \
  --src  models/vibevoice-1.5B.gguf \
  --out  models/vibevoice-1.5B-q8_0.gguf \
  --type q8_0
#
#   2) Mixed: 11 GB -> 6.5 GB, same recall as fp32. FFN at Q6_K, attn
#      at Q5_K, lm_head at Q8_0. Plain Q5_K across the board collapses
#      this model (recall drops to 22%) — FFN weights are the most
#      quant-sensitive piece, attention tolerates Q5_K well.
./build/bin/vibevoice-quantize \
  --src           models/vibevoice-1.5B.gguf \
  --out           models/vibevoice-1.5B-mixed.gguf \
  --type          q6_k    \
  --attn-type     q5_k    \
  --lm-head-type  q8_0

./build/bin/vibevoice-cli tts \
  --model     models/vibevoice-1.5B-q8_0.gguf \
  --tokenizer models/tokenizer.gguf \
  --ref-audio reference-voice.wav \
  --text      "Hello world, this is a test of voice cloning." \
  --out       cloned.wav
```

The same `tts` subcommand handles both model families: pass `--voice
<voice.gguf>` for the realtime-0.5B path, or `--ref-audio <wav>` for
runtime voice cloning on the 1.5B path. The CLI dispatches based on
the loaded gguf's variant metadata; the two flags are mutually
exclusive.

Note: voice cloning **only** works with the 1.5B variant. The
realtime-0.5B weights ship without encoders, so they can't process a
reference WAV at runtime — they only consume pre-baked voice gguf
files (see `scripts/convert_voice_to_gguf.py`).

## Quickstart - ASR

```bash
# ASR model (~14 GB safetensors → ~33 GB fp32 gguf - needs lots of disk)
hf download microsoft/VibeVoice-ASR --local-dir models/vibevoice-asr
python scripts/convert_vibevoice_to_gguf.py \
  --src models/vibevoice-asr \
  --out models/vibevoice-asr.gguf

./build/bin/vibevoice-cli asr \
  --model models/vibevoice-asr.gguf \
  --tokenizer models/tokenizer.gguf \
  --audio my-clip.wav
# -> [{"Start":0.0,"End":6.0,"Speaker":0,"Content":"..."}]
```

## Benchmarks

Long-form ASR on the multi-speaker
[Microsoft VibeVoice samples](https://microsoft.github.io/VibeVoice/),
through `vibevoice-cli asr` with the Q4_K-quantized 7B model. RTF is
inference time / audio duration; lower is faster.

| Sample        | Audio duration | Backend / hardware         | Model | Load time | Inference | RTF       |
|---------------|---------------:|----------------------------|-------|----------:|----------:|----------:|
| `2p_argument` | 68.5 s         | CPU - AMD Ryzen 9950X3D    | Q4_K  | 5.9 s     | 150.4 s   | **2.195** |
| `2p_argument` | 68.5 s         | CUDA - NVIDIA GB10         | Q4_K  | 2.2 s     | 28.0 s    | **0.408** |

Sample transcripts and timing are produced by `vibevoice-cli asr ... 2>&1 | grep "asr: timing"`
(timing breakdown was added in the same release as the backend dispatch).
Reproduce locally:

```bash
hf download mudler/vibevoice.cpp-models --local-dir models
ffmpeg -i sample.mp3 -ac 1 -ar 24000 sample.wav
VIBEVOICE_BACKEND=cuda ./build/bin/vibevoice-cli asr \
    --model models/vibevoice-asr-q4_k.gguf \
    --tokenizer models/tokenizer.gguf \
    --audio sample.wav --max-new-tokens 8192
```

## Tests

```bash
ctest --test-dir build --output-on-failure   # 21 ctest targets
```

For the real-weight tests:

```bash
VIBEVOICE_MODEL=models/vibevoice-realtime-0.5B.gguf \
VIBEVOICE_TOKENIZER=models/tokenizer.gguf \
  ctest --test-dir build --output-on-failure -j 2
```

## Embedding from Go (purego)

`vibevoice.cpp` ships a flat C ABI in [`include/vibevoice_capi.h`](include/vibevoice_capi.h)
designed for `dlopen` / `purego.RegisterLibFunc` consumers. It mirrors
the layout LocalAI's `qwen3-tts-cpp` Go backend uses:

```c
int  vv_capi_load(const char* tts_model, const char* asr_model,
                  const char* tokenizer, const char* voice, int n_threads);
// `voice_path` is for realtime-0.5B, `ref_audio_path` is for 1.5B
// (runtime voice cloning); pass NULL for whichever the loaded model
// doesn't need.
int  vv_capi_tts(const char* text, const char* voice_path, const char* ref_audio_path,
                 const char* dst_wav, int steps, float cfg, int max_speech_frames,
                 uint32_t seed);
int  vv_capi_asr(const char* src_wav, char* out_json, size_t out_capacity,
                 int max_new_tokens);
void vv_capi_unload(void);
```

Build the shared library and call it from Go:

```go
import "github.com/ebitengine/purego"

var (
    Load     func(tts, asr, tok, voice string, threads int) int
    TTS      func(text, voice, refAudio, dst string, steps int, cfg float32, maxFrames int, seed uint32) int
    ASR      func(src string, out []byte, outCap uint64, maxTok int) int
    Unload   func()
)

lib, _ := purego.Dlopen("./libvibevoice.so", purego.RTLD_NOW|purego.RTLD_GLOBAL)
purego.RegisterLibFunc(&Load,   lib, "vv_capi_load")
purego.RegisterLibFunc(&TTS,    lib, "vv_capi_tts")
purego.RegisterLibFunc(&ASR,    lib, "vv_capi_asr")
purego.RegisterLibFunc(&Unload, lib, "vv_capi_unload")
```

Build the shared library with `cmake -DVIBEVOICE_SHARED=ON`.

## Why

VibeVoice's official runtime is Python + Transformers + a vLLM plugin. `vibevoice.cpp` provides:

- A native CPU runtime with no Python at inference time
- Free CUDA / Metal / Vulkan support via ggml backends
- A single `.gguf` weight file + a single binary
- A flat C ABI (`include/vibevoice_capi.h`) for embedding via dlopen / purego / cgo

## Build

```bash
git clone --recursive https://example.com/vibevoice.cpp
cd vibevoice.cpp
cmake -B build -DVIBEVOICE_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Layout

See [`docs/`](docs/) and the project plan for the full architecture.

## Author

Ettore Di Giacinto ([@mudler](https://github.com/mudler)) - also the
maintainer of [LocalAI](https://github.com/mudler/LocalAI). PRs welcome.

## License

MIT - see [LICENSE](LICENSE). Copyright © 2026 Ettore Di Giacinto.
The model weights remain under their upstream license
([microsoft/VibeVoice](https://huggingface.co/microsoft/VibeVoice-1.5B)).
