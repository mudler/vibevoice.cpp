# CUDA / GPU build

`vibevoice.cpp` builds against the embedded `ggml` submodule's CUDA
backend — no source changes needed, just a CMake flag.

> **Status:** the CMake flag enables the CUDA build (verified on a DGX
> Spark / GB10 host). However our current compute path uses
> `ggml_graph_compute_with_ctx`, which is **CPU-only**. ggml builds the
> CUDA plugin (`libggml-cuda.so`) but our binary doesn't dispatch graphs
> to it yet. Real GPU compute requires switching to the
> `ggml_backend_*` API (tracked in the project task list).

## Build

```bash
cmake -B build \
    -DVIBEVOICE_BUILD_TESTS=ON \
    -DVIBEVOICE_BUILD_EXAMPLES=ON \
    -DVIBEVOICE_GGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Forwarded ggml backend flags (any combination):

| CMake flag                     | What it enables                               |
| ------------------------------ | --------------------------------------------- |
| `-DVIBEVOICE_GGML_CUDA=ON`     | NVIDIA CUDA (`ggml-cuda`)                     |
| `-DVIBEVOICE_GGML_METAL=ON`    | Apple Metal (`ggml-metal`)                    |
| `-DVIBEVOICE_GGML_VULKAN=ON`   | cross-vendor Vulkan compute (`ggml-vulkan`)   |
| `-DVIBEVOICE_GGML_HIPBLAS=ON`  | AMD ROCm (`ggml-hipblas`)                     |

## Smoke test

```bash
# Pull the published quantized bundle (Q8_0 ggufs, ~15 GB).
mkdir -p models
hf download mudler/vibevoice.cpp-models --local-dir models

# TTS — should run substantially faster on GPU than CPU.
./build/bin/vibevoice-cli tts \
    --model     models/vibevoice-realtime-0.5B-q8_0.gguf \
    --voice     models/voice-en-Carter_man.gguf \
    --tokenizer models/tokenizer.gguf \
    --text "Hello from CUDA." --out hello.wav

# ASR
./build/bin/vibevoice-cli asr \
    --model     models/vibevoice-asr-q8_0.gguf \
    --tokenizer models/tokenizer.gguf \
    --audio     hello.wav

# Full closed-loop ctest (~16 GB total; uses ~13 GB VRAM with q8_0 ASR)
VIBEVOICE_TTS_MODEL=$PWD/models/vibevoice-realtime-0.5B-q8_0.gguf \
VIBEVOICE_VOICE=$PWD/models/voice-en-Carter_man.gguf \
VIBEVOICE_ASR_MODEL=$PWD/models/vibevoice-asr-q8_0.gguf \
VIBEVOICE_TOKENIZER=$PWD/models/tokenizer.gguf \
VIBEVOICE_CLI=$PWD/build/bin/vibevoice-cli \
ctest --test-dir build -R "test_closed_loop|test_long_form_asr|test_capi|test_encoder_chunked_parity" --output-on-failure
```

## Notes

- ggml dispatches per-op to the available backend; ops that aren't
  implemented on the chosen backend fall back to CPU automatically.
- The `vv_capi_load(... n_threads)` argument is a CPU-thread count; on
  GPU the kernel grids do the parallelism. Pass any reasonable value
  (4 is fine).
- We don't pin the engine to a specific device. Set
  `CUDA_VISIBLE_DEVICES=0` (or similar) before launching `vibevoice-cli`
  if you want it on a specific GPU.
- The closed-loop / long-form / capi tests run unmodified — they just
  exercise the same `vibevoice-cli` binary and hit faster matmul paths
  through ggml.
