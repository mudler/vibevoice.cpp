#ifndef VIBEVOICE_BACKEND_HPP
#define VIBEVOICE_BACKEND_HPP

// Process-wide ggml backend selection + per-graph compute helper.
//
// vibevoice.cpp originally used ggml_graph_compute_with_ctx, the CPU-only
// short-cut. This module replaces that with the backend API so the same
// graphs can run on CUDA / Metal / Vulkan / hipBLAS when those are built
// in (via VIBEVOICE_HAVE_CUDA / _METAL / _VULKAN / _HIPBLAS at compile
// time and dlopen-loaded at runtime via ggml_backend_load_all).
//
// Selection order at runtime:
//   1. The backend named by VIBEVOICE_BACKEND env var (case-insensitive),
//      one of: cuda, metal, vulkan, hipblas, cpu.
//   2. The first GPU-class backend ggml_backend_dev_count reports.
//   3. CPU.
//
// Single global lazy-init: the first call to backend() picks one and
// keeps it for the process lifetime. Pass VIBEVOICE_BACKEND=cpu to
// force CPU even when GPU backends are available.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <string>

namespace vv {

// Returns the singleton backend. Initializes on first call. Never null —
// CPU is the always-available fallback. Lifetime is the process; freed
// at exit.
ggml_backend_t backend();

// Human-readable name of the active backend (e.g. "CUDA", "CPU").
const char* backend_name();

// Compute a graph on the active backend. Allocates intermediate tensors
// on the backend's buffer using a lazily-created `ggml_gallocr_t`, then
// dispatches the compute. Returns true on success.
//
// All graph leaf tensors (inputs + weights) must already live on a
// buffer compatible with the backend (typically allocated via
// allocate_ctx_tensors). For now, weights are CPU-resident and the
// compute ctx is allocated on the active backend; ops that need GPU
// data perform implicit transfers, which is fine for v1 correctness.
bool compute_graph(ggml_cgraph* graph);

// Allocate all tensors in `ctx` on a buffer compatible with the active
// backend. Use after building a no_alloc ggml_context: allocate, then
// memcpy data into each tensor via ggml_backend_tensor_set.
//
// Returns the allocated buffer, or null on failure. The buffer is owned
// by the caller and must outlive any tensor reads/writes; freed with
// ggml_backend_buffer_free.
ggml_backend_buffer_t allocate_ctx_tensors(ggml_context* ctx);

}  // namespace vv

#endif  // VIBEVOICE_BACKEND_HPP
