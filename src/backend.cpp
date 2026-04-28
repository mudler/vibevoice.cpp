#include "backend.hpp"

#include "common.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace vv {

namespace {

ggml_backend_t  g_backend  = nullptr;
ggml_gallocr_t  g_gallocr  = nullptr;
std::string     g_name;
std::once_flag  g_once;

std::string lower(const std::string& s) {
    std::string r = s;
    for (auto& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

// Map a user-facing backend name to a ggml device type. Returns true if
// the name was recognized (even when no matching device is registered).
bool name_to_type(const std::string& name_lower, enum ggml_backend_dev_type* out) {
    if (name_lower == "cpu")      { *out = GGML_BACKEND_DEVICE_TYPE_CPU;  return true; }
    if (name_lower == "cuda")     { *out = GGML_BACKEND_DEVICE_TYPE_GPU;  return true; }
    if (name_lower == "metal")    { *out = GGML_BACKEND_DEVICE_TYPE_GPU;  return true; }
    if (name_lower == "vulkan")   { *out = GGML_BACKEND_DEVICE_TYPE_GPU;  return true; }
    if (name_lower == "hipblas")  { *out = GGML_BACKEND_DEVICE_TYPE_GPU;  return true; }
    if (name_lower == "rocm")     { *out = GGML_BACKEND_DEVICE_TYPE_GPU;  return true; }
    if (name_lower == "gpu")      { *out = GGML_BACKEND_DEVICE_TYPE_GPU;  return true; }
    return false;
}

// Try to init a backend whose ggml_backend_dev_get_name matches the
// requested name (case-insensitive substring). Returns null if no
// matching device is loaded.
ggml_backend_t init_named(const std::string& name_lower) {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const char* dev_name = ggml_backend_dev_name(dev);
        if (!dev_name) continue;
        std::string ln = lower(dev_name);
        if (ln.find(name_lower) != std::string::npos) {
            ggml_backend_t b = ggml_backend_dev_init(dev, /*params=*/nullptr);
            if (b) return b;
        }
    }
    return nullptr;
}

void init() {
    // Load any backend shared libraries (libggml-cuda.so etc.) that ggml
    // ships with. No-op for backends compiled in directly (CPU).
    ggml_backend_load_all();

    const char* env = std::getenv("VIBEVOICE_BACKEND");
    std::string want = env ? lower(env) : "";

    // 1. Honor VIBEVOICE_BACKEND verbatim if it matches a registered
    // device.
    if (!want.empty()) {
        if (want == "cpu") {
            g_backend = ggml_backend_dev_init(
                ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
        } else {
            g_backend = init_named(want);
            if (!g_backend) {
                enum ggml_backend_dev_type t;
                if (name_to_type(want, &t)) {
                    ggml_backend_dev_t dev = ggml_backend_dev_by_type(t);
                    if (dev) g_backend = ggml_backend_dev_init(dev, nullptr);
                }
            }
            if (!g_backend) {
                VV_LOG_WARN("backend: VIBEVOICE_BACKEND=%s requested but no "
                            "matching device registered — falling back",
                            env);
            }
        }
    }

    // 2. Default: best-available, with GPU preferred over CPU.
    if (!g_backend) {
        g_backend = ggml_backend_init_best();
    }
    // 3. Belt-and-braces CPU fallback.
    if (!g_backend) {
        g_backend = ggml_backend_dev_init(
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
    }

    if (!g_backend) {
        // ggml_backend_init_best already falls back to CPU if no GPU; if
        // we still don't have a backend, init_by_type(CPU) at least.
        g_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    if (g_backend) {
        const char* name = ggml_backend_name(g_backend);
        g_name = name ? name : "(unnamed)";
        // Hint the CPU backend to use all available threads.
        if (ggml_backend_is_cpu(g_backend)) {
            ggml_backend_cpu_set_n_threads(g_backend, 0);  // 0 = auto
        }
        // Allocator for graph intermediates. ggml_gallocr_new takes a
        // single buffer type; we use the backend's default.
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g_backend);
        g_gallocr = ggml_gallocr_new(buft);
        VV_LOG_INFO("backend: %s", g_name.c_str());
    } else {
        VV_LOG_ERROR("backend: failed to initialize any backend");
        g_name = "none";
    }
}

}  // namespace

ggml_backend_t backend() {
    std::call_once(g_once, init);
    return g_backend;
}

const char* backend_name() {
    std::call_once(g_once, init);
    return g_name.c_str();
}

bool compute_graph(ggml_cgraph* graph) {
    ggml_backend_t b = backend();
    if (!b || !graph || !g_gallocr) return false;
    if (!ggml_gallocr_alloc_graph(g_gallocr, graph)) {
        VV_LOG_ERROR("backend: gallocr_alloc_graph failed");
        return false;
    }
    return ggml_backend_graph_compute(b, graph) == GGML_STATUS_SUCCESS;
}

ggml_backend_buffer_t allocate_ctx_tensors(ggml_context* ctx) {
    ggml_backend_t b = backend();
    if (!b || !ctx) return nullptr;
    return ggml_backend_alloc_ctx_tensors(ctx, b);
}

}  // namespace vv
