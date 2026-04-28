#ifndef VIBEVOICE_GGML_EXTEND_HPP
#define VIBEVOICE_GGML_EXTEND_HPP

// Minimal ggml helpers for vibevoice.cpp.
// In M1 this is intentionally light; later milestones add graph helpers
// (attention, conv, etc) following the pattern of stable-diffusion.cpp.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <memory>
#include <string>

namespace vv {

struct GgmlCtxDeleter {
    void operator()(struct ggml_context* c) const noexcept {
        if (c) ggml_free(c);
    }
};
using GgmlCtxPtr = std::unique_ptr<struct ggml_context, GgmlCtxDeleter>;

inline GgmlCtxPtr make_ctx(size_t mem_size, bool no_alloc = false) {
    struct ggml_init_params p = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ no_alloc,
    };
    return GgmlCtxPtr(ggml_init(p));
}

}  // namespace vv

#endif  // VIBEVOICE_GGML_EXTEND_HPP
