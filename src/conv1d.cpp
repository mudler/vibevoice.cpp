#include "conv1d.hpp"
#include "common.hpp"

#include <algorithm>
#include <cstring>

namespace vv {

namespace {

inline struct ggml_tensor* maybe_add_bias_t(struct ggml_context* ctx,
                                            struct ggml_tensor*  y,
                                            struct ggml_tensor*  bias) {
    if (!bias) return y;
    // y shape is [T_out, C_out, B]. bias shape is [C_out]. Broadcast on dim 0
    // and dim 2 — ggml_add handles this when the bias is reshaped to [1, C_out, 1].
    struct ggml_tensor* b = ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
    return ggml_add(ctx, y, b);
}

}  // namespace

struct ggml_tensor* sconv1d_causal(struct ggml_context* ctx,
                                   struct ggml_tensor*  x,
                                   struct ggml_tensor*  kernel,
                                   struct ggml_tensor*  bias,
                                   int stride,
                                   int dilation,
                                   int groups) {
    const int K     = static_cast<int>(kernel->ne[0]);
    const int C_in  = static_cast<int>(x->ne[1]);
    const int T_in  = static_cast<int>(x->ne[0]);

    const int pad_total = (K - 1) * dilation - (stride - 1);
    const int extra     = sconv1d_extra_padding(T_in, K, stride, pad_total);

    // Pad on dim 0 (T): left=pad_total, right=extra.
    struct ggml_tensor* xp = ggml_pad_ext(ctx, x,
                                          /*lp0=*/pad_total, /*rp0=*/extra,
                                          /*lp1=*/0,         /*rp1=*/0,
                                          /*lp2=*/0,         /*rp2=*/0,
                                          /*lp3=*/0,         /*rp3=*/0);

    // ggml's im2col-based conv requires the kernel in F16. Cast on the fly
    // when needed; downstream (M3+) we'll store kernels as F16 in the gguf.
    struct ggml_tensor* k = (kernel->type == GGML_TYPE_F16)
                              ? kernel
                              : ggml_cast(ctx, kernel, GGML_TYPE_F16);

    struct ggml_tensor* y;
    if (groups == 1) {
        y = ggml_conv_1d(ctx, k, xp, stride, /*p0=*/0, /*d0=*/dilation);
    } else if (groups == C_in) {
        y = ggml_conv_1d_dw(ctx, k, xp, stride, /*p0=*/0, /*d0=*/dilation);
    } else {
        VV_LOG_ERROR("sconv1d_causal: groups=%d not supported (must be 1 or C_in=%d)",
                     groups, C_in);
        return nullptr;
    }
    return maybe_add_bias_t(ctx, y, bias);
}

struct ggml_tensor* sconv1d_causal_streaming(struct ggml_context* ctx,
                                             struct ggml_tensor*  x,
                                             struct ggml_tensor*  kernel,
                                             struct ggml_tensor*  bias,
                                             int stride,
                                             int dilation,
                                             int groups,
                                             StreamingCache&       cache,
                                             const std::string&    layer_id) {
    const int K        = static_cast<int>(kernel->ne[0]);
    const int C_in     = static_cast<int>(x->ne[1]);
    const int B        = static_cast<int>(x->ne[2]);
    const int T_in     = static_cast<int>(x->ne[0]);
    const int pad_total = (K - 1) * dilation - (stride - 1);
    const int context  = pad_total;  // samples of left context to retain

    auto& entry = cache[layer_id];
    entry.T = context;
    entry.C = C_in;

    // Allocate the cache-prefix leaf tensor; the caller populates it after
    // ggml_backend_alloc_ctx_tensors via ggml_backend_tensor_set, since
    // .data is null in a no_alloc=true ctx until allocation runs.
    struct ggml_tensor* prefix = nullptr;
    if (context > 0) {
        prefix = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, context, C_in, B);
        ggml_set_name(prefix, ("cache_prefix_" + layer_id).c_str());
    }
    entry.prefix = prefix;

    // Concat prefix and x along dim 0 (time). When prefix is null we just
    // run the conv on x directly (= context_size 0, no extra padding needed).
    struct ggml_tensor* xp = x;
    if (prefix) xp = ggml_concat(ctx, prefix, x, /*dim=*/0);

    // Optional right-pad for stride alignment, on the FINAL chunk only.
    int extra = 0;
    if (cache.is_final_chunk) {
        extra = sconv1d_extra_padding(T_in + context, K, stride, /*pad_total=*/0);
    }
    if (extra > 0) {
        xp = ggml_pad_ext(ctx, xp,
                          /*lp0=*/0, /*rp0=*/extra,
                          /*lp1=*/0, /*rp1=*/0,
                          /*lp2=*/0, /*rp2=*/0,
                          /*lp3=*/0, /*rp3=*/0);
    }

    struct ggml_tensor* k = (kernel->type == GGML_TYPE_F16)
                              ? kernel
                              : ggml_cast(ctx, kernel, GGML_TYPE_F16);
    struct ggml_tensor* y;
    if (groups == 1) {
        y = ggml_conv_1d(ctx, k, xp, stride, /*p0=*/0, /*d0=*/dilation);
    } else if (groups == C_in) {
        y = ggml_conv_1d_dw(ctx, k, xp, stride, /*p0=*/0, /*d0=*/dilation);
    } else {
        VV_LOG_ERROR("sconv1d_causal_streaming: groups=%d not supported", groups);
        return nullptr;
    }

    // Register a view of the last `context` samples of the input (BEFORE
    // any right-padding — only the actual input samples live in the cache
    // for the next chunk). Used post-compute to refresh entry.data.
    if (context > 0) {
        const int T_avail = T_in + context;       // pre-extra
        const int start   = std::max(0, T_avail - context);
        // ggml_view_3d operates on bytes; offset = start * nb[0]. Use the
        // pre-extra-pad tensor so we don't include any zeros we just added.
        struct ggml_tensor* base = prefix ? ggml_concat(ctx, prefix, x, /*dim=*/0) : x;
        struct ggml_tensor* view = ggml_view_3d(
            ctx, base,
            /*ne0=*/context,
            /*ne1=*/C_in,
            /*ne2=*/B,
            /*nb1=*/base->nb[1],
            /*nb2=*/base->nb[2],
            /*offset=*/static_cast<size_t>(start) * base->nb[0]);
        entry.next_view = ggml_cont(ctx, view);   // contiguous so copy-out is straightforward
    }

    return maybe_add_bias_t(ctx, y, bias);
}

struct ggml_tensor* sconv_transpose1d_causal(struct ggml_context* ctx,
                                             struct ggml_tensor*  x,
                                             struct ggml_tensor*  kernel,
                                             struct ggml_tensor*  bias,
                                             int stride) {
    const int K = static_cast<int>(kernel->ne[0]);
    const int trim_right = K - stride;  // trim_right_ratio = 1.0 (causal)

    // ggml-cuda's conv_transpose_1d implementation requires F32 kernels
    // (see ggml-cuda/conv-transpose-1d.cu:67). The CPU backend accepts
    // F16 too, but we standardize on F32 here so the same graph works
    // on both backends.
    struct ggml_tensor* k = (kernel->type == GGML_TYPE_F32)
                              ? kernel
                              : ggml_cast(ctx, kernel, GGML_TYPE_F32);

    struct ggml_tensor* y = ggml_conv_transpose_1d(ctx, k, x, stride, /*p0=*/0, /*d0=*/1);

    // y has shape [T_out_full, C_out, B] where T_out_full = (T-1)*stride + K.
    // Trim trim_right samples from the right (dim 0) using a view.
    if (trim_right > 0) {
        const int64_t T_out = y->ne[0] - trim_right;
        y = ggml_view_3d(ctx, y,
                         /*ne0=*/T_out, /*ne1=*/y->ne[1], /*ne2=*/y->ne[2],
                         /*nb1=*/y->nb[1], /*nb2=*/y->nb[2],
                         /*offset=*/0);
        y = ggml_cont(ctx, y);
    }
    return maybe_add_bias_t(ctx, y, bias);
}

}  // namespace vv
