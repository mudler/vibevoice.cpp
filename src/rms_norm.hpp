#ifndef VIBEVOICE_RMS_NORM_HPP
#define VIBEVOICE_RMS_NORM_HPP

#include "ggml.h"

namespace vv {

// RMSNorm with learned per-channel scale. Equivalent to:
//   y = x / sqrt(mean(x^2) + eps) * weight
// `weight` is a 1-D tensor of size matching x->ne[0].
inline struct ggml_tensor* rms_norm(struct ggml_context* ctx,
                                    struct ggml_tensor*  x,
                                    struct ggml_tensor*  weight,
                                    float                eps) {
    struct ggml_tensor* y = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, y, weight);
}

}  // namespace vv

#endif  // VIBEVOICE_RMS_NORM_HPP
