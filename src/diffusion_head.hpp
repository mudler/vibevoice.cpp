#ifndef VIBEVOICE_DIFFUSION_HEAD_HPP
#define VIBEVOICE_DIFFUSION_HEAD_HPP

// VibeVoice diffusion head:
//
//   x = noisy_proj(noisy)                        # [hidden, frames, B]
//   c = cond_proj(cond) + t_embed(t)             # [hidden, frames, B] (t broadcast)
//   for each HeadLayer:
//       (shift, scale, gate) = adaln(silu(c)).chunk(3, axis=-1)
//       x = x + gate * SwiGLU_FFN(modulate(RMSNorm(x), shift, scale))
//   FinalLayer:
//       (shift, scale) = adaln_final(silu(c)).chunk(2, axis=-1)
//       x = proj(modulate(RMSNorm_no_scale(x), shift, scale))
//
// Sinusoidal timestep embedding is pre-computed in C++ (t is a scalar per
// call); the MLP that follows is built into the graph.

#include "ggml.h"
#include "model_loader.hpp"

#include <string>
#include <vector>

namespace vv {

struct HeadLayerWeights {
    struct ggml_tensor* norm     = nullptr;  // [hidden]
    struct ggml_tensor* ffn_gate = nullptr;  // [hidden, ffn_dim]
    struct ggml_tensor* ffn_up   = nullptr;  // [hidden, ffn_dim]
    struct ggml_tensor* ffn_down = nullptr;  // [ffn_dim, hidden]
    struct ggml_tensor* adaln    = nullptr;  // [hidden, 3*hidden]   (no bias)
};

struct DiffusionHeadConfig {
    int   hidden       = 1536;
    int   latent       = 64;
    int   head_layers  = 4;
    float ffn_ratio    = 3.0f;
    int   freq_size    = 256;
    float eps          = 1e-5f;
};

struct DiffusionHeadWeights {
    struct ggml_tensor* noisy_proj   = nullptr;  // [latent, hidden]
    struct ggml_tensor* cond_proj    = nullptr;  // [hidden, hidden]
    struct ggml_tensor* t_embed_lin1 = nullptr;  // [freq_size, hidden]
    struct ggml_tensor* t_embed_lin2 = nullptr;  // [hidden, hidden]
    std::vector<HeadLayerWeights> layers;
    struct ggml_tensor* final_proj   = nullptr;  // [hidden, latent]
    struct ggml_tensor* final_adaln  = nullptr;  // [hidden, 2*hidden]
};

bool load_diffusion_head(const ModelLoader&                m,
                         const std::string&                prefix,
                         const DiffusionHeadConfig&        cfg,
                         DiffusionHeadWeights*             out);

// Compute sinusoidal timestep embedding for `t` (single batch entry).
// Writes `freq_size` floats into `out`.
void timestep_sinusoidal(float t, int freq_size, float* out);

// Build the diffusion-head graph.
//   noisy: [latent, frames, B]
//   cond:  [hidden, frames, B]
//   t_sin: [freq_size, B]      (precomputed sinusoidal embedding)
// Returns: [latent, frames, B]
struct ggml_tensor* diffusion_head_forward(struct ggml_context*       ctx,
                                           struct ggml_tensor*        noisy,
                                           struct ggml_tensor*        cond,
                                           struct ggml_tensor*        t_sin,
                                           const DiffusionHeadWeights& w,
                                           const DiffusionHeadConfig&  cfg);

}  // namespace vv

#endif  // VIBEVOICE_DIFFUSION_HEAD_HPP
