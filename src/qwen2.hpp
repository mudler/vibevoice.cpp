#ifndef VIBEVOICE_QWEN2_HPP
#define VIBEVOICE_QWEN2_HPP

// One Qwen2 transformer layer as a ggml graph builder.
//
// Layer formula:
//   h  = x + Attn(RMSNorm(x))
//   y  = h + FFN(RMSNorm(h))
// where Attn is GQA with RoPE on Q/K, and FFN is SwiGLU.
//
// Weights are stored in PyTorch-style layout (`[out, in]`) — when ggml loads
// them, ne[0] = in (the contiguous dim) and ne[1] = out, which is exactly
// what `ggml_mul_mat(W, X)` expects.

#include "ggml.h"
#include "model_loader.hpp"

#include <string>

namespace vv {

struct Qwen2Hparams {
    int   hidden_size       = 0;
    int   n_heads           = 0;
    int   n_kv_heads        = 0;
    int   head_dim          = 0;
    int   intermediate_size = 0;
    float rope_theta        = 1.0e6f;
    float rms_norm_eps      = 1.0e-6f;
};

struct Qwen2LayerWeights {
    struct ggml_tensor* attn_norm   = nullptr;
    struct ggml_tensor* attn_q      = nullptr;
    struct ggml_tensor* attn_q_bias = nullptr;
    struct ggml_tensor* attn_k      = nullptr;
    struct ggml_tensor* attn_k_bias = nullptr;
    struct ggml_tensor* attn_v      = nullptr;
    struct ggml_tensor* attn_v_bias = nullptr;
    struct ggml_tensor* attn_o      = nullptr;
    struct ggml_tensor* ffn_norm    = nullptr;
    struct ggml_tensor* ffn_gate    = nullptr;
    struct ggml_tensor* ffn_up      = nullptr;
    struct ggml_tensor* ffn_down    = nullptr;
};

// Load layer weights from a gguf using a name prefix (e.g. "" or "blk.0.").
// Tensor names expected: prefix + "weight.attn_norm", "weight.attn_q",
// "weight.attn_q_bias", … (as produced by tests/dump_qwen2_reference.py).
bool qwen2_load_layer(const ModelLoader& m,
                      const std::string& prefix,
                      Qwen2LayerWeights* out);

struct Qwen2LayerOutput {
    struct ggml_tensor* y       = nullptr;  // [hidden, n_tokens, B]
    struct ggml_tensor* k_full  = nullptr;  // [head_dim, n_kv_heads, past+n_tokens, B]
    struct ggml_tensor* v_full  = nullptr;  // same shape
};

// Build a single Qwen2 layer's compute graph. `x` is [hidden, n_tokens, B];
// `pos` is an int32 vector of length n_tokens with ABSOLUTE positions; `mask`
// is the additive attention mask of shape [past+n_tokens, n_tokens, 1, 1].
//
// `k_past` / `v_past` may be null (no cache, prefill from position 0) or
// contain past KV of shape [head_dim, n_kv_heads, past_len, B]. In the
// cached case the new K/V are concatenated with the past along the sequence
// dim before attention.
//
// Returns the layer output AND the updated full K/V so the caller can use
// them as next-step `k_past`/`v_past`.
Qwen2LayerOutput qwen2_layer_forward(struct ggml_context*     ctx,
                                     struct ggml_tensor*      x,
                                     struct ggml_tensor*      pos,
                                     struct ggml_tensor*      mask,
                                     struct ggml_tensor*      k_past,
                                     struct ggml_tensor*      v_past,
                                     const Qwen2LayerWeights& w,
                                     const Qwen2Hparams&      hp);

}  // namespace vv

#endif  // VIBEVOICE_QWEN2_HPP
