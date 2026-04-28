#ifndef VIBEVOICE_ROPE_HPP
#define VIBEVOICE_ROPE_HPP

// Qwen2 uses GPT-NeoX-style RoPE: cos/sin tables of half-dim duplicated to
// full dim, with rotate_half splitting input into two halves. This matches
// HuggingFace's Qwen2RotaryEmbedding + rotate_half exactly.
//
// In ggml, this is the GGML_ROPE_TYPE_NEOX mode of ggml_rope_ext.

#include "ggml.h"

namespace vv {

constexpr int   kRopeMode    = GGML_ROPE_TYPE_NEOX;
constexpr float kRopeBaseQwen2 = 1.0e6f;

}  // namespace vv

#endif  // VIBEVOICE_ROPE_HPP
