#ifndef VIBEVOICE_ACOUSTIC_TOKENIZER_HPP
#define VIBEVOICE_ACOUSTIC_TOKENIZER_HPP

// VibeVoice acoustic tokenizer (VAE encoder + decoder) and the supporting
// Block1D used by both the acoustic and the (encoder-only) semantic
// tokenizer.
//
// Reference: vibevoice/modular/modular_vibevoice_tokenizer.py
//
// Block1D pipeline (input/output [T, C, B] in ggml conv layout):
//   x' = x + γ_mixer * dw_mixer(causal_dwconv(rms_norm_C(x)))
//   y  = x' + γ_ffn   * ffn(rms_norm_C(x'))               (FFN: linear → GELU → linear)
//
// Encoder pipeline:
//   stem (SConv1d k=7) → ( downsample SConv1d k=2r,s=r → stage of N×Block1D )*7
//   → final RMSNorm (over C) → head SConv1d k=7 → permute to [B, T, C]
//
// Decoder pipeline (mirror):
//   permute to [B, C, T] → stem → ( upsample SConvTranspose1d → stage )*7
//   → final RMSNorm → head SConv1d k=1 (channels=1)

#include "conv1d.hpp"
#include "ggml.h"
#include "model_loader.hpp"

#include <string>
#include <vector>

namespace vv {

struct Block1DWeights {
    struct ggml_tensor* norm           = nullptr;
    struct ggml_tensor* mixer_kernel   = nullptr;
    struct ggml_tensor* mixer_bias     = nullptr;
    struct ggml_tensor* gamma          = nullptr;
    struct ggml_tensor* ffn_norm       = nullptr;
    struct ggml_tensor* ffn_linear1    = nullptr;
    struct ggml_tensor* ffn_linear1_b  = nullptr;
    struct ggml_tensor* ffn_linear2    = nullptr;
    struct ggml_tensor* ffn_linear2_b  = nullptr;
    struct ggml_tensor* ffn_gamma      = nullptr;
};

struct StridedConvWeights {
    struct ggml_tensor* kernel = nullptr;
    struct ggml_tensor* bias   = nullptr;
    int                 stride = 1;
};

struct AcousticConfig {
    int              channels    = 1;
    int              vae_dim     = 64;
    int              n_filters   = 32;
    std::vector<int> ratios;            // e.g. [8,5,5,4,2,2]
    std::vector<int> depths;            // e.g. [3,3,3,3,3,3,8]
    float            eps         = 1e-5f;
    int              kernel_stem = 7;
    int              kernel_head = 7;
    int              ffn_mult    = 4;
};

struct EncoderWeights {
    StridedConvWeights              stem;
    std::vector<StridedConvWeights> downs;     // size = ratios.size()
    std::vector<std::vector<Block1DWeights>> stages;  // [depths.size()][depths[i]]
    struct ggml_tensor*             final_norm = nullptr;
    StridedConvWeights              head;
};

struct DecoderWeights {
    StridedConvWeights              stem;
    std::vector<StridedConvWeights> ups;       // SConvTranspose1d, size = ratios.size()
    std::vector<std::vector<Block1DWeights>> stages;
    struct ggml_tensor*             final_norm = nullptr;
    StridedConvWeights              head;
};

bool load_block1d   (const ModelLoader&, const std::string& prefix, Block1DWeights* out);
bool load_strided   (const ModelLoader&, const std::string& prefix, StridedConvWeights* out, int stride);
bool load_encoder   (const ModelLoader&, const std::string& prefix,
                     const AcousticConfig&, EncoderWeights* out);
bool load_decoder   (const ModelLoader&, const std::string& prefix,
                     const AcousticConfig&, DecoderWeights* out);

// x: [T, channels, B] -> latents [T_compressed, vae_dim, B]
struct ggml_tensor* encoder_forward(struct ggml_context*  ctx,
                                    struct ggml_tensor*   x,
                                    const EncoderWeights& w,
                                    const AcousticConfig& cfg);

// z: [T_compressed, vae_dim, B] -> audio [T_full, channels, B]
struct ggml_tensor* decoder_forward(struct ggml_context*  ctx,
                                    struct ggml_tensor*   z,
                                    const DecoderWeights& w,
                                    const AcousticConfig& cfg);

// Apply Block1D forward to x [T, C, B]. Returns [T, C, B].
struct ggml_tensor* block1d_forward(struct ggml_context*    ctx,
                                    struct ggml_tensor*     x,
                                    const Block1DWeights&   w,
                                    float                   eps);

// Streaming variants. Same forward math as the single-shot versions, but
// every causal conv1d reads / writes through `cache` keyed by a unique
// `layer_id_prefix`. The caller is responsible for: (a) calling these in
// chunk order; (b) setting cache.is_first_chunk on the first chunk and
// cache.is_final_chunk on the last; (c) after each ggml_graph_compute,
// copying the populated `next_view` data into each entry's `data` buffer
// so the next chunk picks it up.
struct ggml_tensor* block1d_forward_streaming(struct ggml_context*    ctx,
                                              struct ggml_tensor*     x,
                                              const Block1DWeights&   w,
                                              float                   eps,
                                              StreamingCache&         cache,
                                              const std::string&      layer_id_prefix);

struct ggml_tensor* encoder_forward_streaming(struct ggml_context*    ctx,
                                              struct ggml_tensor*     x,
                                              const EncoderWeights&   w,
                                              const AcousticConfig&   cfg,
                                              StreamingCache&         cache);

}  // namespace vv

#endif  // VIBEVOICE_ACOUSTIC_TOKENIZER_HPP
