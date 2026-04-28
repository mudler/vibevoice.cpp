#include "acoustic_tokenizer.hpp"
#include "common.hpp"
#include "conv1d.hpp"

#include <cstdio>

namespace vv {

namespace {

// RMSNorm over the channel dim of a [T, C, B] tensor.
// Implementation: permute to [C, T, B], use ggml_rms_norm (which normalizes
// over ne[0]), multiply by per-channel weight, permute back.
struct ggml_tensor* rms_norm_channels(struct ggml_context* ctx,
                                      struct ggml_tensor*  x,
                                      struct ggml_tensor*  weight,
                                      float                eps) {
    struct ggml_tensor* y = ggml_permute(ctx, x, 1, 0, 2, 3);     // [C, T, B]
    y = ggml_cont(ctx, y);
    y = ggml_rms_norm(ctx, y, eps);
    y = ggml_mul(ctx, y, weight);                                 // weight [C] broadcasts on dim 0
    y = ggml_permute(ctx, y, 1, 0, 2, 3);                         // back to [T, C, B]
    return ggml_cont(ctx, y);
}

// Multiply x [T, C, B] by per-channel γ [C], broadcasting over T and B.
struct ggml_tensor* mul_per_channel(struct ggml_context* ctx,
                                    struct ggml_tensor*  x,
                                    struct ggml_tensor*  gamma) {
    struct ggml_tensor* g = ggml_reshape_3d(ctx, gamma, 1, gamma->ne[0], 1);
    return ggml_mul(ctx, x, g);
}

}  // namespace

struct ggml_tensor* block1d_forward(struct ggml_context*    ctx,
                                    struct ggml_tensor*     x,
                                    const Block1DWeights&   w,
                                    float                   eps) {
    const int C = static_cast<int>(x->ne[1]);

    // ---- mixer branch ----
    struct ggml_tensor* residual = x;
    struct ggml_tensor* h = rms_norm_channels(ctx, x, w.norm, eps);
    h = sconv1d_causal(ctx, h, w.mixer_kernel, w.mixer_bias,
                       /*stride=*/1, /*dilation=*/1, /*groups=*/C);
    h = mul_per_channel(ctx, h, w.gamma);
    h = ggml_add(ctx, residual, h);

    // ---- FFN branch ----
    residual = h;
    struct ggml_tensor* f = rms_norm_channels(ctx, h, w.ffn_norm, eps);

    // FFN expects channels as innermost dim. f is [T, C, B]; transpose to [C, T, B].
    f = ggml_permute(ctx, f, 1, 0, 2, 3);
    f = ggml_cont(ctx, f);

    // linear1: [C → C_ffn]
    f = ggml_mul_mat(ctx, w.ffn_linear1, f);                       // [C_ffn, T, B]
    if (w.ffn_linear1_b) f = ggml_add(ctx, f, w.ffn_linear1_b);
    f = ggml_gelu(ctx, f);

    // linear2: [C_ffn → C]
    f = ggml_mul_mat(ctx, w.ffn_linear2, f);                       // [C, T, B]
    if (w.ffn_linear2_b) f = ggml_add(ctx, f, w.ffn_linear2_b);

    // ffn-γ — broadcast over [T, B]
    struct ggml_tensor* g = ggml_reshape_3d(ctx, w.ffn_gamma, w.ffn_gamma->ne[0], 1, 1);
    f = ggml_mul(ctx, f, g);

    // permute back to [T, C, B]
    f = ggml_permute(ctx, f, 1, 0, 2, 3);
    f = ggml_cont(ctx, f);

    return ggml_add(ctx, residual, f);
}

// ---------- weight loaders ----------

bool load_block1d(const ModelLoader& m, const std::string& prefix,
                  Block1DWeights* out) {
    auto get = [&](const char* n) { return m.tensor(prefix + ".weight." + n); };
    out->norm           = get("norm");
    out->ffn_norm       = get("ffn_norm");
    out->mixer_kernel   = get("mixer_kernel");
    if (!out->mixer_kernel) out->mixer_kernel = get("mixer_weight");
    out->mixer_bias     = get("mixer_bias");
    out->gamma          = get("gamma");
    out->ffn_gamma      = get("ffn_gamma");
    out->ffn_linear1    = get("ffn_linear1");
    out->ffn_linear1_b  = get("ffn_linear1_bias");
    out->ffn_linear2    = get("ffn_linear2");
    out->ffn_linear2_b  = get("ffn_linear2_bias");
    if (!out->norm || !out->ffn_norm || !out->mixer_kernel || !out->mixer_bias ||
        !out->gamma || !out->ffn_gamma ||
        !out->ffn_linear1 || !out->ffn_linear2) {
        VV_LOG_ERROR("load_block1d: missing tensor under %s", prefix.c_str());
        return false;
    }
    return true;
}

bool load_strided(const ModelLoader& m, const std::string& prefix,
                  StridedConvWeights* out, int stride) {
    // Accept either "<prefix>.kernel" (test fixture) or "<prefix>.weight"
    // (converter from real safetensors).
    out->kernel = m.tensor(prefix + ".kernel");
    if (!out->kernel) out->kernel = m.tensor(prefix + ".weight");
    out->bias   = m.tensor(prefix + ".bias");
    out->stride = stride;
    if (!out->kernel) {
        VV_LOG_ERROR("load_strided: missing %s.kernel/.weight", prefix.c_str());
        return false;
    }
    return true;
}

bool load_encoder(const ModelLoader& m, const std::string& prefix,
                  const AcousticConfig& cfg, EncoderWeights* out) {
    if (!load_strided(m, prefix + ".stem", &out->stem, /*stride=*/1)) return false;

    // Detect 0-based (test fixture: down_0..N-1) vs 1-based (real converter:
    // down_1..N) naming ONCE — based on the LAST index, which only exists
    // in the 1-based scheme. Otherwise iteration 1 in the real model would
    // mis-load `down_1` (which is iteration 0's tensor).
    const bool one_based = [&]() {
        if (cfg.ratios.empty()) return false;
        char nlast[64]; std::snprintf(nlast, sizeof(nlast), "%s.down_%zu",
                                      prefix.c_str(), cfg.ratios.size());
        return m.tensor(std::string(nlast) + ".weight") ||
               m.tensor(std::string(nlast) + ".kernel");
    }();
    // Upstream: `self.ratios = list(reversed(config.ratios))` — the encoder
    // applies ratios in REVERSED order. So down_1 has stride ratios[N-1],
    // down_2 has stride ratios[N-2], …, down_N has stride ratios[0].
    // (The decoder, in contrast, uses original-order ratios.) We previously
    // mismatched this, applying e.g. stride=8 to the kernel that was stored
    // for stride=2 → garbled latents. Fix: reverse for the encoder.
    out->downs.resize(cfg.ratios.size());
    for (size_t i = 0; i < cfg.ratios.size(); ++i) {
        const size_t idx = one_based ? i + 1 : i;
        const int stride = cfg.ratios[cfg.ratios.size() - 1 - i];
        char name[64]; std::snprintf(name, sizeof(name), "%s.down_%zu", prefix.c_str(), idx);
        if (!load_strided(m, name, &out->downs[i], stride)) return false;
    }

    out->stages.assign(cfg.depths.size(), {});
    for (size_t i = 0; i < cfg.depths.size(); ++i) {
        out->stages[i].resize(cfg.depths[i]);
        for (int j = 0; j < cfg.depths[i]; ++j) {
            char name[80];
            std::snprintf(name, sizeof(name), "%s.stage_%zu_block_%d",
                          prefix.c_str(), i, j);
            if (!load_block1d(m, name, &out->stages[i][j])) return false;
        }
    }

    // final_norm is optional (`disable_last_norm: true` in VibeVoice config).
    out->final_norm = m.tensor(prefix + ".final_norm");
    if (!load_strided(m, prefix + ".head", &out->head, /*stride=*/1)) return false;
    return true;
}

bool load_decoder(const ModelLoader& m, const std::string& prefix,
                  const AcousticConfig& cfg, DecoderWeights* out) {
    if (!load_strided(m, prefix + ".stem", &out->stem, /*stride=*/1)) return false;

    // Decoder up-samples are indexed 1..ratios.size() in the upstream layout
    // (upsample_layers[0] is the stem; the strided ones start at index 1).
    // Our converter mirrors that, so try both naming schemes to stay
    // backward-compatible with the random-init test fixture.
    // 0-based (test) vs 1-based (real converter) detection — see load_encoder
    // for context. Detect once via the existence of the last-index tensor.
    const bool one_based = [&]() {
        if (cfg.ratios.empty()) return false;
        char nlast[64]; std::snprintf(nlast, sizeof(nlast), "%s.up_%zu",
                                      prefix.c_str(), cfg.ratios.size());
        return m.tensor(std::string(nlast) + ".weight") ||
               m.tensor(std::string(nlast) + ".kernel");
    }();
    out->ups.resize(cfg.ratios.size());
    for (size_t i = 0; i < cfg.ratios.size(); ++i) {
        const size_t idx = one_based ? i + 1 : i;
        char name[64]; std::snprintf(name, sizeof(name), "%s.up_%zu", prefix.c_str(), idx);
        if (!load_strided(m, name, &out->ups[i], /*stride=*/cfg.ratios[i])) return false;
    }

    out->stages.assign(cfg.depths.size(), {});
    for (size_t i = 0; i < cfg.depths.size(); ++i) {
        out->stages[i].resize(cfg.depths[i]);
        for (int j = 0; j < cfg.depths[i]; ++j) {
            char name[80];
            std::snprintf(name, sizeof(name), "%s.stage_%zu_block_%d",
                          prefix.c_str(), i, j);
            if (!load_block1d(m, name, &out->stages[i][j])) return false;
        }
    }
    // final_norm is optional (VibeVoice's `disable_last_norm: true`).
    out->final_norm = m.tensor(prefix + ".final_norm");
    if (!load_strided(m, prefix + ".head", &out->head, /*stride=*/1)) return false;
    return true;
}

// ---------- forward graphs ----------

namespace {

// strided / non-strided causal SConv1d
struct ggml_tensor* conv_step(struct ggml_context* ctx,
                              struct ggml_tensor*  x,
                              const StridedConvWeights& w) {
    return sconv1d_causal(ctx, x, w.kernel, w.bias,
                          /*stride=*/w.stride, /*dilation=*/1, /*groups=*/1);
}

// causal SConvTranspose1d (trim_right = K - stride; trim_right_ratio = 1)
struct ggml_tensor* convtr_step(struct ggml_context* ctx,
                                struct ggml_tensor*  x,
                                const StridedConvWeights& w) {
    return sconv_transpose1d_causal(ctx, x, w.kernel, w.bias, w.stride);
}

}  // namespace

struct ggml_tensor* encoder_forward(struct ggml_context*  ctx,
                                    struct ggml_tensor*   x,
                                    const EncoderWeights& w,
                                    const AcousticConfig& cfg) {
    struct ggml_tensor* h = conv_step(ctx, x, w.stem);
    for (const auto& blk : w.stages[0]) h = block1d_forward(ctx, h, blk, cfg.eps);

    for (size_t i = 1; i < cfg.depths.size(); ++i) {
        h = conv_step(ctx, h, w.downs[i - 1]);
        for (const auto& blk : w.stages[i]) h = block1d_forward(ctx, h, blk, cfg.eps);
    }

    // final norm over channels
    struct ggml_tensor* y = h;
    if (w.final_norm) {
        struct ggml_tensor* p = ggml_permute(ctx, y, 1, 0, 2, 3);
        p = ggml_cont(ctx, p);
        p = ggml_rms_norm(ctx, p, cfg.eps);
        p = ggml_mul(ctx, p, w.final_norm);
        p = ggml_permute(ctx, p, 1, 0, 2, 3);
        y = ggml_cont(ctx, p);
    }
    y = conv_step(ctx, y, w.head);
    return y;
}

// ---- streaming variants -----------------------------------------------------

struct ggml_tensor* block1d_forward_streaming(struct ggml_context*    ctx,
                                              struct ggml_tensor*     x,
                                              const Block1DWeights&   w,
                                              float                   eps,
                                              StreamingCache&         cache,
                                              const std::string&      layer_id_prefix) {
    const int C = static_cast<int>(x->ne[1]);

    // RMSNorm + cached depthwise mixer + residual.
    struct ggml_tensor* residual = x;
    struct ggml_tensor* h = rms_norm_channels(ctx, x, w.norm, eps);
    h = sconv1d_causal_streaming(ctx, h, w.mixer_kernel, w.mixer_bias,
                                 /*stride=*/1, /*dilation=*/1, /*groups=*/C,
                                 cache, layer_id_prefix + ".mixer");
    h = mul_per_channel(ctx, h, w.gamma);
    h = ggml_add(ctx, residual, h);

    // RMSNorm + FFN (no streaming dependency — pointwise) + residual.
    residual = h;
    struct ggml_tensor* f = rms_norm_channels(ctx, h, w.ffn_norm, eps);
    f = ggml_permute(ctx, f, 1, 0, 2, 3);
    f = ggml_cont(ctx, f);
    f = ggml_mul_mat(ctx, w.ffn_linear1, f);
    if (w.ffn_linear1_b) f = ggml_add(ctx, f, w.ffn_linear1_b);
    f = ggml_gelu(ctx, f);
    f = ggml_mul_mat(ctx, w.ffn_linear2, f);
    if (w.ffn_linear2_b) f = ggml_add(ctx, f, w.ffn_linear2_b);
    {
        struct ggml_tensor* g = ggml_reshape_3d(ctx, w.ffn_gamma, w.ffn_gamma->ne[0], 1, 1);
        f = ggml_mul(ctx, f, g);
    }
    f = ggml_permute(ctx, f, 1, 0, 2, 3);
    f = ggml_cont(ctx, f);
    return ggml_add(ctx, residual, f);
}

struct ggml_tensor* encoder_forward_streaming(struct ggml_context*    ctx,
                                              struct ggml_tensor*     x,
                                              const EncoderWeights&   w,
                                              const AcousticConfig&   cfg,
                                              StreamingCache&         cache) {
    char buf[64];

    struct ggml_tensor* h = sconv1d_causal_streaming(
        ctx, x, w.stem.kernel, w.stem.bias, w.stem.stride, /*dilation=*/1, /*groups=*/1,
        cache, "stem");
    for (size_t j = 0; j < w.stages[0].size(); ++j) {
        std::snprintf(buf, sizeof(buf), "stage_0_block_%zu", j);
        h = block1d_forward_streaming(ctx, h, w.stages[0][j], cfg.eps, cache, buf);
    }
    for (size_t i = 1; i < cfg.depths.size(); ++i) {
        std::snprintf(buf, sizeof(buf), "down_%zu", i);
        h = sconv1d_causal_streaming(
            ctx, h, w.downs[i - 1].kernel, w.downs[i - 1].bias,
            w.downs[i - 1].stride, /*dilation=*/1, /*groups=*/1,
            cache, buf);
        for (size_t j = 0; j < w.stages[i].size(); ++j) {
            std::snprintf(buf, sizeof(buf), "stage_%zu_block_%zu", i, j);
            h = block1d_forward_streaming(ctx, h, w.stages[i][j], cfg.eps, cache, buf);
        }
    }
    struct ggml_tensor* y = h;
    if (w.final_norm) {
        struct ggml_tensor* p = ggml_permute(ctx, y, 1, 0, 2, 3);
        p = ggml_cont(ctx, p);
        p = ggml_rms_norm(ctx, p, cfg.eps);
        p = ggml_mul(ctx, p, w.final_norm);
        p = ggml_permute(ctx, p, 1, 0, 2, 3);
        y = ggml_cont(ctx, p);
    }
    y = sconv1d_causal_streaming(
        ctx, y, w.head.kernel, w.head.bias, w.head.stride, /*dilation=*/1, /*groups=*/1,
        cache, "head");
    return y;
}

struct ggml_tensor* decoder_forward(struct ggml_context*  ctx,
                                    struct ggml_tensor*   z,
                                    const DecoderWeights& w,
                                    const AcousticConfig& cfg) {
    struct ggml_tensor* h = conv_step(ctx, z, w.stem);
    for (const auto& blk : w.stages[0]) h = block1d_forward(ctx, h, blk, cfg.eps);

    for (size_t i = 1; i < cfg.depths.size(); ++i) {
        h = convtr_step(ctx, h, w.ups[i - 1]);
        for (const auto& blk : w.stages[i]) h = block1d_forward(ctx, h, blk, cfg.eps);
    }

    struct ggml_tensor* y = h;
    if (w.final_norm) {
        struct ggml_tensor* p = ggml_permute(ctx, y, 1, 0, 2, 3);
        p = ggml_cont(ctx, p);
        p = ggml_rms_norm(ctx, p, cfg.eps);
        p = ggml_mul(ctx, p, w.final_norm);
        p = ggml_permute(ctx, p, 1, 0, 2, 3);
        y = ggml_cont(ctx, p);
    }
    y = conv_step(ctx, y, w.head);
    return y;
}

}  // namespace vv
