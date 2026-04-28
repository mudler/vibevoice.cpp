#include "diffusion_head.hpp"
#include "common.hpp"

#include <cmath>
#include <cstdio>

namespace vv {

void timestep_sinusoidal(float t, int freq_size, float* out) {
    const int   half        = freq_size / 2;
    const float max_period  = 10000.0f;
    const float log_max     = std::log(max_period);
    for (int k = 0; k < half; ++k) {
        const float freq = std::exp(-log_max * static_cast<float>(k) / half);
        const float arg  = t * freq;
        out[k]        = std::cos(arg);
        out[k + half] = std::sin(arg);
    }
    if (freq_size % 2 == 1) {
        out[freq_size - 1] = 0.0f;
    }
}

namespace {

// Slice along the innermost (ne[0]) dim: returns [chunk, ne1, ne2] view.
inline struct ggml_tensor* slice0(struct ggml_context* ctx,
                                  struct ggml_tensor*  t,
                                  int64_t              chunk,
                                  int64_t              idx) {
    struct ggml_tensor* v = ggml_view_3d(
            ctx, t,
            chunk, t->ne[1], t->ne[2],
            t->nb[1], t->nb[2],
            idx * chunk * ggml_type_size(t->type));
    return ggml_cont(ctx, v);
}

// modulate(x, shift, scale) = x * (1 + scale) + shift
inline struct ggml_tensor* modulate(struct ggml_context* ctx,
                                    struct ggml_tensor*  x,
                                    struct ggml_tensor*  shift,
                                    struct ggml_tensor*  scale) {
    struct ggml_tensor* xs = ggml_mul(ctx, x, scale);
    struct ggml_tensor* xy = ggml_add(ctx, x, xs);
    return ggml_add(ctx, xy, shift);
}

// SwiGLU: down(silu(gate(x)) * up(x))
inline struct ggml_tensor* swiglu(struct ggml_context*     ctx,
                                  struct ggml_tensor*      x,
                                  const HeadLayerWeights&  w) {
    struct ggml_tensor* g = ggml_mul_mat(ctx, w.ffn_gate, x);
    struct ggml_tensor* u = ggml_mul_mat(ctx, w.ffn_up,   x);
    g = ggml_silu(ctx, g);
    g = ggml_mul(ctx, g, u);
    return ggml_mul_mat(ctx, w.ffn_down, g);
}

struct ggml_tensor* head_layer_forward(struct ggml_context*    ctx,
                                       struct ggml_tensor*     x,
                                       struct ggml_tensor*     c,
                                       const HeadLayerWeights& l,
                                       int                     hidden,
                                       float                   eps) {
    // m = adaln(silu(c))   shape [3*hidden, frames, B]
    struct ggml_tensor* m = ggml_silu(ctx, c);
    m = ggml_mul_mat(ctx, l.adaln, m);

    struct ggml_tensor* shift = slice0(ctx, m, hidden, 0);
    struct ggml_tensor* scale = slice0(ctx, m, hidden, 1);
    struct ggml_tensor* gate  = slice0(ctx, m, hidden, 2);

    // RMSNorm + per-channel scale
    struct ggml_tensor* xn = ggml_rms_norm(ctx, x, eps);
    xn = ggml_mul(ctx, xn, l.norm);

    struct ggml_tensor* mod = modulate(ctx, xn, shift, scale);
    struct ggml_tensor* f   = swiglu(ctx, mod, l);
    f = ggml_mul(ctx, f, gate);
    return ggml_add(ctx, x, f);
}

struct ggml_tensor* final_layer_forward(struct ggml_context*       ctx,
                                        struct ggml_tensor*        x,
                                        struct ggml_tensor*        c,
                                        const DiffusionHeadWeights& w,
                                        int                        hidden,
                                        float                      eps) {
    // m = adaln_final(silu(c))   shape [2*hidden, frames, B]
    struct ggml_tensor* m = ggml_silu(ctx, c);
    m = ggml_mul_mat(ctx, w.final_adaln, m);

    struct ggml_tensor* shift = slice0(ctx, m, hidden, 0);
    struct ggml_tensor* scale = slice0(ctx, m, hidden, 1);

    struct ggml_tensor* xn  = ggml_rms_norm(ctx, x, eps);  // no learnable weight
    struct ggml_tensor* mod = modulate(ctx, xn, shift, scale);
    return ggml_mul_mat(ctx, w.final_proj, mod);
}

}  // namespace

bool load_diffusion_head(const ModelLoader&            m,
                         const std::string&            prefix,
                         const DiffusionHeadConfig&    cfg,
                         DiffusionHeadWeights*         out) {
    auto get = [&](const std::string& n) -> struct ggml_tensor* {
        return m.tensor(prefix + n);
    };

    out->noisy_proj   = get("noisy_proj");
    out->cond_proj    = get("cond_proj");
    out->t_embed_lin1 = get("t_embed_lin1");
    out->t_embed_lin2 = get("t_embed_lin2");
    out->final_proj   = get("final.proj");
    out->final_adaln  = get("final.adaln");
    if (!out->noisy_proj || !out->cond_proj || !out->t_embed_lin1 ||
        !out->t_embed_lin2 || !out->final_proj || !out->final_adaln) {
        VV_LOG_ERROR("load_diffusion_head: missing top-level tensor");
        return false;
    }

    out->layers.assign(cfg.head_layers, {});
    for (int i = 0; i < cfg.head_layers; ++i) {
        char p[80]; std::snprintf(p, sizeof(p), "%slayer_%d.", prefix.c_str(), i);
        std::string pp(p);
        out->layers[i].norm     = m.tensor(pp + "norm");
        out->layers[i].ffn_gate = m.tensor(pp + "ffn_gate");
        out->layers[i].ffn_up   = m.tensor(pp + "ffn_up");
        out->layers[i].ffn_down = m.tensor(pp + "ffn_down");
        out->layers[i].adaln    = m.tensor(pp + "adaln");
        if (!out->layers[i].norm || !out->layers[i].ffn_gate ||
            !out->layers[i].ffn_up || !out->layers[i].ffn_down ||
            !out->layers[i].adaln) {
            VV_LOG_ERROR("load_diffusion_head: missing layer %d", i);
            return false;
        }
    }
    return true;
}

struct ggml_tensor* diffusion_head_forward(struct ggml_context*        ctx,
                                           struct ggml_tensor*         noisy,
                                           struct ggml_tensor*         cond,
                                           struct ggml_tensor*         t_sin,
                                           const DiffusionHeadWeights& w,
                                           const DiffusionHeadConfig&  cfg) {
    const int hidden = cfg.hidden;

    // x = noisy_proj @ noisy   [latent → hidden]
    struct ggml_tensor* x = ggml_mul_mat(ctx, w.noisy_proj, noisy);

    // t_emb = lin2(silu(lin1(t_sin)))
    struct ggml_tensor* t_emb = ggml_mul_mat(ctx, w.t_embed_lin1, t_sin);
    t_emb = ggml_silu(ctx, t_emb);
    t_emb = ggml_mul_mat(ctx, w.t_embed_lin2, t_emb);
    // shape [hidden, B]; reshape to [hidden, 1, B] for broadcast over `frames`
    t_emb = ggml_reshape_3d(ctx, t_emb, hidden, 1, t_emb->ne[1]);

    // c = cond_proj @ cond + t_emb(broadcast)
    struct ggml_tensor* c = ggml_mul_mat(ctx, w.cond_proj, cond);
    c = ggml_add(ctx, c, t_emb);

    for (const auto& l : w.layers) {
        x = head_layer_forward(ctx, x, c, l, hidden, cfg.eps);
    }
    x = final_layer_forward(ctx, x, c, w, hidden, cfg.eps);
    return x;
}

}  // namespace vv
