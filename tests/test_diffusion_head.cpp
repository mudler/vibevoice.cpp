// DiffusionHead single-step v-prediction parity test.

#include "diffusion_head.hpp"
#include "ggml-cpu.h"
#include "ggml.h"
#include "model_loader.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
bool file_ok(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    std::fclose(f); return true;
}
}  // namespace

int main() {
#ifndef VV_FIXTURES_DIR
#  define VV_FIXTURES_DIR "tests/fixtures"
#endif
    const std::string path = std::string(VV_FIXTURES_DIR) + "/diffusion.gguf";
    if (!file_ok(path)) {
        std::fprintf(stderr, "skip: missing %s\n  run: python tests/dump_diffusion_reference.py "
                     "--out %s\n", path.c_str(), path.c_str());
        return 77;
    }

    vv::ModelLoader loader;
    if (!loader.load(path)) return 1;

    vv::DiffusionHeadConfig cfg;
    cfg.hidden      = loader.get_i32 ("dh.hidden");
    cfg.latent      = loader.get_i32 ("dh.latent");
    cfg.head_layers = loader.get_i32 ("dh.head_layers");
    cfg.ffn_ratio   = loader.get_f32 ("dh.ffn_ratio");
    cfg.eps         = loader.get_f32 ("dh.eps", 1e-5f);
    cfg.freq_size   = 256;

    const int frames = loader.get_i32("dh.frames");

    vv::DiffusionHeadWeights w;
    if (!vv::load_diffusion_head(loader, "dh.", cfg, &w)) return 2;

    struct ggml_tensor* noisy_t = loader.tensor("test.noisy");
    struct ggml_tensor* cond_t  = loader.tensor("test.cond");
    struct ggml_tensor* t_t     = loader.tensor("test.t");
    struct ggml_tensor* exp_t   = loader.tensor("test.v_out");
    if (!noisy_t || !cond_t || !t_t || !exp_t) return 3;

    struct ggml_init_params p {};
    p.mem_size = 64ull * 1024 * 1024;
    p.no_alloc = false;
    struct ggml_context* ctx = ggml_init(p);

    // input shapes (ggml convention, last dim = batch):
    //   noisy: numpy [B, frames, latent] -> ggml [latent, frames, B]
    //   cond : numpy [B, frames, hidden] -> ggml [hidden, frames, B]
    struct ggml_tensor* noisy = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cfg.latent, frames, 1);
    struct ggml_tensor* cond  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cfg.hidden, frames, 1);
    struct ggml_tensor* t_sin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.freq_size, 1);
    std::memcpy(noisy->data, noisy_t->data, ggml_nbytes(noisy));
    std::memcpy(cond->data,  cond_t->data,  ggml_nbytes(cond));

    const float t_val = static_cast<const float*>(t_t->data)[0];
    vv::timestep_sinusoidal(t_val, cfg.freq_size, static_cast<float*>(t_sin->data));

    struct ggml_tensor* y = vv::diffusion_head_forward(ctx, noisy, cond, t_sin, w, cfg);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx); return 4;
    }

    if (y->ne[0] != exp_t->ne[0] || y->ne[1] != exp_t->ne[1]) {
        std::fprintf(stderr, "shape: got [%lld,%lld] vs [%lld,%lld]\n",
                     (long long)y->ne[0], (long long)y->ne[1],
                     (long long)exp_t->ne[0], (long long)exp_t->ne[1]);
        ggml_free(ctx); return 5;
    }
    const size_t n = static_cast<size_t>(y->ne[0]) * y->ne[1] * y->ne[2];
    const float* a = static_cast<const float*>(y->data);
    const float* e = static_cast<const float*>(exp_t->data);
    double max_abs = 0, sa = 0, sb = 0, sab = 0;
    for (size_t i = 0; i < n; ++i) {
        double aa = a[i], bb = e[i];
        double d = std::fabs(aa - bb);
        if (d > max_abs) max_abs = d;
        sa += aa * aa; sb += bb * bb; sab += aa * bb;
    }
    double cos = sab / (std::sqrt(sa) * std::sqrt(sb) + 1e-12);
    std::printf("diffusion_head: max_abs=%.3e  cos=%.6f  shape=[%lld,%lld,%lld]\n",
                max_abs, cos,
                (long long)y->ne[0], (long long)y->ne[1], (long long)y->ne[2]);
    ggml_free(ctx);

    // No conv → fp32 path. Tighter tolerance.
    return (max_abs < 5e-5 && cos > 0.999999) ? 0 : 6;
}
