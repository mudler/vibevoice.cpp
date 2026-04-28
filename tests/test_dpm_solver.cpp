// DPM-Solver multistep parity test.
//
// Loads tests/fixtures/diffusion.gguf which contains the diffusion-head
// weights, an initial x_T, a cond input, and the PyTorch reference final x_0
// after 20 inference steps. Runs our solver and compares.

#include "diffusion_head.hpp"
#include "dpm_solver.hpp"
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
        std::fprintf(stderr, "skip: missing %s\n", path.c_str());
        return 77;
    }

    vv::ModelLoader loader;
    if (!loader.load(path)) return 1;

    vv::DiffusionHeadConfig hcfg;
    hcfg.hidden      = loader.get_i32 ("dh.hidden");
    hcfg.latent      = loader.get_i32 ("dh.latent");
    hcfg.head_layers = loader.get_i32 ("dh.head_layers");
    hcfg.ffn_ratio   = loader.get_f32 ("dh.ffn_ratio");
    hcfg.eps         = loader.get_f32 ("dh.eps", 1e-5f);
    hcfg.freq_size   = 256;

    const int frames = loader.get_i32("dh.frames");
    const int M      = loader.get_i32("dh.inference_steps");

    vv::DiffusionHeadWeights w;
    if (!vv::load_diffusion_head(loader, "dh.", hcfg, &w)) return 2;

    struct ggml_tensor* xT_t   = loader.tensor("test.x_T");
    struct ggml_tensor* cond_t = loader.tensor("test.cond2");
    struct ggml_tensor* exp_t  = loader.tensor("test.traj_final");
    if (!xT_t || !cond_t || !exp_t) return 3;

    const size_t n_x = static_cast<size_t>(hcfg.latent) * frames * 1;
    const size_t n_c = static_cast<size_t>(hcfg.hidden) * frames * 1;
    std::vector<float> x   (n_x);
    std::vector<float> cond(n_c);
    std::memcpy(x.data(),   xT_t->data,   n_x * sizeof(float));
    std::memcpy(cond.data(), cond_t->data, n_c * sizeof(float));

    vv::DPMSolverConfig scfg;
    scfg.num_train_timesteps = 1000;
    scfg.num_inference_steps = M;
    scfg.solver_order        = 2;
    scfg.lower_order_final   = true;

    vv::DPMSolverState state;
    vv::dpm_solver_init(scfg, &state);

    int rc = vv::dpm_solver_sample(x, hcfg.latent, frames, /*batch=*/1, cond,
                                   hcfg.hidden, w, hcfg, scfg, state);
    if (rc != 0) {
        std::fprintf(stderr, "sampler returned %d\n", rc);
        return 4;
    }

    const float* e = static_cast<const float*>(exp_t->data);
    double max_abs = 0, sa = 0, sb = 0, sab = 0;
    for (size_t i = 0; i < n_x; ++i) {
        double aa = x[i], bb = e[i];
        double d = std::fabs(aa - bb);
        if (d > max_abs) max_abs = d;
        sa += aa * aa; sb += bb * bb; sab += aa * bb;
    }
    double cos = sab / (std::sqrt(sa) * std::sqrt(sb) + 1e-12);
    std::printf("dpm_solver: %d steps  max_abs=%.3e  cos=%.6f  n=%zu\n",
                M, max_abs, cos, n_x);

    // Compounded fp drift over 20 steps; relax tolerance vs single-step.
    return (max_abs < 5e-3 && cos > 0.999) ? 0 : 5;
}
