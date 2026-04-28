// Block1D parity test: PyTorch random-init reference vs our ggml graph.
//
// Tolerance: fp16 floor (~1e-3) since the depthwise conv goes through
// ggml's im2col (kernel cast to fp16 internally).

#include "acoustic_tokenizer.hpp"
#include "ggml-cpu.h"
#include "ggml.h"
#include "model_loader.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

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
    const std::string path = std::string(VV_FIXTURES_DIR) + "/block1d.gguf";
    if (!file_ok(path)) {
        std::fprintf(stderr, "skip: missing %s\n  run: python tests/dump_block1d_reference.py "
                     "--out %s\n", path.c_str(), path.c_str());
        return 77;
    }

    vv::ModelLoader loader;
    if (!loader.load(path)) return 1;

    const int dim = loader.get_i32("block.dim");
    const int T   = loader.get_i32("block.T");
    const float eps = loader.get_f32("block.eps", 1e-5f);

    vv::Block1DWeights w;
    w.norm           = loader.tensor("weight.norm");
    w.ffn_norm       = loader.tensor("weight.ffn_norm");
    w.mixer_kernel   = loader.tensor("weight.mixer_kernel");
    w.mixer_bias     = loader.tensor("weight.mixer_bias");
    w.ffn_linear1    = loader.tensor("weight.ffn_linear1");
    w.ffn_linear1_b  = loader.tensor("weight.ffn_linear1_bias");
    w.ffn_linear2    = loader.tensor("weight.ffn_linear2");
    w.ffn_linear2_b  = loader.tensor("weight.ffn_linear2_bias");
    w.gamma          = loader.tensor("weight.gamma");
    w.ffn_gamma      = loader.tensor("weight.ffn_gamma");
    if (!w.norm || !w.ffn_norm || !w.mixer_kernel || !w.mixer_bias ||
        !w.ffn_linear1 || !w.ffn_linear2 || !w.gamma || !w.ffn_gamma) return 2;

    struct ggml_tensor* in_t  = loader.tensor("test.input");
    struct ggml_tensor* exp_t = loader.tensor("test.expected_output");
    if (!in_t || !exp_t) return 3;

    struct ggml_init_params p {};
    p.mem_size = 64ull * 1024 * 1024;
    p.no_alloc = false;
    struct ggml_context* ctx = ggml_init(p);

    // input shape [T, C, B] in ggml; PyTorch was [B=1, C=dim, T] -> numpy [1, dim, T] -> ggml [T, dim, 1]
    struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, dim, 1);
    std::memcpy(x->data, in_t->data, ggml_nbytes(x));

    struct ggml_tensor* y = vv::block1d_forward(ctx, x, w, eps);
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx); return 4;
    }

    if (y->ne[0] != exp_t->ne[0] || y->ne[1] != exp_t->ne[1]) {
        std::fprintf(stderr, "shape mismatch: got [%lld,%lld] vs [%lld,%lld]\n",
                     (long long)y->ne[0], (long long)y->ne[1],
                     (long long)exp_t->ne[0], (long long)exp_t->ne[1]);
        ggml_free(ctx); return 5;
    }
    const size_t n = static_cast<size_t>(y->ne[0]) * y->ne[1] * y->ne[2];
    const float* a = static_cast<const float*>(y->data);
    const float* e = static_cast<const float*>(exp_t->data);
    double max_abs = 0, sum_a2 = 0, sum_b2 = 0, sum_ab = 0;
    for (size_t i = 0; i < n; ++i) {
        double aa = a[i], bb = e[i];
        double d = std::fabs(aa - bb);
        if (d > max_abs) max_abs = d;
        sum_a2 += aa * aa; sum_b2 += bb * bb; sum_ab += aa * bb;
    }
    double cos = sum_ab / (std::sqrt(sum_a2) * std::sqrt(sum_b2) + 1e-12);
    std::printf("block1d: max_abs=%.3e  cos=%.6f  (dim=%d, T=%d)\n",
                max_abs, cos, dim, T);

    ggml_free(ctx);
    return (max_abs < 2e-3 && cos > 0.9999) ? 0 : 6;
}
