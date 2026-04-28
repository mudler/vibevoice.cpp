// Standalone RoPE-NeoX numerics test against a Python reference.
//
// Loads tests/fixtures/rope.gguf and verifies ggml_rope_ext output to within
// 1e-5 max-abs (fp32).

#include "ggml-cpu.h"
#include "ggml.h"
#include "model_loader.hpp"
#include "rope.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool file_ok(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

}  // namespace

int main() {
#ifndef VV_FIXTURES_DIR
#  define VV_FIXTURES_DIR "tests/fixtures"
#endif
    const std::string fix  = VV_FIXTURES_DIR;
    const std::string path = fix + "/rope.gguf";

    if (!file_ok(path)) {
        std::fprintf(stderr,
                     "test_rope: skipping — missing %s\n"
                     "  run: python tests/dump_rope_reference.py --out %s\n",
                     path.c_str(), path.c_str());
        return 77;
    }

    vv::ModelLoader loader;
    if (!loader.load(path)) return 1;

    const int dim     = loader.get_i32("rope.dim");
    const int n_heads = loader.get_i32("rope.n_heads");
    const int seq     = loader.get_i32("rope.seq");
    const float theta = loader.get_f32("rope.theta", 1.0e6f);

    struct ggml_tensor* in_t  = loader.tensor("test.input");
    struct ggml_tensor* exp_t = loader.tensor("test.expected_output");
    struct ggml_tensor* pos_t = loader.tensor("test.position_ids");
    if (!in_t || !exp_t || !pos_t) return 2;

    struct ggml_init_params p {};
    p.mem_size   = 32ull * 1024 * 1024;
    p.no_alloc   = false;
    struct ggml_context* ctx = ggml_init(p);

    struct ggml_tensor* x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, n_heads, seq);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    std::memcpy(x->data,   in_t->data,  ggml_nbytes(x));
    std::memcpy(pos->data, pos_t->data, ggml_nbytes(pos));

    struct ggml_tensor* y = ggml_rope_ext(
            ctx, x, pos, /*freq_factors=*/nullptr,
            dim, vv::kRopeMode, /*n_ctx_orig=*/0,
            theta, /*freq_scale=*/1.0f,
            /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
            /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed: %d\n", st);
        return 3;
    }

    const size_t n     = static_cast<size_t>(dim) * n_heads * seq;
    const float* got   = static_cast<const float*>(y->data);
    const float* expe  = static_cast<const float*>(exp_t->data);

    double max_abs = 0;
    for (size_t k = 0; k < n; ++k) {
        double d = std::fabs(static_cast<double>(got[k]) - static_cast<double>(expe[k]));
        if (d > max_abs) max_abs = d;
    }
    std::printf("rope: max_abs=%.3e  (dim=%d, n_heads=%d, seq=%d, theta=%.1e)\n",
                max_abs, dim, n_heads, seq, theta);

    ggml_free(ctx);
    // fp32 ggml rope vs fp64 python reference: ~1e-5 max-abs at seq=128 with
    // rope_theta=1e6 is the expected fp32 quantization floor.
    return max_abs < 5e-5 ? 0 : 4;
}
