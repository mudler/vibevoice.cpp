// Qwen2 single-layer parity test.
//
// Loads tests/fixtures/qwen2_layer0.gguf (produced by
// dump_qwen2_reference.py), runs our C++ qwen2_layer_forward graph, and
// compares to the HF-computed expected output.
//
// Tolerance: max-abs < 5e-5 fp32. The fixture is tiny (hidden=128, seq=64,
// random-init weights) so the test runs in well under a second.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "model_loader.hpp"
#include "qwen2.hpp"

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
    const std::string path = fix + "/qwen2_layer0.gguf";

    if (!file_ok(path)) {
        std::fprintf(stderr,
                     "test_qwen2_block: skipping — missing %s\n"
                     "  run python tests/dump_qwen2_reference.py "
                     "--out tests/fixtures/qwen2_layer0.gguf\n",
                     path.c_str());
        return 77;  // ctest skip code
    }

    vv::ModelLoader loader;
    if (!loader.load(path)) return 1;

    vv::Qwen2Hparams hp;
    hp.hidden_size       = loader.get_i32("qwen2.hidden_size");
    hp.n_heads           = loader.get_i32("qwen2.n_heads");
    hp.n_kv_heads        = loader.get_i32("qwen2.n_kv_heads");
    hp.head_dim          = loader.get_i32("qwen2.head_dim");
    hp.intermediate_size = loader.get_i32("qwen2.intermediate_size");
    hp.rope_theta        = loader.get_f32("qwen2.rope_theta", 1.0e6f);
    hp.rms_norm_eps      = loader.get_f32("qwen2.rms_norm_eps", 1.0e-6f);
    const int seq        = loader.get_i32("qwen2.test.seq_len");
    const int batch      = loader.get_i32("qwen2.test.batch");

    vv::Qwen2LayerWeights w;
    if (!vv::qwen2_load_layer(loader, "", &w)) return 2;

    struct ggml_tensor* input_t    = loader.tensor("test.input");
    struct ggml_tensor* expected_t = loader.tensor("test.expected_output");
    struct ggml_tensor* pos_t      = loader.tensor("test.position_ids");
    if (!input_t || !expected_t || !pos_t) {
        std::fprintf(stderr, "missing test.* tensor\n");
        return 3;
    }

    // ---- compute ctx ----
    struct ggml_init_params p {};
    p.mem_size   = 256ull * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc   = false;
    struct ggml_context* ctx = ggml_init(p);

    // Inputs (allocated in this ctx).
    struct ggml_tensor* x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.hidden_size, seq, batch);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
    ggml_set_name(x,    "x");
    ggml_set_name(pos,  "pos");
    ggml_set_name(mask, "mask");

    // Fill inputs.
    // Fixture stored x as PyTorch [B, T, hidden]. ggml shape on load is
    // [hidden, T, B] (innermost dim first). Same memory layout — direct copy.
    std::memcpy(x->data,   input_t->data, ggml_nbytes(x));
    std::memcpy(pos->data, pos_t->data,   ggml_nbytes(pos));

    // Causal mask: shape [seq_kv, seq_q]. mask[j,i] = -inf if j > i (key after query).
    {
        float* m = static_cast<float*>(mask->data);
        for (int i = 0; i < seq; ++i) {
            for (int j = 0; j < seq; ++j) {
                m[i * seq + j] = (j > i) ? -INFINITY : 0.0f;
            }
        }
    }

    // ---- graph ----
    auto out = vv::qwen2_layer_forward(ctx, x, pos, mask, nullptr, nullptr, w, hp);
    struct ggml_tensor* y = out.y;
    ggml_set_name(y, "y");

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    const int n_threads = 4;
    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml compute failed: %d\n", static_cast<int>(st));
        return 4;
    }

    // ---- compare ----
    const size_t n_elems     = static_cast<size_t>(hp.hidden_size) * seq * batch;
    const float* got      = static_cast<const float*>(y->data);
    const float* expected = static_cast<const float*>(expected_t->data);

    double max_abs  = 0.0;
    double sum_abs  = 0.0;
    double sum_a2   = 0.0;
    double sum_b2   = 0.0;
    double sum_ab   = 0.0;
    for (size_t k = 0; k < n_elems; ++k) {
        double a = got[k];
        double b = expected[k];
        double d = std::fabs(a - b);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
        sum_a2 += a * a;
        sum_b2 += b * b;
        sum_ab += a * b;
    }
    const double mean_abs = sum_abs / n_elems;
    const double cos_sim  = sum_ab / (std::sqrt(sum_a2) * std::sqrt(sum_b2) + 1e-12);

    std::printf("qwen2_block: max_abs=%.3e  mean_abs=%.3e  cos_sim=%.6f  "
                "(hidden=%d, n_h=%d/%d, hd=%d, seq=%d)\n",
                max_abs, mean_abs, cos_sim,
                hp.hidden_size, hp.n_heads, hp.n_kv_heads, hp.head_dim, seq);

    ggml_free(ctx);

    const double tol_abs = 5e-5;  // fp32 target
    const double tol_cos = 0.99999;
    if (max_abs > tol_abs || cos_sim < tol_cos) {
        std::fprintf(stderr, "FAIL: tolerance breached\n");
        return 5;
    }
    return 0;
}
