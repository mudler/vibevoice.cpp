// SConv1d (causal Conv1d) parity test against PyTorch.
//
// Three cases: basic (groups=1), strided (groups=1, stride=2 with alignment
// padding), and depthwise (groups=in=out).

#include "conv1d.hpp"
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
    std::fclose(f);
    return true;
}

int run_case(const std::string& path, double tol) {
    if (!file_ok(path)) {
        std::fprintf(stderr, "skip: missing %s\n", path.c_str());
        return 77;
    }
    vv::ModelLoader loader;
    if (!loader.load(path)) return 1;

    const int in_ch    = loader.get_i32("sconv.in_ch");
    const int out_ch   = loader.get_i32("sconv.out_ch");
    const int kernel   = loader.get_i32("sconv.kernel");
    const int stride   = loader.get_i32("sconv.stride");
    const int dilation = loader.get_i32("sconv.dilation");
    const int groups   = loader.get_i32("sconv.groups");
    const int T        = loader.get_i32("sconv.T");

    struct ggml_tensor* in_t  = loader.tensor("test.input");
    struct ggml_tensor* exp_t = loader.tensor("test.expected_output");
    struct ggml_tensor* w_t   = loader.tensor("weight.kernel");
    struct ggml_tensor* b_t   = loader.tensor("weight.bias");
    if (!in_t || !exp_t || !w_t || !b_t) return 2;

    struct ggml_init_params p {};
    p.mem_size = 64ull * 1024 * 1024;
    p.no_alloc = false;
    struct ggml_context* ctx = ggml_init(p);

    struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, in_ch, 1);
    std::memcpy(x->data, in_t->data, ggml_nbytes(x));

    struct ggml_tensor* y = vv::sconv1d_causal(ctx, x, w_t, b_t, stride, dilation, groups);
    if (!y) { ggml_free(ctx); return 3; }
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx); return 4;
    }

    if (y->ne[0] != exp_t->ne[0] || y->ne[1] != exp_t->ne[1]) {
        std::fprintf(stderr, "shape mismatch: got [%lld,%lld,%lld] vs [%lld,%lld,%lld]\n",
                     (long long)y->ne[0], (long long)y->ne[1], (long long)y->ne[2],
                     (long long)exp_t->ne[0], (long long)exp_t->ne[1], (long long)exp_t->ne[2]);
        ggml_free(ctx); return 5;
    }

    const size_t n = static_cast<size_t>(y->ne[0]) * y->ne[1] * y->ne[2];
    const float* a = static_cast<const float*>(y->data);
    const float* e = static_cast<const float*>(exp_t->data);
    double max_abs = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(e[i]));
        if (d > max_abs) max_abs = d;
    }
    std::printf("%-22s max_abs=%.3e  Tin=%d Tout=%d k=%d s=%d g=%d\n",
                path.c_str(), max_abs, T, (int)y->ne[0], kernel, stride, groups);

    ggml_free(ctx);
    return max_abs < tol ? 0 : 6;
}

int run_convt_case(const std::string& path, double tol) {
    if (!file_ok(path)) {
        std::fprintf(stderr, "skip: missing %s\n", path.c_str());
        return 77;
    }
    vv::ModelLoader loader;
    if (!loader.load(path)) return 1;

    const int in_ch  = loader.get_i32("convt.in_ch");
    const int out_ch = loader.get_i32("convt.out_ch");
    const int kernel = loader.get_i32("convt.kernel");
    const int stride = loader.get_i32("convt.stride");
    const int T      = loader.get_i32("convt.T");
    (void)in_ch; (void)out_ch;

    struct ggml_tensor* in_t  = loader.tensor("test.input");
    struct ggml_tensor* exp_t = loader.tensor("test.expected_output");
    struct ggml_tensor* w_t   = loader.tensor("weight.kernel");
    struct ggml_tensor* b_t   = loader.tensor("weight.bias");
    if (!in_t || !exp_t || !w_t || !b_t) return 2;

    struct ggml_init_params p {};
    p.mem_size = 64ull * 1024 * 1024;
    p.no_alloc = false;
    struct ggml_context* ctx = ggml_init(p);

    struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, in_t->ne[1], 1);
    std::memcpy(x->data, in_t->data, ggml_nbytes(x));

    struct ggml_tensor* y = vv::sconv_transpose1d_causal(ctx, x, w_t, b_t, stride);
    if (!y) { ggml_free(ctx); return 3; }
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx); return 4;
    }
    if (y->ne[0] != exp_t->ne[0] || y->ne[1] != exp_t->ne[1]) {
        std::fprintf(stderr, "convt shape mismatch: got [%lld,%lld,%lld] vs [%lld,%lld,%lld]\n",
                     (long long)y->ne[0], (long long)y->ne[1], (long long)y->ne[2],
                     (long long)exp_t->ne[0], (long long)exp_t->ne[1], (long long)exp_t->ne[2]);
        ggml_free(ctx); return 5;
    }
    const size_t n = static_cast<size_t>(y->ne[0]) * y->ne[1] * y->ne[2];
    const float* a = static_cast<const float*>(y->data);
    const float* e = static_cast<const float*>(exp_t->data);
    double max_abs = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(e[i]));
        if (d > max_abs) max_abs = d;
    }
    std::printf("%-22s max_abs=%.3e  Tin=%d Tout=%d k=%d s=%d (transpose)\n",
                path.c_str(), max_abs, T, (int)y->ne[0], kernel, stride);
    ggml_free(ctx);
    return max_abs < tol ? 0 : 6;
}

}  // namespace

int main() {
#ifndef VV_FIXTURES_DIR
#  define VV_FIXTURES_DIR "tests/fixtures"
#endif
    const std::string fix = VV_FIXTURES_DIR;
    int rc = 0;
    int s;
    s = run_case      (fix + "/sconv1d_basic.gguf",    2e-3); if (s != 0 && s != 77) rc = rc ? rc : s;
    s = run_case      (fix + "/sconv1d_strided.gguf",  2e-3); if (s != 0 && s != 77) rc = rc ? rc : s;
    s = run_case      (fix + "/sconv1d_dw.gguf",       2e-3); if (s != 0 && s != 77) rc = rc ? rc : s;
    s = run_convt_case(fix + "/sconvt1d_basic.gguf",   2e-3); if (s != 0 && s != 77) rc = rc ? rc : s;
    s = run_convt_case(fix + "/sconvt1d_long.gguf",    2e-3); if (s != 0 && s != 77) rc = rc ? rc : s;
    return rc;
}
