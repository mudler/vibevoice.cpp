// Acoustic tokenizer encoder + decoder parity tests.
//
// Runs both forward paths against a small random-init PyTorch reference
// (tests/dump_acoustic_reference.py). fp16 conv floor → tolerance ~5e-3
// (a single block was 4e-4; depth 3 + multiple convs accumulate).

#include "acoustic_tokenizer.hpp"
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
    const std::string path = std::string(VV_FIXTURES_DIR) + "/acoustic.gguf";
    if (!file_ok(path)) {
        std::fprintf(stderr, "skip: missing %s\n  run: python tests/dump_acoustic_reference.py "
                     "--out %s\n", path.c_str(), path.c_str());
        return 77;
    }

    vv::ModelLoader loader;
    if (!loader.load(path)) return 1;

    vv::AcousticConfig cfg;
    cfg.channels  = loader.get_i32 ("acoustic.channels");
    cfg.vae_dim   = loader.get_i32 ("acoustic.vae_dim");
    cfg.n_filters = loader.get_i32 ("acoustic.n_filters");
    cfg.eps       = loader.get_f32 ("acoustic.eps", 1e-5f);
    auto ratios   = loader.get_i32_array("acoustic.ratios");
    auto depths   = loader.get_i32_array("acoustic.depths");
    cfg.ratios.assign(ratios.begin(), ratios.end());
    cfg.depths.assign(depths.begin(), depths.end());
    const int T          = loader.get_i32("acoustic.T");
    const int T_compr    = loader.get_i32("acoustic.T_compressed");
    const int T_dec_in   = loader.get_i32("acoustic.T_dec_in");
    const int T_dec_out  = loader.get_i32("acoustic.T_dec_out");
    (void)T_dec_out;

    vv::EncoderWeights ew;
    vv::DecoderWeights dw;
    if (!vv::load_encoder(loader, "enc", cfg, &ew)) return 2;
    if (!vv::load_decoder(loader, "dec", cfg, &dw)) return 3;

    struct ggml_tensor* aud_t      = loader.tensor("test.audio");
    struct ggml_tensor* enc_exp_t  = loader.tensor("test.encoder_output");
    struct ggml_tensor* z_t        = loader.tensor("test.decoder_input");
    struct ggml_tensor* dec_exp_t  = loader.tensor("test.decoder_output");
    if (!aud_t || !enc_exp_t || !z_t || !dec_exp_t) return 4;

    auto run_block = [&](const char* name,
                         struct ggml_tensor* in_t,
                         struct ggml_tensor* exp_t,
                         bool is_decoder,
                         double tol) -> int {
        struct ggml_init_params p {};
        p.mem_size = 256ull * 1024 * 1024;
        p.no_alloc = false;
        struct ggml_context* ctx = ggml_init(p);

        // input shape from gguf: ggml [T, C, B]
        struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                   in_t->ne[0], in_t->ne[1], in_t->ne[2]);
        std::memcpy(x->data, in_t->data, ggml_nbytes(x));

        struct ggml_tensor* y = is_decoder
            ? vv::decoder_forward(ctx, x, dw, cfg)
            : vv::encoder_forward(ctx, x, ew, cfg);

        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx); return 1;
        }

        if (y->ne[0] != exp_t->ne[0] || y->ne[1] != exp_t->ne[1]) {
            std::fprintf(stderr, "%s shape mismatch: got [%lld,%lld] vs [%lld,%lld]\n",
                         name,
                         (long long)y->ne[0], (long long)y->ne[1],
                         (long long)exp_t->ne[0], (long long)exp_t->ne[1]);
            ggml_free(ctx); return 2;
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
        std::printf("%s: max_abs=%.3e  cos=%.6f  shape=[%lld,%lld,%lld]\n",
                    name, max_abs, cos,
                    (long long)y->ne[0], (long long)y->ne[1], (long long)y->ne[2]);
        ggml_free(ctx);
        return (max_abs < tol && cos > 0.999) ? 0 : 3;
    };

    (void)T; (void)T_compr; (void)T_dec_in;
    int rc = 0;
    int s;
    s = run_block("encoder", aud_t, enc_exp_t, /*decoder=*/false, /*tol=*/5e-3);
    if (s != 0) rc = rc ? rc : s;
    s = run_block("decoder", z_t,   dec_exp_t, /*decoder=*/true,  /*tol=*/5e-3);
    if (s != 0) rc = rc ? rc : s;
    return rc;
}
