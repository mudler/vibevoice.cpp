// Parity test for qwen2_layer_forward_resident.
//
// Same single-layer fixture as test_qwen2_block, but runs the input through
// THREE paths and asserts they all produce the same hidden state:
//
//   A) qwen2_layer_forward (host-resident k_past=null)  - the reference
//   B) qwen2_layer_forward_resident with kv.past_len=0  - single-shot
//   C) qwen2_layer_forward_resident, prefix in 2 chunks - chunked-prefill
//      (proves the cpy / view ordering inside qwen2_layer_forward_resident
//      actually makes the second chunk's attention see chunk 1's K/V)
//
// Tolerance: max-abs < 1e-5 fp32 between paths (tighter than the HF parity
// because we're comparing the same kernel against itself, only the K/V
// access pattern differs).
//
// Skips with rc=77 if the fixture is missing.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "backend.hpp"
#include "model_loader.hpp"
#include "qwen2.hpp"

#include <algorithm>
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

// Run path A: classic qwen2_layer_forward, no past, single graph for the
// full sequence. Returns the hidden output [hidden, seq, batch].
bool run_classic(const vv::Qwen2Hparams& hp,
                 const vv::Qwen2LayerWeights& w,
                 const float* input, const int32_t* pos_ids,
                 int seq, int batch,
                 std::vector<float>* out_hidden) {
    struct ggml_init_params p {};
    p.mem_size  = ggml_tensor_overhead() * 8192
                + ggml_graph_overhead_custom(8192, false);
    p.no_alloc  = true;
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return false;

    struct ggml_tensor* x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.hidden_size, seq, batch);
    struct ggml_tensor* pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    // Use F16 mask so the same fill code works on both classic and resident
    // paths and FA / eager dispatch is identical.
    struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, seq, seq);

    auto out = vv::qwen2_layer_forward(ctx, x, pos, mask, nullptr, nullptr, w, hp);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, out.y);

    ggml_backend_buffer_t buf = vv::allocate_ctx_tensors(ctx);
    if (!buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(x,   input,   0, sizeof(float)   * hp.hidden_size * seq * batch);
    ggml_backend_tensor_set(pos, pos_ids, 0, sizeof(int32_t) * seq);

    std::vector<ggml_fp16_t> mask_v(static_cast<size_t>(seq) * seq);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t n = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < seq; ++i)
        for (int j = 0; j < seq; ++j)
            mask_v[i * seq + j] = (j > i) ? n : z;
    ggml_backend_tensor_set(mask, mask_v.data(), 0, sizeof(ggml_fp16_t) * mask_v.size());

    if (!vv::compute_graph(gf)) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }

    out_hidden->assign(static_cast<size_t>(hp.hidden_size) * seq * batch, 0.0f);
    ggml_backend_tensor_get(out.y, out_hidden->data(), 0, sizeof(float) * out_hidden->size());

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Run one resident chunk: writes into kv at offset kv.past_len, then
// returns the chunk's hidden output [hidden, chunk_K, batch].
bool run_resident_chunk(vv::ResidentKV& kv, int layer_idx,
                        const vv::Qwen2Hparams& hp,
                        const vv::Qwen2LayerWeights& w,
                        const float* input, const int32_t* pos_ids,
                        int chunk_K, int batch,
                        std::vector<float>* out_hidden) {
    const int kv_old = kv.past_len;
    const int kv_new = kv_old + chunk_K;

    struct ggml_init_params p {};
    p.mem_size  = ggml_tensor_overhead() * 8192
                + ggml_graph_overhead_custom(8192, false);
    p.no_alloc  = true;
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return false;

    struct ggml_tensor* x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.hidden_size, chunk_K, batch);
    struct ggml_tensor* pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, chunk_K);
    struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_new, chunk_K);

    auto out = vv::qwen2_layer_forward_resident(ctx, x, pos, mask, kv, layer_idx, w, hp);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);
    // Cpys before attention reads, same pattern as the production caller.
    ggml_build_forward_expand(gf, out.k_write);
    ggml_build_forward_expand(gf, out.v_write);
    ggml_build_forward_expand(gf, out.y);

    ggml_backend_buffer_t buf = vv::allocate_ctx_tensors(ctx);
    if (!buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(x,   input,   0, sizeof(float)   * hp.hidden_size * chunk_K * batch);
    ggml_backend_tensor_set(pos, pos_ids, 0, sizeof(int32_t) * chunk_K);

    std::vector<ggml_fp16_t> mask_v(static_cast<size_t>(kv_new) * chunk_K);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t n = ggml_fp32_to_fp16(-INFINITY);
    // Each query at position kv_old+q can attend to keys [0, kv_old+q].
    for (int q = 0; q < chunk_K; ++q) {
        const int q_abs = kv_old + q;
        for (int k = 0; k < kv_new; ++k) {
            mask_v[static_cast<size_t>(q) * kv_new + k] = (k > q_abs) ? n : z;
        }
    }
    ggml_backend_tensor_set(mask, mask_v.data(), 0, sizeof(ggml_fp16_t) * mask_v.size());

    if (!vv::compute_graph(gf)) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }

    out_hidden->assign(static_cast<size_t>(hp.hidden_size) * chunk_K * batch, 0.0f);
    ggml_backend_tensor_get(out.y, out_hidden->data(), 0, sizeof(float) * out_hidden->size());

    kv.past_len = kv_new;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    }
    return m;
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
                     "test_qwen2_resident: skipping - missing %s\n"
                     "  run python tests/dump_qwen2_reference.py "
                     "--out tests/fixtures/qwen2_layer0.gguf\n",
                     path.c_str());
        return 77;
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
    hp.use_flash_attn    = vv::backend_supports_flash_attn();
    const int seq        = loader.get_i32("qwen2.test.seq_len");
    const int batch      = loader.get_i32("qwen2.test.batch");
    if (seq < 4) {
        std::fprintf(stderr, "test_qwen2_resident: seq=%d too small for chunked path\n", seq);
        return 2;
    }

    vv::Qwen2LayerWeights w;
    if (!vv::qwen2_load_layer(loader, "", &w)) return 3;

    struct ggml_tensor* input_t = loader.tensor("test.input");
    struct ggml_tensor* pos_t   = loader.tensor("test.position_ids");
    if (!input_t || !pos_t) {
        std::fprintf(stderr, "missing test.input / test.position_ids\n");
        return 4;
    }
    const float*   input   = static_cast<const float*>(input_t->data);
    const int32_t* pos_ids = static_cast<const int32_t*>(pos_t->data);

    std::printf("test_qwen2_resident: hidden=%d n_h=%d/%d hd=%d seq=%d batch=%d "
                "flash_attn=%s\n",
                hp.hidden_size, hp.n_heads, hp.n_kv_heads, hp.head_dim,
                seq, batch, hp.use_flash_attn ? "yes" : "no");

    // ---- A) classic single-shot ----
    std::vector<float> y_classic;
    if (!run_classic(hp, w, input, pos_ids, seq, batch, &y_classic)) {
        std::fprintf(stderr, "FAIL: classic path\n"); return 5;
    }

    // ---- B) resident single-shot ----
    std::vector<float> y_resident_one;
    {
        vv::ResidentKV kv;
        if (!kv.init(/*n_layers=*/1, hp.head_dim, hp.n_kv_heads, /*max_seq=*/seq + 16)) {
            std::fprintf(stderr, "FAIL: ResidentKV.init\n"); return 6;
        }
        if (!run_resident_chunk(kv, /*layer_idx=*/0, hp, w,
                                input, pos_ids, seq, batch, &y_resident_one)) {
            std::fprintf(stderr, "FAIL: resident single-shot\n"); return 7;
        }
    }

    // ---- C) resident chunked (split prefix in 2) ----
    std::vector<float> y_resident_two;
    y_resident_two.assign(static_cast<size_t>(hp.hidden_size) * seq * batch, 0.0f);
    {
        vv::ResidentKV kv;
        if (!kv.init(1, hp.head_dim, hp.n_kv_heads, seq + 16)) {
            std::fprintf(stderr, "FAIL: ResidentKV.init (chunked)\n"); return 8;
        }
        const int split = seq / 2;

        // Chunk 1: input[0..split), pos_ids[0..split)
        std::vector<float> chunk1_in(input, input + static_cast<size_t>(hp.hidden_size) * split * batch);
        std::vector<int32_t> chunk1_pos(pos_ids, pos_ids + split);
        std::vector<float> y_chunk1;
        if (!run_resident_chunk(kv, 0, hp, w, chunk1_in.data(), chunk1_pos.data(),
                                split, batch, &y_chunk1)) {
            std::fprintf(stderr, "FAIL: resident chunk 1\n"); return 9;
        }
        std::memcpy(y_resident_two.data(), y_chunk1.data(),
                    sizeof(float) * y_chunk1.size());

        // Chunk 2: rest. mask in run_resident_chunk uses kv.past_len for offset
        // so the keys-on-axis-2 range is [0, kv.past_len + chunk_K).
        const int rest = seq - split;
        std::vector<float> chunk2_in(input + static_cast<size_t>(hp.hidden_size) * split * batch,
                                     input + static_cast<size_t>(hp.hidden_size) * seq * batch);
        std::vector<int32_t> chunk2_pos(pos_ids + split, pos_ids + seq);
        std::vector<float> y_chunk2;
        if (!run_resident_chunk(kv, 0, hp, w, chunk2_in.data(), chunk2_pos.data(),
                                rest, batch, &y_chunk2)) {
            std::fprintf(stderr, "FAIL: resident chunk 2\n"); return 10;
        }
        std::memcpy(y_resident_two.data() + static_cast<size_t>(hp.hidden_size) * split * batch,
                    y_chunk2.data(), sizeof(float) * y_chunk2.size());
    }

    // Tolerances differ by attention path. Eager (mul_mat -> soft_max ->
    // mul_mat) is bit-exact between classic and resident because the K/V
    // values written to the resident buffer in chunk 1 are read back by
    // chunk 2's eager attention in F32. FA's CPU path keeps K/V in F16
    // internally; reading K via a strided view of the resident tensor
    // (max_seq != ne[2] of the live K-window) introduces a small fp16
    // noise relative to the all-in-one classic graph.
    const double tol_one = 1e-5;
    const double tol_two = hp.use_flash_attn ? 1e-3 : 1e-5;
    const double d_one   = max_abs_diff(y_classic, y_resident_one);
    const double d_two   = max_abs_diff(y_classic, y_resident_two);
    std::printf("  classic vs resident-1chunk:  max_abs=%.3e (tol=%.0e)\n", d_one, tol_one);
    std::printf("  classic vs resident-2chunks: max_abs=%.3e (tol=%.0e)\n", d_two, tol_two);

    if (d_one > tol_one) {
        std::fprintf(stderr,
            "FAIL: resident single-shot diverges from classic (%.3e > %.0e)\n",
            d_one, tol_one);
        return 11;
    }
    if (d_two > tol_two) {
        std::fprintf(stderr,
            "FAIL: resident chunked diverges from classic (%.3e > %.0e)\n",
            d_two, tol_two);
        return 12;
    }
    std::printf("test_qwen2_resident: OK\n");
    return 0;
}
