// Encoder chunked-vs-single-shot parity test.
//
// The acoustic / semantic encoders are causal stacks of conv1d + ConvNeXt
// blocks. Upstream's streaming path keeps a per-conv1d cache of the last
// `kernel_size - 1 - (stride - 1)` samples and feeds it as left context
// for the next chunk; that makes chunked encoding bit-identical (modulo
// floating-point noise) to a single-shot encode.
//
// Our v1 chunked path (`run_encoder_buf` in src/vibevoice_asr.cpp) is
// streamless: each chunk re-pads causally with zeros on the left, so the
// first ~6 input samples of every non-first chunk are wrong, and that
// error propagates downstream (smaller magnitude with each downsample,
// but visible). This test pins what we want to be true once the streaming
// cache lands: max-abs of (single_shot - chunked) < some small tolerance.
//
// Today this test is EXPECTED TO FAIL — it'll succeed once #41 (streaming-
// cache parity with upstream) ships.
//
// Skips (rc=77) unless VIBEVOICE_ASR_MODEL is set. Gated by
// VIBEVOICE_TEST_LARGE in CMake.

#include "acoustic_tokenizer.hpp"
#include "backend.hpp"
#include "ggml-cpu.h"
#include "ggml.h"
#include "vibevoice_tts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

bool file_ok(const char* p) {
    if (!p || !*p) return false;
    FILE* f = std::fopen(p, "rb");
    if (!f) return false;
    std::fclose(f); return true;
}

// Run the encoder on a full audio buffer in one ggml graph.
// Returns latents in [vae_dim, T_compressed] row-major (vae_dim innermost).
bool encode(const vv::EncoderWeights& w, const vv::AcousticConfig& cfg,
            const std::vector<float>& audio,
            std::vector<float>* out_latents,
            int* out_T_compressed) {
    struct ggml_init_params p {};
    // 64 KB / sample (matches the run_encoder_single_shot pool).
    p.mem_size = std::max<size_t>(1ull << 30,
                                  static_cast<size_t>(audio.size()) * 64ull * 1024);
    p.no_alloc = false;
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return false;
    const int T = static_cast<int>(audio.size());
    struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, 1, 1);
    std::memcpy(x->data, audio.data(), sizeof(float) * T);
    struct ggml_tensor* y_raw = vv::encoder_forward(ctx, x, w, cfg);
    struct ggml_tensor* y = ggml_cont(ctx, ggml_permute(ctx, y_raw, 1, 0, 2, 3));
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, y);
    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx); return false;
    }
    const int latent_dim = static_cast<int>(y->ne[0]);
    *out_T_compressed = static_cast<int>(y->ne[1]);
    out_latents->assign(static_cast<size_t>(latent_dim) * (*out_T_compressed), 0.0f);
    std::memcpy(out_latents->data(), y->data, sizeof(float) * out_latents->size());
    ggml_free(ctx);
    return true;
}

}  // namespace

int main() {
    const char* model_env = std::getenv("VIBEVOICE_ASR_MODEL");
    if (!file_ok(model_env)) {
        std::fprintf(stderr,
            "skip: set VIBEVOICE_ASR_MODEL to a converted asr-7b gguf path\n");
        return 77;
    }

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_env, &model)) {
        std::fprintf(stderr, "FAIL: load %s\n", model_env);
        return 1;
    }
    if (model.variant != "asr-7b") {
        std::fprintf(stderr, "FAIL: expected asr-7b, got %s\n", model.variant.c_str());
        return 2;
    }

    // Synthetic audio: 8 s of mixed sines so each conv layer sees structure
    // (a single tone leaves most channels at zero and hides streaming bugs
    // because zero × anything = 0).
    constexpr int kSampleRate    = 24000;
    constexpr int kCompressRatio = 3200;
    constexpr int kSeconds       = 8;
    const int T = kSeconds * kSampleRate;
    std::vector<float> audio(T);
    for (int i = 0; i < T; ++i) {
        const float t = static_cast<float>(i) / kSampleRate;
        audio[i] = 0.20f * std::sin(2.0f * 3.14159265f * 200.0f * t)
                + 0.10f * std::sin(2.0f * 3.14159265f * 800.0f * t)
                + 0.05f * std::sin(2.0f * 3.14159265f * 3000.0f * t);
    }

    // ---- Reference: single-shot encode of the full 8 s ----
    std::vector<float> ref_latents;
    int ref_Tc = 0;
    std::printf("encoding single-shot (%d samples)…\n", T);
    if (!encode(model.at_enc, model.cfg.acoustic, audio, &ref_latents, &ref_Tc)) {
        std::fprintf(stderr, "FAIL: single-shot encode\n"); return 3;
    }
    const int latent_dim = static_cast<int>(ref_latents.size() / std::max(ref_Tc, 1));
    std::printf("single-shot: shape [%d, %d]\n", ref_Tc, latent_dim);

    // ---- Chunked: split the same audio into N=4 chunks (2 s each), encode
    // each through encoder_forward_streaming with a shared StreamingCache,
    // concat the latents along the time axis. With the streaming cache,
    // every conv1d's left context for chunk K is the tail of chunk K-1, so
    // the concatenated output should match the single-shot reference up to
    // floating-point noise.
    const int chunk_samples = (kSeconds / 4) * kSampleRate;
    std::vector<float> chunked_latents;
    int chunked_Tc = 0;
    vv::StreamingCache cache;
    cache.is_first_chunk = true;
    for (int off = 0; off < T; off += chunk_samples) {
        const int end = std::min(off + chunk_samples, T);
        cache.is_final_chunk = (end == T);
        std::vector<float> seg(audio.begin() + off, audio.begin() + end);

        struct ggml_init_params p {};
        p.mem_size = ggml_tensor_overhead() * 32768
                   + ggml_graph_overhead_custom(32768, false);
        p.no_alloc = true;
        struct ggml_context* ctx = ggml_init(p);
        if (!ctx) return 4;
        struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                    static_cast<int>(seg.size()), 1, 1);
        struct ggml_tensor* y_raw = vv::encoder_forward_streaming(
            ctx, x, model.at_enc, model.cfg.acoustic, cache);
        struct ggml_tensor* y = ggml_cont(ctx, ggml_permute(ctx, y_raw, 1, 0, 2, 3));
        struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
        ggml_build_forward_expand(gf, y);
        for (auto& kv : cache) {
            if (kv.second.next_view) ggml_build_forward_expand(gf, kv.second.next_view);
        }

        ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
        if (!in_buf) { ggml_free(ctx); return 4; }
        ggml_backend_tensor_set(x, seg.data(), 0, sizeof(float) * seg.size());
        for (auto& kv : cache) {
            vv::StreamingCacheEntry& e = kv.second;
            if (!e.prefix || e.T == 0) continue;
            const size_t need = static_cast<size_t>(e.T) * e.C;
            if (cache.is_first_chunk || e.data.size() != need) {
                std::vector<float> zeros(need, 0.0f);
                ggml_backend_tensor_set(e.prefix, zeros.data(), 0, sizeof(float) * need);
            } else {
                ggml_backend_tensor_set(e.prefix, e.data.data(), 0, sizeof(float) * need);
            }
        }

        if (!vv::compute_graph(gf)) {
            ggml_backend_buffer_free(in_buf); ggml_free(ctx); return 4;
        }
        const int seg_dim = static_cast<int>(y->ne[0]);
        const int seg_T   = static_cast<int>(y->ne[1]);
        const size_t n_seg = static_cast<size_t>(seg_dim) * seg_T;
        chunked_latents.resize(chunked_latents.size() + n_seg);
        ggml_backend_tensor_get(y, chunked_latents.data() + chunked_latents.size() - n_seg,
                                0, sizeof(float) * n_seg);
        chunked_Tc += seg_T;
        for (auto& kv : cache) {
            vv::StreamingCacheEntry& e = kv.second;
            if (!e.next_view || e.T == 0) continue;
            const size_t n = static_cast<size_t>(e.T) * e.C;
            e.data.assign(n, 0.0f);
            ggml_backend_tensor_get(e.next_view, e.data.data(), 0, sizeof(float) * n);
            e.next_view = nullptr;
            e.prefix    = nullptr;
        }
        cache.is_first_chunk = false;
        ggml_backend_buffer_free(in_buf);
        ggml_free(ctx);
    }
    std::printf("chunked    : shape [%d, %d] (%d × 2 s chunks, cache=%zu entries)\n",
                chunked_Tc, latent_dim, T / chunk_samples, cache.size());

    if (chunked_Tc != ref_Tc) {
        std::fprintf(stderr,
            "FAIL: chunked frame count %d != single-shot %d "
            "(boundaries off — chunk_samples must be a multiple of %d)\n",
            chunked_Tc, ref_Tc, kCompressRatio);
        return 5;
    }

    // ---- Compare ----
    double max_abs = 0.0, sum_sq_diff = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref_latents.size(); ++i) {
        const double d = static_cast<double>(ref_latents[i]) - chunked_latents[i];
        max_abs = std::max(max_abs, std::fabs(d));
        sum_sq_diff += d * d;
        sum_sq_ref  += static_cast<double>(ref_latents[i]) * ref_latents[i];
    }
    const double rmse = std::sqrt(sum_sq_diff / ref_latents.size());
    const double rms_ref = std::sqrt(sum_sq_ref / ref_latents.size());
    std::printf("parity: max_abs=%.4f rmse=%.4f rms_ref=%.4f rmse/rms=%.3f\n",
                max_abs, rmse, rms_ref, rmse / std::max(rms_ref, 1e-12));

    // With the streaming cache, the only difference from single-shot
    // should be fp16 accumulation noise inside each conv1d. Tolerate up to
    // 1 % relative RMSE — anything worse means the cache is missing a
    // layer or saving the wrong slice.
    if (rmse / std::max(rms_ref, 1e-12) > 0.01) {
        std::fprintf(stderr,
            "FAIL: chunked encoder diverges from single-shot. "
            "max_abs=%.4f rmse=%.4f rms_ref=%.4f rmse/rms=%.3f\n",
            max_abs, rmse, rms_ref, rmse / std::max(rms_ref, 1e-12));
        return 6;
    }
    return 0;
}
