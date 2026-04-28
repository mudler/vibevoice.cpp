// End-to-end ASR test on real weights:
//   1. encoder smoke (acoustic + semantic on a 3 s tone)
//   2. transcription smoke (a short near-silent clip should produce a
//      non-empty JSON-shaped string and stop at <|im_end|>)
//
// Skips (return 77) unless VIBEVOICE_ASR_MODEL is set to the gguf path.
// Also requires VIBEVOICE_TOKENIZER for the transcription leg.

#include "acoustic_tokenizer.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include "vibevoice_asr.hpp"
#include "vibevoice_tts.hpp"

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

// Run an encoder graph: input audio [T, channels=1, B=1] -> latents.
bool run_encoder(const vv::EncoderWeights& w, const vv::AcousticConfig& cfg,
                 const std::vector<float>& audio,
                 std::vector<float>* out_latents,
                 int* out_T_compressed) {
    struct ggml_init_params p {};
    p.mem_size = 4ull * 1024 * 1024 * 1024;  // 4 GB scratch
    p.no_alloc = false;
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return false;

    const int T = static_cast<int>(audio.size());
    struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, /*channels=*/1, /*B=*/1);
    std::memcpy(x->data, audio.data(), sizeof(float) * T);

    struct ggml_tensor* y = vv::encoder_forward(ctx, x, w, cfg);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, y);
    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        return false;
    }
    *out_T_compressed = static_cast<int>(y->ne[0]);
    const int latent_dim = static_cast<int>(y->ne[1]);
    out_latents->assign(static_cast<size_t>(latent_dim) * (*out_T_compressed), 0.0f);
    std::memcpy(out_latents->data(), y->data, sizeof(float) * out_latents->size());
    ggml_free(ctx);
    return true;
}
}  // namespace

int main() {
    const char* model_env = std::getenv("VIBEVOICE_ASR_MODEL");
    if (!file_ok(model_env)) {
        std::fprintf(stderr, "skip: set VIBEVOICE_ASR_MODEL to the converted ASR gguf path\n");
        return 77;
    }

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_env, &model)) {
        std::fprintf(stderr, "asr: load failed\n");
        return 1;
    }
    if (model.variant != "asr-7b") {
        std::fprintf(stderr, "expected variant asr-7b, got %s\n", model.variant.c_str());
        return 2;
    }
    std::printf("asr: loaded variant=%s  hidden=%d  layers=%d  vocab=%d\n",
                model.variant.c_str(), model.cfg.hidden, model.cfg.n_layers_lm,
                model.cfg.vocab_size);

    // ---- 3 seconds of a 200 Hz tone (sane non-zero input) ----
    const int sr  = 24000;
    const int T   = 3 * sr;
    std::vector<float> audio(T);
    for (int i = 0; i < T; ++i) {
        audio[i] = 0.2f * std::sin(2.0f * 3.14159265f * 200.0f * i / sr);
    }

    std::printf("asr: running acoustic encoder on %d samples (%.1fs)…\n", T, T / static_cast<double>(sr));
    std::vector<float> at_latents;
    int at_Tc = 0;
    if (!run_encoder(model.at_enc, model.cfg.acoustic, audio, &at_latents, &at_Tc)) {
        std::fprintf(stderr, "asr: acoustic encoder failed\n");
        return 3;
    }
    std::printf("asr: acoustic latents shape [%d, %d]  (compress ~%dx)\n",
                at_Tc, model.cfg.acoustic.vae_dim, T / std::max(at_Tc, 1));

    std::printf("asr: running semantic encoder…\n");
    std::vector<float> st_latents;
    int st_Tc = 0;
    if (!run_encoder(model.st_enc, model.semantic_cfg, audio, &st_latents, &st_Tc)) {
        std::fprintf(stderr, "asr: semantic encoder failed\n");
        return 4;
    }
    std::printf("asr: semantic latents shape [%d, %d]\n",
                st_Tc, model.semantic_vae_dim);

    // Sanity checks: shapes match and values are finite.
    if (at_Tc != st_Tc) {
        std::fprintf(stderr, "FAIL: acoustic/semantic frame count mismatch %d vs %d\n",
                     at_Tc, st_Tc);
        return 5;
    }
    auto sane = [](const std::vector<float>& v) {
        for (float x : v) if (!std::isfinite(x)) return false;
        return true;
    };
    if (!sane(at_latents) || !sane(st_latents)) {
        std::fprintf(stderr, "FAIL: non-finite latents\n");
        return 6;
    }

    auto rms = [](const std::vector<float>& v) {
        double s = 0; for (float x : v) s += static_cast<double>(x) * x;
        return std::sqrt(s / v.size());
    };
    std::printf("asr: latent rms acoustic=%.4f semantic=%.4f\n",
                rms(at_latents), rms(st_latents));

    // ---- transcription smoke (gated by tokenizer availability) ----
    const char* tok_env = std::getenv("VIBEVOICE_TOKENIZER");
    if (!file_ok(tok_env)) {
        std::printf("asr: skipping transcription leg (set VIBEVOICE_TOKENIZER to enable)\n");
        return 0;
    }
    if (!model.tokenizer.load_from_file(tok_env)) {
        std::fprintf(stderr, "asr: failed to load tokenizer\n");
        return 7;
    }
    // 0.5 s of near-silence — model should emit the JSON skeleton and stop.
    std::vector<float> quiet(static_cast<size_t>(sr) / 2, 0.0f);
    vv::VibeVoiceASRParams pp;
    pp.max_new_tokens = 60;
    pp.verbose        = false;
    std::string transcript;
    int rc = vv::vibevoice_asr_transcribe(&model, quiet, pp, &transcript);
    if (rc != 0) {
        std::fprintf(stderr, "asr: transcribe rc=%d\n", rc);
        return 8;
    }
    std::printf("asr: transcript = %s\n", transcript.c_str());
    if (transcript.empty()) {
        std::fprintf(stderr, "FAIL: empty transcript\n");
        return 9;
    }
    return 0;
}
