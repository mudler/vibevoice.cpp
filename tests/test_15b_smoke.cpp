// Smoke test for the 1.5B TTS path.
//
// Loads the 1.5B gguf, conditions on a short reference WAV, runs
// vibevoice_tts_15b_generate against a fixed text, and asserts the
// output is non-empty and within reasonable bounds. This does NOT
// check audio quality — that's the closed-loop test's job.
//
// Skips with rc=77 unless these env vars all point at valid files:
//   VIBEVOICE_TTS_15B_MODEL  -> 1.5B gguf
//   VIBEVOICE_TOKENIZER      -> tokenizer gguf
//   VIBEVOICE_REF_WAV        -> 24 kHz mono wav (samples/3p_gpt5.wav)

#include "model_loader.hpp"
#include "vibevoice_tts.hpp"
#include "audio_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
bool file_ok(const char* p) {
    if (!p || !*p) return false;
    FILE* f = std::fopen(p, "rb");
    if (!f) return false;
    std::fclose(f); return true;
}
}  // namespace

int main() {
    const char* model_path = std::getenv("VIBEVOICE_TTS_15B_MODEL");
    const char* tok_path   = std::getenv("VIBEVOICE_TOKENIZER");
    const char* ref_wav    = std::getenv("VIBEVOICE_REF_WAV");
    if (!file_ok(model_path) || !file_ok(tok_path) || !file_ok(ref_wav)) {
        std::fprintf(stderr,
            "skip: 1.5b smoke test needs VIBEVOICE_TTS_15B_MODEL + "
            "VIBEVOICE_TOKENIZER + VIBEVOICE_REF_WAV (got model=%s tok=%s wav=%s)\n",
            model_path ? model_path : "(null)",
            tok_path   ? tok_path   : "(null)",
            ref_wav    ? ref_wav    : "(null)");
        return 77;
    }

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_path, &model)) {
        std::fprintf(stderr, "FAIL: load %s\n", model_path);
        return 1;
    }
    if (!model.tokenizer.load_from_file(tok_path)) {
        std::fprintf(stderr, "FAIL: tokenizer load %s\n", tok_path);
        return 2;
    }
    if (model.variant != "1.5b") {
        std::fprintf(stderr, "FAIL: variant=%s want 1.5b\n", model.variant.c_str());
        return 3;
    }

    vv::VibeVoiceTTSParams p;
    p.ref_audio_path    = ref_wav;
    p.max_speech_frames = 64;          // short — smoke only
    p.n_diffusion_steps = 10;          // fewer steps for speed
    p.cfg_scale         = 1.0f;        // no CFG yet
    p.seed              = 12345;
    p.verbose           = true;

    std::vector<float> samples;
    int rc = vv::vibevoice_tts_generate(&model, "Hello world.", p, &samples);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: vibevoice_tts_15b_generate rc=%d\n", rc);
        return 4;
    }
    if (samples.empty()) {
        std::fprintf(stderr, "FAIL: no samples produced\n");
        return 5;
    }

    // Sanity: 1 frame of compressed latent decodes to ~1600 samples (24 kHz /
    // 15-fold compression, the same ratio the realtime decoder produces).
    // p.max_speech_frames = 64 caps the upper bound; we want at least a few
    // hundred samples so we know SOMETHING ran end-to-end.
    if (samples.size() < 1024) {
        std::fprintf(stderr, "FAIL: too few samples (%zu)\n", samples.size());
        return 6;
    }
    // RMS sanity: should be non-silent.
    double sq = 0.0;
    for (float v : samples) sq += static_cast<double>(v) * v;
    const double rms = std::sqrt(sq / samples.size());
    std::printf("[smoke] %zu samples, rms=%.4f\n", samples.size(), rms);
    if (!std::isfinite(rms) || rms < 1e-4) {
        std::fprintf(stderr, "FAIL: silent output (rms=%.6f)\n", rms);
        return 7;
    }

    // Save for inspection if requested.
    const char* out = std::getenv("VIBEVOICE_15B_SMOKE_OUT");
    if (out && out[0]) {
        vv_audio audio_out{};
        audio_out.samples     = samples.data();
        audio_out.n_samples   = samples.size();
        audio_out.sample_rate = 24000;
        audio_out.channels    = 1;
        if (vv::save_wav_pcm16(out, audio_out) == 0) {
            std::printf("[smoke] saved %s\n", out);
        }
    }
    return 0;
}
