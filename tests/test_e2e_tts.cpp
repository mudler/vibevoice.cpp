// End-to-end TTS smoke test.
//
// Runs `vibevoice_tts_generate` on a tiny prompt with max_speech_frames=2
// and 5 diffusion steps, and verifies:
//   - the call succeeds
//   - non-empty audio is produced
//   - amplitude is in a sane (non-silent, non-clipped) range
//
// Skips (return 77) unless both VIBEVOICE_MODEL and VIBEVOICE_TOKENIZER env
// vars point at valid files. The fp32 gguf is large (~3.8 GB for the
// realtime variant) so this isn't run by default in CI.

#include "tokenizer.hpp"
#include "vibevoice_tts.hpp"

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
    const char* model_env = std::getenv("VIBEVOICE_MODEL");
    const char* tok_env   = std::getenv("VIBEVOICE_TOKENIZER");
    if (!file_ok(model_env) || !file_ok(tok_env)) {
        std::fprintf(stderr,
                     "skip: set VIBEVOICE_MODEL and VIBEVOICE_TOKENIZER to "
                     "validated gguf paths.\n");
        return 77;
    }

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_env, &model)) {
        std::fprintf(stderr, "load model failed\n");
        return 1;
    }
    if (!model.tokenizer.load_from_file(tok_env)) {
        std::fprintf(stderr, "load tokenizer failed\n");
        return 2;
    }

    vv::VibeVoiceTTSParams p;
    p.max_speech_frames = 2;
    p.n_diffusion_steps = 5;
    p.seed              = 42;
    p.verbose           = false;

    std::vector<float> samples;
    int rc = vv::vibevoice_tts_generate(&model, "Hi.", p, &samples);
    if (rc != 0) {
        std::fprintf(stderr, "generate rc=%d\n", rc);
        return 3;
    }

    if (samples.empty()) {
        std::fprintf(stderr, "FAIL: empty output\n");
        return 4;
    }
    // 1 frame = 3200 samples at 24 kHz (7.5 Hz frame rate × 1600 base × 2).
    // Expect 2 frames → 6400 samples.
    if (samples.size() < 3200ull || samples.size() > 8000ull) {
        std::fprintf(stderr, "FAIL: unexpected sample count %zu\n", samples.size());
        return 5;
    }

    double sum2 = 0, mn = 1e9, mx = -1e9;
    for (float s : samples) {
        sum2 += static_cast<double>(s) * s;
        if (s < mn) mn = s;
        if (s > mx) mx = s;
    }
    double rms = std::sqrt(sum2 / samples.size());

    std::printf("e2e_tts: %zu samples, rms=%.4f, range=[%.3f, %.3f]\n",
                samples.size(), rms, mn, mx);

    if (rms < 1e-4) {
        std::fprintf(stderr, "FAIL: silent output\n");
        return 6;
    }
    if (mn < -1.0f || mx > 1.0f) {
        std::fprintf(stderr, "FAIL: clipping\n");
        return 7;
    }
    return 0;
}
