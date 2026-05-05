// Smoke test for the flat C ABI in include/vibevoice_capi.h.
//
// Exercises the same TTS → wav → ASR roundtrip as test_closed_loop, but
// through the C API entry points (vv_capi_load / vv_capi_tts / vv_capi_asr)
// rather than the C++ orchestrators directly. This pins the symbols that
// LocalAI's purego backend will dlsym at runtime.
//
// Skips (return 77) unless VIBEVOICE_TTS_MODEL, VIBEVOICE_VOICE,
// VIBEVOICE_ASR_MODEL, VIBEVOICE_TOKENIZER are all set. Gated by
// VIBEVOICE_TEST_LARGE in CMake.

#include "vibevoice_capi.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <string>

namespace {

bool file_ok(const char* p) {
    if (!p || !*p) return false;
    FILE* f = std::fopen(p, "rb");
    if (!f) return false;
    std::fclose(f); return true;
}

std::set<std::string> word_set(const std::string& s) {
    std::string clean;
    clean.reserve(s.size());
    for (char c : s) {
        clean += std::isalnum(static_cast<unsigned char>(c))
                 ? static_cast<char>(std::tolower(static_cast<unsigned char>(c)))
                 : ' ';
    }
    std::set<std::string> out;
    std::istringstream iss(clean);
    for (std::string w; iss >> w; ) out.insert(w);
    return out;
}

std::string extract_content(const std::string& s) {
    const std::string key = "\"Content\":\"";
    std::string out;
    for (size_t pos = 0; (pos = s.find(key, pos)) != std::string::npos; ) {
        pos += key.size();
        size_t end = pos;
        while (end < s.size() && s[end] != '"') {
            end += (s[end] == '\\' && end + 1 < s.size()) ? 2 : 1;
        }
        if (!out.empty()) out += ' ';
        out.append(s, pos, end - pos);
        pos = end;
    }
    return out;
}

}  // namespace

int main() {
    const char* tts   = std::getenv("VIBEVOICE_TTS_MODEL");
    const char* voice = std::getenv("VIBEVOICE_VOICE");
    const char* asr   = std::getenv("VIBEVOICE_ASR_MODEL");
    const char* tok   = std::getenv("VIBEVOICE_TOKENIZER");
    if (!file_ok(tts) || !file_ok(voice) || !file_ok(asr) || !file_ok(tok)) {
        std::fprintf(stderr,
            "skip: capi test needs VIBEVOICE_{TTS_MODEL,VOICE,ASR_MODEL,"
            "TOKENIZER} all set.\n");
        return 77;
    }

    std::printf("[capi] %s\n", vv_capi_version());

    int rc = vv_capi_load(tts, asr, tok, voice, /*n_threads=*/0);
    if (rc != 0) { std::fprintf(stderr, "FAIL: vv_capi_load rc=%d\n", rc); return 1; }

    const std::string source = "Hello world this is a capi smoke test.";
    const std::string wav    = "/tmp/vibevoice_capi_smoke.wav";

    rc = vv_capi_tts(source.c_str(),
                     /*voice_path=*/nullptr, /*ref_audio_path=*/nullptr,
                     wav.c_str(),
                     /*steps=*/20, /*cfg=*/1.3f,
                     /*max_speech_frames=*/200, /*seed=*/0xCAFE);
    if (rc != 0) { std::fprintf(stderr, "FAIL: vv_capi_tts rc=%d\n", rc); return 2; }

    char buf[8192] = {0};
    rc = vv_capi_asr(wav.c_str(), buf, sizeof(buf), /*max_new_tokens=*/256);
    if (rc <= 0) { std::fprintf(stderr, "FAIL: vv_capi_asr rc=%d\n", rc); return 3; }
    std::printf("[capi] asr: %s\n", buf);

    const std::string content = extract_content(buf);
    if (content.empty()) {
        std::fprintf(stderr, "FAIL: empty Content field\n"); return 4;
    }
    auto src_w = word_set(source);
    auto out_w = word_set(content);
    size_t hits = 0;
    for (const auto& w : src_w) if (out_w.count(w)) ++hits;
    const double recall = static_cast<double>(hits)
                        / static_cast<double>(src_w.size());
    std::printf("[capi] %zu/%zu source words recovered (%.1f%%)\n",
                hits, src_w.size(), recall * 100.0);
    if (recall < 0.6) {  // looser than closed-loop because TTS+ASR in same proc
        std::fprintf(stderr, "FAIL: recall %.2f < 0.60\n", recall);
        return 5;
    }

    vv_capi_unload();
    std::remove(wav.c_str());

    // ---- 1.5B path (optional) ----
    // If VIBEVOICE_TTS_15B_MODEL + VIBEVOICE_REF_WAV are set, exercise
    // the unified vv_capi_tts entry point with `ref_audio_path` (runtime
    // voice cloning) and round-trip through ASR. Skipped silently
    // otherwise so the existing realtime-only test still passes on
    // contributors who haven't downloaded the 1.5B gguf.
    const char* tts15b  = std::getenv("VIBEVOICE_TTS_15B_MODEL");
    const char* ref_wav = std::getenv("VIBEVOICE_REF_WAV");
    if (file_ok(tts15b) && file_ok(ref_wav)) {
        std::printf("[capi] 1.5b: loading %s\n", tts15b);
        rc = vv_capi_load(tts15b, asr, tok, /*voice_path=*/nullptr,
                          /*n_threads=*/0);
        if (rc != 0) { std::fprintf(stderr, "FAIL: vv_capi_load (1.5b) rc=%d\n", rc); return 6; }

        const std::string source15b = "Hello world this is a capi smoke test for runtime voice cloning.";
        const std::string wav15b    = "/tmp/vibevoice_capi_15b.wav";
        rc = vv_capi_tts(source15b.c_str(),
                         /*voice_path=*/nullptr, /*ref_audio_path=*/ref_wav,
                         wav15b.c_str(),
                         /*steps=*/20, /*cfg_scale=*/1.3f,
                         /*max_speech_frames=*/200, /*seed=*/0xCAFE);
        if (rc != 0) { std::fprintf(stderr, "FAIL: vv_capi_tts (1.5b) rc=%d\n", rc); return 7; }

        std::memset(buf, 0, sizeof(buf));
        rc = vv_capi_asr(wav15b.c_str(), buf, sizeof(buf), /*max_new_tokens=*/256);
        if (rc <= 0) { std::fprintf(stderr, "FAIL: vv_capi_asr (1.5b) rc=%d\n", rc); return 8; }
        std::printf("[capi] 1.5b asr: %s\n", buf);

        const std::string content15b = extract_content(buf);
        auto src15b = word_set(source15b);
        auto out15b = word_set(content15b);
        size_t hits15b = 0;
        for (const auto& w : src15b) if (out15b.count(w)) ++hits15b;
        const double recall15b = static_cast<double>(hits15b)
                                / static_cast<double>(src15b.size());
        std::printf("[capi] 1.5b %zu/%zu source words recovered (%.1f%%)\n",
                    hits15b, src15b.size(), recall15b * 100.0);
        if (recall15b < 0.6) {
            std::fprintf(stderr, "FAIL: 1.5b recall %.2f < 0.60\n", recall15b);
            return 9;
        }

        vv_capi_unload();
        std::remove(wav15b.c_str());
    }
    return 0;
}
