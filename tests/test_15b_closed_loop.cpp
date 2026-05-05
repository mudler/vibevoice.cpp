// Closed-loop test for the 1.5B TTS path:
//
//   1. load 1.5B + ASR-7B
//   2. TTS_15B(ref_wav, "Hello world this is a test of voice cloning.")
//      -> WAV that should sound like the speaker in ref_wav saying that line
//   3. ASR_7B(WAV) -> transcript
//   4. assert >=80% source-word recall
//
// This is the regression gate for runtime voice cloning: cloning produces
// a usable, intelligible voice. Skips with rc=77 unless these env vars all
// point at valid files:
//
//   VIBEVOICE_TTS_15B_MODEL  -> 1.5B gguf
//   VIBEVOICE_ASR_MODEL      -> ASR-7B gguf
//   VIBEVOICE_TOKENIZER      -> tokenizer gguf
//   VIBEVOICE_REF_WAV        -> 24 kHz mono wav (any speaker, ~5 s is enough)

#include "audio_io.hpp"
#include "vibevoice_asr.hpp"
#include "vibevoice_tts.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <vector>

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
    const char* tts15b   = std::getenv("VIBEVOICE_TTS_15B_MODEL");
    const char* asr_path = std::getenv("VIBEVOICE_ASR_MODEL");
    const char* tok_path = std::getenv("VIBEVOICE_TOKENIZER");
    const char* ref_wav  = std::getenv("VIBEVOICE_REF_WAV");

    if (!file_ok(tts15b) || !file_ok(asr_path)
        || !file_ok(tok_path) || !file_ok(ref_wav)) {
        std::fprintf(stderr,
            "skip: closed-loop needs VIBEVOICE_{TTS_15B_MODEL,ASR_MODEL,"
            "TOKENIZER,REF_WAV} all set to valid files (got 15b=%s asr=%s "
            "tok=%s wav=%s)\n",
            tts15b   ? tts15b   : "(null)",
            asr_path ? asr_path : "(null)",
            tok_path ? tok_path : "(null)",
            ref_wav  ? ref_wav  : "(null)");
        return 77;
    }

    // ---- 1. load both models + tokenizer ----
    std::printf("[loop] loading 1.5B model %s\n", tts15b);
    vv::VibeVoiceModel m15;
    if (!vv::vibevoice_load(tts15b, &m15)) {
        std::fprintf(stderr, "FAIL: load 1.5B\n"); return 1;
    }
    if (!m15.tokenizer.load_from_file(tok_path)) {
        std::fprintf(stderr, "FAIL: tokenizer load\n"); return 2;
    }
    if (m15.variant != "1.5b") {
        std::fprintf(stderr, "FAIL: 1.5B model reports variant=%s\n",
                     m15.variant.c_str());
        return 3;
    }

    // ---- 2. TTS the source phrase, conditioning on ref_wav ----
    const std::string source =
        "Hello world this is a test of voice cloning.";

    vv::VibeVoiceTTSParams p;
    p.ref_audio_path    = ref_wav;
    p.max_speech_frames = 200;
    p.n_diffusion_steps = 20;
    p.cfg_scale         = 1.3f;
    p.seed              = 12345;
    p.verbose           = false;

    std::vector<float> samples;
    if (vv::vibevoice_tts_generate(&m15, source, p, &samples) != 0
        || samples.empty()) {
        std::fprintf(stderr, "FAIL: vibevoice_tts_generate (1.5b) returned no audio\n"); return 4;
    }
    std::printf("[loop] TTS produced %zu samples (%.2fs)\n",
                samples.size(), samples.size() / 24000.0);

    const std::string out_wav = "/tmp/vibevoice_15b_loop.wav";
    {
        vv_audio a{};
        a.samples     = samples.data();
        a.n_samples   = samples.size();
        a.sample_rate = 24000;
        a.channels    = 1;
        if (vv::save_wav_pcm16(out_wav.c_str(), a) != 0) {
            std::fprintf(stderr, "FAIL: save wav\n"); return 5;
        }
    }

    // ---- 3. transcribe with ASR-7B and assert recall ----
    std::printf("[loop] loading ASR model %s\n", asr_path);
    vv::VibeVoiceModel masr;
    if (!vv::vibevoice_load(asr_path, &masr)) {
        std::fprintf(stderr, "FAIL: load ASR\n"); return 6;
    }
    if (!masr.tokenizer.load_from_file(tok_path)) {
        std::fprintf(stderr, "FAIL: ASR tokenizer load\n"); return 7;
    }

    vv::VibeVoiceASRParams ap;
    ap.max_new_tokens     = 256;
    ap.repetition_penalty = 1.0f;
    ap.no_repeat_ngram    = 0;
    ap.verbose            = false;

    std::string transcript;
    if (vv::vibevoice_asr_transcribe(&masr, samples, ap, &transcript) != 0) {
        std::fprintf(stderr, "FAIL: ASR transcribe\n"); return 8;
    }
    const std::string content = extract_content(transcript);
    std::printf("[loop] transcript: %s\n", content.c_str());

    const auto src_w = word_set(source);
    const auto out_w = word_set(content);
    size_t hits = 0;
    for (const auto& w : src_w) if (out_w.count(w)) ++hits;
    const double recall = static_cast<double>(hits) /
                          std::max<size_t>(src_w.size(), 1);
    std::printf("[loop] %zu/%zu source words recovered (%.1f%%)\n",
                hits, src_w.size(), recall * 100.0);
    if (recall < 0.8) {
        std::fprintf(stderr,
            "FAIL: cloned-voice closed-loop recall %.2f < 0.80\n"
            "  source: %s\n  output: %s\n",
            recall, source.c_str(), content.c_str());
        return 9;
    }
    return 0;
}
