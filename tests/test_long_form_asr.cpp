// Long-form ASR test (>60 s of audio).
//
// Upstream's encode_speech splits inputs longer than `streaming_segment_duration`
// (60 s by default) into chunks and streams them through the encoder, because
// the conv1d activations explode past ~2^32 elements on a single shot. Our v1
// `run_encoder_buf` is single-shot; this test pins the long-form behavior we
// want to ship next.
//
// Strategy:
//   1. shell out to `vibevoice-cli tts` to synthesize a known phrase to a wav
//   2. read the wav with our audio_io helpers, concatenate the samples N
//      times to produce > 70 s of audio, save it back as a long wav
//   3. shell out to `vibevoice-cli asr` to transcribe the long wav
//   4. assert the source phrase appears multiple times in the transcript
//
// Skips (return 77) unless all of:
//   VIBEVOICE_TTS_MODEL  -> path to converted realtime-0.5b gguf
//   VIBEVOICE_VOICE      -> path to converted voice gguf
//   VIBEVOICE_ASR_MODEL  -> path to converted asr-7b gguf
//   VIBEVOICE_TOKENIZER  -> path to tokenizer gguf
//   VIBEVOICE_CLI        -> path to the vibevoice-cli binary
//
// Gated by VIBEVOICE_TEST_LARGE in CMake.

#include "audio_io.hpp"
#include "vibevoice.h"

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

std::string run_capture(const std::string& cmd) {
    std::string buf;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return buf;
    char chunk[4096];
    while (size_t n = std::fread(chunk, 1, sizeof(chunk), p)) buf.append(chunk, n);
    pclose(p);
    return buf;
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

// Count non-overlapping occurrences of `needle` (lowercase, alphanumeric only)
// within `haystack` after the same normalization.
int count_occurrences(const std::string& haystack_raw, const std::string& needle_raw) {
    auto normalize = [](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            out += std::isalnum(static_cast<unsigned char>(c))
                   ? static_cast<char>(std::tolower(static_cast<unsigned char>(c)))
                   : ' ';
        }
        // Collapse runs of whitespace to a single space so word boundaries match.
        std::string compact;
        bool in_ws = false;
        for (char c : out) {
            if (c == ' ') {
                if (!in_ws) { compact += ' '; in_ws = true; }
            } else { compact += c; in_ws = false; }
        }
        return compact;
    };
    const std::string h = normalize(haystack_raw);
    const std::string n = normalize(needle_raw);
    if (n.empty()) return 0;
    int count = 0;
    size_t pos = 0;
    while ((pos = h.find(n, pos)) != std::string::npos) {
        ++count;
        pos += n.size();
    }
    return count;
}

}  // namespace

int main() {
    const char* tts   = std::getenv("VIBEVOICE_TTS_MODEL");
    const char* voice = std::getenv("VIBEVOICE_VOICE");
    const char* asr   = std::getenv("VIBEVOICE_ASR_MODEL");
    const char* tok   = std::getenv("VIBEVOICE_TOKENIZER");
    const char* cli   = std::getenv("VIBEVOICE_CLI");
    if (!file_ok(tts) || !file_ok(voice) || !file_ok(asr) ||
        !file_ok(tok) || !file_ok(cli)) {
        std::fprintf(stderr,
            "skip: long-form ASR test needs VIBEVOICE_{TTS_MODEL,VOICE,"
            "ASR_MODEL,TOKENIZER,CLI} all set to valid paths.\n");
        return 77;
    }

    const std::string source =
        "Hello world this is a test of the synthesis system.";
    const std::string short_wav = "/tmp/vibevoice_long_form_short.wav";
    const std::string long_wav  = "/tmp/vibevoice_long_form_long.wav";

    // ---- 1. TTS → short wav ------------------------------------------------
    {
        std::string cmd = std::string(cli) + " tts"
            + " --model "     + tts
            + " --voice "     + voice
            + " --tokenizer " + tok
            + " --text \""    + source + "\""
            + " --seed 12345"
            + " --out "       + short_wav
            + " 2>&1";
        std::printf("[tts] %s\n", cmd.c_str());
        if (std::system(cmd.c_str()) != 0) {
            std::fprintf(stderr, "FAIL: tts\n"); return 1;
        }
    }

    // ---- 2. read + concatenate to >70 s ------------------------------------
    std::vector<float> short_samples;
    if (vv::load_wav_24k_mono(short_wav, &short_samples) != 0 || short_samples.empty()) {
        std::fprintf(stderr, "FAIL: load %s\n", short_wav.c_str());
        return 2;
    }
    const double short_dur = static_cast<double>(short_samples.size()) / 24000.0;
    // Just over 60 s so we cross the streaming-segment boundary, but keep the
    // total LM decode work bounded — every extra second of audio adds ~7.5
    // speech-pad tokens to the prompt and an unbounded number of generated
    // tokens (the model wants to transcribe the whole repeating phrase).
    const double target_dur = 65.0;
    const int repeats = static_cast<int>(target_dur / short_dur) + 1;
    std::vector<float> long_samples;
    long_samples.reserve(short_samples.size() * static_cast<size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        long_samples.insert(long_samples.end(), short_samples.begin(), short_samples.end());
    }
    const double long_dur = static_cast<double>(long_samples.size()) / 24000.0;
    std::printf("[concat] %d × %.2fs → %.2fs (%zu samples)\n",
                repeats, short_dur, long_dur, long_samples.size());
    if (long_dur < 60.0) {
        std::fprintf(stderr, "FAIL: concat too short (%.2fs); need >60s\n", long_dur);
        return 3;
    }

    // ---- 3. write the long wav --------------------------------------------
    vv_audio audio_out{};
    audio_out.samples     = long_samples.data();
    audio_out.n_samples   = long_samples.size();
    audio_out.sample_rate = 24000;
    audio_out.channels    = 1;
    if (vv::save_wav_pcm16(long_wav, audio_out) != 0) {
        std::fprintf(stderr, "FAIL: save %s\n", long_wav.c_str());
        return 4;
    }

    // ---- 4. ASR ← long wav ------------------------------------------------
    std::string asr_out;
    {
        std::string cmd = std::string(cli) + " asr"
            + " --model "     + asr
            + " --tokenizer " + tok
            + " --audio "     + long_wav
            + " --max-new-tokens 512";  // enough for several repeats
        std::printf("[asr] %s\n", cmd.c_str());
        asr_out = run_capture(cmd);
        if (asr_out.empty()) {
            std::fprintf(stderr, "FAIL: asr produced no output\n"); return 5;
        }
        std::printf("[asr] %s\n", asr_out.c_str());
    }

    // ---- 5. compare -------------------------------------------------------
    const std::string content = extract_content(asr_out);
    if (content.empty()) {
        std::fprintf(stderr, "FAIL: empty Content field\n"); return 6;
    }

    // Word-set recall — at least 80% of source words anywhere in the
    // transcription. This catches the "model produced [Noise] for the
    // out-of-context boundary frames" failure mode.
    const auto src_w = word_set(source);
    const auto out_w = word_set(content);
    size_t hits = 0;
    for (const auto& w : src_w) if (out_w.count(w)) ++hits;
    const double recall = static_cast<double>(hits)
                        / std::max<size_t>(src_w.size(), 1);
    std::printf("long-form: %zu/%zu source words recovered (%.1f%%)\n",
                hits, src_w.size(), recall * 100.0);
    if (recall < 0.8) {
        std::fprintf(stderr,
            "FAIL: recall %.2f < 0.80\n  source: %s\n  output: %s\n",
            recall, source.c_str(), content.c_str());
        return 7;
    }

    // Multi-segment check: the source phrase should be transcribed at least
    // twice (we synthesized the same phrase ≥17 times in the long wav).
    // 80% of those transcriptions should preserve enough words to match the
    // last 4 source words ("of the synthesis system").
    const std::string anchor = "of the synthesis system";
    const int hits_anchor = count_occurrences(content, anchor);
    std::printf("long-form: anchor phrase '%s' found %d × in transcript "
                "(synth'd %d ×)\n", anchor.c_str(), hits_anchor, repeats);
    if (hits_anchor < 2) {
        std::fprintf(stderr,
            "FAIL: anchor phrase appears %d × — long-form encoding is dropping "
            "content. Expected ≥2.\n  output: %s\n",
            hits_anchor, content.c_str());
        return 8;
    }

    std::remove(short_wav.c_str());
    std::remove(long_wav.c_str());
    return 0;
}
