// Closed-loop TTS → ASR roundtrip test.
//
// 1. Invoke `vibevoice-cli tts` to synthesize a known phrase to a wav.
// 2. Invoke `vibevoice-cli asr` to transcribe that wav.
// 3. Assert the recovered transcript covers ≥80% of the source words.
//
// This shells out to the CLI binary (via system()) instead of running both
// models in-process. ggml keeps a fair amount of static state and the 0.5B
// + 7B models exhaust each other's compute pools when loaded back-to-back,
// but two short-lived processes work fine — and it's how anyone using this
// project will actually wire TTS → ASR anyway.
//
// Skips (return 77) unless all of the following env vars point at valid files:
//   VIBEVOICE_TTS_MODEL  -> realtime-0.5b gguf
//   VIBEVOICE_VOICE      -> voice gguf (e.g. Carter_man)
//   VIBEVOICE_ASR_MODEL  -> asr-7b gguf
//   VIBEVOICE_TOKENIZER  -> tokenizer gguf
//   VIBEVOICE_CLI        -> path to the vibevoice-cli binary
//
// Models total ~16 GB so this is gated by VIBEVOICE_TEST_LARGE in CMake.

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

// Capture stdout from a shell command. Returns "" on failure.
std::string run_capture(const std::string& cmd) {
    std::string buf;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return buf;
    char chunk[4096];
    while (size_t n = std::fread(chunk, 1, sizeof(chunk), p)) buf.append(chunk, n);
    pclose(p);
    return buf;
}

// Pull "Content":"..." substrings out of the ASR JSON output.
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
    const char* cli   = std::getenv("VIBEVOICE_CLI");
    if (!file_ok(tts) || !file_ok(voice) || !file_ok(asr) ||
        !file_ok(tok) || !file_ok(cli)) {
        std::fprintf(stderr,
            "skip: closed-loop needs VIBEVOICE_{TTS_MODEL,VOICE,ASR_MODEL,"
            "TOKENIZER,CLI} all set to valid paths.\n");
        return 77;
    }

    const std::string source =
        "Hello world this is a test of the synthesis system.";
    const std::string wav = "/tmp/vibevoice_closed_loop.wav";

    // 1. TTS → wav. Pin --seed so the test is deterministic; without it
    // each run samples different latents and the ASR-recovered text varies
    // enough to make the recall threshold flap.
    {
        std::string cmd = std::string(cli) + " tts"
            + " --model "     + tts
            + " --voice "     + voice
            + " --tokenizer " + tok
            + " --text \""    + source + "\""
            + " --seed 12345"
            + " --out "       + wav
            + " 2>&1";
        std::printf("[tts] %s\n", cmd.c_str());
        int rc = std::system(cmd.c_str());
        if (rc != 0) {
            std::fprintf(stderr, "FAIL: tts rc=%d\n", rc);
            return 1;
        }
        std::ifstream f(wav, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() < 1000) {
            std::fprintf(stderr, "FAIL: tts produced empty/missing wav at %s\n", wav.c_str());
            return 2;
        }
        std::printf("[tts] wrote %s (%lld bytes)\n", wav.c_str(),
                    static_cast<long long>(f.tellg()));
    }

    // 2. ASR ← wav
    std::string asr_out;
    {
        std::string cmd = std::string(cli) + " asr"
            + " --model "     + asr
            + " --tokenizer " + tok
            + " --audio "     + wav;
        std::printf("[asr] %s\n", cmd.c_str());
        asr_out = run_capture(cmd);
        if (asr_out.empty()) {
            std::fprintf(stderr, "FAIL: asr produced no output\n");
            return 3;
        }
        std::printf("[asr] %s\n", asr_out.c_str());
    }

    // 3. compare
    const std::string content = extract_content(asr_out);
    if (content.empty()) {
        std::fprintf(stderr, "FAIL: empty Content field in transcript\n");
        return 4;
    }
    const auto src_w = word_set(source);
    const auto out_w = word_set(content);
    size_t hits = 0;
    for (const auto& w : src_w) if (out_w.count(w)) ++hits;
    const double recall = static_cast<double>(hits)
                        / std::max<size_t>(src_w.size(), 1);
    std::printf("closed-loop: %zu/%zu source words recovered (%.1f%%)\n",
                hits, src_w.size(), recall * 100.0);

    if (recall < 0.8) {
        std::fprintf(stderr,
            "FAIL: closed-loop recall %.2f < 0.80\n  source: %s\n  output: %s\n",
            recall, source.c_str(), content.c_str());
        return 5;
    }
    std::remove(wav.c_str());
    return 0;
}
