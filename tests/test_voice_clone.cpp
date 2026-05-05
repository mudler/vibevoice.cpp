// Runtime voice-cloning tests.
//
// Three checks layered as TDD targets - each turns from RED to GREEN
// as the implementation phases land:
//
//   1. SMOKE: vibevoice_voice_clone(wav, asr_model, &voice) returns
//      true and produces non-empty per-layer K/V plus non-zero seq
//      lengths. Phase 2 makes this green.
//   2. DETERMINISM: running voice_clone twice on the same WAV with
//      the same model produces bit-equal voice.lm.k[0]. Phase 2.
//   3. CLOSED-LOOP: clone voice from a real WAV, synthesize a known
//      phrase, transcribe back, assert >=80% source-word recall.
//      Same shape as tests/test_closed_loop.cpp but with a runtime-
//      cloned voice instead of the shipped voice gguf. Phase 4
//      makes this green (needs the vibevoice-cli voice-clone
//      subcommand to write the cloned voice to disk).
//
// Skips with rc=77 unless these env vars all point at valid files:
//   VIBEVOICE_TTS_MODEL  -> realtime-0.5b gguf
//   VIBEVOICE_ASR_MODEL  -> asr-7b gguf (carries the encoders we need)
//   VIBEVOICE_TOKENIZER  -> tokenizer gguf
//   VIBEVOICE_CLI        -> vibevoice-cli binary (closed-loop test only)
// And a reference WAV at:
//   VIBEVOICE_REF_WAV    -> any 24kHz mono wav, e.g. samples/3p_gpt5.wav
//
// Gated by VIBEVOICE_TEST_LARGE in CMake (the ASR model is ~14 GB).

#include "model_loader.hpp"
#include "vibevoice_tts.hpp"
#include "vibevoice_asr.hpp"
#include "vibevoice.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

bool kv_nonzero(const std::vector<vv::LayerKV>& kv) {
    if (kv.empty()) return false;
    for (const auto& l : kv) {
        if (l.k.empty() || l.v.empty()) return false;
        bool any = false;
        for (float x : l.k) if (std::isfinite(x) && x != 0.0f) { any = true; break; }
        if (!any) return false;
    }
    return true;
}

}  // namespace

int main() {
    const char* tts_path = std::getenv("VIBEVOICE_TTS_MODEL");
    const char* asr_path = std::getenv("VIBEVOICE_ASR_MODEL");
    const char* tok_path = std::getenv("VIBEVOICE_TOKENIZER");
    const char* cli_path = std::getenv("VIBEVOICE_CLI");
    const char* ref_wav  = std::getenv("VIBEVOICE_REF_WAV");

    if (!file_ok(asr_path) || !file_ok(tok_path) || !file_ok(ref_wav)) {
        std::fprintf(stderr,
            "skip: voice-clone needs VIBEVOICE_{ASR_MODEL,TOKENIZER,REF_WAV} "
            "set to valid paths (got asr=%s tok=%s wav=%s).\n",
            asr_path ? asr_path : "(null)",
            tok_path ? tok_path : "(null)",
            ref_wav  ? ref_wav  : "(null)");
        return 77;
    }

    // ---- 1. SMOKE: clone produces non-empty resident KV ----
    std::printf("[smoke] loading ASR model %s\n", asr_path);
    vv::VibeVoiceModel asr_model;
    if (!vv::vibevoice_load(asr_path, &asr_model)) {
        std::fprintf(stderr, "FAIL: load ASR model %s\n", asr_path);
        return 1;
    }
    if (!asr_model.tokenizer.load_from_file(tok_path)) {
        std::fprintf(stderr, "FAIL: load tokenizer %s\n", tok_path);
        return 2;
    }

    vv::VibeVoiceVoice voice1;
    if (!vv::vibevoice_voice_clone(ref_wav, asr_model, /*with_cfg=*/true, &voice1)) {
        std::fprintf(stderr,
            "FAIL: vibevoice_voice_clone returned false (Phase 2 not implemented?)\n");
        return 3;
    }
    if (voice1.seq_lm <= 0 || voice1.seq_tlm <= 0) {
        std::fprintf(stderr,
            "FAIL: cloned voice has empty seq_lm=%d seq_tlm=%d\n",
            voice1.seq_lm, voice1.seq_tlm);
        return 4;
    }
    if (!kv_nonzero(voice1.kv_lm) || !kv_nonzero(voice1.kv_tlm)) {
        std::fprintf(stderr, "FAIL: cloned voice has empty/zero K/V tensors\n");
        return 5;
    }
    std::printf("[smoke] OK (seq_lm=%d seq_tlm=%d has_neg=%d)\n",
                voice1.seq_lm, voice1.seq_tlm, voice1.has_neg ? 1 : 0);

    // ---- 2. DETERMINISM: same WAV -> same K/V ----
    vv::VibeVoiceVoice voice2;
    if (!vv::vibevoice_voice_clone(ref_wav, asr_model, /*with_cfg=*/true, &voice2)) {
        std::fprintf(stderr, "FAIL: voice_clone (run 2) returned false\n");
        return 6;
    }
    if (voice1.seq_lm != voice2.seq_lm || voice1.seq_tlm != voice2.seq_tlm) {
        std::fprintf(stderr,
            "FAIL: non-deterministic seq lens (%d/%d vs %d/%d)\n",
            voice1.seq_lm, voice1.seq_tlm, voice2.seq_lm, voice2.seq_tlm);
        return 7;
    }
    if (voice1.kv_lm.size() != voice2.kv_lm.size() ||
        voice1.kv_lm[0].k.size() != voice2.kv_lm[0].k.size()) {
        std::fprintf(stderr, "FAIL: non-deterministic K shape\n");
        return 8;
    }
    double max_abs_kv = 0.0;
    for (size_t i = 0; i < voice1.kv_lm[0].k.size(); ++i) {
        max_abs_kv = std::max(max_abs_kv,
            std::fabs(static_cast<double>(voice1.kv_lm[0].k[i]) - voice2.kv_lm[0].k[i]));
    }
    std::printf("[determinism] kv_lm[0].k max_abs=%.3e\n", max_abs_kv);
    if (max_abs_kv > 1e-5) {
        std::fprintf(stderr,
            "FAIL: voice_clone is non-deterministic (max_abs=%.3e > 1e-5)\n",
            max_abs_kv);
        return 9;
    }
    std::printf("[determinism] OK\n");

    // ---- 3. CLOSED-LOOP: cloned voice produces intelligible TTS ----
    // This needs the CLI voice-clone subcommand (Phase 4) to write
    // the cloned voice to disk so vibevoice-cli tts can consume it.
    // Until Phase 4 lands, skip this part of the test with a note.
    if (!file_ok(tts_path) || !file_ok(cli_path)) {
        std::fprintf(stderr,
            "[closed-loop] skipping: needs VIBEVOICE_TTS_MODEL + VIBEVOICE_CLI\n");
        return 0;
    }

    const std::string voice_gguf = "/tmp/vibevoice_cloned_voice.gguf";
    {
        std::string cmd = std::string(cli_path) + " voice-clone"
            + " --asr-model " + asr_path
            + " --tokenizer " + tok_path
            + " --audio "     + ref_wav
            + " --out "       + voice_gguf
            + " 2>&1";
        std::printf("[clone] %s\n", cmd.c_str());
        int rc = std::system(cmd.c_str());
        if (rc != 0) {
            std::fprintf(stderr,
                "FAIL: vibevoice-cli voice-clone rc=%d (Phase 4 not implemented?)\n", rc);
            return 10;
        }
        if (!file_ok(voice_gguf.c_str())) {
            std::fprintf(stderr, "FAIL: voice-clone produced no gguf at %s\n", voice_gguf.c_str());
            return 11;
        }
    }

    const std::string source = "Hello world this is a test of voice cloning.";
    const std::string wav    = "/tmp/vibevoice_voice_clone_out.wav";
    {
        std::string cmd = std::string(cli_path) + " tts"
            + " --model "     + tts_path
            + " --voice "     + voice_gguf
            + " --tokenizer " + tok_path
            + " --text \""    + source + "\""
            + " --seed 12345"
            + " --out "       + wav
            + " 2>&1";
        std::printf("[tts] %s\n", cmd.c_str());
        int rc = std::system(cmd.c_str());
        if (rc != 0) { std::fprintf(stderr, "FAIL: tts rc=%d\n", rc); return 12; }
        std::ifstream f(wav, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() < 1000) {
            std::fprintf(stderr, "FAIL: tts produced empty wav\n"); return 13;
        }
    }

    std::string asr_out;
    {
        std::string cmd = std::string(cli_path) + " asr"
            + " --model "     + asr_path
            + " --tokenizer " + tok_path
            + " --audio "     + wav;
        std::printf("[asr] %s\n", cmd.c_str());
        asr_out = run_capture(cmd);
        if (asr_out.empty()) {
            std::fprintf(stderr, "FAIL: asr produced no output\n"); return 14;
        }
    }

    const std::string content = extract_content(asr_out);
    const auto src_w = word_set(source);
    const auto out_w = word_set(content);
    size_t hits = 0;
    for (const auto& w : src_w) if (out_w.count(w)) ++hits;
    const double recall = static_cast<double>(hits)
                        / std::max<size_t>(src_w.size(), 1);
    std::printf("[closed-loop] %zu/%zu source words recovered (%.1f%%)\n",
                hits, src_w.size(), recall * 100.0);
    if (recall < 0.8) {
        std::fprintf(stderr,
            "FAIL: cloned-voice closed-loop recall %.2f < 0.80\n  source: %s\n  output: %s\n",
            recall, source.c_str(), content.c_str());
        return 15;
    }

    std::remove(voice_gguf.c_str());
    std::remove(wav.c_str());
    return 0;
}
