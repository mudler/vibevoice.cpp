// vibevoice-cli — text-to-speech and (later) ASR command-line front-end.

#include "audio_io.hpp"
#include "model_loader.hpp"
#include "tokenizer.hpp"
#include "vibevoice.h"
#include "vibevoice_asr.hpp"
#include "vibevoice_tts.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void print_usage(const char* argv0) {
    std::printf(
        "usage: %s <command> [options]\n"
        "\n"
        "commands:\n"
        "  tts     synthesize speech from text\n"
        "  asr     transcribe audio to JSON\n"
        "  version print version and exit\n"
        "  help    print this message\n"
        "\n"
        "tts options:\n"
        "  --model <path>      path to vibevoice gguf (required)\n"
        "  --tokenizer <path>  path to tokenizer.gguf (required)\n"
        "  --voice <path>      pre-baked voice.gguf — use with realtime-0.5B\n"
        "                      models. Mutually exclusive with --ref-audio.\n"
        "  --ref-audio <path>  reference WAV (24 kHz mono, ~5 s) — runtime\n"
        "                      voice cloning. Use with VibeVoice-1.5B models.\n"
        "                      Mutually exclusive with --voice.\n"
        "  --text <string>     input text\n"
        "  --text-file <path>  read text from file\n"
        "  --out <path>        output WAV path (default: out.wav)\n"
        "  --max-frames N      cap speech frames (default 200)\n"
        "  --steps N           DPM-Solver inference steps (default 20)\n"
        "  --cfg X             classifier-free guidance scale (default 1.3,\n"
        "                      1.0 disables CFG)\n"
        "  --seed N            RNG seed for noise (default random)\n"
        "  --verbose           print per-frame progress\n"
        "\n"
        "asr options:\n"
        "  --model <path>      path to vibevoice-asr.gguf (required)\n"
        "  --tokenizer <path>  path to tokenizer.gguf (required)\n"
        "  --audio <path>      path to input WAV (required, mono ≥16 kHz)\n"
        "  --max-new-tokens N  cap generated tokens (default 256)\n"
        "  --verbose           print encoder + decode stats\n"
        "\n"
        "%s\n",
        argv0, vv_version());
}

int cmd_version() { std::printf("%s\n", vv_version()); return 0; }

const char* arg_value(int argc, char** argv, int i) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", argv[i]);
        return nullptr;
    }
    return argv[i + 1];
}

bool slurp(const std::string& path, std::string* out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out->resize(n > 0 ? static_cast<size_t>(n) : 0);
    if (n > 0) std::fread(out->data(), 1, n, f);
    std::fclose(f);
    return true;
}

int cmd_tts(int argc, char** argv) {
    std::string model_path, tok_path, voice_path, ref_audio;
    std::string text, text_file, out_path = "out.wav";
    int   max_frames = 200, steps = 20;
    float cfg_scale = 1.3f;
    uint32_t seed = 0;
    bool  verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"      && (i + 1 < argc)) { model_path = argv[++i]; }
        else if (a == "--tokenizer"  && (i + 1 < argc)) { tok_path   = argv[++i]; }
        else if (a == "--voice"      && (i + 1 < argc)) { voice_path = argv[++i]; }
        else if (a == "--ref-audio"  && (i + 1 < argc)) { ref_audio  = argv[++i]; }
        else if (a == "--text"       && (i + 1 < argc)) { text       = argv[++i]; }
        else if (a == "--text-file"  && (i + 1 < argc)) { text_file  = argv[++i]; }
        else if (a == "--out"        && (i + 1 < argc)) { out_path   = argv[++i]; }
        else if (a == "--max-frames" && (i + 1 < argc)) { max_frames = std::atoi(argv[++i]); }
        else if (a == "--steps"      && (i + 1 < argc)) { steps      = std::atoi(argv[++i]); }
        else if (a == "--seed"       && (i + 1 < argc)) { seed       = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10)); }
        else if (a == "--cfg"        && (i + 1 < argc)) { cfg_scale = static_cast<float>(std::atof(argv[++i])); }
        else if (a == "--verbose")                       { verbose = true; }
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr, "see `%s help`\n", argv[0]); return 0;
        }
        else {
            std::fprintf(stderr, "tts: unknown arg: %s\n", a.c_str());
            return 1;
        }
    }

    if (model_path.empty() || tok_path.empty()) {
        std::fprintf(stderr, "tts: --model and --tokenizer are required\n");
        return 1;
    }
    if (!voice_path.empty() && !ref_audio.empty()) {
        std::fprintf(stderr, "tts: --voice and --ref-audio are mutually exclusive\n");
        return 1;
    }
    if (text.empty()) {
        if (text_file.empty()) {
            std::fprintf(stderr, "tts: provide --text or --text-file\n");
            return 1;
        }
        if (!slurp(text_file, &text)) {
            std::fprintf(stderr, "tts: failed to read %s\n", text_file.c_str());
            return 1;
        }
    }

    std::fprintf(stderr, "vibevoice-cli tts: loading model %s\n", model_path.c_str());
    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_path, &model)) {
        std::fprintf(stderr, "tts: failed to load model\n");
        return 2;
    }

    std::fprintf(stderr, "vibevoice-cli tts: loading tokenizer %s\n", tok_path.c_str());
    if (!model.tokenizer.load_from_file(tok_path)) {
        std::fprintf(stderr, "tts: failed to load tokenizer\n");
        return 3;
    }

    // Validate inputs against the loaded model's variant. The gguf
    // already says which kind of conditioning it expects; the CLI is
    // a thin wrapper around that.
    const bool is_15b = (model.variant == "1.5b");
    if (is_15b && ref_audio.empty()) {
        std::fprintf(stderr,
                     "tts: 1.5b model requires --ref-audio (raw 24 kHz mono "
                     "WAV). Pre-baked --voice gguf files are realtime-0.5B "
                     "only.\n");
        return 1;
    }
    if (!is_15b && !ref_audio.empty()) {
        std::fprintf(stderr,
                     "tts: --ref-audio only applies to 1.5b models; this "
                     "model is variant=%s. Use --voice with a voice gguf "
                     "instead.\n", model.variant.c_str());
        return 1;
    }

    vv::VibeVoiceVoice voice;
    bool have_voice = false;
    if (!is_15b) {
        if (!voice_path.empty()) {
            if (!vv::vibevoice_voice_load(voice_path, model, &voice)) {
                std::fprintf(stderr, "tts: failed to load voice %s\n", voice_path.c_str());
                return 6;
            }
            have_voice = true;
        } else {
            std::fprintf(stderr,
                         "tts: WARNING — no --voice specified; output will be "
                         "low-amplitude / incoherent. Convert one with "
                         "scripts/convert_voice_to_gguf.py.\n");
        }
    }

    vv::VibeVoiceTTSParams p;
    p.voice             = have_voice ? &voice : nullptr;
    p.ref_audio_path    = ref_audio;
    p.max_speech_frames = max_frames;
    p.n_diffusion_steps = steps;
    p.cfg_scale         = cfg_scale;
    p.seed              = seed;
    p.verbose           = verbose;

    std::vector<float> samples;
    int rc = vv::vibevoice_tts_generate(&model, text, p, &samples);
    if (rc != 0) {
        std::fprintf(stderr, "tts: generate failed (rc=%d)\n", rc);
        return 4;
    }
    std::fprintf(stderr, "tts: generated %zu samples (%.2fs at %d Hz)\n",
                 samples.size(),
                 static_cast<double>(samples.size()) / model.cfg.sample_rate,
                 model.cfg.sample_rate);

    vv_audio out;
    out.samples     = samples.data();
    out.n_samples   = samples.size();
    out.sample_rate = model.cfg.sample_rate;
    out.channels    = 1;
    if (vv_save_wav(out_path.c_str(), &out) != VV_OK) {
        std::fprintf(stderr, "tts: failed to write %s\n", out_path.c_str());
        return 5;
    }
    std::fprintf(stderr, "tts: wrote %s\n", out_path.c_str());
    return 0;
}

int cmd_asr(int argc, char** argv) {
    std::string model_path, tok_path, wav_path;
    int  max_new_tokens = 256;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"     && (i + 1 < argc)) model_path = argv[++i];
        else if (a == "--tokenizer" && (i + 1 < argc)) tok_path   = argv[++i];
        else if (a == "--audio"     && (i + 1 < argc)) wav_path   = argv[++i];
        else if (a == "--max-new-tokens" && (i + 1 < argc)) max_new_tokens = std::atoi(argv[++i]);
        else if (a == "--verbose")                       verbose = true;
        else if (a == "-h" || a == "--help") { std::fprintf(stderr, "see `%s help`\n", argv[0]); return 0; }
        else { std::fprintf(stderr, "asr: unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || tok_path.empty() || wav_path.empty()) {
        std::fprintf(stderr, "asr: --model, --tokenizer, --audio are required\n");
        return 1;
    }

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    std::vector<float> audio;
    if (vv::load_wav_24k_mono(wav_path, &audio) != VV_OK || audio.empty()) {
        std::fprintf(stderr, "asr: failed to load %s\n", wav_path.c_str());
        return 2;
    }
    const double audio_sec = audio.size() / 24000.0;
    std::fprintf(stderr, "asr: loaded %zu samples (%.2fs)\n", audio.size(), audio_sec);

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_path, &model)) {
        std::fprintf(stderr, "asr: failed to load model\n");
        return 3;
    }
    if (!model.tokenizer.load_from_file(tok_path)) {
        std::fprintf(stderr, "asr: failed to load tokenizer\n");
        return 4;
    }
    auto t1 = clk::now();

    vv::VibeVoiceASRParams p;
    p.max_new_tokens = max_new_tokens;
    p.verbose        = verbose;

    std::string transcript;
    int rc = vv::vibevoice_asr_transcribe(&model, audio, p, &transcript);
    if (rc != 0) { std::fprintf(stderr, "asr: rc=%d\n", rc); return 5; }
    auto t2 = clk::now();

    std::printf("%s\n", transcript.c_str());

    using ms = std::chrono::duration<double, std::milli>;
    const double load_ms = ms(t1 - t0).count();
    const double infer_ms = ms(t2 - t1).count();
    std::fprintf(stderr,
                 "asr: timing  load=%.1fs  inference=%.1fs  audio=%.1fs  RTF=%.3f\n",
                 load_ms / 1000.0, infer_ms / 1000.0, audio_sec,
                 (infer_ms / 1000.0) / std::max(audio_sec, 1e-9));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }
    std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help" || cmd == "help") { print_usage(argv[0]); return 0; }
    if (cmd == "-v" || cmd == "--version" || cmd == "version") return cmd_version();
    if (cmd == "tts") return cmd_tts(argc - 1, argv + 1);
    if (cmd == "asr") return cmd_asr(argc - 1, argv + 1);
    std::fprintf(stderr, "unknown command: %s\n", argv[1]);
    print_usage(argv[0]);
    return 1;
}
