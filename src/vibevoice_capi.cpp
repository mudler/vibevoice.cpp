// vibevoice_capi.cpp - flat C ABI implementation.
//
// Wraps the C++ orchestrators (vv::vibevoice_load, vv::vibevoice_tts_generate,
// vv::vibevoice_asr_transcribe) in a single global-state surface that
// LocalAI's purego backends consume (see vibevoice_capi.h).

#include "vibevoice_capi.h"

#include "audio_io.hpp"
#include "common.hpp"
#include "vibevoice_asr.hpp"
#include "vibevoice_tts.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

namespace {

struct GlobalEngine {
    std::unique_ptr<vv::VibeVoiceModel> tts;
    std::unique_ptr<vv::VibeVoiceModel> asr;
    std::unique_ptr<vv::VibeVoiceVoice> voice;
    std::string                          voice_path_loaded;
    std::string                          ref_audio_path_loaded;  // 1.5b
    int                                  n_threads = 4;
    std::mutex                           mu;
};

GlobalEngine& engine() {
    static GlobalEngine g;
    return g;
}

bool ensure_voice_loaded(GlobalEngine& g, const char* voice_path) {
    if (!voice_path || !voice_path[0]) return g.voice != nullptr;
    if (g.voice && g.voice_path_loaded == voice_path) return true;
    if (!g.tts) {
        VV_LOG_ERROR("vv_capi: TTS model required to load a voice");
        return false;
    }
    auto v = std::make_unique<vv::VibeVoiceVoice>();
    if (!vv::vibevoice_voice_load(voice_path, *g.tts, v.get())) {
        VV_LOG_ERROR("vv_capi: voice load failed: %s", voice_path);
        return false;
    }
    g.voice              = std::move(v);
    g.voice_path_loaded  = voice_path;
    return true;
}

}  // namespace

extern "C" {

int vv_capi_load(const char* tts_model_path,
                 const char* asr_model_path,
                 const char* tokenizer_path,
                 const char* voice_path,
                 int         n_threads) {
    auto& g = engine();
    std::lock_guard<std::mutex> lk(g.mu);

    if (!tokenizer_path || !tokenizer_path[0]) {
        VV_LOG_ERROR("vv_capi_load: tokenizer_path is required");
        return -2;
    }
    if ((!tts_model_path || !tts_model_path[0]) &&
        (!asr_model_path || !asr_model_path[0])) {
        VV_LOG_ERROR("vv_capi_load: at least one of tts_model_path or asr_model_path is required");
        return -2;
    }

    g.tts.reset();
    g.asr.reset();
    g.voice.reset();
    g.voice_path_loaded.clear();
    g.n_threads = n_threads > 0 ? n_threads : 4;

    // Each VibeVoiceModel has its own Tokenizer member, but the tokenizer
    // backing data is loaded from the same gguf each time and `load_from_file`
    // refuses to run twice on a single Tokenizer instance. We use a fresh
    // Tokenizer per model — both load OK because each is a different
    // instance.
    if (tts_model_path && tts_model_path[0]) {
        auto m = std::make_unique<vv::VibeVoiceModel>();
        if (!vv::vibevoice_load(tts_model_path, m.get())) {
            VV_LOG_ERROR("vv_capi_load: TTS model load failed: %s", tts_model_path);
            return -3;
        }
        if (!m->tokenizer.load_from_file(tokenizer_path)) {
            VV_LOG_ERROR("vv_capi_load: TTS tokenizer load failed: %s", tokenizer_path);
            return -7;
        }
        g.tts = std::move(m);
    }

    if (asr_model_path && asr_model_path[0]) {
        auto m = std::make_unique<vv::VibeVoiceModel>();
        if (!vv::vibevoice_load(asr_model_path, m.get())) {
            VV_LOG_ERROR("vv_capi_load: ASR model load failed: %s", asr_model_path);
            return -3;
        }
        if (!m->tokenizer.load_from_file(tokenizer_path)) {
            VV_LOG_ERROR("vv_capi_load: ASR tokenizer load failed: %s", tokenizer_path);
            return -7;
        }
        g.asr = std::move(m);
    }

    if (voice_path && voice_path[0]) {
        if (!ensure_voice_loaded(g, voice_path)) return -3;
    }
    return 0;
}

int vv_capi_tts(const char* text,
                const char* voice_path,
                const char* ref_audio_path,
                const char* dst_wav_path,
                int         n_diffusion_steps,
                float       cfg_scale,
                int         max_speech_frames,
                uint32_t    seed) {
    auto& g = engine();
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.tts)             return -3;
    if (!text || !dst_wav_path) return -2;

    const bool is_15b = (g.tts->variant == "1.5b");

    vv::VibeVoiceTTSParams p;
    p.cfg_scale          = cfg_scale > 0.0f ? cfg_scale : 1.3f;
    p.n_diffusion_steps  = n_diffusion_steps > 0 ? n_diffusion_steps : 20;
    p.max_speech_frames  = max_speech_frames > 0 ? max_speech_frames : 200;
    p.seed               = seed;
    p.verbose            = false;

    if (is_15b) {
        // 1.5B path: ref_audio_path is required (per call or via load).
        const std::string ref =
            (ref_audio_path && ref_audio_path[0]) ? ref_audio_path
                                                  : g.ref_audio_path_loaded;
        if (ref.empty()) {
            VV_LOG_ERROR("vv_capi_tts: 1.5b model needs ref_audio_path "
                         "here or via vv_capi_load");
            return -2;
        }
        p.ref_audio_path = ref;
    } else {
        // realtime-0.5b path: voice_path required (per call or via load).
        if (voice_path && voice_path[0]) {
            if (!ensure_voice_loaded(g, voice_path)) return -3;
        }
        if (!g.voice) {
            VV_LOG_ERROR("vv_capi_tts: no voice loaded — pass voice_path "
                         "here or to vv_capi_load");
            return -2;
        }
        p.voice = g.voice.get();
    }

    std::vector<float> samples;
    int rc = vv::vibevoice_tts_generate(g.tts.get(), text, p, &samples);
    if (rc != 0) {
        VV_LOG_ERROR("vv_capi_tts: generate rc=%d", rc);
        return rc;
    }

    vv_audio audio_out{};
    audio_out.samples     = samples.data();
    audio_out.n_samples   = samples.size();
    audio_out.sample_rate = 24000;
    audio_out.channels    = 1;
    return vv::save_wav_pcm16(dst_wav_path, audio_out);
}

int vv_capi_asr(const char* src_wav_path,
                char*       out_json,
                size_t      out_capacity,
                int         max_new_tokens) {
    auto& g = engine();
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.asr)                    return -3;
    if (!src_wav_path || !out_json || out_capacity == 0) return -2;

    std::vector<float> audio;
    if (vv::load_wav_24k_mono(src_wav_path, &audio) != 0 || audio.empty()) {
        VV_LOG_ERROR("vv_capi_asr: failed to load wav: %s", src_wav_path);
        return -8;
    }

    vv::VibeVoiceASRParams p;
    p.max_new_tokens     = max_new_tokens > 0 ? max_new_tokens : 256;
    p.repetition_penalty = 1.0f;
    p.no_repeat_ngram    = 0;
    p.verbose            = false;

    std::string transcript;
    int rc = vv::vibevoice_asr_transcribe(g.asr.get(), audio, p, &transcript);
    if (rc != 0) {
        VV_LOG_ERROR("vv_capi_asr: transcribe rc=%d", rc);
        return rc;
    }
    if (transcript.empty()) {
        out_json[0] = '\0';
        return 0;
    }

    const size_t need = transcript.size() + 1;  // +1 for NUL
    if (need > out_capacity) {
        // Write the prefix that fits, NUL-terminate, and report the
        // required size negatively so the caller can grow + retry.
        std::memcpy(out_json, transcript.data(), out_capacity - 1);
        out_json[out_capacity - 1] = '\0';
        return -static_cast<int>(need);
    }
    std::memcpy(out_json, transcript.data(), transcript.size());
    out_json[transcript.size()] = '\0';
    return static_cast<int>(transcript.size());
}

void vv_capi_unload(void) {
    auto& g = engine();
    std::lock_guard<std::mutex> lk(g.mu);
    g.voice.reset();
    g.tts.reset();
    g.asr.reset();
    g.voice_path_loaded.clear();
}

const char* vv_capi_version(void) { return "vibevoice.cpp 0.1.0 (capi)"; }

int vv_capi_voice_clone(const char* /*src_wav_path*/,
                        const char* /*dst_voice_gguf_path*/,
                        int         /*with_cfg*/) {
    VV_LOG_ERROR("vv_capi_voice_clone: not supported. The realtime-0.5B "
                 "weights ship without encoders, so we cannot prep a voice "
                 "gguf at runtime. Load a 1.5B gguf and pass ref_audio_path "
                 "to vv_capi_tts for runtime voice cloning instead.");
    return -2;
}

}  // extern "C"
