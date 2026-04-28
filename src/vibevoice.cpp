// Public C API implementation. M1 scope: log + version + default params.
// Real init/inference paths are stubbed with VV_ERR_NOT_IMPLEMENTED until M5/M6.

#include "common.hpp"
#include "vibevoice.h"

#include <cstdlib>
#include <cstring>

extern "C" {

vv_context_params vv_context_default_params(void) {
    vv_context_params p;
    p.model_path  = nullptr;
    p.n_threads   = 0;
    p.gpu_layers  = 0;
    p.use_f16_kv  = true;
    p.seed        = 0;
    return p;
}

vv_tts_params vv_tts_default_params(void) {
    vv_tts_params p;
    p.voice             = nullptr;
    p.n_diffusion_steps = 20;
    p.cfg_scale         = 1.3f;
    p.max_new_tokens    = 0;
    p.seed              = 0;
    return p;
}

vv_asr_params vv_asr_default_params(void) {
    vv_asr_params p;
    p.temperature    = 0.0f;
    p.max_new_tokens = 0;
    p.greedy         = true;
    return p;
}

vv_context* vv_init(const vv_context_params* /*params*/) {
    VV_LOG_WARN("vv_init: model loading not yet implemented (M1)");
    return nullptr;
}

void vv_free(vv_context* /*ctx*/) {}

vv_voice* vv_voice_load(vv_context* /*ctx*/, const char* /*path*/) {
    VV_LOG_WARN("vv_voice_load: not yet implemented (M5)");
    return nullptr;
}
void vv_voice_free(vv_voice* /*v*/) {}

void vv_audio_free(vv_audio* a) {
    if (!a) return;
    if (a->samples) std::free(a->samples);
    a->samples     = nullptr;
    a->n_samples   = 0;
    a->sample_rate = 0;
    a->channels    = 0;
}

void vv_string_free(char* s) {
    if (s) std::free(s);
}

int vv_tts(vv_context* /*ctx*/,
           const char* /*text*/,
           const vv_tts_params* /*p*/,
           vv_audio* /*out*/) {
    return VV_ERR_NOT_IMPLEMENTED;
}

int vv_asr(vv_context* /*ctx*/,
           const float* /*samples*/,
           size_t /*n_samples*/,
           int /*sample_rate*/,
           const vv_asr_params* /*p*/,
           char** /*out_utf8*/) {
    return VV_ERR_NOT_IMPLEMENTED;
}

}  // extern "C"
