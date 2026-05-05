#ifndef VIBEVOICE_TTS_HPP
#define VIBEVOICE_TTS_HPP

// VibeVoice realtime-TTS orchestrator.
//
// Loads everything from a single .gguf produced by
// scripts/convert_vibevoice_to_gguf.py and exposes a `generate(text)` call
// that returns 24 kHz mono audio.
//
// Architecture (mirrors microsoft/VibeVoice-Realtime-0.5B):
//   text_ids
//     -> embed via lm.tok_embd
//     -> language_model (lower 4 layers, no final norm)         [stack 1]
//     -> spliced into TTS LM tail + tts_input_types[1] (text)
//     -> tts_language_model (upper 20 layers, with final norm)  [stack 2]
//     -> last hidden = condition for diffusion head
//     -> DPM-Solver++ samples a 64-D speech latent
//     -> acoustic decoder produces ~1600 samples per latent
//     -> acoustic_connector(latent) becomes next-step input embed
//     -> stack 2 again with type=speech, then EOS classifier check
//     -> repeat until EOS or max_speech_frames
//
// v1 limitations:
//   - no CFG (cfg_scale = 1.0; only the positive condition is sampled)
//   - no voice prompt (initial KV caches start empty); without a learned
//     voice the generated audio will be incoherent. The wiring is correct;
//     a follow-up will add voice-cache loading.

#include "acoustic_tokenizer.hpp"
#include "diffusion_head.hpp"
#include "dpm_solver.hpp"
#include "model_loader.hpp"
#include "qwen2.hpp"
#include "tokenizer.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vv {

struct VibeVoiceConfig {
    // LM (Qwen2)
    int     hidden        = 0;
    int     n_layers_lm   = 0;
    int     n_layers_tlm  = 0;
    int     n_heads       = 0;
    int     n_kv_heads    = 0;
    int     head_dim      = 0;
    int     vocab_size    = 0;
    float   rope_theta    = 1.0e6f;
    float   rms_norm_eps  = 1.0e-6f;
    // Diffusion head
    int     latent        = 64;
    int     head_layers   = 4;
    float   ffn_ratio     = 3.0f;
    int     freq_size     = 256;
    // Acoustic decoder
    int     vae_dim       = 64;
    AcousticConfig acoustic;
    // Audio
    int     sample_rate   = 24000;
    // Speech latent normalization
    float   speech_scaling = 1.0f;
    float   speech_bias    = 0.0f;
};

struct VibeVoiceWeights {
    // ---- LM ----
    struct ggml_tensor*               lm_tok_embd  = nullptr;
    std::vector<Qwen2LayerWeights>    lm_layers;             // size n_layers_lm
    std::vector<Qwen2LayerWeights>    tlm_layers;            // size n_layers_tlm
    struct ggml_tensor*               tlm_output_norm = nullptr;
    struct ggml_tensor*               tts_input_types = nullptr;  // [hidden, 2]

    // ---- connector / EOS ----
    struct ggml_tensor* ac_fc1_w = nullptr, *ac_fc1_b = nullptr;
    struct ggml_tensor* ac_norm  = nullptr;
    struct ggml_tensor* ac_fc2_w = nullptr, *ac_fc2_b = nullptr;
    struct ggml_tensor* eos_fc1_w = nullptr, *eos_fc1_b = nullptr;
    struct ggml_tensor* eos_fc2_w = nullptr, *eos_fc2_b = nullptr;

    // ---- diffusion head ----
    DiffusionHeadWeights dh;

    // ---- acoustic decoder ----
    DecoderWeights at_dec;
};

// Per-layer KV cache stored on the CPU as raw float buffers. Each layer
// holds k, v of shape [head_dim, n_kv_heads, past_len, B=1] in ggml's
// convention. Shared by both the TTS and ASR orchestrators.
struct LayerKV {
    std::vector<float> k;
    std::vector<float> v;
    int                past_len = 0;
};

struct VibeVoiceModel {
    ModelLoader      loader;
    VibeVoiceConfig  cfg;
    VibeVoiceWeights w;
    Tokenizer        tokenizer;        // optional, set externally

    // Set during vibevoice_load: "realtime-0.5b", "asr-7b", or "vibepod-1.5b".
    std::string      variant;

    // ---- ASR-specific weights (only populated when variant == "asr-7b") ----
    EncoderWeights   at_enc;
    EncoderWeights   st_enc;
    AcousticConfig   semantic_cfg;
    int              semantic_vae_dim = 128;
    struct ggml_tensor* sc_fc1_w  = nullptr; struct ggml_tensor* sc_fc1_b = nullptr;
    struct ggml_tensor* sc_norm   = nullptr;
    struct ggml_tensor* sc_fc2_w  = nullptr; struct ggml_tensor* sc_fc2_b = nullptr;
    struct ggml_tensor* lm_head   = nullptr;
};

bool vibevoice_load(const std::string& gguf_path, VibeVoiceModel* out);

// A pre-baked voice prompt (system prompt + voice audio prefix) loaded from
// `convert_voice_to_gguf.py` output. Without this the orchestrator runs
// without speaker context and produces low-amplitude/incoherent audio.
struct VibeVoiceVoice {
    int                  seq_lm  = 0;
    int                  seq_tlm = 0;
    std::vector<LayerKV> kv_lm;
    std::vector<LayerKV> kv_tlm;
    std::vector<float>   tlm_last_hidden;       // [hidden]

    // Negative branch (for classifier-free guidance). Optional — older
    // voice .gguf files won't have these populated.
    bool                 has_neg     = false;
    int                  seq_neg_lm  = 0;
    int                  seq_neg_tlm = 0;
    std::vector<LayerKV> kv_neg_lm;
    std::vector<LayerKV> kv_neg_tlm;
    std::vector<float>   neg_tlm_last_hidden;   // [hidden]
};

bool vibevoice_voice_load(const std::string&     path,
                          const VibeVoiceModel&  model,
                          VibeVoiceVoice*        out);

struct VibeVoiceTTSParams {
    // realtime-0.5b conditioning: a pre-baked voice gguf wrapped in
    // VibeVoiceVoice (load with vibevoice_voice_load). Ignored when the
    // model is a 1.5b variant.
    const VibeVoiceVoice* voice = nullptr;

    // 1.5b conditioning: raw reference WAV path. Encoded inline
    // through at_enc + st_enc + connectors at synthesis time. Ignored
    // when the model is a realtime-0.5b variant.
    std::string ref_audio_path = "";

    int      max_speech_frames = 200;
    float    cfg_scale         = 1.3f;
    int      n_diffusion_steps = 20;
    uint32_t seed              = 0;
    bool     verbose           = false;
};

// Generate audio for `text`. Dispatches on `model->variant`:
//   * realtime-0.5b -> uses `p.voice` (pre-baked voice gguf state).
//   * 1.5b          -> uses `p.ref_audio_path` (raw reference WAV;
//                     runtime voice cloning, no separate voice gguf).
// Output samples are 24 kHz mono float32. Returns 0 on success.
int vibevoice_tts_generate(VibeVoiceModel*           model,
                           const std::string&        text,
                           const VibeVoiceTTSParams& p,
                           std::vector<float>*       samples);

}  // namespace vv

#endif  // VIBEVOICE_TTS_HPP
