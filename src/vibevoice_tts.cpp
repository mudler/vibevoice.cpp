#include "vibevoice_tts.hpp"
#include "vibevoice_speech_helpers.hpp"
#include "audio_io.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "rms_norm.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

namespace vv {

// ============================================================================
//  Loaders
// ============================================================================

namespace {

bool load_qwen2_layer(const ModelLoader& m, const std::string& prefix,
                      Qwen2LayerWeights* out) {
    auto get = [&](const char* n) { return m.tensor(prefix + n); };
    out->attn_norm   = get("attn_norm.weight");
    out->attn_q      = get("attn_q.weight");
    out->attn_q_bias = get("attn_q.bias");
    out->attn_k      = get("attn_k.weight");
    out->attn_k_bias = get("attn_k.bias");
    out->attn_v      = get("attn_v.weight");
    out->attn_v_bias = get("attn_v.bias");
    out->attn_o      = get("attn_o.weight");
    out->ffn_norm    = get("ffn_norm.weight");
    out->ffn_gate    = get("ffn_gate.weight");
    out->ffn_up      = get("ffn_up.weight");
    out->ffn_down    = get("ffn_down.weight");
    return out->attn_norm && out->attn_q && out->attn_k && out->attn_v &&
           out->attn_o && out->ffn_norm && out->ffn_gate && out->ffn_up &&
           out->ffn_down;
}

bool load_qwen2_stack(const ModelLoader& m, const std::string& prefix,
                      int n_layers, std::vector<Qwen2LayerWeights>* out) {
    out->assign(n_layers, {});
    for (int i = 0; i < n_layers; ++i) {
        char buf[64]; std::snprintf(buf, sizeof(buf), "%sblk.%d.", prefix.c_str(), i);
        if (!load_qwen2_layer(m, buf, &(*out)[i])) {
            VV_LOG_ERROR("load_qwen2_stack: layer %d incomplete (prefix %s)", i, prefix.c_str());
            return false;
        }
    }
    return true;
}

}  // namespace

bool vibevoice_load(const std::string& path, VibeVoiceModel* out) {
    if (!out->loader.load(path)) return false;
    auto& m = out->loader;
    // ggml's element-wise ops don't accept mixed f32/f16; promote all small
    // tensors up front so RMSNorm scales, biases, gammas, scaling buffers and
    // tiny embeddings work in fp32 even when the gguf is fp16. Matmul weights
    // (large 2-D tensors) stay f16.
    m.promote_small_f16_to_f32();

    out->variant = m.get_str("vibevoice.variant", "realtime-0.5b");
    const bool is_asr      = (out->variant == "asr-7b");
    const bool is_realtime = (out->variant == "realtime-0.5b");
    const bool is_15b      = (out->variant == "1.5b");

    auto& c = out->cfg;
    c.hidden       = m.get_i32 ("vibevoice.hidden");
    c.n_layers_lm  = m.get_i32 ("vibevoice.n_layers_lm");
    c.n_layers_tlm = m.get_i32 ("vibevoice.n_layers_tlm");
    c.n_heads      = m.get_i32 ("vibevoice.n_heads");
    c.n_kv_heads   = m.get_i32 ("vibevoice.n_kv_heads");
    c.head_dim     = m.get_i32 ("vibevoice.head_dim");
    c.vocab_size   = m.get_i32 ("vibevoice.vocab_size");
    c.rope_theta   = m.get_f32 ("vibevoice.rope_theta",   1.0e6f);
    c.rms_norm_eps = m.get_f32 ("vibevoice.rms_norm_eps", 1.0e-6f);
    c.latent       = m.get_i32 ("vibevoice.diffusion.latent",      64);
    c.head_layers  = m.get_i32 ("vibevoice.diffusion.head_layers",  4);
    c.ffn_ratio    = m.get_f32 ("vibevoice.diffusion.ffn_ratio",  3.0f);
    c.vae_dim      = m.get_i32 ("vibevoice.acoustic.vae_dim",      64);
    c.sample_rate  = m.get_i32 ("vibevoice.sample_rate",        24000);

    // Acoustic decoder config (we only use the decoder side here)
    c.acoustic.channels   = 1;
    c.acoustic.vae_dim    = c.vae_dim;
    c.acoustic.eps        = m.get_f32("vibevoice.acoustic.eps", 1e-5f);
    auto ratios = m.get_i32_array("vibevoice.acoustic.encoder_ratios");
    auto depths = m.get_i32_array("vibevoice.acoustic.decoder_depths");
    if (ratios.empty() || depths.empty()) {
        VV_LOG_ERROR("vibevoice_load: acoustic ratios/depths missing");
        return false;
    }
    c.acoustic.ratios.assign(ratios.begin(), ratios.end());
    c.acoustic.depths.assign(depths.begin(), depths.end());
    c.acoustic.kernel_stem = 7;
    c.acoustic.kernel_head = 7;
    c.acoustic.ffn_mult    = 4;

    // ---- LM stacks ----
    auto& w = out->w;
    w.lm_tok_embd = m.tensor("lm.tok_embd.weight");
    if (!w.lm_tok_embd) {
        VV_LOG_ERROR("vibevoice_load: missing lm.tok_embd.weight");
        return false;
    }
    if (!load_qwen2_stack(m, "lm.",  c.n_layers_lm,  &w.lm_layers))  return false;
    if (is_realtime) {
        if (!load_qwen2_stack(m, "tlm.", c.n_layers_tlm, &w.tlm_layers)) return false;
        w.tlm_output_norm = m.tensor("tlm.output_norm.weight");
        w.tts_input_types = m.tensor("tts.input_types.weight");
        if (!w.tlm_output_norm || !w.tts_input_types) {
            VV_LOG_ERROR("vibevoice_load: missing tlm.output_norm or tts.input_types");
            return false;
        }
    }

    // ---- acoustic connector (always present) ----
    w.ac_fc1_w = m.tensor("ac.fc1.weight");
    w.ac_fc1_b = m.tensor("ac.fc1.bias");
    w.ac_norm  = m.tensor("ac.norm.weight");
    w.ac_fc2_w = m.tensor("ac.fc2.weight");
    w.ac_fc2_b = m.tensor("ac.fc2.bias");
    if (!w.ac_fc1_w || !w.ac_fc2_w) {
        VV_LOG_ERROR("vibevoice_load: acoustic connector missing");
        return false;
    }

    if (is_realtime) {
        // ---- TTS-specific: EOS classifier + diffusion head + acoustic decoder ----
        w.eos_fc1_w = m.tensor("eos.fc1.weight");
        w.eos_fc1_b = m.tensor("eos.fc1.bias");
        w.eos_fc2_w = m.tensor("eos.fc2.weight");
        w.eos_fc2_b = m.tensor("eos.fc2.bias");
        if (!w.eos_fc1_w || !w.eos_fc2_w) {
            VV_LOG_ERROR("vibevoice_load: eos classifier missing");
            return false;
        }
        DiffusionHeadConfig dhc;
        dhc.hidden      = c.hidden;
        dhc.latent      = c.latent;
        dhc.head_layers = c.head_layers;
        dhc.ffn_ratio   = c.ffn_ratio;
        dhc.eps         = c.rms_norm_eps;
        dhc.freq_size   = 256;
        if (!load_diffusion_head(m, "dh.", dhc, &w.dh)) return false;
        if (!load_decoder(m, "at.dec", c.acoustic, &w.at_dec)) return false;
    }

    if (is_15b) {
        // ---- 1.5B: single-stack LM + diffusion head + decoder + encoders ----
        // No EOS classifier (uses LM logits + speech_end token instead).
        DiffusionHeadConfig dhc;
        dhc.hidden      = c.hidden;
        dhc.latent      = c.latent;
        dhc.head_layers = c.head_layers;
        dhc.ffn_ratio   = c.ffn_ratio;
        dhc.eps         = c.rms_norm_eps;
        dhc.freq_size   = 256;
        if (!load_diffusion_head(m, "dh.", dhc, &w.dh)) return false;
        if (!load_decoder(m, "at.dec", c.acoustic, &w.at_dec)) return false;
        // Single LM stack: stash its final RMSNorm in tlm_output_norm so the
        // run_qwen2_stack call site can reuse the same hook the realtime path
        // uses for the upper stack.
        w.tlm_output_norm = m.tensor("lm.output_norm.weight");
        if (!w.tlm_output_norm) {
            VV_LOG_ERROR("vibevoice_load: missing lm.output_norm.weight");
            return false;
        }
        out->lm_head = m.tensor("lm_head.weight");
        if (!out->lm_head) {
            VV_LOG_ERROR("vibevoice_load: missing lm_head.weight");
            return false;
        }
    }

    if (is_asr || is_15b) {
        // ---- ASR-specific: encoders + semantic connector + lm_head ----
        // Encoder depths are forward order (3,3,3,3,3,3,8) — different from
        // the (reversed) decoder depths the TTS path uses.
        AcousticConfig enc_cfg = c.acoustic;
        auto enc_depths = m.get_i32_array("vibevoice.acoustic.encoder_depths");
        if (!enc_depths.empty()) enc_cfg.depths.assign(enc_depths.begin(), enc_depths.end());

        if (!load_encoder(m, "at.enc", enc_cfg, &out->at_enc)) {
            VV_LOG_ERROR("vibevoice_load: acoustic encoder load failed");
            return false;
        }
        // Update the model's acoustic config so callers (e.g. the test) get the
        // right depths when running encoder_forward.
        c.acoustic = enc_cfg;
        // Semantic config (separate ratios/depths possible, but in practice same)
        out->semantic_vae_dim = m.get_i32("vibevoice.semantic.vae_dim", 128);
        out->semantic_cfg.channels = 1;
        out->semantic_cfg.vae_dim  = out->semantic_vae_dim;
        out->semantic_cfg.eps      = m.get_f32("vibevoice.acoustic.eps", 1e-5f);
        auto sm_ratios = m.get_i32_array("vibevoice.semantic.encoder_ratios");
        auto sm_depths = m.get_i32_array("vibevoice.semantic.encoder_depths");
        if (sm_ratios.empty()) sm_ratios = ratios;
        if (sm_depths.empty()) sm_depths.assign(c.acoustic.depths.begin(), c.acoustic.depths.end());
        out->semantic_cfg.ratios.assign(sm_ratios.begin(), sm_ratios.end());
        out->semantic_cfg.depths.assign(sm_depths.begin(), sm_depths.end());
        out->semantic_cfg.kernel_stem = 7;
        out->semantic_cfg.kernel_head = 7;
        out->semantic_cfg.ffn_mult    = 4;
        if (!load_encoder(m, "st.enc", out->semantic_cfg, &out->st_enc)) {
            VV_LOG_ERROR("vibevoice_load: semantic encoder load failed");
            return false;
        }
        out->sc_fc1_w = m.tensor("sc.fc1.weight");
        out->sc_fc1_b = m.tensor("sc.fc1.bias");
        out->sc_norm  = m.tensor("sc.norm.weight");
        out->sc_fc2_w = m.tensor("sc.fc2.weight");
        out->sc_fc2_b = m.tensor("sc.fc2.bias");
        out->lm_head  = m.tensor("lm_head.weight");
        if (!out->sc_fc1_w || !out->sc_fc2_w || !out->lm_head) {
            VV_LOG_ERROR("vibevoice_load: semantic connector or lm_head missing");
            return false;
        }
        // ASR also has full output_norm
        if (!w.tlm_output_norm) w.tlm_output_norm = m.tensor("lm.output_norm.weight");
    }

    // ---- speech scaling buffers (handle fp32 or fp16) ----
    // Tensors live on the active backend's buffer; read the first scalar
    // via ggml_backend_tensor_get (memcpy on CPU, DtoH on GPU).
    auto load_scalar = [](struct ggml_tensor* t) -> float {
        if (!t) return 0.0f;
        if (t->type == GGML_TYPE_F32) {
            float v = 0.0f;
            ggml_backend_tensor_get(t, &v, 0, sizeof(float));
            return v;
        }
        if (t->type == GGML_TYPE_F16) {
            ggml_fp16_t h = 0;
            ggml_backend_tensor_get(t, &h, 0, sizeof(ggml_fp16_t));
            return ggml_fp16_to_fp32(h);
        }
        return 0.0f;
    };
    c.speech_scaling = load_scalar(m.tensor("speech.scaling"));
    c.speech_bias    = load_scalar(m.tensor("speech.bias"));

    VV_LOG_INFO("vibevoice_load: hidden=%d  layers=%d+%d  vocab=%d  scaling=%.4f bias=%.4f",
                c.hidden, c.n_layers_lm, c.n_layers_tlm, c.vocab_size,
                static_cast<double>(c.speech_scaling),
                static_cast<double>(c.speech_bias));
    return true;
}

bool vibevoice_voice_load(const std::string&    path,
                          const VibeVoiceModel& model,
                          VibeVoiceVoice*       out) {
    if (!out) return false;
    static thread_local ModelLoader v_loader;
    if (!v_loader.load(path)) {
        VV_LOG_ERROR("voice_load: failed to open %s", path.c_str());
        return false;
    }
    auto& m = v_loader;

    const int hidden    = m.get_i32("voice.hidden");
    const int head_dim  = m.get_i32("voice.head_dim");
    const int n_kv      = m.get_i32("voice.n_kv_heads");
    const int n_lm_l    = m.get_i32("voice.lm.n_layers");
    const int n_tlm_l   = m.get_i32("voice.tts_lm.n_layers");
    out->seq_lm  = m.get_i32("voice.lm.seq_len");
    out->seq_tlm = m.get_i32("voice.tts_lm.seq_len");

    if (hidden != model.cfg.hidden || head_dim != model.cfg.head_dim ||
        n_kv   != model.cfg.n_kv_heads ||
        n_lm_l  != model.cfg.n_layers_lm ||
        n_tlm_l != model.cfg.n_layers_tlm) {
        VV_LOG_ERROR("voice_load: shape mismatch with model "
                     "(hidden %d/%d head_dim %d/%d n_kv %d/%d "
                     "lm_layers %d/%d tlm_layers %d/%d)",
                     hidden, model.cfg.hidden, head_dim, model.cfg.head_dim,
                     n_kv, model.cfg.n_kv_heads,
                     n_lm_l, model.cfg.n_layers_lm,
                     n_tlm_l, model.cfg.n_layers_tlm);
        return false;
    }

    auto load_stack = [&](const char* prefix, int n_layers, int seq_len,
                          std::vector<LayerKV>* dst) {
        dst->assign(n_layers, {});
        const size_t n = static_cast<size_t>(head_dim) * n_kv * seq_len;
        for (int i = 0; i < n_layers; ++i) {
            char nk[64], nv[64];
            std::snprintf(nk, sizeof(nk), "%s.k.%d", prefix, i);
            std::snprintf(nv, sizeof(nv), "%s.v.%d", prefix, i);
            struct ggml_tensor* k = m.tensor(nk);
            struct ggml_tensor* v = m.tensor(nv);
            if (!k || !v) {
                VV_LOG_ERROR("voice_load: missing %s or %s", nk, nv);
                return false;
            }
            (*dst)[i].k.assign(n, 0.0f);
            (*dst)[i].v.assign(n, 0.0f);
            ggml_backend_tensor_get(k, (*dst)[i].k.data(), 0, sizeof(float) * n);
            ggml_backend_tensor_get(v, (*dst)[i].v.data(), 0, sizeof(float) * n);
            (*dst)[i].past_len = seq_len;
        }
        return true;
    };
    if (!load_stack("voice.lm",     n_lm_l,  out->seq_lm,  &out->kv_lm))  return false;
    if (!load_stack("voice.tts_lm", n_tlm_l, out->seq_tlm, &out->kv_tlm)) return false;

    struct ggml_tensor* tlm_h = m.tensor("voice.tts_lm.last_hidden");
    if (!tlm_h) {
        VV_LOG_ERROR("voice_load: missing voice.tts_lm.last_hidden");
        return false;
    }
    auto last_col = [&](struct ggml_tensor* t, int seq, std::vector<float>* dst) {
        dst->resize(hidden);
        ggml_backend_tensor_get(t, dst->data(),
                                sizeof(float) * static_cast<size_t>(hidden) * (seq - 1),
                                sizeof(float) * hidden);
    };
    last_col(tlm_h, out->seq_tlm, &out->tlm_last_hidden);

    // Optional negative branch (CFG).
    out->has_neg = m.get_bool("voice.has_neg", false);
    if (out->has_neg) {
        out->seq_neg_lm  = m.get_i32("voice.neg_lm.seq_len");
        out->seq_neg_tlm = m.get_i32("voice.neg_tts_lm.seq_len");
        if (!load_stack("voice.neg_lm",     n_lm_l,  out->seq_neg_lm,  &out->kv_neg_lm))  out->has_neg = false;
        if (!load_stack("voice.neg_tts_lm", n_tlm_l, out->seq_neg_tlm, &out->kv_neg_tlm)) out->has_neg = false;
        struct ggml_tensor* nh = m.tensor("voice.neg_tts_lm.last_hidden");
        if (out->has_neg && nh) {
            last_col(nh, out->seq_neg_tlm, &out->neg_tlm_last_hidden);
        }
    }

    VV_LOG_INFO("voice_load: %s  lm=%dx%d  tlm=%dx%d  neg=%s",
                path.c_str(), n_lm_l, out->seq_lm, n_tlm_l, out->seq_tlm,
                out->has_neg ? "on" : "off");
    return true;
}

// ============================================================================
//  Inference
// ============================================================================

namespace {


// Build & run one forward pass through a Qwen2 stack.
// `inputs_embeds`: [hidden, n_new_tokens, B=1]
// `pos_start`:     absolute position of the first new token
// `kvs`:           resident K/V cache; new K/V written via ggml_cpy at
//                  offset kvs.past_len, then kvs.past_len is advanced.
// `out_hidden`:    if non-null, all output hidden states ([hidden * n_new_tokens])
// `out_hidden_last`: if non-null, only the last token's hidden state ([hidden])
bool run_qwen2_stack(struct ggml_context* /*ctx_ext*/,
                     const VibeVoiceConfig& cfg,
                     const std::vector<Qwen2LayerWeights>& layers,
                     struct ggml_tensor*  output_norm,
                     int                  pos_start,
                     int                  n_new_tokens,
                     const float*         inputs_embeds,
                     ResidentKV*           kvs,
                     std::vector<float>*   out_hidden,            // optional
                     std::vector<float>*   out_hidden_last) {     // optional
    const int hidden = cfg.hidden;

    Qwen2Hparams hp;
    hp.hidden_size       = hidden;
    hp.n_heads           = cfg.n_heads;
    hp.n_kv_heads        = cfg.n_kv_heads;
    hp.head_dim          = cfg.head_dim;
    hp.intermediate_size = 0;
    hp.rope_theta        = cfg.rope_theta;
    hp.rms_norm_eps      = cfg.rms_norm_eps;
    hp.use_flash_attn    = vv::backend_supports_flash_attn();

    const int kv_len_old = kvs->past_len;
    const int kv_len_new = kv_len_old + n_new_tokens;

    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 16384
               + ggml_graph_overhead_custom(16384, false);
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return false;

    struct ggml_tensor* x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, n_new_tokens, 1);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_new_tokens);
    struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len_new, n_new_tokens);

    struct ggml_tensor* h = x;
    std::vector<struct ggml_tensor*> k_writes(layers.size()),
                                     v_writes(layers.size());
    for (size_t li = 0; li < layers.size(); ++li) {
        auto out = qwen2_layer_forward_resident(ctx, h, pos, mask,
                                                *kvs, static_cast<int>(li),
                                                layers[li], hp);
        h            = out.y;
        k_writes[li] = out.k_write;
        v_writes[li] = out.v_write;
    }
    if (output_norm) {
        h = rms_norm(ctx, h, output_norm, cfg.rms_norm_eps);
    }

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
    // Interleave per-layer cpys (cpy_v_li MUST be in the graph before
    // attention_(li+1) is added through k_write[li+1]'s expansion - same
    // ordering rule as the ASR prefill, see test_qwen2_resident chain28
    // one-graph case).
    for (size_t li = 0; li < layers.size(); ++li) {
        ggml_build_forward_expand(gf, k_writes[li]);
        ggml_build_forward_expand(gf, v_writes[li]);
    }
    ggml_build_forward_expand(gf, h);

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(x, inputs_embeds, 0, sizeof(float) * hidden * n_new_tokens);
    std::vector<int32_t> pos_v(n_new_tokens);
    for (int i = 0; i < n_new_tokens; ++i) pos_v[i] = pos_start + i;
    ggml_backend_tensor_set(pos, pos_v.data(), 0, sizeof(int32_t) * n_new_tokens);
    std::vector<ggml_fp16_t> mask_v(static_cast<size_t>(kv_len_new) * n_new_tokens);
    const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n_new_tokens; ++i) {
        const int qabs = pos_start + i;
        for (int j = 0; j < kv_len_new; ++j) {
            mask_v[i * kv_len_new + j] = (j > qabs) ? f16_ninf : f16_zero;
        }
    }
    ggml_backend_tensor_set(mask, mask_v.data(), 0, sizeof(ggml_fp16_t) * mask_v.size());

    if (!vv::compute_graph(gf)) {
        ggml_backend_buffer_free(in_buf); ggml_free(ctx); return false;
    }

    if (out_hidden) {
        out_hidden->resize(static_cast<size_t>(hidden) * n_new_tokens);
        ggml_backend_tensor_get(h, out_hidden->data(), 0,
                                sizeof(float) * hidden * n_new_tokens);
    }
    if (out_hidden_last) {
        out_hidden_last->resize(hidden);
        ggml_backend_tensor_get(h, out_hidden_last->data(),
                                sizeof(float) * hidden * (n_new_tokens - 1),
                                sizeof(float) * hidden);
    }

    kvs->past_len = kv_len_new;

    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return true;
}

// SpeechConnector: y = fc2(rmsnorm_last_dim(fc1(x)))
// `x` is [latent, B], output is [hidden, B].
std::vector<float> run_speech_connector(const VibeVoiceConfig&  cfg,
                                        const VibeVoiceWeights& w,
                                        const float*            x,
                                        int                     batch) {
    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead();
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);

    const int latent = cfg.latent;
    const int hidden = cfg.hidden;
    struct ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent, batch);

    struct ggml_tensor* h = ggml_mul_mat(ctx, w.ac_fc1_w, in);
    if (w.ac_fc1_b) h = ggml_add(ctx, h, w.ac_fc1_b);
    h = ggml_rms_norm(ctx, h, 1e-6f);
    h = ggml_mul(ctx, h, w.ac_norm);
    h = ggml_mul_mat(ctx, w.ac_fc2_w, h);
    if (w.ac_fc2_b) h = ggml_add(ctx, h, w.ac_fc2_b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, h);

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return {}; }
    ggml_backend_tensor_set(in, x, 0, sizeof(float) * latent * batch);

    std::vector<float> out;
    if (vv::compute_graph(gf)) {
        out.assign(static_cast<size_t>(hidden) * batch, 0.0f);
        ggml_backend_tensor_get(h, out.data(), 0, sizeof(float) * out.size());
    }
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return out;
}

// EOS classifier: sigmoid(fc2(relu(fc1(x))))
float run_eos_classifier(const VibeVoiceConfig&  /*cfg*/,
                         const VibeVoiceWeights& w,
                         const float*            x_hidden) {
    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);

    const int hidden = w.eos_fc1_w->ne[0];
    struct ggml_tensor* in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);

    struct ggml_tensor* h = ggml_mul_mat(ctx, w.eos_fc1_w, in);
    if (w.eos_fc1_b) h = ggml_add(ctx, h, w.eos_fc1_b);
    h = ggml_relu(ctx, h);
    h = ggml_mul_mat(ctx, w.eos_fc2_w, h);
    if (w.eos_fc2_b) h = ggml_add(ctx, h, w.eos_fc2_b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, h);

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return 0.0f; }
    ggml_backend_tensor_set(in, x_hidden, 0, sizeof(float) * hidden);

    float logit = 0.0f;
    if (vv::compute_graph(gf)) {
        ggml_backend_tensor_get(h, &logit, 0, sizeof(float));
    }
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return 1.0f / (1.0f + std::exp(-logit));
}

// Look up the tts_input_types embedding for a given mask (0=speech, 1=text)
// and add it (broadcast across n_tokens) into x [hidden * n_tokens].
void add_input_type_embedding(const VibeVoiceConfig& cfg,
                              const VibeVoiceWeights& w,
                              int n_tokens,
                              int type_id,                // 0 or 1
                              float* x) {
    const int hidden = cfg.hidden;
    // tts_input_types lives on the active backend's buffer; pull the row
    // once via ggml_backend_tensor_get into a CPU staging buffer.
    const size_t row_bytes = (w.tts_input_types->type == GGML_TYPE_F32)
                             ? sizeof(float) * hidden
                             : sizeof(ggml_fp16_t) * hidden;
    std::vector<uint8_t> row(row_bytes);
    ggml_backend_tensor_get(w.tts_input_types, row.data(),
                            row_bytes * static_cast<size_t>(type_id),
                            row_bytes);
    if (w.tts_input_types->type == GGML_TYPE_F32) {
        const float* emb = reinterpret_cast<const float*>(row.data());
        for (int t = 0; t < n_tokens; ++t)
            for (int i = 0; i < hidden; ++i)
                x[t * hidden + i] += emb[i];
    } else if (w.tts_input_types->type == GGML_TYPE_F16) {
        const ggml_fp16_t* emb = reinterpret_cast<const ggml_fp16_t*>(row.data());
        for (int t = 0; t < n_tokens; ++t)
            for (int i = 0; i < hidden; ++i)
                x[t * hidden + i] += ggml_fp16_to_fp32(emb[i]);
    }
}

// Decode a SEQUENCE of N speech latents into audio samples in a single
// decoder pass. The decoder is causal — its convolutions need to see the
// full latent trajectory to produce coherent audio. Decoding frame-by-frame
// independently zeros out the receptive field across frames and yields
// "lyric"-style noise instead of intelligible speech.
//
// `scaled_latents` has shape [vae_dim * n_frames] in row-major (latent
// fastest), matching what `ggml_new_tensor_3d(ctx, F32, n_frames, vae_dim, 1)`
// expects when ne[0] = n_frames is the contiguous dim.
std::vector<float> decode_latent_sequence(const VibeVoiceConfig&  cfg,
                                          const VibeVoiceWeights& w,
                                          const float*            scaled_latents,
                                          int                     n_frames) {
    if (n_frames <= 0) return {};
    // Backend-aware compute: build the graph in a no_alloc ctx, allocate
    // leaf tensors on the active backend's buffer, upload input via
    // ggml_backend_tensor_set, and let vv::compute_graph allocate the
    // intermediates and dispatch.
    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);

    struct ggml_tensor* z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_frames, cfg.vae_dim, 1);
    ggml_set_name(z, "decode_z");

    struct ggml_tensor* y = decoder_forward(ctx, z, w.at_dec, cfg.acoustic);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return {}; }
    ggml_backend_tensor_set(z, scaled_latents, 0,
                            sizeof(float) * cfg.vae_dim * n_frames);

    if (!vv::compute_graph(gf)) {
        ggml_backend_buffer_free(in_buf);
        ggml_free(ctx);
        return {};
    }
    const int T_full = static_cast<int>(y->ne[0]);
    std::vector<float> samples(T_full);
    ggml_backend_tensor_get(y, samples.data(), 0, sizeof(float) * T_full);
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return samples;
}

}  // namespace

// Forward declaration — implementation later in this file.
namespace {
int tts_15b_generate(VibeVoiceModel*            model,
                     const std::string&         ref_wav_path,
                     const std::string&         text,
                     const VibeVoiceTTSParams&  p,
                     std::vector<float>*        samples);
}

int vibevoice_tts_generate(VibeVoiceModel*           model,
                           const std::string&        text,
                           const VibeVoiceTTSParams& p,
                           std::vector<float>*       samples) {
    if (!model || !samples) return -1;

    // Variant dispatch — the gguf already knows which path it wants;
    // callers don't need a separate entry point per model family.
    if (model->variant == "1.5b") {
        if (p.ref_audio_path.empty()) {
            VV_LOG_ERROR("vibevoice_tts_generate: 1.5b model requires "
                         "VibeVoiceTTSParams::ref_audio_path");
            return -1;
        }
        return tts_15b_generate(model, p.ref_audio_path, text, p, samples);
    }

    const auto& cfg = model->cfg;
    const auto& w   = model->w;

    // ---- 1. tokenize ----
    if (!model->tokenizer.vocab_size()) {
        VV_LOG_ERROR("vibevoice_tts_generate: tokenizer not loaded");
        return -2;
    }
    // mlx-audio convention: append "\n" to terminate the user turn so the
    // model knows the input is complete and starts the speech reply. The
    // upstream voice prefix ends mid-conversation; without a separator, the
    // model treats the new text as a continuation of the prior turn.
    std::string text_with_sep = text;
    while (!text_with_sep.empty() &&
           (text_with_sep.back() == ' ' || text_with_sep.back() == '\t' ||
            text_with_sep.back() == '\r' || text_with_sep.back() == '\n')) {
        text_with_sep.pop_back();
    }
    text_with_sep += "\n";
    const auto text_ids = model->tokenizer.encode(text_with_sep);
    if (text_ids.empty()) {
        VV_LOG_ERROR("tokenizer produced no tokens");
        return -3;
    }
    if (p.verbose) {
        std::fprintf(stderr, "[tts] %zu input text tokens\n", text_ids.size());
    }

    // ---- 2. embed text via lm.tok_embd ----
    // tok_embd is [hidden, vocab_size] in ggml after gguf load. May be fp32
    // or fp16 depending on the converter --dtype. Lives on the active
    // backend's buffer, so each row is fetched via ggml_backend_tensor_get
    // (a memcpy on CPU; a DtoH transfer on CUDA / Metal / Vulkan).
    const int hidden = cfg.hidden;
    const int n_text = static_cast<int>(text_ids.size());
    std::vector<float> text_embeds(static_cast<size_t>(hidden) * n_text);

    if (w.lm_tok_embd->type == GGML_TYPE_F32) {
        const size_t row = sizeof(float) * hidden;
        for (int t = 0; t < n_text; ++t) {
            const int id = text_ids[t];
            if (id < 0 || id >= cfg.vocab_size) {
                VV_LOG_ERROR("token id out of range: %d", id);
                return -5;
            }
            ggml_backend_tensor_get(w.lm_tok_embd, &text_embeds[hidden * t],
                                    row * static_cast<size_t>(id), row);
        }
    } else if (w.lm_tok_embd->type == GGML_TYPE_F16) {
        const size_t row = sizeof(ggml_fp16_t) * hidden;
        std::vector<ggml_fp16_t> staged(hidden);
        for (int t = 0; t < n_text; ++t) {
            const int id = text_ids[t];
            if (id < 0 || id >= cfg.vocab_size) {
                VV_LOG_ERROR("token id out of range: %d", id);
                return -5;
            }
            ggml_backend_tensor_get(w.lm_tok_embd, staged.data(),
                                    row * static_cast<size_t>(id), row);
            for (int i = 0; i < hidden; ++i) {
                text_embeds[hidden * t + i] = ggml_fp16_to_fp32(staged[i]);
            }
        }
    } else {
        VV_LOG_ERROR("vibevoice_tts_generate: lm.tok_embd unsupported dtype %d",
                     static_cast<int>(w.lm_tok_embd->type));
        return -4;
    }

    // CFG parallel state probe (reads from voice prompt).
    const bool use_cfg = p.voice && p.voice->has_neg && p.cfg_scale > 1.0f;

    // Resident K/V caches. lm and tlm grow with text + (tlm only) speech
    // frames. neg_tlm starts at the negative voice's seq_neg_tlm and
    // grows by the same speech-frame count as tlm. We size for the
    // pessimistic upper bound: voice prefix + n_text + max_speech_frames.
    const int hd   = cfg.head_dim;
    const int n_kv = cfg.n_kv_heads;
    const int max_lm_seq      = (p.voice ? p.voice->seq_lm     : 0) + n_text                       + 32;
    const int max_tlm_seq     = (p.voice ? p.voice->seq_tlm    : 0) + n_text + p.max_speech_frames + 32;
    const int max_neg_tlm_seq = (use_cfg ? p.voice->seq_neg_tlm : 0)         + p.max_speech_frames + 32;

    ResidentKV kv_lm, kv_tlm, kv_neg_tlm;
    if (!kv_lm.init (cfg.n_layers_lm,  hd, n_kv, max_lm_seq))  return -5;
    if (!kv_tlm.init(cfg.n_layers_tlm, hd, n_kv, max_tlm_seq)) return -5;
    if (use_cfg && !kv_neg_tlm.init(cfg.n_layers_tlm, hd, n_kv, max_neg_tlm_seq)) return -5;

    // Upload voice-prompt KV into the resident buffers. After this the
    // host-side voice->kv_* vectors are no longer touched - the resident
    // tensors are the source of truth for the rest of the call.
    auto upload_kv = [hd, n_kv](ResidentKV& dst,
                                const std::vector<LayerKV>& src,
                                int seq_len) {
        for (size_t li = 0; li < src.size(); ++li) {
            const size_t per = static_cast<size_t>(hd) * n_kv * seq_len;
            ggml_backend_tensor_set(dst.k[li], src[li].k.data(), 0, sizeof(float) * per);
            ggml_backend_tensor_set(dst.v[li], src[li].v.data(), 0, sizeof(float) * per);
        }
        dst.past_len = seq_len;
    };
    if (p.voice) {
        upload_kv(kv_lm,  p.voice->kv_lm,  p.voice->seq_lm);
        upload_kv(kv_tlm, p.voice->kv_tlm, p.voice->seq_tlm);
        if (use_cfg) upload_kv(kv_neg_tlm, p.voice->kv_neg_tlm, p.voice->seq_neg_tlm);
    }

    int                  lm_pos  = p.voice ? p.voice->seq_lm  : 0;
    int                  tlm_pos = p.voice ? p.voice->seq_tlm : 0;
    std::vector<float>   tlm_hidden_last = p.voice ? p.voice->tlm_last_hidden
                                                   : std::vector<float>(hidden, 0.0f);
    int                  neg_tlm_pos = use_cfg ? p.voice->seq_neg_tlm : 0;
    std::vector<float>   neg_tlm_hidden_last = use_cfg ? p.voice->neg_tlm_last_hidden
                                                       : std::vector<float>{};
    if (p.verbose) std::fprintf(stderr, "[tts] cfg=%s (scale=%.2f)\n",
                                use_cfg ? "on" : "off",
                                static_cast<double>(p.cfg_scale));

    // mlx-audio + upstream alternate text windows (5 tokens) with speech
    // windows (6 frames). The model is trained on this pattern; processing
    // all text up front then generating all speech misleads it.
    constexpr int kTextWindow   = 5;
    constexpr int kSpeechWindow = 6;

    // ---- diffusion + RNG setup ----
    DPMSolverConfig solver_cfg;
    solver_cfg.num_train_timesteps = 1000;
    solver_cfg.num_inference_steps = p.n_diffusion_steps;
    solver_cfg.solver_order        = 2;
    solver_cfg.lower_order_final   = true;
    DPMSolverState solver_state;
    dpm_solver_init(solver_cfg, &solver_state);

    DiffusionHeadConfig dh_cfg;
    dh_cfg.hidden      = hidden;
    dh_cfg.latent      = cfg.latent;
    dh_cfg.head_layers = cfg.head_layers;
    dh_cfg.ffn_ratio   = cfg.ffn_ratio;
    dh_cfg.eps         = cfg.rms_norm_eps;
    dh_cfg.freq_size   = 256;

    std::mt19937 rng(p.seed ? p.seed : std::random_device{}());
    std::normal_distribution<float> norm(0.0f, 1.0f);

    samples->clear();
    std::vector<float> all_latents;
    all_latents.reserve(static_cast<size_t>(p.max_speech_frames) * cfg.latent);

    int  text_pos = 0;
    bool finished = false;
    int  total_frames = 0;

    while (!finished && total_frames < p.max_speech_frames) {
        // ---- text window (up to 5 tokens) ----
        if (text_pos < n_text) {
            const int win = std::min(kTextWindow, n_text - text_pos);
            // Slice the text embeddings for this window.
            std::vector<float> emb_win(static_cast<size_t>(hidden) * win);
            std::memcpy(emb_win.data(),
                        text_embeds.data() + static_cast<size_t>(hidden) * text_pos,
                        sizeof(float) * hidden * win);

            // LM forward
            std::vector<float> lm_hidden;
            if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, /*output_norm=*/nullptr,
                                 lm_pos, win, emb_win.data(), &kv_lm,
                                 &lm_hidden, nullptr)) return -6;
            lm_pos += win;

            // TTS-LM input = LM hidden + text-type embedding
            std::vector<float> tlm_in = lm_hidden;
            add_input_type_embedding(cfg, w, win, /*type=*/1, tlm_in.data());

            if (!run_qwen2_stack(nullptr, cfg, w.tlm_layers,
                                 /*output_norm=*/w.tlm_output_norm,
                                 tlm_pos, win, tlm_in.data(), &kv_tlm,
                                 nullptr, &tlm_hidden_last)) return -8;
            tlm_pos += win;

            text_pos += win;
            if (p.verbose) std::fprintf(stderr,
                "[tts] text window: %d/%d tokens consumed\n", text_pos, n_text);
        }

        // ---- speech window (up to 6 frames or until EOS) ----
        const int sp_budget = std::min(kSpeechWindow, p.max_speech_frames - total_frames);
        for (int sp = 0; sp < sp_budget; ++sp) {
            const int frame = total_frames;
        // 5a. Sample one speech latent from the diffusion head.
        std::vector<float> z(cfg.latent);
        for (auto& v : z) v = norm(rng);

        std::vector<float> cond(static_cast<size_t>(hidden));
        std::memcpy(cond.data(), tlm_hidden_last.data(), sizeof(float) * hidden);

        // dpm_solver_sample expects shape [latent * frames * batch] with
        // frames=1, B=1 here. cond shape: [hidden * 1 * 1].
        std::vector<float> cond_neg;
        if (use_cfg) {
            cond_neg.assign(neg_tlm_hidden_last.begin(), neg_tlm_hidden_last.end());
        }
        if (dpm_solver_sample(z, cfg.latent, /*frames=*/1, /*batch=*/1,
                              cond, hidden,
                              w.dh, dh_cfg, solver_cfg, solver_state,
                              cond_neg, p.cfg_scale) != 0) {
            VV_LOG_ERROR("dpm_solver_sample failed at frame %d", frame);
            return -9;
        }

        // 5b. Buffer this latent (decoder runs on the full sequence later).
        all_latents.insert(all_latents.end(), z.begin(), z.end());

        // 5c. Project latent through connector → next-step embedding.
        auto ac_embed = run_speech_connector(cfg, w, z.data(), /*batch=*/1);
        // Add tts_input_types[0] (speech type)
        add_input_type_embedding(cfg, w, /*n_tokens=*/1, /*type=*/0, ac_embed.data());

        // 5d. Step TTS-LM by one position (positive branch).
            if (!run_qwen2_stack(nullptr, cfg, w.tlm_layers, /*output_norm=*/w.tlm_output_norm,
                                 tlm_pos, /*n_new=*/1, ac_embed.data(), &kv_tlm,
                                 nullptr, &tlm_hidden_last)) {
                VV_LOG_ERROR("TTS-LM speech step failed at frame %d", frame);
                return -10;
            }
            tlm_pos += 1;

            if (use_cfg) {
                std::vector<float> ac_embed_neg(ac_embed);
                if (!run_qwen2_stack(nullptr, cfg, w.tlm_layers,
                                     /*output_norm=*/w.tlm_output_norm,
                                     neg_tlm_pos, /*n_new=*/1, ac_embed_neg.data(),
                                     &kv_neg_tlm, nullptr, &neg_tlm_hidden_last)) {
                    VV_LOG_ERROR("neg TTS-LM speech step failed at frame %d", frame);
                    return -10;
                }
                neg_tlm_pos += 1;
            }

            const float eos = run_eos_classifier(cfg, w, tlm_hidden_last.data());
            if (p.verbose && (total_frames % 4 == 0 || eos > 0.5f)) {
                std::fprintf(stderr, "[tts] frame %d: eos=%.3f, %d latents\n",
                             total_frames, eos,
                             static_cast<int>(all_latents.size() / cfg.latent));
            }
            ++total_frames;
            if (eos > 0.5f) {
                if (p.verbose) std::fprintf(stderr, "[tts] EOS at frame %d\n", frame);
                finished = true;
                break;
            }
        }

        // If we've consumed all text AND not finished, the outer loop will
        // continue with empty-text iterations (just speech) until EOS or cap.
        if (text_pos >= n_text && !finished) {
            // No more text — keep generating speech until EOS or budget.
        }
    }

    // ---- 6. decode the full latent sequence in one pass ----
    const int n_frames = static_cast<int>(all_latents.size() / cfg.latent);
    if (n_frames > 0) {
        // scaled = latent / speech_scaling - speech_bias  (mlx-audio order)
        std::vector<float> scaled(all_latents.size());
        for (size_t i = 0; i < all_latents.size(); ++i) {
            scaled[i] = all_latents[i] / cfg.speech_scaling - cfg.speech_bias;
        }
        // The decoder expects the latent dim as the contiguous (innermost)
        // axis in ggml. Our `all_latents` is stored as N consecutive
        // [latent]-sized chunks in time order, which is exactly that layout
        // when reshaped to [n_frames, vae_dim] (numpy) -> ggml [vae_dim,
        // n_frames] -> oh wait we need to transpose. Re-pack as
        // [latent fastest, frames slower] (ggml ne[0]=n_frames, ne[1]=vae_dim
        // is wrong — should be ne[0]=vae_dim).
        //
        // Actually our existing decoder fixture / forward expects ne[0] =
        // T_compressed (frames). Let's check by tracing:
        //   encoder_forward output has ne = [T_compr, vae_dim, B]
        //   decoder_forward input  has ne = [T_compr, vae_dim, B]   (same)
        // So ne[0] is frames, ne[1] is vae_dim. Memory: frame fastest.
        // We append [latent] vectors per frame so memory is "latent fastest"
        // = mismatched. Need to transpose.
        std::vector<float> packed(scaled.size());
        for (int t = 0; t < n_frames; ++t) {
            for (int d = 0; d < cfg.latent; ++d) {
                packed[d * n_frames + t] = scaled[t * cfg.latent + d];
            }
        }
        auto audio = decode_latent_sequence(cfg, w, packed.data(), n_frames);
        *samples = std::move(audio);
    }

    if (p.verbose) std::fprintf(stderr, "[tts] decoded %zu samples from %d latents\n",
                                samples->size(), n_frames);
    return 0;
}

// ============================================================================
// 1.5B path
// ============================================================================
//
// The 1.5B model is structurally simpler than realtime: a single Qwen2.5-1.5B
// LM stack feeds the diffusion head directly (no `tts_lm` split, no `eos`
// classifier, no `tts.input_types` type embedding). Voice cloning is built
// in: the prompt has placeholder `<|vision_pad|>` tokens whose embeddings are
// replaced inline with features from the reference audio's at_enc + st_enc
// + connectors (summed). The LM emits `<|vision_end|>` when it has finished
// generating speech.
//
// Token IDs (from Qwen/Qwen2.5-1.5B vocab; the `VibeVoiceTextTokenizer`
// repurposes the unused `vision_*` tokens for speech):
//
//   speech_start     = <|vision_start|>     = 151652
//   speech_end       = <|vision_end|>       = 151653
//   speech_diffusion = <|vision_pad|>       = 151654

namespace {

constexpr int kSpeech15bStartId = 151652;  // <|vision_start|>
constexpr int kSpeech15bEndId   = 151653;  // <|vision_end|>
constexpr int kSpeech15bDiffId  = 151654;  // <|vision_pad|>
constexpr int kSpeech15bImgPadId = 151655; // <|image_pad|> — negative branch
constexpr int kSpeech15bCompressRatio = 3200;

// Build the 1.5B prompt as a literal byte-BPE string. The tokenizer's
// special-token matching rewrites the bracketed markers into single token
// IDs (151652 / 151653 / 151654 etc.) while the rest is regular text.
//
// Mirrors `VibeVoiceProcessor._create_voice_prompt` +
// `VibeVoiceProcessor.__call__` from the upstream reference repo.
std::string build_prompt_15b(int vae_tok_len, const std::string& text) {
    std::string out;
    out.reserve(2048 + vae_tok_len * 16);
    out += " Transform the text provided by various speakers into speech "
           "output, utilizing the distinct voice of each respective speaker.\n";
    out += " Voice input:\n";
    out += " Speaker 0:";
    out += "<|vision_start|>";
    for (int i = 0; i < vae_tok_len; ++i) out += "<|vision_pad|>";
    out += "<|vision_end|>\n";
    out += " Text input:\n";
    out += " Speaker 0:";
    out += text;
    out += "\n";
    out += " Speech output:\n";
    out += "<|vision_start|>";
    return out;
}

// Fetch a single row from `tok_embd` (handles fp32 / fp16). Caller passes
// a destination buffer of `hidden` floats.
void embed_row(struct ggml_tensor* tok_embd, int id, int hidden, float* dst) {
    if (tok_embd->type == GGML_TYPE_F32) {
        const size_t row = sizeof(float) * hidden;
        ggml_backend_tensor_get(tok_embd, dst,
                                row * static_cast<size_t>(id), row);
    } else {
        // fp16
        const size_t row = sizeof(ggml_fp16_t) * hidden;
        std::vector<ggml_fp16_t> staged(hidden);
        ggml_backend_tensor_get(tok_embd, staged.data(),
                                row * static_cast<size_t>(id), row);
        for (int i = 0; i < hidden; ++i) {
            dst[i] = ggml_fp16_to_fp32(staged[i]);
        }
    }
}

}  // namespace

namespace {
int tts_15b_generate(VibeVoiceModel*            model,
                     const std::string&         ref_wav_path,
                     const std::string&         text,
                     const VibeVoiceTTSParams&  p,
                     std::vector<float>*        samples) {
    if (!model || !samples) return -1;
    if (model->variant != "1.5b") {
        VV_LOG_ERROR("tts_15b: model is variant=%s, expected 1.5b",
                     model->variant.c_str());
        return -1;
    }
    if (!model->tokenizer.vocab_size()) {
        VV_LOG_ERROR("tts_15b: tokenizer not loaded");
        return -2;
    }

    const auto& cfg = model->cfg;
    const auto& w   = model->w;
    const int hidden = cfg.hidden;

    samples->clear();

    // ---- 1. load reference audio ----
    std::vector<float> ref_audio;
    if (load_wav_24k_mono(ref_wav_path.c_str(), &ref_audio) != 0 ||
        ref_audio.empty()) {
        VV_LOG_ERROR("tts_15b: failed to load reference WAV: %s",
                     ref_wav_path.c_str());
        return -3;
    }

    // RMS-normalize to -25 dBFS, mirrors the upstream AudioNormalizer +
    // matches what the ASR encoder expects (same encoders, same convention).
    {
        const float target_dB_FS = -25.0f, eps = 1e-6f;
        double sq = 0.0;
        for (float v : ref_audio) sq += static_cast<double>(v) * v;
        const float rms = static_cast<float>(std::sqrt(sq / std::max<size_t>(ref_audio.size(), 1)));
        const float target_lin = std::pow(10.0f, target_dB_FS / 20.0f);
        const float scalar = target_lin / (rms + eps);
        for (auto& v : ref_audio) v *= scalar;
        float maxabs = 0.0f;
        for (float v : ref_audio) maxabs = std::max(maxabs, std::fabs(v));
        if (maxabs > 1.0f) {
            const float clip_div = maxabs + eps;
            for (auto& v : ref_audio) v /= clip_div;
        }
    }

    // ---- 2. acoustic + semantic encoders ----
    std::vector<float> ac_lat, sm_lat;
    int Tc_a = 0, Tc_s = 0;
    if (!detail::run_encoder_buf(model->at_enc, cfg.acoustic, ref_audio,
                                  &ac_lat, &Tc_a)) return -4;
    if (!detail::run_encoder_buf(model->st_enc, model->semantic_cfg, ref_audio,
                                  &sm_lat, &Tc_s)) return -4;
    if (Tc_a != Tc_s) {
        VV_LOG_ERROR("tts_15b: encoder frame mismatch %d vs %d", Tc_a, Tc_s);
        return -5;
    }
    const int Tc = Tc_a;
    if (p.verbose) std::fprintf(stderr,
        "[tts_15b] ref %zu samples -> %d compressed frames\n",
        ref_audio.size(), Tc);

    // ---- 3. connectors -> per-frame speech features ----
    auto ac_emb = detail::run_connector(w.ac_fc1_w, w.ac_fc1_b, w.ac_norm,
                                         w.ac_fc2_w, w.ac_fc2_b,
                                         ac_lat, cfg.vae_dim, Tc, hidden);
    auto sm_emb = detail::run_connector(model->sc_fc1_w, model->sc_fc1_b, model->sc_norm,
                                         model->sc_fc2_w, model->sc_fc2_b,
                                         sm_lat, model->semantic_vae_dim, Tc, hidden);
    if (ac_emb.size() != sm_emb.size()
        || ac_emb.size() != static_cast<size_t>(hidden) * Tc) {
        VV_LOG_ERROR("tts_15b: connector output size mismatch");
        return -6;
    }
    std::vector<float> speech_features(static_cast<size_t>(hidden) * Tc);
    for (size_t i = 0; i < speech_features.size(); ++i) {
        speech_features[i] = ac_emb[i] + sm_emb[i];
    }

    // ---- 4. build prompt + tokenize ----
    const std::string prompt = build_prompt_15b(Tc, text);
    const auto input_ids = model->tokenizer.encode(prompt);
    if (input_ids.empty()) {
        VV_LOG_ERROR("tts_15b: tokenizer returned no tokens");
        return -7;
    }
    const int N = static_cast<int>(input_ids.size());

    std::vector<int> pad_positions;
    pad_positions.reserve(Tc);
    for (int i = 0; i < N; ++i) {
        if (input_ids[i] == kSpeech15bDiffId) pad_positions.push_back(i);
    }
    if (static_cast<int>(pad_positions.size()) != Tc) {
        VV_LOG_ERROR("tts_15b: prompt has %zu vision_pad tokens, want %d "
                     "(tokenizer mis-decoding the speech markers?)",
                     pad_positions.size(), Tc);
        return -8;
    }
    if (p.verbose) std::fprintf(stderr,
        "[tts_15b] prompt %d tokens, %d are speech-pad\n", N, Tc);

    // ---- 5. embed prompt + splice speech features ----
    std::vector<float> embeds(static_cast<size_t>(hidden) * N);
    for (int t = 0; t < N; ++t) {
        const int id = input_ids[t];
        if (id < 0 || id >= cfg.vocab_size) {
            VV_LOG_ERROR("tts_15b: token id out of range: %d", id);
            return -9;
        }
        embed_row(w.lm_tok_embd, id, hidden, &embeds[hidden * t]);
    }
    for (int k = 0; k < Tc; ++k) {
        const int pos = pad_positions[k];
        std::memcpy(&embeds[static_cast<size_t>(hidden) * pos],
                    &speech_features[static_cast<size_t>(hidden) * k],
                    sizeof(float) * hidden);
    }

    // ---- 6. resident KV cache + LM prefill ----
    const int max_seq = N + p.max_speech_frames + 32;
    ResidentKV kv_lm;
    if (!kv_lm.init(cfg.n_layers_lm, cfg.head_dim, cfg.n_kv_heads, max_seq)) {
        VV_LOG_ERROR("tts_15b: resident KV init failed");
        return -10;
    }

    std::vector<float> hidden_last;
    if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                         /*past_len=*/0, /*n_new=*/N, embeds.data(),
                         &kv_lm, /*all_hidden_out=*/nullptr, &hidden_last)) {
        VV_LOG_ERROR("tts_15b: prefill LM forward failed");
        return -11;
    }
    int lm_pos = N;

    // ---- 6b. negative branch (CFG) ----
    // Build a parallel LM forward over the SAME prompt structure as
    // positive, but with each <|vision_pad|> position embedded as
    // <|image_pad|> instead of as the reference-audio feature. That
    // keeps the negative LM's hidden-state distribution close to
    // positive's (same prompt length, same KV depth at each step) so
    // the diffusion head sees an in-distribution `cond_neg` — just
    // one without the specific voice/text conditioning.
    //
    // (A simpler "single image_pad token" negative — the literal
    // analog of the streaming model's pre-baked neg_tts_lm state —
    // collapses for the 1.5B model: its diffusion head was trained
    // against full-length conditioning hidden states, not the 1-token
    // state the streaming TLM uses. Mixing v predictions from such a
    // low-rank baseline produces noise even at cfg_scale=1.05.)
    //
    // Empirical sweet spot for cfg_scale on the closed-loop word-recall
    // benchmark (1.5B Q8_0, samples/2p_argument 5s ref):
    //   cfg=1.0  -> 100%   (CFG off)
    //   cfg=1.3  -> 88.9%
    //   cfg=2.0  -> 100%
    //   cfg=3.0  -> 100%
    //   cfg>=5   -> noise / [Music] artifact
    const bool use_cfg = (p.cfg_scale > 1.0f);
    ResidentKV         kv_neg;
    std::vector<float> neg_hidden_last;
    int                neg_pos = 0;
    if (use_cfg) {
        if (!kv_neg.init(cfg.n_layers_lm, cfg.head_dim, cfg.n_kv_heads, max_seq)) {
            VV_LOG_ERROR("tts_15b: neg KV init failed");
            return -10;
        }
        // Copy positive embeds, then overwrite vision_pad positions
        // with image_pad embeddings.
        std::vector<float> neg_embeds(embeds);
        std::vector<float> img_pad_embed(hidden);
        embed_row(w.lm_tok_embd, kSpeech15bImgPadId, hidden, img_pad_embed.data());
        for (int k = 0; k < Tc; ++k) {
            const int pos = pad_positions[k];
            std::memcpy(&neg_embeds[static_cast<size_t>(hidden) * pos],
                        img_pad_embed.data(),
                        sizeof(float) * hidden);
        }
        if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                             /*past_len=*/0, /*n_new=*/N, neg_embeds.data(),
                             &kv_neg, /*all_hidden_out=*/nullptr,
                             &neg_hidden_last)) {
            VV_LOG_ERROR("tts_15b: neg prefill failed");
            return -11;
        }
        neg_pos = N;
        if (p.verbose) std::fprintf(stderr,
            "[tts_15b] CFG on, scale=%.2f (neg branch prefilled %d tokens)\n",
            static_cast<double>(p.cfg_scale), N);
    } else if (p.verbose) {
        std::fprintf(stderr, "[tts_15b] CFG off (cfg_scale=%.2f)\n",
                     static_cast<double>(p.cfg_scale));
    }

    // ---- 7. speech-generation loop ----
    DPMSolverConfig solver_cfg;
    solver_cfg.num_train_timesteps = 1000;
    solver_cfg.num_inference_steps = p.n_diffusion_steps > 0 ? p.n_diffusion_steps : 20;
    solver_cfg.solver_order        = 2;
    solver_cfg.lower_order_final   = true;
    DPMSolverState solver_state;
    dpm_solver_init(solver_cfg, &solver_state);

    DiffusionHeadConfig dh_cfg;
    dh_cfg.hidden      = hidden;
    dh_cfg.latent      = cfg.latent;
    dh_cfg.head_layers = cfg.head_layers;
    dh_cfg.ffn_ratio   = cfg.ffn_ratio;
    dh_cfg.eps         = cfg.rms_norm_eps;
    dh_cfg.freq_size   = 256;

    std::mt19937 rng(p.seed ? p.seed : std::random_device{}());
    std::normal_distribution<float> norm(0.0f, 1.0f);

    std::vector<float> all_latents;
    all_latents.reserve(static_cast<size_t>(p.max_speech_frames) * cfg.latent);

    int  total_frames = 0;
    bool finished     = false;

    while (!finished && total_frames < p.max_speech_frames) {
        // 7a. sample one speech latent via DPM-Solver, optionally with CFG.
        std::vector<float> z(cfg.latent);
        for (auto& v : z) v = norm(rng);

        std::vector<float> cond(static_cast<size_t>(hidden));
        std::memcpy(cond.data(), hidden_last.data(), sizeof(float) * hidden);

        std::vector<float> cond_neg;
        if (use_cfg) cond_neg.assign(neg_hidden_last.begin(), neg_hidden_last.end());

        if (dpm_solver_sample(z, cfg.latent, /*frames=*/1, /*batch=*/1,
                              cond, hidden,
                              w.dh, dh_cfg, solver_cfg, solver_state,
                              cond_neg, p.cfg_scale) != 0) {
            VV_LOG_ERROR("tts_15b: dpm_solver_sample failed at frame %d", total_frames);
            return -12;
        }
        all_latents.insert(all_latents.end(), z.begin(), z.end());

        // 7b. project latent -> next-step LM input embedding.
        auto step_embed = run_speech_connector(cfg, w, z.data(), /*batch=*/1);

        // 7c. step LM by 1 position with the speech embedding (positive).
        if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                             lm_pos, /*n_new=*/1, step_embed.data(),
                             &kv_lm, nullptr, &hidden_last)) {
            VV_LOG_ERROR("tts_15b: LM step failed at frame %d", total_frames);
            return -13;
        }
        ++lm_pos;

        // 7c'. negative branch sees the same speech embed.
        if (use_cfg) {
            if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                                 neg_pos, /*n_new=*/1, step_embed.data(),
                                 &kv_neg, nullptr, &neg_hidden_last)) {
                VV_LOG_ERROR("tts_15b: neg LM step failed at frame %d", total_frames);
                return -13;
            }
            ++neg_pos;
        }
        ++total_frames;

        // 7d. speech-end detection from LM logits.
        auto logits = detail::lm_head_logits_last(model->lm_head, hidden_last,
                                                   hidden, cfg.vocab_size);
        if (static_cast<int>(logits.size()) == cfg.vocab_size) {
            int   best_id = 0;
            float best_v  = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < cfg.vocab_size; ++i) {
                if (logits[i] > best_v) { best_v = logits[i]; best_id = i; }
            }
            if (p.verbose && (total_frames % 8 == 0 || best_id == kSpeech15bEndId)) {
                std::fprintf(stderr, "[tts_15b] frame %d: argmax=%d (end=%d)\n",
                             total_frames, best_id, kSpeech15bEndId);
            }
            if (best_id == kSpeech15bEndId) {
                if (p.verbose) std::fprintf(stderr,
                    "[tts_15b] speech_end at frame %d\n", total_frames);
                finished = true;
            }
        }
    }

    // ---- 8. decode latents to waveform ----
    const int n_frames = static_cast<int>(all_latents.size() / cfg.latent);
    if (n_frames > 0) {
        std::vector<float> scaled(all_latents.size());
        for (size_t i = 0; i < all_latents.size(); ++i) {
            scaled[i] = all_latents[i] / cfg.speech_scaling - cfg.speech_bias;
        }
        // Repack from per-frame [latent] to ggml-order [vae_dim, n_frames].
        std::vector<float> packed(scaled.size());
        for (int t = 0; t < n_frames; ++t) {
            for (int d = 0; d < cfg.latent; ++d) {
                packed[d * n_frames + t] = scaled[t * cfg.latent + d];
            }
        }
        *samples = decode_latent_sequence(cfg, w, packed.data(), n_frames);
    }
    if (p.verbose) std::fprintf(stderr,
        "[tts_15b] %d frames -> %zu samples\n", n_frames, samples->size());
    return 0;
}
}  // namespace

}  // namespace vv
