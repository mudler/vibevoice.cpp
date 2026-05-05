#include "vibevoice_asr.hpp"
#include "vibevoice_speech_helpers.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "rms_norm.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <unordered_map>

namespace vv {

namespace {

constexpr int kSpeechCompressRatio = 3200;
constexpr int kImStartId   = 151644;
constexpr int kImEndId     = 151645;
constexpr int kEndOfTextId = 151643;
constexpr int kSpeechStartId = 151646;  // <|object_ref_start|>
constexpr int kSpeechEndId   = 151647;  // <|object_ref_end|>
constexpr int kSpeechPadId   = 151648;  // <|box_start|>

// Build the ASR prompt as a literal string; the byte-level BPE tokenizer's
// special-token matching emits the right IDs for the bracketed markers.
std::string build_prompt(int vae_tok_len, double audio_seconds) {
    std::string speech_marker = "<|object_ref_start|>";
    speech_marker.reserve(speech_marker.size() + vae_tok_len * 13);
    for (int i = 0; i < vae_tok_len; ++i) speech_marker += "<|box_start|>";
    speech_marker += "<|object_ref_end|>";

    char suffix[256];
    std::snprintf(suffix, sizeof(suffix),
                  "\nThis is a %.2f seconds audio, please transcribe it "
                  "with these keys: Start time, End time, Speaker ID, Content",
                  audio_seconds);

    std::ostringstream os;
    os << "<|im_start|>system\n"
       << "You are a helpful assistant that transcribes audio input into text output in JSON format."
       << "<|im_end|>\n"
       << "<|im_start|>user\n"
       << speech_marker << suffix
       << "<|im_end|>\n"
       << "<|im_start|>assistant\n";
    return os.str();
}

// Run an encoder graph on a [T, 1, 1] audio buffer.
//
// The encoder produces a tensor with ggml ne = [T_compr, vae_dim, B], i.e.
// T-fastest memory layout. The connector's `ggml_mul_mat(fc1_w, in)` needs
// `in.ne[0]` to equal `fc1_w.ne[0] = vae_dim`, so we permute to swap the
// first two dims (mirrors upstream `latents.permute(0, 2, 1)`).
//
// `single_shot` runs one ggml graph for the whole buffer; `run_encoder_buf`
// dispatches to chunked processing when the audio is too long for a single
// graph to fit in a sane memory pool. Encoder activations grow ~linearly
// with input length (the conv stack has fixed depth) so the pool size is
// sized per-sample with headroom.
bool run_encoder_single_shot(const EncoderWeights& w, const AcousticConfig& cfg,
                             const std::vector<float>& audio,
                             std::vector<float>* latents,
                             int* T_compressed) {
    const int T = static_cast<int>(audio.size());
    // Backend-aware compute: build graph in a no_alloc ctx, allocate the
    // input tensor on the active backend's buffer (intermediates handled
    // by gallocr inside vv::compute_graph), then upload audio + compute
    // + read back via backend tensor APIs.
    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 32768
               + ggml_graph_overhead_custom(32768, false);
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return false;
    struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, 1, 1);
    ggml_set_name(x, "audio_in");
    struct ggml_tensor* y_raw = encoder_forward(ctx, x, w, cfg);
    // Permute (T_compr, vae_dim, B) -> (vae_dim, T_compr, B). Mirrors
    // upstream `latents.permute(0, 2, 1)`. Required because the connector's
    // ggml_mul_mat needs ne[0] = vae_dim.
    struct ggml_tensor* y = ggml_cont(ctx, ggml_permute(ctx, y_raw, 1, 0, 2, 3));
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(x, audio.data(), 0, sizeof(float) * T);

    if (!vv::compute_graph(gf)) {
        ggml_backend_buffer_free(in_buf);
        ggml_free(ctx);
        return false;
    }
    const int latent_dim = static_cast<int>(y->ne[0]);
    *T_compressed        = static_cast<int>(y->ne[1]);
    latents->assign(static_cast<size_t>(latent_dim) * (*T_compressed), 0.0f);
    ggml_backend_tensor_get(y, latents->data(), 0, sizeof(float) * latents->size());
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return true;
}

// One streaming chunk: builds the encoder graph against `cache`, runs it,
// extracts the per-conv "next view" data into the cache for the next call.
bool run_encoder_chunk_streaming(const EncoderWeights& w, const AcousticConfig& cfg,
                                  const std::vector<float>& audio,
                                  StreamingCache&  cache,
                                  std::vector<float>* latents,
                                  int* T_compressed) {
    const int T = static_cast<int>(audio.size());
    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 32768
               + ggml_graph_overhead_custom(32768, false);
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return false;
    struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, 1, 1);
    ggml_set_name(x, "audio_chunk_in");
    struct ggml_tensor* y_raw = encoder_forward_streaming(ctx, x, w, cfg, cache);
    struct ggml_tensor* y = ggml_cont(ctx, ggml_permute(ctx, y_raw, 1, 0, 2, 3));
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
    ggml_build_forward_expand(gf, y);
    // Make the per-conv "last context" views part of the forward graph so
    // ggml computes them and their memory lives until we read it back.
    for (auto& kv : cache) {
        if (kv.second.next_view) ggml_build_forward_expand(gf, kv.second.next_view);
    }

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(x, audio.data(), 0, sizeof(float) * T);
    // Populate per-conv cache prefixes — zeros on the first chunk, the
    // previous chunk's tail thereafter. .data was null until the alloc
    // above, so we couldn't memcpy them inside sconv1d_causal_streaming.
    for (auto& kv : cache) {
        StreamingCacheEntry& e = kv.second;
        if (!e.prefix || e.T == 0) continue;
        const size_t need = static_cast<size_t>(e.T) * e.C;
        if (cache.is_first_chunk || e.data.size() != need) {
            std::vector<float> zeros(need, 0.0f);
            ggml_backend_tensor_set(e.prefix, zeros.data(), 0, sizeof(float) * need);
        } else {
            ggml_backend_tensor_set(e.prefix, e.data.data(), 0, sizeof(float) * need);
        }
    }

    if (!vv::compute_graph(gf)) {
        ggml_backend_buffer_free(in_buf);
        ggml_free(ctx); return false;
    }
    const int latent_dim = static_cast<int>(y->ne[0]);
    *T_compressed        = static_cast<int>(y->ne[1]);
    latents->assign(static_cast<size_t>(latent_dim) * (*T_compressed), 0.0f);
    ggml_backend_tensor_get(y, latents->data(), 0, sizeof(float) * latents->size());
    // Pull each conv's last-context view back into the cache entry's
    // flat CPU buffer for the next chunk to prepend.
    for (auto& kv : cache) {
        StreamingCacheEntry& e = kv.second;
        if (!e.next_view || e.T == 0) continue;
        const size_t n = static_cast<size_t>(e.T) * e.C;
        e.data.assign(n, 0.0f);
        ggml_backend_tensor_get(e.next_view, e.data.data(), 0, sizeof(float) * n);
        e.next_view = nullptr;
        e.prefix    = nullptr;
    }
    cache.is_first_chunk = false;
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return true;
}

// Long-form encode. Splits the input audio into ≤kSegmentSamples chunks
// aligned to the kSpeechCompressRatio (3200 samples = one output frame),
// runs them through encoder_forward_streaming with a shared StreamingCache
// so each chunk's conv1d layers see the previous chunk's tail as left
// context. Concatenates the per-chunk latents along the time axis. The
// streaming cache makes chunked output bit-exact with single-shot — see
// VibeVoiceTokenizerStreamingCache in upstream
// vibevoice/modular/modular_vibevoice_tokenizer.py.
bool run_encoder_buf(const EncoderWeights& w, const AcousticConfig& cfg,
                     const std::vector<float>& audio,
                     std::vector<float>* latents,
                     int* T_compressed) {
    // Chunk size depends on the active backend:
    //   * CPU:  10 s (240 k samples) — bounded by per-chunk pool memory.
    //   * CUDA:  2 s ( 48 k samples) — bounded by ggml-cuda's IM2COL kernel,
    //            which encodes the stem conv's output time as gridDim.y
    //            (capped at 65535). At 24 kHz, 2.73 s already maxes that
    //            out, so 2 s gives a small safety margin.
    // The streaming cache (entry.prefix + entry.next_view) makes
    // chunk-vs-single-shot output bit-exact regardless of chunk size.
    constexpr int kSampleRate = 24000;
    const ggml_backend_t b = vv::backend();
    const bool is_cuda = b && std::string(ggml_backend_name(b)).find("CUDA") != std::string::npos;
    const int kSecondsPerSegment = is_cuda ? 2 : 10;
    const int kSegmentSamples    = kSecondsPerSegment * kSampleRate;

    const int T = static_cast<int>(audio.size());
    if (T <= kSegmentSamples) {
        return run_encoder_single_shot(w, cfg, audio, latents, T_compressed);
    }

    // Chunk on multiples of the compression ratio so each chunk produces a
    // whole number of latent frames and the concatenation lines up.
    const int chunk_samples = (kSegmentSamples / kSpeechCompressRatio)
                              * kSpeechCompressRatio;

    StreamingCache cache;
    cache.is_first_chunk = true;

    std::vector<float> all;
    int total_frames = 0;
    int latent_dim = 0;

    for (int off = 0; off < T; off += chunk_samples) {
        const int end = std::min(off + chunk_samples, T);
        cache.is_final_chunk = (end == T);
        std::vector<float> seg(audio.begin() + off, audio.begin() + end);
        std::vector<float> seg_lat;
        int seg_T = 0;
        if (!run_encoder_chunk_streaming(w, cfg, seg, cache, &seg_lat, &seg_T))
            return false;
        if (seg_T == 0) continue;
        if (latent_dim == 0)
            latent_dim = static_cast<int>(seg_lat.size() / seg_T);
        all.insert(all.end(), seg_lat.begin(), seg_lat.end());
        total_frames += seg_T;
    }

    if (latent_dim == 0 || total_frames == 0) return false;
    *latents       = std::move(all);
    *T_compressed  = total_frames;
    return true;
}

// Run SpeechConnector: y = fc2(rmsnorm(fc1(x))).
//   x: [latent_dim, T_compressed], output: [hidden, T_compressed]
std::vector<float> run_connector(struct ggml_tensor* fc1_w,
                                 struct ggml_tensor* fc1_b,
                                 struct ggml_tensor* norm_w,
                                 struct ggml_tensor* fc2_w,
                                 struct ggml_tensor* fc2_b,
                                 const std::vector<float>& x,
                                 int latent_dim, int T, int hidden) {
    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 256
               + ggml_graph_overhead();
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);

    struct ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent_dim, T);
    ggml_set_name(in, "connector_in");
    struct ggml_tensor* h = ggml_mul_mat(ctx, fc1_w, in);
    if (fc1_b) h = ggml_add(ctx, h, fc1_b);
    h = ggml_rms_norm(ctx, h, 1e-6f);
    if (norm_w) h = ggml_mul(ctx, h, norm_w);
    h = ggml_mul_mat(ctx, fc2_w, h);
    if (fc2_b) h = ggml_add(ctx, h, fc2_b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, h);

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return {}; }
    ggml_backend_tensor_set(in, x.data(), 0, sizeof(float) * x.size());

    std::vector<float> out;
    if (vv::compute_graph(gf)) {
        out.assign(static_cast<size_t>(hidden) * T, 0.0f);
        ggml_backend_tensor_get(h, out.data(), 0, sizeof(float) * out.size());
    }
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return out;
}

// Compute lm_head logits for the LAST token only.
std::vector<float> lm_head_logits_last(struct ggml_tensor* lm_head_w,
                                       const std::vector<float>& hidden_last,
                                       int hidden, int vocab) {
    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);

    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
    ggml_set_name(x, "lm_head_in");
    struct ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_w, x);
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return {}; }
    ggml_backend_tensor_set(x, hidden_last.data(), 0, sizeof(float) * hidden);

    std::vector<float> out;
    if (vv::compute_graph(gf)) {
        out.assign(vocab, 0.0f);
        ggml_backend_tensor_get(logits, out.data(), 0, sizeof(float) * vocab);
    }
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return out;
}

}  // namespace

namespace detail {

// Thin wrappers exposing the file-local helpers above so the 1.5B TTS
// path (in vibevoice_tts.cpp) can reuse them without copy-pasting the
// chunking / streaming-cache logic.
bool run_encoder_buf(const EncoderWeights& w, const AcousticConfig& cfg,
                     const std::vector<float>& audio,
                     std::vector<float>* latents,
                     int* T_compressed) {
    return ::vv::run_encoder_buf(w, cfg, audio, latents, T_compressed);
}
std::vector<float> run_connector(struct ggml_tensor* fc1_w,
                                 struct ggml_tensor* fc1_b,
                                 struct ggml_tensor* norm_w,
                                 struct ggml_tensor* fc2_w,
                                 struct ggml_tensor* fc2_b,
                                 const std::vector<float>& x,
                                 int latent_dim, int T, int hidden) {
    return ::vv::run_connector(fc1_w, fc1_b, norm_w, fc2_w, fc2_b,
                               x, latent_dim, T, hidden);
}
std::vector<float> lm_head_logits_last(struct ggml_tensor* lm_head_w,
                                       const std::vector<float>& hidden_last,
                                       int hidden, int vocab) {
    return ::vv::lm_head_logits_last(lm_head_w, hidden_last, hidden, vocab);
}

}  // namespace detail

int vibevoice_asr_transcribe(VibeVoiceModel*           model,
                             const std::vector<float>& audio,
                             const VibeVoiceASRParams& p,
                             std::string*              transcript) {
    if (!model || !transcript) return -1;
    if (model->variant != "asr-7b") {
        VV_LOG_ERROR("asr_transcribe: model is variant=%s, expected asr-7b",
                     model->variant.c_str());
        return -2;
    }
    if (!model->tokenizer.vocab_size()) {
        VV_LOG_ERROR("asr_transcribe: tokenizer not loaded");
        return -3;
    }
    if (audio.empty()) return -4;

    const auto& cfg = model->cfg;
    const auto& w   = model->w;
    const int   hidden = cfg.hidden;

    // ---- 0. RMS-normalize to -25 dBFS + avoid clipping (mirrors upstream AudioNormalizer) ----
    std::vector<float> audio_norm = audio;
    {
        const float target_dB_FS = -25.0f;
        const float eps          = 1e-6f;
        double sq = 0.0;
        for (float v : audio_norm) sq += static_cast<double>(v) * v;
        const float rms = static_cast<float>(std::sqrt(sq / std::max<size_t>(audio_norm.size(), 1)));
        const float target_lin = std::pow(10.0f, target_dB_FS / 20.0f);
        const float scalar = target_lin / (rms + eps);
        for (auto& v : audio_norm) v *= scalar;
        // avoid_clipping: if any |v|>1, divide by (max_abs + eps)
        float maxabs = 0.0f;
        for (float v : audio_norm) maxabs = std::max(maxabs, std::fabs(v));
        if (maxabs > 1.0f) {
            const float clip_div = maxabs + eps;
            for (auto& v : audio_norm) v /= clip_div;
        }
        if (p.verbose) {
            std::fprintf(stderr, "[asr] rms-normalize: in_rms=%.4f scalar=%.4f maxabs=%.4f -> target=%.4f\n",
                         static_cast<double>(rms), static_cast<double>(scalar),
                         static_cast<double>(maxabs), static_cast<double>(target_lin));
        }
    }

    // ---- 1. encoders ----
    std::vector<float> ac_latents, sm_latents;
    int Tc_a = 0, Tc_s = 0;
    if (!run_encoder_buf(model->at_enc, cfg.acoustic, audio_norm, &ac_latents, &Tc_a))
        return -5;
    if (!run_encoder_buf(model->st_enc, model->semantic_cfg, audio_norm, &sm_latents, &Tc_s))
        return -6;
    if (Tc_a != Tc_s) {
        VV_LOG_ERROR("asr: encoder frame mismatch %d vs %d", Tc_a, Tc_s);
        return -7;
    }
    const int Tc = Tc_a;
    if (p.verbose) {
        // Quick stats so we can see latent magnitudes / distribution.
        auto stats = [](const std::vector<float>& v) {
            double sum = 0, sq = 0;
            for (float x : v) { sum += x; sq += x*x; }
            const double mean = sum / std::max<size_t>(v.size(), 1);
            const double std  = std::sqrt(sq / v.size() - mean * mean);
            return std::pair<double, double>{mean, std};
        };
        auto [a_mu, a_std] = stats(ac_latents);
        auto [s_mu, s_std] = stats(sm_latents);
        std::fprintf(stderr,
                     "[asr] %d frames  acoustic[mu=%.3f std=%.3f]  "
                     "semantic[mu=%.3f std=%.3f]\n",
                     Tc, a_mu, a_std, s_mu, s_std);
    }

    // NOTE: mlx-audio's ASR pipeline does NOT add posterior noise — it just
    // uses the encoder mean directly. The official PyTorch path samples with
    // `dist_type='gaussian'` (mean + std/0.8 * randn(B,) * randn_like(mean))
    // but adding inference-time noise at scale ~0.5 onto features that are
    // mostly mean-driven made our outputs unstable. Match mlx-audio: skip
    // sampling entirely. (semantic also uses mean — std_dist_type='none'.)

    // ---- 3. connectors ----
    auto ac_embed = run_connector(w.ac_fc1_w, w.ac_fc1_b, w.ac_norm,
                                  w.ac_fc2_w, w.ac_fc2_b,
                                  ac_latents, cfg.vae_dim, Tc, hidden);
    auto sm_embed = run_connector(model->sc_fc1_w, model->sc_fc1_b, model->sc_norm,
                                  model->sc_fc2_w, model->sc_fc2_b,
                                  sm_latents, model->semantic_vae_dim, Tc, hidden);

    // ---- 4. sum into speech_features[Tc, hidden] ----
    std::vector<float> speech_features(static_cast<size_t>(hidden) * Tc);
    for (size_t i = 0; i < speech_features.size(); ++i)
        speech_features[i] = ac_embed[i] + sm_embed[i];

    if (p.verbose) {
        auto stats = [](const std::vector<float>& v) {
            double sum = 0, sq = 0;
            for (float x : v) { sum += x; sq += x*x; }
            const double m = sum / std::max<size_t>(v.size(), 1);
            return std::pair<double,double>{m, std::sqrt(sq / v.size() - m*m)};
        };
        auto [a_mu, a_std] = stats(ac_embed);
        auto [s_mu, s_std] = stats(sm_embed);
        auto [c_mu, c_std] = stats(speech_features);
        std::fprintf(stderr,
                     "[asr] connector ac[mu=%.3f std=%.3f]  sm[mu=%.3f std=%.3f]  "
                     "sum[mu=%.3f std=%.3f]\n",
                     a_mu, a_std, s_mu, s_std, c_mu, c_std);
    }

    // ---- 5. build & tokenize prompt ----
    const double audio_seconds = static_cast<double>(audio.size()) / cfg.sample_rate;
    std::string prompt = build_prompt(Tc, audio_seconds);
    auto input_ids = model->tokenizer.encode(prompt);
    if (input_ids.empty()) return -8;
    const int N = static_cast<int>(input_ids.size());

    // Find the speech_pad positions (must be exactly Tc of them).
    std::vector<int> pad_positions;
    pad_positions.reserve(Tc);
    for (int i = 0; i < N; ++i) if (input_ids[i] == kSpeechPadId) pad_positions.push_back(i);
    if (static_cast<int>(pad_positions.size()) != Tc) {
        VV_LOG_ERROR("asr: pad count %zu != %d compressed frames",
                     pad_positions.size(), Tc);
        return -9;
    }

    // ---- 6. embed input_ids and splice speech features ----
    // lm_tok_embd lives on the active backend's buffer; read each row
    // via ggml_backend_tensor_get (memcpy on CPU, DtoH on GPU).
    std::vector<float> text_embeds(static_cast<size_t>(hidden) * N);
    const size_t row_bytes_embd = (w.lm_tok_embd->type == GGML_TYPE_F32)
                                  ? sizeof(float) * hidden
                                  : sizeof(ggml_fp16_t) * hidden;
    std::vector<uint8_t> row_buf(row_bytes_embd);
    auto fill_one = [&](int pos, int id) {
        ggml_backend_tensor_get(w.lm_tok_embd, row_buf.data(),
                                row_bytes_embd * static_cast<size_t>(id),
                                row_bytes_embd);
        if (w.lm_tok_embd->type == GGML_TYPE_F32) {
            std::memcpy(&text_embeds[hidden * pos], row_buf.data(), row_bytes_embd);
        } else if (w.lm_tok_embd->type == GGML_TYPE_F16) {
            const ggml_fp16_t* src = reinterpret_cast<const ggml_fp16_t*>(row_buf.data());
            for (int i = 0; i < hidden; ++i)
                text_embeds[hidden * pos + i] = ggml_fp16_to_fp32(src[i]);
        }
    };
    for (int i = 0; i < N; ++i) fill_one(i, input_ids[i]);
    // Overwrite at speech_pad positions with the connector outputs.
    for (int k = 0; k < Tc; ++k) {
        std::memcpy(&text_embeds[hidden * pad_positions[k]],
                    &speech_features[hidden * k],
                    sizeof(float) * hidden);
    }

    // ---- 7. LM prefill ----
    // Two memory optimizations layered together (see docs/long-form-asr.md):
    //   * Flash-attention path in qwen2_layer_forward (when the active
    //     backend supports it). FA never materializes the [seq_kv, seq_q,
    //     n_h] scores tensor, so peak activation memory becomes O(seq*hd)
    //     per layer instead of O(seq^2 * n_h) - 6 GB FP32 per layer at
    //     N=7500 in eager mode.
    //   * Chunked prefill: process the prompt+audio prefix in batches of
    //     kPrefillBatch tokens, accumulating each layer's K/V cache as we
    //     go. This bounds the K/V tensor staging memory to O(K * past)
    //     and keeps the activation pool size constant across chunks.
    // Both fall back gracefully: FA -> eager (set VIBEVOICE_FLASH_ATTN=0
    // or just run on a backend that doesn't advertise the op), chunked ->
    // single-shot (set VIBEVOICE_PREFILL_BATCH >= N).
    // Resident K/V cache lives on the active backend's buffer for the
    // whole transcribe() call - prefill and decode both write into it
    // via ggml_cpy, no host round-trip. Sized to fit the prefix plus the
    // entire decode budget. See tests/test_qwen2_resident.cpp for the
    // parity proof (one-graph 28-layer is bit-exact between classic
    // and resident on eager, and within fp16 noise on FA).
    ResidentKV kv_lm;
    {
        const int max_seq_total = N + p.max_new_tokens + 32;
        if (!kv_lm.init(cfg.n_layers_lm, cfg.head_dim, cfg.n_kv_heads,
                              max_seq_total)) {
            VV_LOG_ERROR("asr: failed to allocate resident K/V cache "
                         "(layers=%d max_seq=%d)",
                         cfg.n_layers_lm, max_seq_total);
            return -10;
        }
    }
    std::vector<float> hidden_last(hidden);

    {
        Qwen2Hparams hp;
        hp.hidden_size    = hidden;
        hp.n_heads        = cfg.n_heads;
        hp.n_kv_heads     = cfg.n_kv_heads;
        hp.head_dim       = cfg.head_dim;
        hp.rope_theta     = cfg.rope_theta;
        hp.rms_norm_eps   = cfg.rms_norm_eps;
        hp.use_flash_attn = vv::backend_supports_flash_attn();

        const char* env_batch = std::getenv("VIBEVOICE_PREFILL_BATCH");
        int kPrefillBatch = (env_batch && std::atoi(env_batch) > 0)
                            ? std::atoi(env_batch) : 512;
        if (kPrefillBatch > N) kPrefillBatch = N;

        if (p.verbose) {
            std::fprintf(stderr,
                         "[asr] prefill: N=%d batch=%d flash_attn=%s "
                         "(resident KV)\n",
                         N, kPrefillBatch, hp.use_flash_attn ? "yes" : "no");
        }

        for (int chunk_start = 0; chunk_start < N; chunk_start += kPrefillBatch) {
            const int chunk_K = std::min(kPrefillBatch, N - chunk_start);
            const int kv_new  = chunk_start + chunk_K;

            struct ggml_init_params ip {};
            ip.mem_size = ggml_tensor_overhead() * 32768
                        + ggml_graph_overhead_custom(32768, false);
            ip.no_alloc = true;
            struct ggml_context* ctx = ggml_init(ip);

            struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                       hidden, chunk_K, 1);
            ggml_set_name(x, "prefill_x");
            struct ggml_tensor* posv = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, chunk_K);
            ggml_set_name(posv, "prefill_pos");
            struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
                                                          kv_new, chunk_K);
            ggml_set_name(mask, "prefill_mask");

            kv_lm.past_len = chunk_start;

            struct ggml_tensor* h = x;
            std::vector<struct ggml_tensor*> k_writes(w.lm_layers.size()),
                                             v_writes(w.lm_layers.size());
            for (size_t li = 0; li < w.lm_layers.size(); ++li) {
                auto out = qwen2_layer_forward_resident(ctx, h, posv, mask,
                                                        kv_lm,
                                                        static_cast<int>(li),
                                                        w.lm_layers[li], hp);
                h = out.y;
                k_writes[li] = out.k_write;
                v_writes[li] = out.v_write;
            }
            if (w.tlm_output_norm) {
                h = rms_norm(ctx, h, w.tlm_output_norm, cfg.rms_norm_eps);
            }
            struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
            // Interleave k/v cpys per layer (NOT all k_writes then all
            // v_writes). Expanding k_writes[li+1] transitively pulls in
            // layer-li's attention chain, which reads v_full_li - so
            // cpy_v_li must already be in the graph by then or the read
            // gets stale V. Verified by tests/test_qwen2_resident
            // chain28 one-graph case: bit-exact 0.000e+00 with
            // interleaving, ~28 max_abs (catastrophic) without.
            for (size_t li = 0; li < w.lm_layers.size(); ++li) {
                ggml_build_forward_expand(gf, k_writes[li]);
                ggml_build_forward_expand(gf, v_writes[li]);
            }
            ggml_build_forward_expand(gf, h);

            ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
            if (!in_buf) { ggml_free(ctx); return -10; }

            ggml_backend_tensor_set(x,
                text_embeds.data() + static_cast<size_t>(hidden) * chunk_start,
                0, sizeof(float) * hidden * chunk_K);

            std::vector<int32_t> pos_v(chunk_K);
            for (int i = 0; i < chunk_K; ++i) pos_v[i] = chunk_start + i;
            ggml_backend_tensor_set(posv, pos_v.data(), 0,
                                    sizeof(int32_t) * chunk_K);

            std::vector<ggml_fp16_t> mask_v(static_cast<size_t>(kv_new) * chunk_K);
            const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);
            for (int q = 0; q < chunk_K; ++q) {
                const int q_abs = chunk_start + q;
                for (int k = 0; k < kv_new; ++k) {
                    mask_v[static_cast<size_t>(q) * kv_new + k] =
                        (k > q_abs) ? f16_ninf : f16_zero;
                }
            }
            ggml_backend_tensor_set(mask, mask_v.data(), 0,
                                    sizeof(ggml_fp16_t) * mask_v.size());

            if (!vv::compute_graph(gf)) {
                ggml_backend_buffer_free(in_buf);
                ggml_free(ctx);
                return -10;
            }

            if (chunk_start + chunk_K == N) {
                ggml_backend_tensor_get(h, hidden_last.data(),
                    sizeof(float) * hidden * (chunk_K - 1),
                    sizeof(float) * hidden);
            }

            kv_lm.past_len = kv_new;

            ggml_backend_buffer_free(in_buf);
            ggml_free(ctx);

            if (p.verbose) {
                std::fprintf(stderr, "[asr] prefill chunk %d/%d: tokens [%d,%d)\n",
                             (chunk_start / kPrefillBatch) + 1,
                             (N + kPrefillBatch - 1) / kPrefillBatch,
                             chunk_start, kv_new);
            }
        }

    }
    if (p.verbose) std::fprintf(stderr, "[asr] prefill %d tokens done\n", N);

    // ---- 8. greedy decode ----
    std::vector<int> generated;
    int pos = N;

    // Per-token penalty for previously generated ids (incl. prompt).
    std::unordered_map<int, int> seen_count;
    for (int id : input_ids) ++seen_count[id];

    for (int step = 0; step < p.max_new_tokens; ++step) {
        auto logits = lm_head_logits_last(model->lm_head, hidden_last, hidden, cfg.vocab_size);

        // Repetition penalty (HF GenerationConfig style):
        //   if logit > 0: logit /= penalty
        //   else:         logit *= penalty
        if (p.repetition_penalty > 1.0f) {
            for (const auto& kv : seen_count) {
                if (kv.first < 0 || kv.first >= cfg.vocab_size) continue;
                float& l = logits[kv.first];
                l = l > 0 ? l / p.repetition_penalty : l * p.repetition_penalty;
            }
        }

        // No-repeat-ngram: ban any id that would close an n-gram already seen.
        if (p.no_repeat_ngram > 1 && static_cast<int>(generated.size()) >= p.no_repeat_ngram - 1) {
            const int n = p.no_repeat_ngram;
            // The (n-1)-token suffix we're about to extend.
            std::vector<int> suffix(generated.end() - (n - 1), generated.end());
            // Search through the prompt + generated tokens for any prior occurrence
            // of `suffix` and ban the token that followed it.
            std::vector<int> all = input_ids;
            all.insert(all.end(), generated.begin(), generated.end());
            for (int s = 0; s + n <= static_cast<int>(all.size()); ++s) {
                bool match = true;
                for (int k = 0; k < n - 1; ++k) {
                    if (all[s + k] != suffix[k]) { match = false; break; }
                }
                if (match) {
                    int banned = all[s + n - 1];
                    if (banned >= 0 && banned < cfg.vocab_size) {
                        logits[banned] = -1e30f;
                    }
                }
            }
        }

        int best_id = 0;
        float best = logits[0];
        for (int v = 1; v < cfg.vocab_size; ++v) {
            if (logits[v] > best) { best = logits[v]; best_id = v; }
        }
        if (p.verbose && (step % 16 == 0 || best_id == kImEndId || best_id == kEndOfTextId))
            std::fprintf(stderr, "[asr] step %d: id=%d (logit=%.2f)\n",
                         step, best_id, static_cast<double>(best));

        if (best_id == kImEndId || best_id == kEndOfTextId) break;
        generated.push_back(best_id);
        ++seen_count[best_id];

        // Embed via the lm_tok_embd row. lm_tok_embd lives on the active
        // backend's buffer, so read it via ggml_backend_tensor_get into a
        // CPU staging buffer (on CPU backend that's a memcpy; on GPU it's
        // a DtoH transfer).
        std::vector<float> emb(hidden);
        const size_t row_bytes = (w.lm_tok_embd->type == GGML_TYPE_F32)
                               ? sizeof(float) * hidden
                               : sizeof(ggml_fp16_t) * hidden;
        std::vector<uint8_t> row(row_bytes);
        ggml_backend_tensor_get(w.lm_tok_embd, row.data(),
                                row_bytes * static_cast<size_t>(best_id),
                                row_bytes);
        if (w.lm_tok_embd->type == GGML_TYPE_F32) {
            std::memcpy(emb.data(), row.data(), row_bytes);
        } else {
            const ggml_fp16_t* src = reinterpret_cast<const ggml_fp16_t*>(row.data());
            for (int i = 0; i < hidden; ++i) emb[i] = ggml_fp16_to_fp32(src[i]);
        }

        // Single-token decode against the resident K/V cache. Same
        // pattern as the prefill chunk loop, just with chunk_K=1.
        Qwen2Hparams hp;
        hp.hidden_size    = hidden;
        hp.n_heads        = cfg.n_heads;
        hp.n_kv_heads     = cfg.n_kv_heads;
        hp.head_dim       = cfg.head_dim;
        hp.rope_theta     = cfg.rope_theta;
        hp.rms_norm_eps   = cfg.rms_norm_eps;
        hp.use_flash_attn = vv::backend_supports_flash_attn();

        struct ggml_init_params ip {};
        ip.mem_size = ggml_tensor_overhead() * 32768
                    + ggml_graph_overhead_custom(32768, false);
        ip.no_alloc = true;
        struct ggml_context* ctx = ggml_init(ip);

        const int kv_new = kv_lm.past_len + 1;

        struct ggml_tensor* x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, 1, 1);
        struct ggml_tensor* posv = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_new, 1);

        struct ggml_tensor* h = x;
        std::vector<struct ggml_tensor*> k_writes(w.lm_layers.size()),
                                         v_writes(w.lm_layers.size());
        for (size_t li = 0; li < w.lm_layers.size(); ++li) {
            auto out = qwen2_layer_forward_resident(ctx, h, posv, mask,
                                                    kv_lm, static_cast<int>(li),
                                                    w.lm_layers[li], hp);
            h = out.y;
            k_writes[li] = out.k_write;
            v_writes[li] = out.v_write;
        }
        if (w.tlm_output_norm) {
            h = rms_norm(ctx, h, w.tlm_output_norm, cfg.rms_norm_eps);
        }
        struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
        // Interleave k/v cpys per layer - see the prefill loop comment for
        // the rationale.
        for (size_t li = 0; li < w.lm_layers.size(); ++li) {
            ggml_build_forward_expand(gf, k_writes[li]);
            ggml_build_forward_expand(gf, v_writes[li]);
        }
        ggml_build_forward_expand(gf, h);

        ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
        if (!in_buf) { ggml_free(ctx); return -11; }
        ggml_backend_tensor_set(x, emb.data(), 0, sizeof(float) * hidden);
        const int32_t pos_v = pos;
        ggml_backend_tensor_set(posv, &pos_v, 0, sizeof(int32_t));
        std::vector<ggml_fp16_t> mask_v(kv_new);
        const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int j = 0; j < kv_new; ++j) mask_v[j] = (j > pos) ? f16_ninf : f16_zero;
        ggml_backend_tensor_set(mask, mask_v.data(), 0, sizeof(ggml_fp16_t) * kv_new);

        if (!vv::compute_graph(gf)) {
            ggml_backend_buffer_free(in_buf); ggml_free(ctx); return -11;
        }

        ggml_backend_tensor_get(h, hidden_last.data(), 0, sizeof(float) * hidden);
        kv_lm.past_len = kv_new;

        ggml_backend_buffer_free(in_buf);
        ggml_free(ctx);
        ++pos;
    }

    *transcript = model->tokenizer.decode(generated);
    return 0;
}

}  // namespace vv
