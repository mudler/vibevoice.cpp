// Smoke test that the gguf produced by convert_vibevoice_to_gguf.py
// for `microsoft/VibeVoice-1.5B` loads cleanly and exposes the tensor
// families that the upcoming 1.5B TTS path will consume.
//
// Skips with rc=77 when VIBEVOICE_TTS_15B_MODEL is missing.

#include "model_loader.hpp"
#include "vibevoice_tts.hpp"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool file_ok(const char* p) {
    if (!p || !*p) return false;
    FILE* f = std::fopen(p, "rb");
    if (!f) return false;
    std::fclose(f); return true;
}

}  // namespace

int main() {
    const char* path = std::getenv("VIBEVOICE_TTS_15B_MODEL");
    if (!file_ok(path)) {
        std::fprintf(stderr,
            "skip: set VIBEVOICE_TTS_15B_MODEL to a 1.5B gguf "
            "(produced by scripts/convert_vibevoice_to_gguf.py from "
            "microsoft/VibeVoice-1.5B safetensors).\n");
        return 77;
    }

    vv::ModelLoader m;
    if (!m.load(path)) {
        std::fprintf(stderr, "FAIL: load %s\n", path);
        return 1;
    }

    const std::string variant = m.get_str("vibevoice.variant");
    const int n_lm  = m.get_i32("vibevoice.n_layers_lm",  -1);
    const int n_tlm = m.get_i32("vibevoice.n_layers_tlm", -1);
    const int hidden = m.get_i32("vibevoice.hidden", -1);
    const int n_h    = m.get_i32("vibevoice.n_heads", -1);
    const int n_kv_h = m.get_i32("vibevoice.n_kv_heads", -1);
    const int hd     = m.get_i32("vibevoice.head_dim", -1);
    const int latent = m.get_i32("vibevoice.diffusion.latent", -1);
    const int head_layers = m.get_i32("vibevoice.diffusion.head_layers", -1);

    std::printf("variant=%s hidden=%d heads=%d/%d hd=%d lm=%d+%d latent=%d head_layers=%d\n",
                variant.c_str(), hidden, n_h, n_kv_h, hd, n_lm, n_tlm, latent, head_layers);

    if (variant != "1.5b") {
        std::fprintf(stderr, "FAIL: variant=%s, want 1.5b\n", variant.c_str());
        return 2;
    }
    if (hidden != 1536 || n_lm != 28 || n_tlm != 0 ||
        n_h != 12 || n_kv_h != 2 || hd != 128 || latent != 64 || head_layers != 4) {
        std::fprintf(stderr,
            "FAIL: 1.5b dims wrong; expected hidden=1536 lm=28+0 heads=12/2 hd=128 "
            "latent=64 head_layers=4\n");
        return 3;
    }

    std::vector<std::string> required = {
        // LM stack (single-LM, 28 layers; no tlm.*)
        "lm.tok_embd.weight",
        "lm.output_norm.weight",
        "lm_head.weight",  // synthesised from tied embeddings

        // Speech scaling
        "speech.scaling",
        "speech.bias",

        // Acoustic + semantic connectors
        "ac.fc1.weight",  "ac.fc1.bias",
        "ac.norm.weight",
        "ac.fc2.weight",  "ac.fc2.bias",
        "sc.fc1.weight",  "sc.fc1.bias",
        "sc.norm.weight",
        "sc.fc2.weight",  "sc.fc2.bias",

        // Diffusion head
        "dh.noisy_proj", "dh.cond_proj",
        "dh.t_embed_lin1", "dh.t_embed_lin2",
        "dh.final.proj",   "dh.final.adaln",

        // Acoustic decoder
        "at.dec.stem.weight", "at.dec.stem.bias",
        "at.dec.head.weight", "at.dec.head.bias",
    };

    // Per-layer Qwen2 attn + ffn weights
    for (int i = 0; i < n_lm; ++i) {
        char b[64]; std::snprintf(b, sizeof(b), "lm.blk.%d.", i);
        std::string p(b);
        for (const char* s : {"attn_q.weight","attn_q.bias",
                              "attn_k.weight","attn_k.bias",
                              "attn_v.weight","attn_v.bias",
                              "attn_o.weight",
                              "attn_norm.weight","ffn_norm.weight",
                              "ffn_gate.weight","ffn_up.weight","ffn_down.weight"}) {
            required.push_back(p + s);
        }
    }
    for (int i = 0; i < head_layers; ++i) {
        char b[40]; std::snprintf(b, sizeof(b), "dh.layer_%d.", i);
        std::string pp(b);
        for (const char* s : {"norm","ffn_gate","ffn_up","ffn_down","adaln"}) {
            required.push_back(pp + s);
        }
    }
    for (int i = 1; i <= 6; ++i) {
        char b[40]; std::snprintf(b, sizeof(b), "at.dec.up_%d.", i);
        required.push_back(std::string(b) + "weight");
        required.push_back(std::string(b) + "bias");
    }

    // Encoder side: 1.5B has both at.enc and st.enc (used for voice prep).
    // Spot-check stem + head; the per-stage blocks are exercised by the
    // existing ASR encoder tests.
    for (const char* s : {"at.enc.stem.weight", "at.enc.stem.bias",
                          "at.enc.head.weight", "at.enc.head.bias",
                          "st.enc.stem.weight", "st.enc.stem.bias",
                          "st.enc.head.weight", "st.enc.head.bias"}) {
        required.push_back(s);
    }

    // Tensors that MUST NOT be present in 1.5B (they're realtime-only).
    const std::vector<std::string> forbidden = {
        "tlm.tok_embd.weight",
        "tlm.output_norm.weight",
        "tts.input_types.weight",
        "eos.fc1.weight",
        "eos.fc2.weight",
    };

    int missing = 0;
    for (const auto& n : required) {
        if (!m.has(n)) {
            if (missing < 20) std::fprintf(stderr, "MISSING: %s\n", n.c_str());
            ++missing;
        }
    }
    int unexpected = 0;
    for (const auto& n : forbidden) {
        if (m.has(n)) {
            std::fprintf(stderr, "UNEXPECTED: %s\n", n.c_str());
            ++unexpected;
        }
    }
    std::printf("required=%zu missing=%d unexpected=%d total_tensors=%zu\n",
                required.size(), missing, unexpected, m.tensor_names().size());

    if (missing > 0)    return 4;
    if (unexpected > 0) return 5;

    // Also verify the higher-level orchestrator load path branches to the
    // 1.5b variant correctly: encoders, connectors, lm_head, diffusion head
    // and decoder all populated; no realtime-only weights touched.
    vv::VibeVoiceModel mm;
    if (!vv::vibevoice_load(path, &mm)) {
        std::fprintf(stderr, "FAIL: vv::vibevoice_load returned false\n");
        return 6;
    }
    if (mm.variant != "1.5b") {
        std::fprintf(stderr, "FAIL: model.variant=%s want 1.5b\n", mm.variant.c_str());
        return 7;
    }
    if (mm.cfg.hidden != 1536 || mm.cfg.n_layers_lm != 28 || mm.cfg.n_layers_tlm != 0 ||
        mm.cfg.head_dim != 128 || mm.cfg.latent != 64) {
        std::fprintf(stderr, "FAIL: cfg dims wrong\n");
        return 8;
    }
    if (!mm.w.lm_tok_embd || !mm.w.tlm_output_norm || !mm.lm_head ||
        !mm.w.ac_fc1_w || !mm.sc_fc1_w ||
        !mm.w.dh.cond_proj || !mm.w.at_dec.head.kernel) {
        std::fprintf(stderr, "FAIL: required weights null after vibevoice_load\n");
        return 9;
    }
    // Realtime-only weights must remain null:
    if (mm.w.tts_input_types || mm.w.eos_fc1_w) {
        std::fprintf(stderr, "FAIL: realtime-only weights wrongly populated for 1.5b\n");
        return 10;
    }
    if (mm.at_enc.stem.kernel == nullptr || mm.st_enc.stem.kernel == nullptr) {
        std::fprintf(stderr, "FAIL: at_enc / st_enc not populated\n");
        return 11;
    }

    std::printf("vibevoice_load OK: variant=%s lm_layers=%d head_dim=%d "
                "encoders+connectors+dh+decoder all live\n",
                mm.variant.c_str(), mm.cfg.n_layers_lm, mm.cfg.head_dim);
    return 0;
}
