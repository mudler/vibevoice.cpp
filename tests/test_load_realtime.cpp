// Smoke test that the real-weight gguf produced by
// scripts/convert_vibevoice_to_gguf.py loads cleanly and contains every
// tensor our orchestrator will need.
//
// Skips (return 77) if the gguf is missing.

#include "model_loader.hpp"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Expect {
    std::string name;
    bool        required;
};

void expect_lm(std::vector<Expect>* out, const std::string& prefix, int n_layers) {
    out->push_back({prefix + ".tok_embd.weight", true});
    for (int i = 0; i < n_layers; ++i) {
        char b[64]; std::snprintf(b, sizeof(b), "%s.blk.%d.", prefix.c_str(), i);
        std::string p(b);
        for (const char* s : {"attn_q.weight","attn_q.bias",
                              "attn_k.weight","attn_k.bias",
                              "attn_v.weight","attn_v.bias",
                              "attn_o.weight",
                              "attn_norm.weight","ffn_norm.weight",
                              "ffn_gate.weight","ffn_up.weight","ffn_down.weight"}) {
            out->push_back({p + s, true});
        }
    }
}

}  // namespace

int main() {
    const char* path = std::getenv("VIBEVOICE_MODEL");
    std::string p   = path ? path : "models/vibevoice-realtime-0.5B.gguf";

    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "skip: missing %s (run scripts/convert_vibevoice_to_gguf.py)\n",
                     p.c_str());
        return 77;
    }
    std::fclose(f);

    vv::ModelLoader m;
    if (!m.load(p)) {
        std::fprintf(stderr, "load failed\n");
        return 1;
    }

    const int n_lm  = m.get_i32("vibevoice.n_layers_lm",  -1);
    const int n_tlm = m.get_i32("vibevoice.n_layers_tlm", -1);
    const int hidden = m.get_i32("vibevoice.hidden", -1);
    const int n_h    = m.get_i32("vibevoice.n_heads", -1);
    const int n_kv_h = m.get_i32("vibevoice.n_kv_heads", -1);
    const int hd     = m.get_i32("vibevoice.head_dim", -1);
    const int latent = m.get_i32("vibevoice.diffusion.latent", -1);

    std::printf("loaded %s\n", p.c_str());
    std::printf("  variant=%s  hidden=%d  heads=%d/%d  hd=%d\n",
                m.get_str("vibevoice.variant").c_str(),
                hidden, n_h, n_kv_h, hd);
    std::printf("  lm=%d layers, tlm=%d layers, latent=%d\n", n_lm, n_tlm, latent);
    std::printf("  total tensors: %zu\n", m.tensor_names().size());

    if (n_lm <= 0 || n_tlm <= 0 || hidden <= 0) {
        std::fprintf(stderr, "FAIL: bad metadata\n");
        return 2;
    }

    std::vector<Expect> expect;
    expect_lm(&expect, "lm",  n_lm);
    expect_lm(&expect, "tlm", n_tlm);
    expect.push_back({"tlm.output_norm.weight", true});

    expect.push_back({"tts.input_types.weight", true});
    expect.push_back({"speech.scaling", true});
    expect.push_back({"speech.bias", true});

    expect.push_back({"ac.fc1.weight", true});
    expect.push_back({"ac.fc1.bias",   true});
    expect.push_back({"ac.norm.weight",true});
    expect.push_back({"ac.fc2.weight", true});
    expect.push_back({"ac.fc2.bias",   true});

    // Diffusion head (4 layers, head_layers cfg)
    const int head_layers = m.get_i32("vibevoice.diffusion.head_layers", 4);
    expect.push_back({"dh.noisy_proj",   true});
    expect.push_back({"dh.cond_proj",    true});
    expect.push_back({"dh.t_embed_lin1", true});
    expect.push_back({"dh.t_embed_lin2", true});
    expect.push_back({"dh.final.proj",   true});
    expect.push_back({"dh.final.adaln",  true});
    for (int i = 0; i < head_layers; ++i) {
        char b[40]; std::snprintf(b, sizeof(b), "dh.layer_%d.", i);
        std::string pp(b);
        for (const char* s : {"norm","ffn_gate","ffn_up","ffn_down","adaln"}) {
            expect.push_back({pp + s, true});
        }
    }

    // EOS classifier (2-layer MLP)
    for (const char* s : {"eos.fc1.weight","eos.fc1.bias","eos.fc2.weight","eos.fc2.bias"}) {
        expect.push_back({s, true});
    }

    // Acoustic decoder: stem + 6 transposed upsamples + 7 stages + head
    expect.push_back({"at.dec.stem.weight",       true});
    expect.push_back({"at.dec.stem.bias",         true});
    expect.push_back({"at.dec.head.weight",       true});
    expect.push_back({"at.dec.head.bias",         true});
    for (int i = 1; i <= 6; ++i) {
        char b[40]; std::snprintf(b, sizeof(b), "at.dec.up_%d.", i);
        expect.push_back({std::string(b) + "weight", true});
        expect.push_back({std::string(b) + "bias",   true});
    }

    int missing = 0;
    for (const auto& e : expect) {
        if (!m.has(e.name)) {
            if (e.required) {
                if (missing < 20) std::fprintf(stderr, "MISSING: %s\n", e.name.c_str());
                ++missing;
            }
        }
    }
    std::printf("required tensors: %zu  missing: %d\n", expect.size(), missing);

    if (missing > 0) return 3;
    return 0;
}
