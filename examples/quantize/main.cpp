// vibevoice-quantize — selective tensor quantization tool.
//
// Reads a vibevoice gguf, quantizes the LM matmul tensors (attention
// q/k/v/o + FFN gate/up/down + lm_head, plus optionally tok_embd) to a
// target ggml_type via ggml_quantize_chunk, leaves everything else as
// the source dtype, and writes a new gguf.
//
// Why selective quantization (mirrors scripts/quantize_gguf.py):
//   ggml_mul_mat handles quantized weights natively; our conv1d wrapper
//   inline-casts kernels to fp16 (it doesn't dequantize on the fly), so
//   quantizing those would silently produce wrong outputs.
//
// Why this in addition to the python script: gguf-py only implements
// Q4_0 / Q5_0 / Q8_0 / BF16 in pure python. K-quants (Q4_K, Q5_K, Q6_K)
// need libggml linkage — that's what this tool provides.
//
// Usage:
//   vibevoice-quantize --src in.gguf --out out.gguf --type q4_k_m

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

namespace {

struct QuantSpec { const char* name; ggml_type type; };
const QuantSpec kKnown[] = {
    {"q4_0",   GGML_TYPE_Q4_0},
    {"q4_1",   GGML_TYPE_Q4_1},
    {"q5_0",   GGML_TYPE_Q5_0},
    {"q5_1",   GGML_TYPE_Q5_1},
    {"q8_0",   GGML_TYPE_Q8_0},
    {"q2_k",   GGML_TYPE_Q2_K},
    {"q3_k",   GGML_TYPE_Q3_K},
    {"q4_k",   GGML_TYPE_Q4_K},
    {"q5_k",   GGML_TYPE_Q5_K},
    {"q6_k",   GGML_TYPE_Q6_K},
    {"f16",    GGML_TYPE_F16},
    {"f32",    GGML_TYPE_F32},
    {"bf16",   GGML_TYPE_BF16},
};

ggml_type parse_type(const std::string& s) {
    for (const auto& q : kKnown) {
        if (s == q.name) return q.type;
    }
    return GGML_TYPE_COUNT;
}

const std::vector<std::regex>& quantizable() {
    static const std::vector<std::regex> v = {
        std::regex(R"(^(lm|tlm)\.blk\.\d+\.attn_[qkvo]\.weight$)"),
        std::regex(R"(^(lm|tlm)\.blk\.\d+\.ffn_(gate|up|down)\.weight$)"),
        std::regex(R"(^lm_head\.weight$)"),
    };
    return v;
}

bool should_quantize(const std::string& name, bool include_embed) {
    for (const auto& re : quantizable()) {
        if (std::regex_match(name, re)) return true;
    }
    if (include_embed && name == "lm.tok_embd.weight") return true;
    return false;
}

// Convert a source tensor's data (in F32, F16, or BF16) to a flat F32
// vector for input to ggml_quantize_chunk. The source data lives inside
// a gguf-allocated I8 buffer; ggml_get_tensor returns a tensor that knows
// its type and points into that buffer.
std::vector<float> to_f32(const ggml_tensor* t) {
    const size_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
        return out;
    }
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = static_cast<const ggml_fp16_t*>(t->data);
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(src[i]);
        return out;
    }
    if (t->type == GGML_TYPE_BF16) {
        // BF16 → F32 via the standard upper-half cast.
        const uint16_t* src = static_cast<const uint16_t*>(t->data);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            out[i] = f;
        }
        return out;
    }
    std::fprintf(stderr, "to_f32: unsupported source dtype %d for %s\n",
                 static_cast<int>(t->type), t->name);
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    std::string src, out, type_str = "q4_k";
    std::string attn_type_str, ffn_type_str, lm_head_type_str;
    bool include_embed = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--src"           && i + 1 < argc) src              = argv[++i];
        else if (a == "--out"           && i + 1 < argc) out              = argv[++i];
        else if (a == "--type"          && i + 1 < argc) type_str         = argv[++i];
        else if (a == "--attn-type"     && i + 1 < argc) attn_type_str    = argv[++i];
        else if (a == "--ffn-type"      && i + 1 < argc) ffn_type_str     = argv[++i];
        else if (a == "--lm-head-type"  && i + 1 < argc) lm_head_type_str = argv[++i];
        else if (a == "--include-embed")                  include_embed = true;
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr,
                "usage: %s --src in.gguf --out out.gguf --type <type>\n"
                "  --type one of q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k f16 f32 bf16\n"
                "  --attn-type <type>     override quant type for attention weights\n"
                "                         (lm/tlm.blk.*.attn_[qkvo].weight) only.\n"
                "  --ffn-type <type>      override quant type for FFN weights\n"
                "                         (lm/tlm.blk.*.ffn_[gate|up|down].weight) only.\n"
                "  --lm-head-type <type>  override quant type for lm_head.weight only.\n"
                "                         lm_head drives speech-end + text logits and is\n"
                "                         sensitive — for the 1.5B path, e.g.\n"
                "                         --type q5_k --lm-head-type q8_0 lifts closed-loop\n"
                "                         recall from 22%% to ~78%%.\n"
                "  --include-embed   also quantize lm.tok_embd.weight (default: keep at source dtype)\n",
                argv[0]);
            return 0;
        }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (src.empty() || out.empty()) {
        std::fprintf(stderr, "--src and --out are required\n"); return 1;
    }
    const ggml_type target = parse_type(type_str);
    if (target == GGML_TYPE_COUNT) {
        std::fprintf(stderr, "unknown --type: %s\n", type_str.c_str()); return 1;
    }
    auto resolve_override = [&](const std::string& s, const char* flag) -> ggml_type {
        if (s.empty()) return target;
        const ggml_type t = parse_type(s);
        if (t == GGML_TYPE_COUNT) {
            std::fprintf(stderr, "unknown %s: %s\n", flag, s.c_str());
            std::exit(1);
        }
        return t;
    };
    const ggml_type attn_target    = resolve_override(attn_type_str,    "--attn-type");
    const ggml_type ffn_target     = resolve_override(ffn_type_str,     "--ffn-type");
    const ggml_type lm_head_target = resolve_override(lm_head_type_str, "--lm-head-type");

    if (ggml_quantize_requires_imatrix(target)) {
        std::fprintf(stderr,
            "type %s requires an importance matrix (imatrix) — not supported here\n",
            type_str.c_str());
        return 1;
    }

    struct ggml_context* src_ctx = nullptr;
    struct gguf_init_params p {};
    p.no_alloc = false;
    p.ctx      = &src_ctx;
    struct gguf_context* src_gguf = gguf_init_from_file(src.c_str(), p);
    if (!src_gguf) {
        std::fprintf(stderr, "failed to read %s\n", src.c_str()); return 2;
    }

    const int64_t n = gguf_get_n_tensors(src_gguf);
    std::printf("loaded %s: %lld tensors\n", src.c_str(), static_cast<long long>(n));

    // Build a fresh gguf and a sibling ggml_context that holds tensors with
    // the new (possibly quantized) types pointing at our own buffers. Doing
    // this instead of mutating the source gguf keeps the control flow
    // simple: the source ctx owns the original I8 read buffer untouched,
    // and we hand the new gguf only fully-allocated tensors.
    struct gguf_context* dst_gguf = gguf_init_empty();
    gguf_set_kv(dst_gguf, src_gguf);
    // Drop the architecture key that gguf_init_empty added; gguf_set_kv
    // already copied the source's "general.architecture" so we'd have a
    // duplicate without this. Actually gguf_set_kv overrides it; kept for
    // future reference.

    const size_t mem = ggml_tensor_overhead() * (n + 1) + 16ull * 1024 * 1024;
    struct ggml_init_params ip {};
    ip.mem_size = mem;
    ip.no_alloc = true;
    struct ggml_context* dst_ctx = ggml_init(ip);
    if (!dst_ctx) {
        std::fprintf(stderr, "ggml_init(dst) failed\n");
        gguf_free(src_gguf); gguf_free(dst_gguf); ggml_free(src_ctx);
        return 2;
    }

    std::vector<std::vector<uint8_t>> kept;  // owns our quantized / passthrough buffers
    size_t bytes_in = 0, bytes_out = 0, n_quant = 0;

    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(src_gguf, i);
        if (!name) continue;
        ggml_tensor* st = ggml_get_tensor(src_ctx, name);
        if (!st) continue;
        const size_t in_size = ggml_nbytes(st);
        bytes_in += in_size;

        const std::string sname = name;
        ggml_type tensor_target = target;
        if (sname == "lm_head.weight") {
            tensor_target = lm_head_target;
        } else {
            // Match attn vs ffn by the per-block tensor naming convention
            // used by the convert script (attn_[qkvo] / ffn_[gate|up|down]).
            const auto blk_attn = std::regex(R"(^(lm|tlm)\.blk\.\d+\.attn_[qkvo]\.weight$)");
            const auto blk_ffn  = std::regex(R"(^(lm|tlm)\.blk\.\d+\.ffn_(gate|up|down)\.weight$)");
            if (std::regex_match(sname, blk_attn))      tensor_target = attn_target;
            else if (std::regex_match(sname, blk_ffn))  tensor_target = ffn_target;
        }
        const int blk      = ggml_blck_size(tensor_target);
        const bool wantq   = should_quantize(name, include_embed);
        const bool can_quant = (st->ne[0] % blk == 0);
        const bool do_quant  = wantq && can_quant;
        if (wantq && !can_quant) {
            std::fprintf(stderr,
                "skip %s [ne0=%lld] — row not divisible by block size %d\n",
                name, static_cast<long long>(st->ne[0]), blk);
        }

        ggml_type out_type = do_quant ? tensor_target : st->type;
        // Allocate the destination tensor in dst_ctx (no_alloc=true means
        // .data is null until we set it manually).
        ggml_tensor* dt = ggml_new_tensor(dst_ctx, out_type, ggml_n_dims(st), st->ne);
        ggml_set_name(dt, name);

        const int64_t n_per_row = st->ne[0];
        const int64_t nrows     = static_cast<int64_t>(ggml_nelements(st)) / n_per_row;
        const size_t out_size = do_quant
                                ? ggml_row_size(tensor_target, n_per_row) * nrows
                                : in_size;
        kept.emplace_back();
        kept.back().resize(out_size);
        dt->data = kept.back().data();

        if (do_quant) {
            std::vector<float> src_f32 = to_f32(st);
            if (src_f32.empty()) { return 3; }
            ggml_quantize_init(tensor_target);
            const size_t produced = ggml_quantize_chunk(
                tensor_target, src_f32.data(), kept.back().data(),
                /*start=*/0, nrows, n_per_row, /*imatrix=*/nullptr);
            if (produced != out_size) {
                std::fprintf(stderr,
                    "ggml_quantize_chunk(%s, %s): produced %zu, expected %zu\n",
                    name, ggml_type_name(tensor_target), produced, out_size);
                return 4;
            }
            ++n_quant;
        } else {
            std::memcpy(kept.back().data(), st->data, in_size);
        }
        bytes_out += out_size;

        gguf_add_tensor(dst_gguf, dt);
    }

    if (!gguf_write_to_file(dst_gguf, out.c_str(), /*only_meta=*/false)) {
        std::fprintf(stderr, "failed to write %s\n", out.c_str()); return 5;
    }

    const double saved_gb = (static_cast<double>(bytes_in) - bytes_out) / (1024.0*1024.0*1024.0);
    std::printf("wrote %s: quantized %zu tensors → %s, saved %.2f GB (%.1f%%)\n",
                out.c_str(), n_quant, type_str.c_str(), saved_gb,
                100.0 * (1.0 - static_cast<double>(bytes_out) / bytes_in));

    ggml_quantize_free();
    gguf_free(dst_gguf);
    gguf_free(src_gguf);
    ggml_free(dst_ctx);
    ggml_free(src_ctx);
    return 0;
}
