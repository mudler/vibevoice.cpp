#ifndef VIBEVOICE_CONV1D_HPP
#define VIBEVOICE_CONV1D_HPP

// Causal Conv1d / ConvTranspose1d helpers, mirroring VibeVoice's SConv1d /
// SConvTranspose1d:
//
//   SConv1d:  pad_total = (k - 1) * dilation - (stride - 1)
//             extra     = stride alignment so output length = ceil((T + pad_total - k) / stride) + 1
//             pad LEFT by pad_total (zeros, causal), pad RIGHT by extra (zeros)
//             then apply Conv1d(stride, p=0, d).
//
//   SConvTranspose1d (causal, trim_right_ratio=1):
//             apply ConvTranspose1d(stride), output length = (T - 1) * stride + k
//             trim RIGHT by (k - stride) zeros — left untouched, right fully trimmed.
//
// Layout: ggml expects data as [T, C_in, B]; kernel as [K, C_in/groups, C_out]
// (non-depthwise) or [K, 1, C] (depthwise, groups=C). Both are obtained
// directly from PyTorch shapes when written through gguf.

#include "ggml.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace vv {

inline int sconv1d_extra_padding(int T_in, int kernel, int stride, int pad_total) {
    // Equivalent to: ceil((T - k + pad_total) / stride + 1) → ideal length
    long n_num = static_cast<long>(T_in) - kernel + pad_total;
    long n_den = stride;
    long n_frames_ceil = (n_num + n_den - 1) / n_den + 1;
    long ideal = (n_frames_ceil - 1) * stride + (kernel - pad_total);
    long extra = ideal - T_in;
    return extra > 0 ? static_cast<int>(extra) : 0;
}

// Per-conv1d streaming cache, mirrors upstream's
// VibeVoiceTokenizerStreamingCache (vibevoice/modular/modular_vibevoice_tokenizer.py:192).
// One entry per causal conv1d in the encoder; the entry holds the last
// `context_size = (k-1)*d - (s-1)` samples of that conv's INPUT, in
// row-major [T, C, B] = ggml ne layout, so the next chunk can prepend
// them as left context and produce bit-exact output vs a single-shot pass.
//
// `next_*` tensors are populated during graph build by sconv1d_causal_streaming
// and are leaves expanded into the forward graph; the caller copies them
// out after `ggml_graph_compute_with_ctx` and stores into `data` for the
// next chunk.
struct StreamingCacheEntry {
    std::vector<float> data;           // size = context_size * C * B (B=1)
    int                T = 0;          // context_size
    int                C = 0;
    struct ggml_tensor* next_view = nullptr;  // last-context view, captured during graph build
    struct ggml_tensor* prefix    = nullptr;  // input prefix leaf, filled by caller after alloc
};

class StreamingCache {
public:
    bool is_first_chunk = true;
    bool is_final_chunk = false;

    StreamingCacheEntry& operator[](const std::string& key) { return entries_[key]; }
    bool   has(const std::string& key) const { return entries_.count(key) > 0; }
    size_t size() const                       { return entries_.size(); }
    void   clear()                            { entries_.clear(); }

    // Iterate entries — callers use this after graph compute to copy
    // next_view tensor data back into the entry's `data` buffer.
    auto begin() { return entries_.begin(); }
    auto end()   { return entries_.end(); }

private:
    std::unordered_map<std::string, StreamingCacheEntry> entries_;
};

// Causal SConv1d: zero left-pad + zero right-extra-pad + ggml_conv_1d.
// `groups` of 1 = standard conv; groups == C_in == C_out = depthwise.
struct ggml_tensor* sconv1d_causal(struct ggml_context* ctx,
                                   struct ggml_tensor*  x,        // [T, C_in, B]
                                   struct ggml_tensor*  kernel,   // [K, C_in/groups, C_out]
                                   struct ggml_tensor*  bias,     // [C_out] or null
                                   int stride,
                                   int dilation,
                                   int groups);

// Streaming variant. Reads the cache entry for `layer_id`, prepends those
// samples as left context (or zeros for the first chunk), runs the conv,
// and registers a view of "the last context_size samples of the
// concatenated input" as `cache.entries_[layer_id].next_view` so the
// caller can copy it out post-compute.
//
// Right-side extra-padding for stride alignment is applied ONLY when
// `cache.is_final_chunk` is true — non-final chunks omit it so the
// concatenated stream stays aligned across boundaries.
struct ggml_tensor* sconv1d_causal_streaming(struct ggml_context* ctx,
                                             struct ggml_tensor*  x,
                                             struct ggml_tensor*  kernel,
                                             struct ggml_tensor*  bias,
                                             int stride,
                                             int dilation,
                                             int groups,
                                             StreamingCache&       cache,
                                             const std::string&    layer_id);

// Causal SConvTranspose1d with trim_right_ratio = 1.0.
// kernel layout: [K, C_out, C_in] (PyTorch [C_in, C_out, K] reversed by gguf).
struct ggml_tensor* sconv_transpose1d_causal(struct ggml_context* ctx,
                                             struct ggml_tensor*  x,      // [T, C_in, B]
                                             struct ggml_tensor*  kernel, // [K, C_out, C_in]
                                             struct ggml_tensor*  bias,   // [C_out] or null
                                             int stride);

}  // namespace vv

#endif  // VIBEVOICE_CONV1D_HPP
