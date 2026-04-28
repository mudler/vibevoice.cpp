#ifndef VIBEVOICE_MODEL_LOADER_HPP
#define VIBEVOICE_MODEL_LOADER_HPP

// Lightweight gguf reader. Wraps ggml's gguf_* API with a name→tensor map
// and typed metadata accessors. Used by every component that needs to load
// weights (tokenizer, qwen2, acoustic_tokenizer, diffusion_head, ...).

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace vv {

class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();

    ModelLoader(const ModelLoader&)            = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    // Open the gguf file and load all tensor metadata + weight bytes into a
    // newly-created ggml_context. Tensors are looked up by name with `tensor`.
    bool load(const std::string& path);

    // Tensor lookup. Returns nullptr if not found.
    struct ggml_tensor* tensor(const std::string& name) const;
    bool                has(const std::string& name) const;

    // Cast all small (1-D / scalar / few-row) f16 tensors to f32 in-place.
    // ggml's CPU element-wise ops don't auto-cast f32 + f16, so norm scales,
    // biases, gammas and tiny embeddings need to be promoted up front when
    // the gguf was written in fp16. Large matmul weights stay fp16.
    //
    // `max_elems` is the size threshold (in scalar element count) for promotion.
    void promote_small_f16_to_f32(size_t max_elems = 65536);

    const std::vector<std::string>& tensor_names() const { return tensor_names_; }

    // Metadata accessors. Return the provided default if the key is absent or
    // wrong-typed.
    int32_t     get_i32(const std::string& key, int32_t       def = 0)  const;
    int64_t     get_i64(const std::string& key, int64_t       def = 0)  const;
    uint32_t    get_u32(const std::string& key, uint32_t      def = 0)  const;
    float       get_f32(const std::string& key, float         def = 0)  const;
    bool        get_bool(const std::string& key, bool         def = false) const;
    std::string get_str(const std::string& key, const std::string& def = {}) const;
    std::vector<int32_t> get_i32_array(const std::string& key) const;
    std::vector<std::string> get_str_array(const std::string& key) const;

    // Raw access if a caller needs more.
    struct gguf_context* gguf_ctx() const { return gguf_; }
    struct ggml_context* ggml_ctx() const { return ctx_; }

private:
    struct gguf_context*      gguf_           = nullptr;
    struct ggml_context*      ctx_            = nullptr;  // tensor metadata
    struct ggml_context*      promote_ctx_    = nullptr;  // f32 promoted tensors
    ggml_backend_buffer_t     backend_buffer_ = nullptr;  // owns the actual weight data
    ggml_backend_buffer_t     promote_buffer_ = nullptr;  // owns promoted weight data
    std::vector<std::string>  tensor_names_;
    std::unordered_map<std::string, struct ggml_tensor*> tensor_by_name_;
};

}  // namespace vv

#endif  // VIBEVOICE_MODEL_LOADER_HPP
