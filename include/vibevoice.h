#ifndef VIBEVOICE_H
#define VIBEVOICE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(VIBEVOICE_SHARED)
#  ifdef VIBEVOICE_BUILD
#    define VV_API __declspec(dllexport)
#  else
#    define VV_API __declspec(dllimport)
#  endif
#else
#  define VV_API
#endif

/* ---------- status codes ---------- */
typedef enum {
    VV_OK                  =  0,
    VV_ERR_GENERIC         = -1,
    VV_ERR_INVALID_ARG     = -2,
    VV_ERR_FILE_NOT_FOUND  = -3,
    VV_ERR_FILE_FORMAT     = -4,
    VV_ERR_OUT_OF_MEMORY   = -5,
    VV_ERR_NOT_IMPLEMENTED = -6,
    VV_ERR_TOKENIZER       = -7,
    VV_ERR_AUDIO           = -8
} vv_status;

/* ---------- logging ---------- */
typedef enum {
    VV_LOG_ERROR = 0,
    VV_LOG_WARN  = 1,
    VV_LOG_INFO  = 2,
    VV_LOG_DEBUG = 3
} vv_log_level;

typedef void (*vv_log_cb)(vv_log_level lvl, const char* msg, void* user_data);

VV_API void        vv_set_log_callback(vv_log_cb cb, void* user_data);
VV_API const char* vv_version(void);

/* ---------- forward decls ---------- */
typedef struct vv_context vv_context;
typedef struct vv_voice   vv_voice;

/* ---------- context ---------- */
typedef struct {
    const char* model_path;     /* path to a .gguf model file */
    int         n_threads;      /* 0 = use all logical cores */
    int         gpu_layers;     /* reserved; 0 = CPU only */
    bool        use_f16_kv;     /* default true */
    uint32_t    seed;           /* 0 = random */
} vv_context_params;

VV_API vv_context_params vv_context_default_params(void);
VV_API vv_context*       vv_init(const vv_context_params* params);
VV_API void              vv_free(vv_context* ctx);

/* ---------- audio buffer ---------- */
typedef struct {
    float*  samples;
    size_t  n_samples;
    int     sample_rate;
    int     channels;
} vv_audio;

VV_API void vv_audio_free(vv_audio* a);

VV_API int vv_load_wav(const char* path, vv_audio* out);
VV_API int vv_save_wav(const char* path, const vv_audio* in);

/* ---------- voices (TTS) ---------- */
VV_API vv_voice* vv_voice_load(vv_context* ctx, const char* path);
VV_API void      vv_voice_free(vv_voice* v);

/* ---------- TTS ---------- */
typedef struct {
    const vv_voice* voice;
    int             n_diffusion_steps;  /* default 20 */
    float           cfg_scale;          /* default 1.3 */
    int             max_new_tokens;     /* safety cap; 0 = unbounded */
    uint32_t        seed;               /* 0 = use ctx seed */
} vv_tts_params;

VV_API vv_tts_params vv_tts_default_params(void);
VV_API int           vv_tts(vv_context* ctx,
                            const char* text,
                            const vv_tts_params* p,
                            vv_audio* out);

/* ---------- ASR ---------- */
typedef struct {
    float temperature;     /* 0.0 = greedy */
    int   max_new_tokens;  /* 0 = unbounded */
    bool  greedy;
} vv_asr_params;

VV_API vv_asr_params vv_asr_default_params(void);
VV_API int           vv_asr(vv_context* ctx,
                            const float* samples,
                            size_t n_samples,
                            int sample_rate,
                            const vv_asr_params* p,
                            char** out_utf8);
VV_API void          vv_string_free(char* s);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* VIBEVOICE_H */
