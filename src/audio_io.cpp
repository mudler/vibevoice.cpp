#include "audio_io.hpp"
#include "common.hpp"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace vv {

int load_wav_mono_f32(const std::string& path, vv_audio* out) {
    if (!out) return VV_ERR_INVALID_ARG;

    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        VV_LOG_ERROR("load_wav: failed to open %s", path.c_str());
        return VV_ERR_FILE_NOT_FOUND;
    }

    const size_t total = static_cast<size_t>(wav.totalPCMFrameCount);
    const int    ch    = static_cast<int>(wav.channels);
    const int    sr    = static_cast<int>(wav.sampleRate);

    std::vector<float> interleaved(total * ch);
    drwav_read_pcm_frames_f32(&wav, total, interleaved.data());
    drwav_uninit(&wav);

    auto* mono = static_cast<float*>(std::malloc(sizeof(float) * total));
    if (!mono) return VV_ERR_OUT_OF_MEMORY;

    if (ch == 1) {
        std::memcpy(mono, interleaved.data(), sizeof(float) * total);
    } else {
        const float inv = 1.0f / static_cast<float>(ch);
        for (size_t i = 0; i < total; ++i) {
            float s = 0.f;
            for (int c = 0; c < ch; ++c) s += interleaved[i * ch + c];
            mono[i] = s * inv;
        }
    }

    out->samples     = mono;
    out->n_samples   = total;
    out->sample_rate = sr;
    out->channels    = 1;
    return VV_OK;
}

int save_wav_pcm16(const std::string& path, const vv_audio& a) {
    if (!a.samples || a.n_samples == 0) return VV_ERR_INVALID_ARG;

    drwav_data_format fmt{};
    fmt.container     = drwav_container_riff;
    fmt.format        = DR_WAVE_FORMAT_PCM;
    fmt.channels      = static_cast<drwav_uint32>(a.channels > 0 ? a.channels : 1);
    fmt.sampleRate    = static_cast<drwav_uint32>(a.sample_rate);
    fmt.bitsPerSample = 16;

    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &fmt, nullptr)) {
        VV_LOG_ERROR("save_wav: failed to open %s for write", path.c_str());
        return VV_ERR_AUDIO;
    }

    const size_t n = a.n_samples * fmt.channels;
    std::vector<int16_t> pcm(n);
    for (size_t i = 0; i < n; ++i) {
        float s = a.samples[i];
        s = std::max(-1.0f, std::min(1.0f, s));
        pcm[i] = static_cast<int16_t>(std::lround(s * 32767.0f));
    }

    drwav_write_pcm_frames(&wav, a.n_samples, pcm.data());
    drwav_uninit(&wav);
    return VV_OK;
}

std::vector<float> resample_linear(const std::vector<float>& in,
                                   int src_rate, int dst_rate) {
    if (in.empty() || src_rate == dst_rate) return in;
    if (src_rate <= 0 || dst_rate <= 0) return {};

    const double ratio = static_cast<double>(dst_rate) / src_rate;
    const size_t out_n = static_cast<size_t>(std::llround(in.size() * ratio));
    std::vector<float> out(out_n);

    for (size_t i = 0; i < out_n; ++i) {
        const double t  = i / ratio;
        const size_t i0 = static_cast<size_t>(t);
        const double f  = t - i0;
        const float  a  = in[std::min(i0,     in.size() - 1)];
        const float  b  = in[std::min(i0 + 1, in.size() - 1)];
        out[i] = static_cast<float>(a + (b - a) * f);
    }
    return out;
}

int load_wav_24k_mono(const std::string& path, std::vector<float>* out) {
    if (!out) return VV_ERR_INVALID_ARG;

    vv_audio a{};
    int rc = load_wav_mono_f32(path, &a);
    if (rc != VV_OK) return rc;

    std::vector<float> samples(a.samples, a.samples + a.n_samples);
    const int src_sr = a.sample_rate;
    vv_audio_free(&a);

    if (src_sr == kVibeVoiceSampleRate) {
        *out = std::move(samples);
    } else {
        *out = resample_linear(samples, src_sr, kVibeVoiceSampleRate);
    }
    return VV_OK;
}

}  // namespace vv

extern "C" {

int vv_load_wav(const char* path, vv_audio* out) {
    if (!path || !out) return VV_ERR_INVALID_ARG;
    return vv::load_wav_mono_f32(path, out);
}

int vv_save_wav(const char* path, const vv_audio* in) {
    if (!path || !in) return VV_ERR_INVALID_ARG;
    return vv::save_wav_pcm16(path, *in);
}

}  // extern "C"
