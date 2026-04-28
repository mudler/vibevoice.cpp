#ifndef VIBEVOICE_AUDIO_IO_HPP
#define VIBEVOICE_AUDIO_IO_HPP

// Minimal WAV I/O + linear resampler. The runtime works at 24 kHz mono
// internally; this header is the single converging point for audio I/O.

#include "vibevoice.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vv {

constexpr int kVibeVoiceSampleRate = 24000;

// Load a WAV file into a vv_audio (mono float32). Multi-channel input is
// downmixed to mono. Caller owns vv_audio_free(out).
int load_wav_mono_f32(const std::string& path, vv_audio* out);

// Save a mono float32 vv_audio as a 16-bit PCM WAV.
int save_wav_pcm16(const std::string& path, const vv_audio& a);

// Linear resampler — good enough for our 24 kHz pipeline. We never resample
// inside the model graph; this is only at I/O boundaries.
std::vector<float> resample_linear(const std::vector<float>& in,
                                   int src_rate, int dst_rate);

// Convenience: read a WAV from disk and produce a 24 kHz mono float32 buffer.
int load_wav_24k_mono(const std::string& path, std::vector<float>* out);

}  // namespace vv

#endif  // VIBEVOICE_AUDIO_IO_HPP
