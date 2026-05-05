// Runtime voice cloning: encode a raw reference WAV into the same
// VibeVoiceVoice format that scripts/convert_voice_to_gguf.py emits
// from upstream's pre-baked .pt files.
//
// Phase 0: stubs only - declarations land in src/vibevoice_tts.hpp +
// include/vibevoice_capi.h, the actual encode/forward path lands in
// Phase 2. Tests in tests/test_voice_clone.cpp drive the
// implementation via TDD.

#include "vibevoice_tts.hpp"
#include "common.hpp"

#include <cstdio>
#include <string>

namespace vv {

bool vibevoice_voice_clone(const std::string&    /*wav_path*/,
                           const VibeVoiceModel& /*model*/,
                           bool                  /*with_cfg*/,
                           VibeVoiceVoice*       /*out*/) {
    VV_LOG_ERROR("vibevoice_voice_clone: not implemented yet "
                 "(Phase 2 - see plan)");
    return false;
}

}  // namespace vv
