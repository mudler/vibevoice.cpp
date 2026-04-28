// Smoke test — proves tests compile & link against libvibevoice.

#include "vibevoice.h"

#include <cstdio>
#include <cstring>

int main() {
    const char* v = vv_version();
    if (!v || std::strlen(v) == 0) {
        std::fprintf(stderr, "vv_version() returned empty\n");
        return 1;
    }
    auto p = vv_context_default_params();
    if (p.use_f16_kv != true) return 4;
    auto tts = vv_tts_default_params();
    auto asr = vv_asr_default_params();
    if (tts.n_diffusion_steps != 20) return 2;
    if (asr.greedy != true)         return 3;
    std::printf("smoke ok: %s\n", v);
    return 0;
}
