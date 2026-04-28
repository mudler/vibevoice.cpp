#ifndef VIBEVOICE_ASR_HPP
#define VIBEVOICE_ASR_HPP

// VibeVoice ASR transcription:
//   audio (24 kHz mono) -> acoustic + semantic encoders -> connectors -> sum
//                       -> embed_tokens(prompt with <|box_start|> placeholders)
//                       -> splice speech features at placeholder positions
//                       -> 28-layer Qwen2 LM + final norm + lm_head
//                       -> greedy decode until <|im_end|> or <|endoftext|>
//
// Prompt format (mirrors vibevoice/processor/vibevoice_asr_processor.py and
// vibevoice/modular/modular_vibevoice_text_tokenizer.py:VibeVoiceASRTextTokenizerFast):
//
//   <|im_start|>system
//   You are a helpful assistant that transcribes audio input into text output in JSON format.<|im_end|>
//   <|im_start|>user
//   <|object_ref_start|><|box_start|>×N<|object_ref_end|>
//   This is a X.XX seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>
//   <|im_start|>assistant
//
//   where N = ceil(audio_samples / 3200) and speech_start/end/pad use the
//   reused vision-token IDs:
//     <|object_ref_start|> = 151646   speech_start
//     <|object_ref_end|>   = 151647   speech_end
//     <|box_start|>        = 151648   speech_pad

#include "vibevoice_tts.hpp"   // shares VibeVoiceModel + Qwen2 stack runner

#include <string>
#include <vector>

namespace vv {

struct VibeVoiceASRParams {
    int      max_new_tokens     = 512;
    float    temperature        = 0.0f;   // 0 = greedy
    float    repetition_penalty = 1.05f;  // 1.0 = off; small penalty avoids
                                          // long greedy loops without
                                          // forcing hallucinations
    int      no_repeat_ngram    = 0;      // 0 = off
    uint32_t seed               = 0;
    bool     verbose            = false;
};

// Transcribe `audio_24k_mono` into `transcript`. Returns 0 on success.
int vibevoice_asr_transcribe(VibeVoiceModel*           model,
                             const std::vector<float>& audio_24k_mono,
                             const VibeVoiceASRParams& p,
                             std::string*              transcript);

}  // namespace vv

#endif  // VIBEVOICE_ASR_HPP
