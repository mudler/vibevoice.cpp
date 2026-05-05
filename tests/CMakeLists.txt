# Unit / integration tests. Each test_*.cpp becomes an executable + ctest entry.
# Large or weight-dependent tests are gated by VIBEVOICE_TEST_LARGE.

set(VV_FIXTURES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

function(vv_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE vibevoice)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
    )
    target_compile_definitions(${name} PRIVATE
        VV_FIXTURES_DIR="${VV_FIXTURES_DIR}"
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Tests are added by later milestones; a placeholder smoke test
# proves the test machinery is wired up.
vv_add_test(test_smoke)
vv_add_test(test_audio_io)
vv_add_test(test_tokenizer)
vv_add_test(test_rope)
vv_add_test(test_qwen2_block)
vv_add_test(test_qwen2_resident)
vv_add_test(test_sconv1d)
vv_add_test(test_block1d)
vv_add_test(test_acoustic)
vv_add_test(test_diffusion_head)
vv_add_test(test_dpm_solver)
vv_add_test(test_load_realtime)
vv_add_test(test_load_15b)
vv_add_test(test_e2e_tts)
vv_add_test(test_e2e_asr)
# Closed-loop TTS→ASR roundtrip. Requires both model gguf files (~16 GB),
# so it's gated by VIBEVOICE_TEST_LARGE. When the env vars are missing the
# test exits 77 (= ctest skip) so it stays green for everyone else.
if(VIBEVOICE_TEST_LARGE)
    vv_add_test(test_closed_loop)
    vv_add_test(test_long_form_asr)
    vv_add_test(test_capi)
    vv_add_test(test_encoder_chunked_parity)
    vv_add_test(test_15b_smoke)
    vv_add_test(test_15b_closed_loop)
    set_tests_properties(test_closed_loop test_long_form_asr test_capi
                         test_encoder_chunked_parity
                         test_15b_smoke test_15b_closed_loop
                         PROPERTIES SKIP_RETURN_CODE 77)
endif()
set_tests_properties(test_tokenizer test_rope test_qwen2_block test_qwen2_resident
                     test_sconv1d test_block1d test_acoustic test_diffusion_head
                     test_dpm_solver test_load_realtime test_load_15b
                     test_e2e_tts test_e2e_asr
                     PROPERTIES SKIP_RETURN_CODE 77)
