add_executable(vibevoice-quantize main.cpp)
target_link_libraries(vibevoice-quantize PRIVATE vibevoice)
target_include_directories(vibevoice-quantize PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_options(vibevoice-quantize PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)
set_target_properties(vibevoice-quantize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
