add_executable(vibevoice-cli main.cpp)
target_link_libraries(vibevoice-cli PRIVATE vibevoice)
target_include_directories(vibevoice-cli PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_options(vibevoice-cli PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

set_target_properties(vibevoice-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
