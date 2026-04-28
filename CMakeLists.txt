cmake_minimum_required(VERSION 3.18)
project(vibevoice
        VERSION   0.0.1
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------- options ----------
option(VIBEVOICE_BUILD_TESTS    "Build vibevoice tests"    OFF)
option(VIBEVOICE_BUILD_EXAMPLES "Build vibevoice examples" ON)
option(VIBEVOICE_BUILD_SERVER   "Build vibevoice server"   OFF)
option(VIBEVOICE_TEST_LARGE     "Run large/E2E tests"      OFF)
option(VIBEVOICE_SHARED         "Build shared library"     OFF)

# Backend toggles — forwarded to the embedded ggml submodule. Off by
# default; enable with `-DVIBEVOICE_GGML_CUDA=ON`, etc.
option(VIBEVOICE_GGML_CUDA      "Build ggml with CUDA backend"     OFF)
option(VIBEVOICE_GGML_METAL     "Build ggml with Metal backend"    OFF)
option(VIBEVOICE_GGML_VULKAN    "Build ggml with Vulkan backend"   OFF)
option(VIBEVOICE_GGML_HIPBLAS   "Build ggml with hipBLAS backend"  OFF)

# ---------- ggml ----------
# We embed ggml as a submodule under third_party/ggml. We forward common
# GGML_* options so users can drive backends from the top-level cache.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(VIBEVOICE_GGML_CUDA)
    set(GGML_CUDA   ON CACHE BOOL "" FORCE)
endif()
if(VIBEVOICE_GGML_METAL)
    set(GGML_METAL  ON CACHE BOOL "" FORCE)
endif()
if(VIBEVOICE_GGML_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(VIBEVOICE_GGML_HIPBLAS)
    set(GGML_HIPBLAS ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# ---------- libvibevoice ----------
set(VIBEVOICE_SOURCES
    src/common.cpp
    src/backend.cpp
    src/vibevoice.cpp
    src/vibevoice_capi.cpp
    src/audio_io.cpp
    src/model_loader.cpp
    src/tokenizer.cpp
    src/qwen2.cpp
    src/conv1d.cpp
    src/acoustic_tokenizer.cpp
    src/diffusion_head.cpp
    src/dpm_solver.cpp
    src/vibevoice_tts.cpp
    src/vibevoice_asr.cpp
)

set(VIBEVOICE_PUBLIC_HEADERS
    include/vibevoice.h
    include/vibevoice_capi.h
)

if(VIBEVOICE_SHARED)
    add_library(vibevoice SHARED ${VIBEVOICE_SOURCES})
    target_compile_definitions(vibevoice PUBLIC VIBEVOICE_SHARED)
    target_compile_definitions(vibevoice PRIVATE VIBEVOICE_BUILD)
else()
    add_library(vibevoice STATIC ${VIBEVOICE_SOURCES})
endif()

target_include_directories(vibevoice
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(vibevoice PUBLIC ggml)

# When a ggml backend (CUDA / Metal / Vulkan / hipBLAS / blas) is enabled,
# explicitly link its target so the static library carries the symbols
# (mirrors the pattern in LocalAI's qwen3-tts-cpp wrapper).
foreach(_be cuda metal vulkan hipblas blas)
    if(TARGET ggml-${_be})
        target_link_libraries(vibevoice PUBLIC ggml-${_be})
        string(TOUPPER ${_be} _BE)
        target_compile_definitions(vibevoice PUBLIC VIBEVOICE_HAVE_${_BE})
    endif()
endforeach()
if(VIBEVOICE_GGML_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_link_libraries(vibevoice PUBLIC CUDA::cudart)
    endif()
endif()

target_compile_options(vibevoice PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
)

# ---------- subdirs ----------
if(VIBEVOICE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(VIBEVOICE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------- install ----------
include(GNUInstallDirs)

install(TARGETS vibevoice
        EXPORT  vibevoiceTargets
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES ${VIBEVOICE_PUBLIC_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
