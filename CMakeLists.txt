cmake_minimum_required(VERSION 3.14)
project(FunASR-GGML LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU 支持（可选）
option(FUNASR_CUDA "Enable CUDA GPU support" OFF)

if(FUNASR_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    add_compile_definitions(FUNASR_USE_CUDA)
endif()

# ============================================================
# third_party: ggml
# ============================================================
add_subdirectory(third_party/ggml)

# ============================================================
# funasr_core (header-only)
# ============================================================
add_library(funasr_core INTERFACE)
target_include_directories(funasr_core INTERFACE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/ggml/include
    ${CMAKE_SOURCE_DIR}/third_party              # miniaudio.h
)
target_link_libraries(funasr_core INTERFACE ggml)

# ============================================================
# funasr_model
# ============================================================
add_library(funasr_model STATIC
    model/loader.cpp
    model/tokenizer.cpp
)
target_link_libraries(funasr_model PUBLIC funasr_core)

# ============================================================
# funasr_compute
# ============================================================
add_library(funasr_compute STATIC
    compute/fbank.cpp
    compute/encoder_ops.cpp
    compute/adaptor_ops.cpp
    compute/llm_ops.cpp
)
if(FUNASR_CUDA)
    target_sources(funasr_compute PRIVATE compute/llm_ops_gpu.cpp)
endif()
target_link_libraries(funasr_compute PUBLIC funasr_core)

# ============================================================
# funasr_pipeline
# ============================================================
add_library(funasr_pipeline STATIC
    pipeline/pipeline.cpp
    pipeline/audio_capture.cpp
)
target_link_libraries(funasr_pipeline PUBLIC funasr_model funasr_compute)
if(UNIX AND NOT APPLE)
    target_link_libraries(funasr_pipeline PUBLIC pthread dl m)
endif()

# ============================================================
# 测试目标
# ============================================================
add_executable(test_config test/test_config.cpp)
target_link_libraries(test_config PRIVATE funasr_core)

add_executable(test_loader test/test_loader.cpp)
target_link_libraries(test_loader PRIVATE funasr_model)

add_executable(test_tokenizer test/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE funasr_model)

add_executable(test_fbank test/test_fbank.cpp)
target_link_libraries(test_fbank PRIVATE funasr_compute)

add_executable(test_encoder test/test_encoder.cpp)
target_link_libraries(test_encoder PRIVATE funasr_model funasr_compute)

add_executable(test_adaptor test/test_adaptor.cpp)
target_link_libraries(test_adaptor PRIVATE funasr_model funasr_compute)

add_executable(test_llm test/test_llm.cpp)
target_link_libraries(test_llm PRIVATE funasr_model funasr_compute)

add_executable(test_pipeline test/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE funasr_pipeline)

add_executable(test_realtime test/test_realtime.cpp)
target_link_libraries(test_realtime PRIVATE funasr_pipeline)

if(FUNASR_CUDA)
    add_executable(test_gpu test/test_gpu.cpp)
    target_link_libraries(test_gpu PRIVATE funasr_pipeline)
endif()