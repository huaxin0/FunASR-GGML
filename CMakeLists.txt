cmake_minimum_required(VERSION 3.14)
project(FunASR-GGML LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

# GPU 支持（可选）
option(FUNASR_CUDA "Enable CUDA GPU support" OFF)
option(FUNASR_FLASH_ATTN "Enable GGML FlashAttention for CUDA LLM attention" ON)
option(FUNASR_BUILD_TESTS "Build local test executables" ON)

if(FUNASR_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    add_compile_definitions(FUNASR_USE_CUDA)
    if(FUNASR_FLASH_ATTN)
        add_compile_definitions(FUNASR_USE_FLASH_ATTN)
    endif()
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
set_target_properties(funasr_model PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ============================================================
# funasr_compute
# ============================================================
add_library(funasr_compute STATIC
    compute/fbank.cpp
    compute/encoder_ops.cpp
    compute/adaptor_ops.cpp
    compute/llm_ops.cpp
    compute/silero_vad.cpp
)
if(FUNASR_CUDA)
    target_sources(funasr_compute PRIVATE compute/llm_ops_gpu.cpp)
endif()
target_link_libraries(funasr_compute PUBLIC funasr_core)
set_target_properties(funasr_compute PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ============================================================
# funasr_pipeline
# ============================================================
add_library(funasr_pipeline STATIC
    pipeline/pipeline.cpp
    pipeline/audio_capture.cpp
    pipeline/chunking.cpp
    pipeline/offline_batching.cpp
)
target_link_libraries(funasr_pipeline PUBLIC funasr_model funasr_compute)
if(UNIX AND NOT APPLE)
    target_link_libraries(funasr_pipeline PUBLIC pthread dl m)
endif()
set_target_properties(funasr_pipeline PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ============================================================
# funasr_sdk (C ABI wrapper for host applications)
# ============================================================
add_library(funasr_sdk SHARED sdk/funasr_sdk.cpp)
target_link_libraries(funasr_sdk PRIVATE funasr_pipeline)
target_include_directories(funasr_sdk PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(funasr_sdk PRIVATE FUNASR_SDK_BUILD)
set_target_properties(funasr_sdk PROPERTIES
    OUTPUT_NAME funasr_sdk
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
)

add_library(funasr_sdk_static STATIC sdk/funasr_sdk.cpp)
target_link_libraries(funasr_sdk_static PUBLIC funasr_pipeline)
target_include_directories(funasr_sdk_static PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(funasr_sdk_static PUBLIC FUNASR_SDK_STATIC)
set_target_properties(funasr_sdk_static PROPERTIES OUTPUT_NAME funasr_sdk_static)

add_executable(sdk_ptt_demo examples/sdk_ptt_demo.cpp)
target_link_libraries(sdk_ptt_demo PRIVATE funasr_sdk_static)

add_executable(funasr-cli cli/funasr_cli.cpp)
target_link_libraries(funasr-cli PRIVATE funasr_pipeline)

if(FUNASR_BUILD_TESTS AND EXISTS "${CMAKE_SOURCE_DIR}/test/test_config.cpp")
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

    add_executable(test_offline_batching test/test_offline_batching.cpp)
    target_link_libraries(test_offline_batching PRIVATE funasr_pipeline)

    add_executable(test_offline_scheduler test/test_offline_scheduler.cpp)
    target_link_libraries(test_offline_scheduler PRIVATE funasr_pipeline)

    add_executable(test_realtime test/test_realtime.cpp)
    target_link_libraries(test_realtime PRIVATE funasr_pipeline)

    add_executable(test_push_to_talk test/test_push_to_talk.cpp)
    target_link_libraries(test_push_to_talk PRIVATE funasr_pipeline)

    add_executable(test_sdk_api test/test_sdk_api.cpp)
    target_link_libraries(test_sdk_api PRIVATE funasr_sdk_static)

    add_executable(test_benchmark test/test_benchmark.cpp)
    target_link_libraries(test_benchmark PRIVATE funasr_pipeline)

    add_executable(test_cli_options test/test_cli_options.cpp)
    target_link_libraries(test_cli_options PRIVATE funasr_pipeline)
endif()

if(FUNASR_CUDA)
    if(FUNASR_BUILD_TESTS AND EXISTS "${CMAKE_SOURCE_DIR}/test/test_gpu.cpp")
        add_executable(test_gpu test/test_gpu.cpp)
        target_link_libraries(test_gpu PRIVATE funasr_pipeline)

        add_executable(test_encoder_gpu test/test_encoder_gpu.cpp)
        target_link_libraries(test_encoder_gpu PRIVATE funasr_pipeline)
    endif()
endif()
