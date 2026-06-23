#ifndef FUNASR_SDK_H
#define FUNASR_SDK_H

#include <stdint.h>

#if defined(_WIN32) && !defined(FUNASR_SDK_STATIC)
#  ifdef FUNASR_SDK_BUILD
#    define FUNASR_SDK_API __declspec(dllexport)
#  else
#    define FUNASR_SDK_API __declspec(dllimport)
#  endif
#else
#  define FUNASR_SDK_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* FunasrHandle;

typedef struct FunasrConfig {
    const char* model_path;
    int use_gpu;
    int gpu_id;
    int ctx_size;
    int max_new_tokens;
    int max_record_ms;
    int sample_rate;
    int n_threads;
} FunasrConfig;

typedef struct FunasrResult {
    float audio_sec;
    float total_ms;
    float first_token_ms;
    float rtf;
    int truncated;
    int decode_tokens;
    int prefill_tokens;
} FunasrResult;

FUNASR_SDK_API void funasr_get_default_config(FunasrConfig* config);

FUNASR_SDK_API FunasrHandle funasr_create(void);
FUNASR_SDK_API void funasr_destroy(FunasrHandle handle);

FUNASR_SDK_API int funasr_init(FunasrHandle handle, const FunasrConfig* config);
FUNASR_SDK_API const char* funasr_last_error(FunasrHandle handle);

FUNASR_SDK_API int funasr_set_hotwords(FunasrHandle handle, const char* hotwords_utf8);
FUNASR_SDK_API int funasr_load_hotwords_file(FunasrHandle handle, const char* path_utf8);

FUNASR_SDK_API int funasr_ptt_start(FunasrHandle handle);
FUNASR_SDK_API int funasr_ptt_feed_f32(FunasrHandle handle,
                                       const float* samples,
                                       int sample_count);
FUNASR_SDK_API int funasr_ptt_feed_i16(FunasrHandle handle,
                                       const int16_t* samples,
                                       int sample_count);
FUNASR_SDK_API int funasr_ptt_stop_and_transcribe(FunasrHandle handle,
                                                  char* text_out,
                                                  int text_out_size,
                                                  FunasrResult* result_out);

FUNASR_SDK_API int funasr_transcribe_f32(FunasrHandle handle,
                                         const float* samples,
                                         int sample_count,
                                         char* text_out,
                                         int text_out_size,
                                         FunasrResult* result_out);

#ifdef __cplusplus
}
#endif

#endif // FUNASR_SDK_H
