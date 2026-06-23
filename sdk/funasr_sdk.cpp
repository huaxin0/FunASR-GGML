#include "funasr_sdk.h"

#include "pipeline/recognizer.hpp"
#include "pipeline/realtime.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int FUNASR_OK = 0;
constexpr int FUNASR_ERR_INVALID_ARGUMENT = -1;
constexpr int FUNASR_ERR_NOT_INITIALIZED = -2;
constexpr int FUNASR_ERR_INIT_FAILED = -3;
constexpr int FUNASR_ERR_TRANSCRIBE_FAILED = -4;

struct FunasrSdkContext {
    std::mutex mutex;
    funasr::Recognizer recognizer;
    funasr::InferenceConfig inference;
    std::unique_ptr<funasr::PushToTalkRecorder> recorder;
    std::string last_error;
    int sample_rate = 16000;
    bool initialized = false;
};

FunasrSdkContext* as_context(FunasrHandle handle) {
    return static_cast<FunasrSdkContext*>(handle);
}

void set_error(FunasrSdkContext* ctx, const std::string& message) {
    if (ctx) {
        ctx->last_error = message;
    }
}

void clear_result(FunasrResult* result) {
    if (result) {
        std::memset(result, 0, sizeof(*result));
        result->first_token_ms = -1.0f;
    }
}

void clear_text(char* text_out, int text_out_size) {
    if (text_out && text_out_size > 0) {
        text_out[0] = '\0';
    }
}

int copy_text(const std::string& text, char* text_out, int text_out_size) {
    if (!text_out || text_out_size <= 0) {
        return FUNASR_ERR_INVALID_ARGUMENT;
    }
    size_t capacity = static_cast<size_t>(text_out_size);
    size_t to_copy = std::min(text.size(), capacity - 1);
    std::memcpy(text_out, text.data(), to_copy);
    text_out[to_copy] = '\0';
    return text.size() < capacity ? FUNASR_OK : 1;
}

std::string trim_ascii(std::string value) {
    size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin]))) {
        begin++;
    }
    size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        end--;
    }
    return value.substr(begin, end - begin);
}

std::vector<std::string> parse_hotwords_text(const char* text) {
    std::vector<std::string> hotwords;
    if (!text) {
        return hotwords;
    }

    std::string token;
    for (const char* p = text; *p; ++p) {
        char ch = *p;
        if (ch == '\n' || ch == '\r' || ch == ',' || ch == ';' || ch == '\t') {
            std::string word = trim_ascii(token);
            if (!word.empty()) {
                hotwords.push_back(word);
            }
            token.clear();
        } else {
            token.push_back(ch);
        }
    }

    std::string word = trim_ascii(token);
    if (!word.empty()) {
        hotwords.push_back(word);
    }
    return hotwords;
}

int transcribe_samples(FunasrSdkContext* ctx,
                       const float* samples,
                       int sample_count,
                       char* text_out,
                       int text_out_size,
                       FunasrResult* result_out,
                       int truncated) {
    clear_text(text_out, text_out_size);
    clear_result(result_out);

    if (!ctx || !samples || sample_count <= 0 || !text_out || text_out_size <= 0) {
        set_error(ctx, "invalid argument");
        return FUNASR_ERR_INVALID_ARGUMENT;
    }
    if (!ctx->initialized) {
        set_error(ctx, "SDK is not initialized");
        if (result_out) {
            result_out->audio_sec = static_cast<float>(sample_count) /
                                    static_cast<float>(std::max(1, ctx->sample_rate));
            result_out->truncated = truncated;
        }
        return FUNASR_ERR_NOT_INITIALIZED;
    }

    float first_token_ms = -1.0f;
    auto t_start = std::chrono::high_resolution_clock::now();
    funasr::InferenceResult result = ctx->recognizer.transcribe_audio(
        samples, static_cast<size_t>(sample_count), ctx->inference,
        [&](int, const std::string&, bool is_final) {
            if (!is_final && first_token_ms < 0.0f) {
                auto t_first = std::chrono::high_resolution_clock::now();
                first_token_ms = static_cast<float>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        t_first - t_start).count());
            }
        });
    auto t_end = std::chrono::high_resolution_clock::now();

    float total_ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
    float audio_sec = static_cast<float>(sample_count) /
                      static_cast<float>(std::max(1, ctx->sample_rate));

    if (result_out) {
        result_out->audio_sec = audio_sec;
        result_out->total_ms = total_ms;
        result_out->first_token_ms = first_token_ms;
        result_out->rtf = audio_sec > 0.0f ? total_ms / (audio_sec * 1000.0f) : 0.0f;
        result_out->truncated = truncated;
        result_out->decode_tokens = result.decode_tokens;
        result_out->prefill_tokens = result.prefill_tokens;
    }

    if (result.text.empty()) {
        set_error(ctx, "transcription produced empty text");
        return FUNASR_ERR_TRANSCRIBE_FAILED;
    }

    int copy_rc = copy_text(result.text, text_out, text_out_size);
    if (copy_rc < 0) {
        set_error(ctx, "invalid text output buffer");
        return copy_rc;
    }
    ctx->last_error.clear();
    return copy_rc;
}

} // namespace

extern "C" {

void funasr_get_default_config(FunasrConfig* config) {
    if (!config) return;
    config->model_path = nullptr;
    config->use_gpu = 0;
    config->gpu_id = 0;
    config->ctx_size = 4096;
    config->max_new_tokens = 220;
    config->max_record_ms = 60000;
    config->sample_rate = 16000;
    config->n_threads = 4;
}

FunasrHandle funasr_create(void) {
    auto* ctx = new FunasrSdkContext();
    FunasrConfig config;
    funasr_get_default_config(&config);
    ctx->sample_rate = config.sample_rate;
    ctx->inference.max_new_tokens = config.max_new_tokens;
    ctx->inference.kv_cache_size = config.ctx_size;
    ctx->inference.n_threads = config.n_threads;
    ctx->recorder = std::make_unique<funasr::PushToTalkRecorder>(
        config.sample_rate, config.max_record_ms);
    return ctx;
}

void funasr_destroy(FunasrHandle handle) {
    delete as_context(handle);
}

int funasr_init(FunasrHandle handle, const FunasrConfig* config) {
    auto* ctx = as_context(handle);
    if (!ctx || !config || !config->model_path || config->model_path[0] == '\0') {
        set_error(ctx, "model_path is required");
        return FUNASR_ERR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->initialized = false;
    ctx->sample_rate = config->sample_rate > 0 ? config->sample_rate : 16000;

    int ctx_size = config->ctx_size > 0 ? config->ctx_size : 4096;
    int max_new_tokens = config->max_new_tokens > 0 ? config->max_new_tokens : 220;
    int max_record_ms = config->max_record_ms > 0 ? config->max_record_ms : 60000;
    int n_threads = config->n_threads > 0 ? config->n_threads : 4;
    int gpu_id = config->gpu_id >= 0 ? config->gpu_id : 0;

    ctx->recorder = std::make_unique<funasr::PushToTalkRecorder>(
        ctx->sample_rate, max_record_ms);
    ctx->inference.max_new_tokens = max_new_tokens;
    ctx->inference.kv_cache_size = ctx_size;
    ctx->inference.n_threads = n_threads;
    ctx->inference.use_gpu = config->use_gpu != 0;
    ctx->inference.gpu_id = gpu_id;

    if (!ctx->recognizer.init(config->model_path, false)) {
        set_error(ctx, ctx->recognizer.last_error());
        return FUNASR_ERR_INIT_FAILED;
    }

    if (ctx->inference.use_gpu) {
        if (!ctx->recognizer.init_gpu(ctx_size, gpu_id, 1)) {
            set_error(ctx, ctx->recognizer.last_error());
            return FUNASR_ERR_INIT_FAILED;
        }
    }

    ctx->initialized = true;
    ctx->last_error.clear();
    return FUNASR_OK;
}

const char* funasr_last_error(FunasrHandle handle) {
    auto* ctx = as_context(handle);
    if (!ctx) return "invalid handle";
    return ctx->last_error.c_str();
}

int funasr_set_hotwords(FunasrHandle handle, const char* hotwords_utf8) {
    auto* ctx = as_context(handle);
    if (!ctx || !hotwords_utf8) {
        set_error(ctx, "invalid argument");
        return FUNASR_ERR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->inference.prompt.hotwords = parse_hotwords_text(hotwords_utf8);
    if (!ctx->inference.prompt.hotwords.empty()) {
        ctx->inference.prompt.language = "中文";
        ctx->inference.prompt.itn = true;
    }
    ctx->last_error.clear();
    return FUNASR_OK;
}

int funasr_load_hotwords_file(FunasrHandle handle, const char* path_utf8) {
    auto* ctx = as_context(handle);
    if (!ctx || !path_utf8 || path_utf8[0] == '\0') {
        set_error(ctx, "invalid argument");
        return FUNASR_ERR_INVALID_ARGUMENT;
    }

    std::ifstream in(path_utf8, std::ios::binary);
    if (!in) {
        set_error(ctx, std::string("failed to open hotwords file: ") + path_utf8);
        return FUNASR_ERR_INVALID_ARGUMENT;
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();
    return funasr_set_hotwords(handle, buffer.str().c_str());
}

int funasr_ptt_start(FunasrHandle handle) {
    auto* ctx = as_context(handle);
    if (!ctx || !ctx->recorder) {
        set_error(ctx, "invalid handle");
        return FUNASR_ERR_INVALID_ARGUMENT;
    }
    ctx->recorder->start();
    ctx->last_error.clear();
    return FUNASR_OK;
}

int funasr_ptt_feed_f32(FunasrHandle handle, const float* samples, int sample_count) {
    auto* ctx = as_context(handle);
    if (!ctx || !ctx->recorder || !samples || sample_count < 0) {
        set_error(ctx, "invalid argument");
        return FUNASR_ERR_INVALID_ARGUMENT;
    }
    ctx->recorder->feed_audio(samples, static_cast<size_t>(sample_count));
    return FUNASR_OK;
}

int funasr_ptt_feed_i16(FunasrHandle handle, const int16_t* samples, int sample_count) {
    auto* ctx = as_context(handle);
    if (!ctx || !ctx->recorder || !samples || sample_count < 0) {
        set_error(ctx, "invalid argument");
        return FUNASR_ERR_INVALID_ARGUMENT;
    }

    std::vector<float> converted(static_cast<size_t>(sample_count));
    for (int i = 0; i < sample_count; i++) {
        converted[static_cast<size_t>(i)] = static_cast<float>(samples[i]) / 32768.0f;
    }
    ctx->recorder->feed_audio(converted.data(), converted.size());
    return FUNASR_OK;
}

int funasr_ptt_stop_and_transcribe(FunasrHandle handle,
                                   char* text_out,
                                   int text_out_size,
                                   FunasrResult* result_out) {
    auto* ctx = as_context(handle);
    if (!ctx || !ctx->recorder) {
        clear_text(text_out, text_out_size);
        clear_result(result_out);
        set_error(ctx, "invalid handle");
        return FUNASR_ERR_INVALID_ARGUMENT;
    }

    std::vector<float> audio = ctx->recorder->stop();
    int truncated = ctx->recorder->was_truncated() ? 1 : 0;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    return transcribe_samples(ctx, audio.data(), static_cast<int>(audio.size()),
                              text_out, text_out_size, result_out, truncated);
}

int funasr_transcribe_f32(FunasrHandle handle,
                          const float* samples,
                          int sample_count,
                          char* text_out,
                          int text_out_size,
                          FunasrResult* result_out) {
    auto* ctx = as_context(handle);
    if (!ctx) {
        clear_text(text_out, text_out_size);
        clear_result(result_out);
        return FUNASR_ERR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    return transcribe_samples(ctx, samples, sample_count,
                              text_out, text_out_size, result_out, 0);
}

} // extern "C"
