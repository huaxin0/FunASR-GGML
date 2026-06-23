// funasr-cli: user-facing command line transcription tool
//
// Usage:
//   funasr-cli -m <model> -f <input> [options]
//
#include "pipeline/recognizer.hpp"
#include "pipeline/chunking.hpp"
#include "pipeline/offline_batching.hpp"
#include "compute/silero_vad.hpp"
#include "miniaudio.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <cstdlib>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

enum class OutputFormat {
    Text,
    Srt,
    Tsv,
};

enum class ChunkMode {
    None,
    Window,
    Vad,
};

enum class OfflineKVMode {
    Continuous,
    Paged,
};

struct Options {
    std::string model_path;
    std::string input_path;
    std::string output_path;
    bool output_path_set = false;
    bool format_set = false;
    OutputFormat format = OutputFormat::Text;
    int srt_max_chars = 20;

    bool use_gpu = false;
    int gpu_id = 0;
    int ctx_size = 2048;
    int n_threads = 4;
    int max_tokens = 100;

    bool use_vad = false;
    bool chunk_mode_set = false;
    ChunkMode chunk_mode = ChunkMode::None;
    int chunk_sec = 30;
    std::string vad_model_path;
    int max_segment_sec = 5;
    bool max_segment_sec_set = false;
    int min_silence_ms = 600;
    int segment_pad_ms = 200;
    float energy_threshold = 0.002f;
    funasr::SileroVadParams silero_vad;

    bool offline_scheduler = false;
    bool offline_profile = false;
    bool offline_paged_opts = true;
    int offline_batch_size = 12;
    OfflineKVMode offline_kv_mode = OfflineKVMode::Continuous;
    int offline_kv_block_size = 64;
    int offline_kv_num_blocks = 0;

    bool help = false;
};

struct SegmentResult {
    std::string text;
    float start_sec = 0.0f;
    float end_sec = 0.0f;
};

struct TranscriptionItem {
    std::string path;
    std::string filename;
    std::string utt_id;
    std::string text;
    float duration_sec = 0.0f;
    float inference_ms = 0.0f;
    std::vector<SegmentResult> segments;
    bool ok = false;
};

class ScopedStdoutToStderr {
public:
    ScopedStdoutToStderr() {
#ifndef _WIN32
        std::fflush(stdout);
        saved_stdout_ = dup(fileno(stdout));
        if (saved_stdout_ >= 0) {
            dup2(fileno(stderr), fileno(stdout));
            active_ = true;
        }
#endif
    }

    ~ScopedStdoutToStderr() {
#ifndef _WIN32
        if (active_) {
            std::fflush(stdout);
            dup2(saved_stdout_, fileno(stdout));
            close(saved_stdout_);
        }
#endif
    }

    ScopedStdoutToStderr(const ScopedStdoutToStderr&) = delete;
    ScopedStdoutToStderr& operator=(const ScopedStdoutToStderr&) = delete;

private:
#ifndef _WIN32
    int saved_stdout_ = -1;
    bool active_ = false;
#endif
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s -m <model> -f <input> [options]\n"
        "\n"
        "Required:\n"
        "  -m, --model <path>     Model file path (GGUF)\n"
        "  -f, --file <path>      Input WAV file or directory\n"
        "\n"
        "Output:\n"
        "  -otxt                  Output plain text (.txt)\n"
        "  -osrt                  Output SRT subtitles (.srt)\n"
        "  -otsv                  Output TSV (utt_id<TAB>text)\n"
        "  -o, --output <path>    Output file path (format inferred by extension)\n"
        "  --srt-max-chars <n>    Max UTF-8 chars per SRT subtitle (default: 20)\n"
        "\n"
        "GPU:\n"
        "  --gpu                  Enable GPU inference\n"
        "  --gpu-id <id>          CUDA device ID (default: 0)\n"
        "  --ctx-size <n>         LLM KV cache context size (default: 2048)\n"
        "\n"
        "Long audio:\n"
        "  --chunk-mode <mode>    Chunking mode: none, window, vad (default: none)\n"
        "  --chunk-sec <n>        Fixed window size in seconds for --chunk-mode window (default: 30)\n"
        "  --vad                  Shortcut for --chunk-mode vad\n"
        "  --vad-model <path>     Use Silero VAD model file for segmentation\n"
        "  --vad-threshold <f>    Silero speech probability threshold (default: 0.5)\n"
        "  --vad-min-speech-ms <n> Silero minimum speech duration (default: 250)\n"
        "  --vad-min-silence-ms <n> Silero minimum silence duration (default: 100)\n"
        "  --vad-max-speech-sec <f> Silero maximum speech duration (default: 30)\n"
        "  --vad-speech-pad-ms <n> Silero speech padding in milliseconds (default: 30)\n"
        "  --max-segment-sec <n>  Max segment duration in seconds (default: 5)\n"
        "  --min-silence-ms <n>   Minimum silence for splitting (default: 600)\n"
        "  --segment-pad-ms <n>   Audio padding around VAD segments (default: 200)\n"
        "  --energy-threshold <f> RMS speech threshold (default: 0.002)\n"
        "\n"
        "Offline scheduler:\n"
        "  --offline-scheduler    Use continuous offline scheduler for chunked audio\n"
        "  --offline-profile      Print scheduler and paged decode profile lines\n"
        "  --offline-preset <n>   Preset: long-video\n"
        "  --no-offline-paged-opts Disable default paged decode optimization env flags\n"
        "  --batch-size <n>       Offline scheduler active request limit (default: 12)\n"
        "  --kv-mode <mode>       Offline KV mode: continuous, paged (default: continuous)\n"
        "  --kv-block-size <n>    Paged KV block size (default: 64)\n"
        "  --kv-num-blocks <n>    Paged KV block count (default: derived)\n"
        "\n"
        "Other:\n"
        "  -t, --threads <n>      CPU threads (default: 4)\n"
        "  --max-tokens <n>       Max generated tokens (default: 100)\n"
        "  -h, --help             Show this help\n",
        argv0);
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool has_wav_extension(const fs::path& path) {
    return to_lower(path.extension().string()) == ".wav";
}

bool has_audio_extension(const fs::path& path) {
    std::string ext = to_lower(path.extension().string());
    return ext == ".wav" || ext == ".mp3" || ext == ".flac";
}

std::string get_utterance_id(const std::string& path) {
    return fs::path(path).stem().string();
}

OutputFormat infer_format_from_extension(const std::string& path) {
    std::string ext = to_lower(fs::path(path).extension().string());
    if (ext == ".srt") return OutputFormat::Srt;
    if (ext == ".tsv") return OutputFormat::Tsv;
    return OutputFormat::Text;
}

std::string extension_for_format(OutputFormat format) {
    switch (format) {
        case OutputFormat::Srt: return ".srt";
        case OutputFormat::Tsv: return ".tsv";
        case OutputFormat::Text:
        default: return ".txt";
    }
}

std::string default_output_path(const std::string& input_path, OutputFormat format, bool batch_mode) {
    fs::path in(input_path);
    fs::path out;
    if (batch_mode) {
        out = in;
        out += extension_for_format(format);
    } else {
        out = in;
        out.replace_extension(extension_for_format(format));
    }
    return out.string();
}

bool parse_positive_int(const std::string& value, int& out) {
    char* end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > 1000000) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

bool parse_nonnegative_int(const std::string& value, int& out) {
    char* end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed < 0 || parsed > 1000000) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

bool parse_positive_float(const std::string& value, float& out) {
    char* end = nullptr;
    float parsed = std::strtof(value.c_str(), &end);
    if (!end || *end != '\0' || parsed <= 0.0f) {
        return false;
    }
    out = parsed;
    return true;
}

bool parse_chunk_mode(const std::string& value, ChunkMode& out) {
    std::string mode = to_lower(value);
    if (mode == "none") {
        out = ChunkMode::None;
        return true;
    }
    if (mode == "window") {
        out = ChunkMode::Window;
        return true;
    }
    if (mode == "vad") {
        out = ChunkMode::Vad;
        return true;
    }
    return false;
}

bool parse_offline_kv_mode(const std::string& value, OfflineKVMode& out) {
    std::string mode = to_lower(value);
    if (mode == "continuous") {
        out = OfflineKVMode::Continuous;
        return true;
    }
    if (mode == "paged") {
        out = OfflineKVMode::Paged;
        return true;
    }
    return false;
}

bool apply_offline_preset(const std::string& value, Options& opt) {
    std::string preset = to_lower(value);
    if (preset != "long-video") {
        return false;
    }

    opt.offline_scheduler = true;
    opt.offline_profile = true;
    opt.offline_batch_size = 12;
    opt.offline_kv_mode = OfflineKVMode::Paged;
    opt.offline_kv_block_size = 128;
    opt.ctx_size = 4096;
    opt.chunk_mode = ChunkMode::Window;
    opt.chunk_mode_set = true;
    opt.chunk_sec = 30;
    opt.max_tokens = 220;
    return true;
}

const char* chunk_mode_name(ChunkMode mode) {
    switch (mode) {
        case ChunkMode::Window: return "window";
        case ChunkMode::Vad:    return "vad";
        case ChunkMode::None:
        default:                return "none";
    }
}

const char* offline_kv_mode_name(OfflineKVMode mode) {
    switch (mode) {
        case OfflineKVMode::Paged:      return "paged";
        case OfflineKVMode::Continuous:
        default:                        return "continuous";
    }
}

bool should_use_offline_scheduler(const Options& opt, bool loaded, size_t chunk_count) {
    return opt.offline_scheduler && opt.chunk_mode != ChunkMode::None && loaded && chunk_count > 0;
}

bool should_enable_offline_paged_opts(const Options& opt) {
    return opt.offline_paged_opts &&
           opt.use_gpu &&
           opt.offline_scheduler &&
           opt.chunk_mode != ChunkMode::None &&
           opt.offline_kv_mode == OfflineKVMode::Paged;
}

int gpu_init_slots(const Options& opt) {
    if (opt.offline_scheduler && opt.chunk_mode != ChunkMode::None) {
        return std::max(1, opt.offline_batch_size);
    }
    return 1;
}

funasr::OfflineBatchConfig make_offline_batch_config(const Options& opt) {
    funasr::OfflineBatchConfig cfg;
    cfg.batch_size = opt.offline_batch_size;
    cfg.ctx_size = opt.ctx_size;
    cfg.max_tokens = opt.max_tokens;
    cfg.n_threads = opt.n_threads;
    cfg.use_gpu = opt.use_gpu;
    cfg.gpu_id = opt.gpu_id;
    cfg.use_paged_kv = opt.offline_kv_mode == OfflineKVMode::Paged;
    cfg.kv_block_size = opt.offline_kv_block_size;
    cfg.kv_num_blocks = opt.offline_kv_num_blocks;
    return cfg;
}

void apply_offline_paged_env_defaults(const Options& opt) {
    if (!should_enable_offline_paged_opts(opt)) {
        return;
    }
#ifndef _WIN32
    setenv("FUNASR_PAGED_KV_WRITE_OP", "1", 0);
    setenv("FUNASR_PAGED_DECODE_BUCKET_MAX_KV", "1", 0);
    setenv("FUNASR_PAGED_DECODE_GRAPH_CACHE", "1", 0);
#endif
}

bool parse_args(int argc, char* argv[], Options& opt) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            opt.help = true;
            return true;
        } else if (arg == "-m" || arg == "--model") {
            const char* value = need_value(arg.c_str());
            if (!value) return false;
            opt.model_path = value;
        } else if (arg == "-f" || arg == "--file") {
            const char* value = need_value(arg.c_str());
            if (!value) return false;
            opt.input_path = value;
        } else if (arg == "-o" || arg == "--output") {
            const char* value = need_value(arg.c_str());
            if (!value) return false;
            opt.output_path = value;
            opt.output_path_set = true;
            if (!opt.format_set) {
                opt.format = infer_format_from_extension(opt.output_path);
            }
        } else if (arg == "-otxt") {
            opt.format = OutputFormat::Text;
            opt.format_set = true;
        } else if (arg == "-osrt") {
            opt.format = OutputFormat::Srt;
            opt.format_set = true;
        } else if (arg == "-otsv") {
            opt.format = OutputFormat::Tsv;
            opt.format_set = true;
        } else if (arg == "--srt-max-chars") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.srt_max_chars)) {
                std::fprintf(stderr, "Invalid --srt-max-chars value\n");
                return false;
            }
        } else if (arg == "--gpu") {
            opt.use_gpu = true;
        } else if (arg == "--gpu-id") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_nonnegative_int(value, opt.gpu_id)) {
                std::fprintf(stderr, "Invalid --gpu-id value\n");
                return false;
            }
        } else if (arg == "--ctx-size") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.ctx_size)) {
                std::fprintf(stderr, "Invalid --ctx-size value\n");
                return false;
            }
        } else if (arg == "-t" || arg == "--threads") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.n_threads)) {
                std::fprintf(stderr, "Invalid thread count\n");
                return false;
            }
        } else if (arg == "--max-tokens") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.max_tokens)) {
                std::fprintf(stderr, "Invalid --max-tokens value\n");
                return false;
            }
        } else if (arg == "--chunk-mode") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_chunk_mode(value, opt.chunk_mode)) {
                std::fprintf(stderr, "Invalid --chunk-mode value: use none, window, or vad\n");
                return false;
            }
            opt.chunk_mode_set = true;
        } else if (arg == "--chunk-sec") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.chunk_sec)) {
                std::fprintf(stderr, "Invalid --chunk-sec value\n");
                return false;
            }
        } else if (arg == "--vad") {
            opt.use_vad = true;
        } else if (arg == "--vad-model") {
            const char* value = need_value(arg.c_str());
            if (!value) return false;
            opt.vad_model_path = value;
        } else if (arg == "--vad-threshold") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_float(value, opt.silero_vad.threshold)) {
                std::fprintf(stderr, "Invalid --vad-threshold value\n");
                return false;
            }
        } else if (arg == "--vad-min-speech-ms") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.silero_vad.min_speech_duration_ms)) {
                std::fprintf(stderr, "Invalid --vad-min-speech-ms value\n");
                return false;
            }
        } else if (arg == "--vad-min-silence-ms") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.silero_vad.min_silence_duration_ms)) {
                std::fprintf(stderr, "Invalid --vad-min-silence-ms value\n");
                return false;
            }
        } else if (arg == "--vad-max-speech-sec") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_float(value, opt.silero_vad.max_speech_duration_s)) {
                std::fprintf(stderr, "Invalid --vad-max-speech-sec value\n");
                return false;
            }
        } else if (arg == "--vad-speech-pad-ms") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_nonnegative_int(value, opt.silero_vad.speech_pad_ms)) {
                std::fprintf(stderr, "Invalid --vad-speech-pad-ms value\n");
                return false;
            }
        } else if (arg == "--max-segment-sec") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.max_segment_sec)) {
                std::fprintf(stderr, "Invalid --max-segment-sec value\n");
                return false;
            }
            opt.max_segment_sec_set = true;
        } else if (arg == "--min-silence-ms") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.min_silence_ms)) {
                std::fprintf(stderr, "Invalid --min-silence-ms value\n");
                return false;
            }
        } else if (arg == "--segment-pad-ms") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_nonnegative_int(value, opt.segment_pad_ms)) {
                std::fprintf(stderr, "Invalid --segment-pad-ms value\n");
                return false;
            }
        } else if (arg == "--energy-threshold") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_float(value, opt.energy_threshold)) {
                std::fprintf(stderr, "Invalid --energy-threshold value\n");
                return false;
            }
        } else if (arg == "--offline-scheduler") {
            opt.offline_scheduler = true;
        } else if (arg == "--offline-profile") {
            opt.offline_profile = true;
        } else if (arg == "--no-offline-paged-opts") {
            opt.offline_paged_opts = false;
        } else if (arg == "--offline-preset") {
            const char* value = need_value(arg.c_str());
            if (!value || !apply_offline_preset(value, opt)) {
                std::fprintf(stderr, "Invalid --offline-preset value: use long-video\n");
                return false;
            }
        } else if (arg == "--batch-size") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.offline_batch_size)) {
                std::fprintf(stderr, "Invalid --batch-size value\n");
                return false;
            }
        } else if (arg == "--kv-mode") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_offline_kv_mode(value, opt.offline_kv_mode)) {
                std::fprintf(stderr, "Invalid --kv-mode value: use continuous or paged\n");
                return false;
            }
        } else if (arg == "--kv-block-size") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.offline_kv_block_size)) {
                std::fprintf(stderr, "Invalid --kv-block-size value\n");
                return false;
            }
        } else if (arg == "--kv-num-blocks") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_nonnegative_int(value, opt.offline_kv_num_blocks)) {
                std::fprintf(stderr, "Invalid --kv-num-blocks value\n");
                return false;
            }
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    if (opt.model_path.empty() || opt.input_path.empty()) {
        std::fprintf(stderr, "Missing required -m/--model or -f/--file\n");
        return false;
    }
    if (!opt.chunk_mode_set) {
        if (opt.use_vad || !opt.vad_model_path.empty()) {
            opt.chunk_mode = ChunkMode::Vad;
        }
    }
    opt.use_vad = opt.chunk_mode == ChunkMode::Vad;
    return true;
}

std::vector<std::string> collect_audio_files(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file() && has_audio_extension(entry.path())) {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

uint32_t read_u32_le(const unsigned char* data) {
    return static_cast<uint32_t>(data[0])
         | (static_cast<uint32_t>(data[1]) << 8)
         | (static_cast<uint32_t>(data[2]) << 16)
         | (static_cast<uint32_t>(data[3]) << 24);
}

float get_wav_duration_sec(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return 0.0f;
    }

    unsigned char riff_header[12] = {};
    in.read(reinterpret_cast<char*>(riff_header), sizeof(riff_header));
    if (in.gcount() < static_cast<std::streamsize>(sizeof(riff_header)) ||
        std::string(reinterpret_cast<char*>(riff_header), 4) != "RIFF" ||
        std::string(reinterpret_cast<char*>(riff_header + 8), 4) != "WAVE") {
        return 0.0f;
    }

    uint32_t sample_rate = 0;
    uint32_t data_bytes = 0;

    while (in.good()) {
        unsigned char chunk_header[8] = {};
        in.read(reinterpret_cast<char*>(chunk_header), sizeof(chunk_header));
        if (in.gcount() < static_cast<std::streamsize>(sizeof(chunk_header))) {
            break;
        }

        std::string chunk_id(reinterpret_cast<char*>(chunk_header), 4);
        uint32_t chunk_size = read_u32_le(chunk_header + 4);

        if (chunk_id == "fmt ") {
            std::vector<unsigned char> fmt(chunk_size);
            in.read(reinterpret_cast<char*>(fmt.data()), chunk_size);
            if (fmt.size() >= 8) {
                sample_rate = read_u32_le(fmt.data() + 4);
            }
        } else if (chunk_id == "data") {
            data_bytes = chunk_size;
            break;
        } else {
            in.seekg(chunk_size, std::ios::cur);
        }

        if (chunk_size % 2 == 1) {
            in.seekg(1, std::ios::cur);
        }
    }

    if (sample_rate == 0 || data_bytes == 0) {
        return 0.0f;
    }
    return static_cast<float>(data_bytes) / (static_cast<float>(sample_rate) * 2.0f);
}

bool load_audio_file(const std::string& path, std::vector<float>& samples, uint32_t& sample_rate) {
    samples.clear();
    sample_rate = 16000;

    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, sample_rate);
    ma_decoder decoder;
    if (ma_decoder_init_file(path.c_str(), &config, &decoder) != MA_SUCCESS) {
        return false;
    }

    ma_uint64 total_frames = 0;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames) == MA_SUCCESS &&
        total_frames > 0) {
        samples.resize(static_cast<size_t>(total_frames));
        ma_uint64 frames_read = 0;
        ma_result result = ma_decoder_read_pcm_frames(
            &decoder, samples.data(), total_frames, &frames_read);
        samples.resize(static_cast<size_t>(frames_read));
        ma_decoder_uninit(&decoder);
        return result == MA_SUCCESS && frames_read > 0;
    }

    constexpr ma_uint64 chunk_frames = 16000;
    std::vector<float> chunk(static_cast<size_t>(chunk_frames));
    while (true) {
        ma_uint64 frames_read = 0;
        ma_result result = ma_decoder_read_pcm_frames(
            &decoder, chunk.data(), chunk_frames, &frames_read);
        if (frames_read > 0) {
            samples.insert(samples.end(), chunk.begin(), chunk.begin() + frames_read);
        }
        if (result != MA_SUCCESS || frames_read == 0) {
            break;
        }
    }

    ma_decoder_uninit(&decoder);
    return !samples.empty();
}

std::string format_timestamp(float seconds) {
    if (seconds < 0.0f) seconds = 0.0f;

    int total_ms = static_cast<int>(seconds * 1000.0f + 0.5f);
    int ms = total_ms % 1000;
    int total_sec = total_ms / 1000;
    int sec = total_sec % 60;
    int total_min = total_sec / 60;
    int min = total_min % 60;
    int hour = total_min / 60;

    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(2) << hour << ":"
        << std::setw(2) << min << ":"
        << std::setw(2) << sec << ","
        << std::setw(3) << ms;
    return oss.str();
}

std::string append_without_overlap(const std::string& base, const std::string& next) {
    if (base.empty()) return next;
    if (next.empty()) return base;

    size_t max_overlap = std::min(base.size(), next.size());
    for (size_t len = max_overlap; len > 0; len--) {
        if (base.compare(base.size() - len, len, next, 0, len) == 0) {
            return base + next.substr(len);
        }
    }
    return base + next;
}

std::string trim_ascii_space(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
        begin++;
    }
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        end--;
    }
    return s.substr(begin, end - begin);
}

std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if ((c & 0x80) == 0) {
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        }
        if (i + len > s.size()) {
            len = 1;
        }
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

int utf8_char_count(const std::string& s) {
    return static_cast<int>(utf8_chars(s).size());
}

bool is_one_of(const std::string& ch, const std::vector<std::string>& values) {
    return std::find(values.begin(), values.end(), ch) != values.end();
}

std::string normalize_for_repeat_compare(const std::string& text) {
    const std::vector<std::string> ignored = {
        " ", "\t", "\n", "\r", "，", "。", "！", "？", "；", "：", "、",
        ",", ".", "!", "?", ";", ":"
    };

    std::string normalized;
    for (const auto& ch : utf8_chars(text)) {
        if (!is_one_of(ch, ignored)) {
            normalized += ch;
        }
    }
    return normalized;
}

bool chars_equal_range(const std::vector<std::string>& chars,
                       size_t a, size_t b, size_t len) {
    if (a + len > chars.size() || b + len > chars.size()) {
        return false;
    }
    for (size_t i = 0; i < len; i++) {
        if (chars[a + i] != chars[b + i]) {
            return false;
        }
    }
    return true;
}

std::string collapse_repeated_char_runs(const std::string& text) {
    auto chars = utf8_chars(text);
    if (chars.size() < 12) {
        return text;
    }

    std::string out;
    for (size_t i = 0; i < chars.size();) {
        bool collapsed = false;
        size_t max_unit = std::min<size_t>(32, (chars.size() - i) / 3);
        for (size_t unit = 4; unit <= max_unit; unit++) {
            size_t repeats = 1;
            while (i + (repeats + 1) * unit <= chars.size() &&
                   chars_equal_range(chars, i, i + repeats * unit, unit)) {
                repeats++;
            }
            if (repeats >= 3) {
                for (size_t k = 0; k < unit; k++) {
                    out += chars[i + k];
                }
                i += repeats * unit;
                collapsed = true;
                break;
            }
        }
        if (!collapsed) {
            out += chars[i++];
        }
    }
    return out;
}

std::string collapse_adjacent_repeated_clauses(const std::string& text) {
    const std::vector<std::string> delimiters = {
        "。", "！", "？", "；", "，", "、", "：",
        ".", "!", "?", ";", ",", ":"
    };

    std::vector<std::string> clauses;
    std::string current;
    for (const auto& ch : utf8_chars(text)) {
        current += ch;
        if (is_one_of(ch, delimiters)) {
            clauses.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        clauses.push_back(current);
    }

    if (clauses.size() < 2) {
        return text;
    }

    std::string out;
    std::string prev_norm;
    int dropped = 0;
    for (const auto& clause : clauses) {
        std::string norm = normalize_for_repeat_compare(clause);
        if (!norm.empty() && norm == prev_norm && utf8_char_count(norm) >= 4) {
            dropped++;
            continue;
        }
        out += clause;
        prev_norm = norm;
    }

    return dropped > 0 ? out : text;
}

std::string cleanup_repeated_text(const std::string& text) {
    std::string cleaned = collapse_adjacent_repeated_clauses(text);
    cleaned = collapse_repeated_char_runs(cleaned);
    cleaned = collapse_adjacent_repeated_clauses(cleaned);
    return trim_ascii_space(cleaned);
}

std::vector<std::string> split_by_punctuation(
    const std::string& text,
    const std::vector<std::string>& delimiters,
    int min_chars
) {
    std::vector<std::string> result;
    std::string current;
    int current_chars = 0;

    for (const auto& ch : utf8_chars(text)) {
        current += ch;
        current_chars++;
        if (is_one_of(ch, delimiters) && current_chars >= min_chars) {
            result.push_back(current);
            current.clear();
            current_chars = 0;
        }
    }

    if (!current.empty()) {
        if (!result.empty() && current_chars < min_chars) {
            result.back() += current;
        } else {
            result.push_back(current);
        }
    }

    if (result.empty()) {
        result.push_back(text);
    }
    return result;
}

// 按中文标点拆分文本，返回子段文本列表
std::vector<std::string> split_subtitle_text(const std::string& text, int max_chars = 20) {
    if (text.empty() || utf8_char_count(text) <= max_chars) {
        return {text};
    }

    constexpr int min_chars = 4;
    constexpr int comma_split_threshold = 30;
    const std::vector<std::string> sentence_delims = {"。", "！", "？", "；"};
    const std::vector<std::string> comma_delims = {"，", "、", "："};

    std::vector<std::string> sentence_parts =
        split_by_punctuation(text, sentence_delims, min_chars);

    std::vector<std::string> result;
    for (const auto& part : sentence_parts) {
        if (utf8_char_count(part) > comma_split_threshold) {
            auto comma_parts = split_by_punctuation(part, comma_delims, min_chars);
            result.insert(result.end(), comma_parts.begin(), comma_parts.end());
        } else {
            result.push_back(part);
        }
    }

    return result.empty() ? std::vector<std::string>{text} : result;
}

void write_text(std::ostream& out, const std::vector<TranscriptionItem>& items) {
    for (const auto& item : items) {
        if (item.ok) {
            out << item.text << "\n";
        }
    }
}

void write_tsv(std::ostream& out, const std::vector<TranscriptionItem>& items) {
    for (const auto& item : items) {
        if (item.ok) {
            out << item.utt_id << "\t" << item.text << "\n";
        }
    }
}

void write_srt_segment(std::ostream& out, int& index,
                       float start_sec, float end_sec,
                       const std::string& text, int max_chars) {
    if (text.empty()) {
        return;
    }

    auto sub_texts = split_subtitle_text(text, max_chars);
    if (sub_texts.size() <= 1) {
        out << index++ << "\n";
        out << format_timestamp(start_sec) << " --> "
            << format_timestamp(end_sec) << "\n";
        out << text << "\n\n";
        return;
    }

    int total_chars = utf8_char_count(text);
    float total_duration = std::max(0.0f, end_sec - start_sec);
    float cursor = start_sec;

    for (size_t k = 0; k < sub_texts.size(); k++) {
        int sub_chars = utf8_char_count(sub_texts[k]);
        float sub_duration = total_chars > 0
            ? total_duration * static_cast<float>(sub_chars) / static_cast<float>(total_chars)
            : 0.0f;
        float sub_start = cursor;
        float sub_end = (k + 1 < sub_texts.size())
            ? cursor + sub_duration
            : end_sec;

        out << index++ << "\n";
        out << format_timestamp(sub_start) << " --> "
            << format_timestamp(sub_end) << "\n";
        out << sub_texts[k] << "\n\n";

        cursor = sub_end;
    }
}

void write_srt(std::ostream& out, const std::vector<TranscriptionItem>& items, int max_chars) {
    int index = 1;
    for (const auto& item : items) {
        if (!item.ok) continue;
        if (!item.segments.empty()) {
            for (const auto& segment : item.segments) {
                write_srt_segment(out, index,
                                  segment.start_sec, segment.end_sec,
                                  segment.text, max_chars);
            }
        } else {
            write_srt_segment(out, index, 0.0f, item.duration_sec,
                              item.text, max_chars);
        }
    }
}

void write_results(std::ostream& out, OutputFormat format,
                   const std::vector<TranscriptionItem>& items,
                   int srt_max_chars) {
    switch (format) {
        case OutputFormat::Srt:
            write_srt(out, items, srt_max_chars);
            break;
        case OutputFormat::Tsv:
            write_tsv(out, items);
            break;
        case OutputFormat::Text:
        default:
            write_text(out, items);
            break;
    }
}

void print_batch_progress(int current, int total, int success, int failed, float total_ms) {
    float progress = total > 0 ? current * 100.0f / total : 100.0f;
    float avg_ms = success > 0 ? total_ms / success : 0.0f;
    std::fprintf(stderr, "\r[%d/%d] %.1f%% | OK: %d, Failed: %d | Avg: %.0f ms/file",
                 current, total, progress, success, failed, avg_ms);
    std::fflush(stderr);
}

void print_vad_progress(int current, int total, const funasr::AudioChunk& segment, float audio_sec) {
    float done = audio_sec > 0.0f ? segment.end_sec * 100.0f / audio_sec : 100.0f;
    if (done > 100.0f) done = 100.0f;
    std::fprintf(stderr, "\r[段 %d/%d] %s - %s | 已完成 %.1f%%",
                 current, total,
                 format_timestamp(segment.start_sec).substr(0, 8).c_str(),
                 format_timestamp(segment.end_sec).substr(0, 8).c_str(),
                 done);
    std::fflush(stderr);
}

void print_summary(int total, int success, int failed, float total_sec, float total_ms,
                   const std::string& output_path) {
    std::fprintf(stderr, "\n\n");
    std::fprintf(stderr, "========================================\n");
    std::fprintf(stderr, "  FunASR CLI Summary\n");
    std::fprintf(stderr, "========================================\n");
    std::fprintf(stderr, "  Total files:    %d\n", total);
    std::fprintf(stderr, "  Success:        %d\n", success);
    std::fprintf(stderr, "  Failed:         %d\n", failed);
    std::fprintf(stderr, "  Total time:     %.1f sec\n", total_sec);
    std::fprintf(stderr, "  Avg per file:   %.0f ms\n", success > 0 ? total_ms / success : 0.0f);
    std::fprintf(stderr, "  Throughput:     %.2f files/sec\n", total_sec > 0.0f ? total / total_sec : 0.0f);
    if (!output_path.empty()) {
        std::fprintf(stderr, "  Output:         %s\n", output_path.c_str());
    }
    std::fprintf(stderr, "========================================\n");
}

void print_offline_profile(const funasr::OfflineBatchStats& stats,
                           const funasr::Recognizer& recognizer,
                           const Options& opt) {
    std::fprintf(stderr,
                 "[OfflineCLI] offline_stats: chunks=%d batch=%d kv=%s "
                 "block_size=%d blocks_peak=%d/%d decode_steps=%d "
                 "grouped_calls=%d fallback_calls=%d avg_active=%.2f\n",
                 stats.total_chunks,
                 stats.batch_size,
                 stats.use_paged_kv ? "paged" : "continuous",
                 stats.kv_block_size,
                 stats.peak_blocks_in_use,
                 stats.kv_block_capacity,
                 stats.decode_steps,
                 stats.decode_group_calls,
                 stats.decode_fallback_calls,
                 stats.average_active_batch());

    std::fprintf(stderr,
                 "[OfflineCLI] fallback_reasons: single=%d token_id=%d "
                 "host_embed=%d serial_env=%d invalid=%d\n",
                 stats.fallback_reasons.single_request,
                 stats.fallback_reasons.token_id_fast_path_unavailable,
                 stats.fallback_reasons.host_embedding_batch_unavailable,
                 stats.fallback_reasons.serial_env_forced,
                 stats.fallback_reasons.invalid_paged_input);

    std::fprintf(stderr,
                 "[OfflineCLI] scheduler_profile: admit=%d/%d no_kv=%d "
                 "admit_rounds=%d avg_admit_round=%.2f max_admit_round=%d "
                 "avg_prefill_wall=%.3fms avg_decode_dispatch=%.3fms idle_steps=%d\n",
                 stats.admit_success,
                 stats.admit_attempts,
                 stats.admit_no_kv_capacity,
                 stats.admit_rounds,
                 stats.average_admit_round_size(),
                 stats.max_admit_round_size,
                 stats.average_prefill_wall_ms(),
                 stats.average_decode_dispatch_ms(),
                 stats.scheduler_idle_steps);

    if (!(opt.use_gpu && stats.use_paged_kv)) {
        return;
    }

    const funasr::PagedDecodeProfile profile = recognizer.gpu_paged_decode_profile();
    if (profile.calls <= 0) {
        return;
    }

    const double calls = static_cast<double>(profile.calls);
    std::fprintf(stderr,
                 "[OfflineCLI] paged_profile: calls=%ld build=%.3fms "
                 "alloc=%.3fms set=%.3fms compute=%.3fms get=%.3fms total=%.3fms\n",
                 profile.calls,
                 profile.build_ms / calls,
                 profile.alloc_ms / calls,
                 profile.set_input_ms / calls,
                 profile.compute_ms / calls,
                 profile.get_ms / calls,
                 profile.avg_total_ms());

    if (profile.paged_attn_calls > 0) {
        std::fprintf(stderr,
                     "[OfflineCLI] paged_attn_profile: calls=%ld total=%.3fms "
                     "avg_kernel=%.6fms graph_compute_share=%.2f%%\n",
                     profile.paged_attn_calls,
                     profile.paged_attn_ms,
                     profile.avg_paged_attn_ms(),
                     profile.compute_ms > 0.0
                        ? 100.0 * profile.paged_attn_ms / profile.compute_ms
                        : 0.0);
    }

    if (profile.graph_cache_probe_calls > 0) {
        std::fprintf(stderr,
                     "[OfflineCLI] paged_graph_cache_probe: calls=%ld "
                     "shape_hits=%ld shape_hit_rate=%.2f%% "
                     "param_hits=%ld param_hit_rate=%.2f%% "
                     "full_hits=%ld full_hit_rate=%.2f%% "
                     "cache_hits=%ld cache_misses=%ld cache_hit_rate=%.2f%%\n",
                     profile.graph_cache_probe_calls,
                     profile.shape_cache_probe_hits,
                     profile.shape_cache_probe_hit_rate(),
                     profile.param_cache_probe_hits,
                     profile.param_cache_probe_hit_rate(),
                     profile.full_graph_cache_probe_hits,
                     profile.full_graph_cache_probe_hit_rate(),
                     profile.graph_cache_hits,
                     profile.graph_cache_misses,
                     profile.graph_cache_hit_rate());
    }
}

} // namespace

int main(int argc, char* argv[]) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }
    if (opt.help) {
        print_usage(argv[0]);
        return 0;
    }

    if (!fs::exists(opt.model_path)) {
        std::fprintf(stderr, "Model file not found: %s\n", opt.model_path.c_str());
        return 1;
    }
    if (!fs::exists(opt.input_path)) {
        std::fprintf(stderr, "Input path not found: %s\n", opt.input_path.c_str());
        return 1;
    }
    if (!opt.vad_model_path.empty()) {
        if (!fs::exists(opt.vad_model_path)) {
            std::fprintf(stderr, "VAD model file not found: %s\n", opt.vad_model_path.c_str());
            return 1;
        }
    }

    bool batch_mode = fs::is_directory(opt.input_path);
    std::vector<std::string> audio_files;
    if (batch_mode) {
        audio_files = collect_audio_files(opt.input_path);
        if (audio_files.empty()) {
            std::fprintf(stderr, "No audio files found in: %s\n", opt.input_path.c_str());
            return 1;
        }
    } else {
        if (!fs::is_regular_file(opt.input_path) || !has_audio_extension(opt.input_path)) {
            std::fprintf(stderr, "Input file must be .wav, .mp3, or .flac: %s\n",
                         opt.input_path.c_str());
            return 1;
        }
        audio_files.push_back(opt.input_path);
    }

    bool write_to_file = opt.output_path_set || opt.format_set;
    if (write_to_file && !opt.output_path_set) {
        opt.output_path = default_output_path(opt.input_path, opt.format, batch_mode);
        opt.output_path_set = true;
    }

    std::fprintf(stderr, "[1] Loading model: %s\n", opt.model_path.c_str());
    funasr::Recognizer recognizer;
    {
        ScopedStdoutToStderr redirect;
        if (!recognizer.init(opt.model_path, false)) {
            std::fprintf(stderr, "Failed to init model: %s\n", recognizer.last_error().c_str());
            return 1;
        }
    }

    if (opt.use_gpu) {
        std::fprintf(stderr, "[2] Initializing GPU device %d...\n", opt.gpu_id);
        int gpu_slots = gpu_init_slots(opt);
        ScopedStdoutToStderr redirect;
        if (!recognizer.init_gpu(opt.ctx_size, opt.gpu_id, gpu_slots)) {
            std::fprintf(stderr, "Warning: GPU init failed, falling back to CPU\n");
            opt.use_gpu = false;
        }
    }
    std::fprintf(stderr, "[2] Mode: %s\n", opt.use_gpu ? "GPU" : "CPU");
    apply_offline_paged_env_defaults(opt);

    std::fprintf(stderr, "[3] Chunk mode: %s", chunk_mode_name(opt.chunk_mode));
    if (opt.chunk_mode == ChunkMode::Window) {
        std::fprintf(stderr, " (%d sec)", opt.chunk_sec);
    }
    std::fprintf(stderr, "\n");

    funasr::SileroVAD silero_vad;
    bool use_silero_vad = opt.chunk_mode == ChunkMode::Vad && !opt.vad_model_path.empty();
    if (use_silero_vad) {
        std::fprintf(stderr, "[4] Loading Silero VAD: %s\n", opt.vad_model_path.c_str());
        if (!silero_vad.init(opt.vad_model_path)) {
            std::fprintf(stderr, "Failed to init Silero VAD model\n");
            return 1;
        }
    }

    funasr::InferenceConfig config;
    config.use_gpu = opt.use_gpu;
    config.gpu_id = opt.gpu_id;
    config.kv_cache_size = opt.ctx_size;
    config.n_threads = opt.n_threads;
    config.max_new_tokens = opt.max_tokens;

    std::vector<TranscriptionItem> items;
    items.reserve(audio_files.size());

    int success = 0;
    int failed = 0;
    float total_inference_ms = 0.0f;
    auto t_all_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < audio_files.size(); i++) {
        const std::string& audio_path = audio_files[i];
        TranscriptionItem item;
        item.path = audio_path;
        item.filename = fs::path(audio_path).filename().string();
        item.utt_id = get_utterance_id(audio_path);

        auto t_start = std::chrono::high_resolution_clock::now();
        std::vector<float> samples;
        uint32_t sample_rate = 0;
        bool loaded = load_audio_file(audio_path, samples, sample_rate);
        item.duration_sec = loaded && sample_rate > 0
            ? static_cast<float>(samples.size()) / sample_rate
            : 0.0f;

        if (loaded && opt.chunk_mode == ChunkMode::None) {
            if (opt.offline_scheduler) {
                std::fprintf(stderr,
                             "Warning: --offline-scheduler requested with "
                             "--chunk-mode none; using single-call transcription\n");
            }
            funasr::InferenceResult result;
            {
                ScopedStdoutToStderr redirect;
                result = recognizer.transcribe_audio(samples.data(), samples.size(), config);
            }
            item.text = cleanup_repeated_text(result.text);
        } else if (loaded) {
            std::vector<funasr::AudioChunk> segments;
            if (opt.chunk_mode == ChunkMode::Window) {
                segments = funasr::split_audio_by_window(
                    samples, static_cast<int>(sample_rate), opt.chunk_sec);
                std::fprintf(stderr, "%s[Window] %s: %zu chunks (%d sec)\n",
                             batch_mode ? "\n" : "", item.filename.c_str(),
                             segments.size(), opt.chunk_sec);
            } else if (use_silero_vad) {
                auto vad_segments = silero_vad.detect(
                    samples.data(), samples.size(),
                    static_cast<int>(sample_rate),
                    opt.silero_vad);
                segments = funasr::convert_silero_segments_to_chunks(
                    vad_segments, samples.size(), static_cast<int>(sample_rate),
                    opt.silero_vad.samples_overlap);
                std::fprintf(stderr, "%s[Silero VAD] %s: %zu segments\n",
                             batch_mode ? "\n" : "", item.filename.c_str(), segments.size());
            } else if (opt.chunk_mode == ChunkMode::Vad) {
                segments = funasr::split_audio_by_energy_vad(samples, static_cast<int>(sample_rate),
                                              opt.max_segment_sec,
                                              opt.min_silence_ms,
                                              opt.segment_pad_ms,
                                              opt.energy_threshold);
                std::fprintf(stderr, "%s[VAD] %s: %zu segments\n",
                             batch_mode ? "\n" : "", item.filename.c_str(), segments.size());
            } else {
                segments = funasr::split_audio_by_window(
                    samples, static_cast<int>(sample_rate), opt.chunk_sec);
            }

            int max_llm_segment_sec = std::max(5, opt.ctx_size / 128);
            size_t before_split = segments.size();
            segments = funasr::split_chunks_by_max_duration(
                segments, samples.size(), static_cast<int>(sample_rate), max_llm_segment_sec);
            if (segments.size() != before_split) {
                std::fprintf(stderr,
                             "[VAD] Split long segments for LLM context: %zu -> %zu "
                             "(max %d sec)\n",
                             before_split, segments.size(), max_llm_segment_sec);
            }

            if (should_use_offline_scheduler(opt, loaded, segments.size())) {
                if (opt.offline_kv_mode == OfflineKVMode::Paged && !opt.use_gpu) {
                    std::fprintf(stderr,
                                 "Warning: --kv-mode %s requested without GPU; "
                                 "using CPU scheduler compatibility path\n",
                                 offline_kv_mode_name(opt.offline_kv_mode));
                }

                funasr::OfflineBatchConfig offline_cfg = make_offline_batch_config(opt);
                funasr::OfflineBatchTranscriber transcriber(recognizer);
                std::vector<funasr::OfflineChunkResult> offline_results;
                {
                    ScopedStdoutToStderr redirect;
                    offline_results = transcriber.transcribe(
                        samples, static_cast<int>(sample_rate), segments, offline_cfg);
                }

                for (const auto& result : offline_results) {
                    if (!result.ok || result.text.empty()) {
                        continue;
                    }
                    std::string cleaned_text = cleanup_repeated_text(result.text);
                    if (cleaned_text.empty()) {
                        continue;
                    }
                    SegmentResult seg_result;
                    seg_result.text = cleaned_text;
                    seg_result.start_sec = result.start_sec;
                    seg_result.end_sec = result.end_sec;
                    item.segments.push_back(seg_result);
                    item.text = append_without_overlap(item.text, cleaned_text);
                }

                if (opt.offline_profile) {
                    print_offline_profile(transcriber.last_stats(), recognizer, opt);
                }
            } else {
                for (size_t seg_idx = 0; seg_idx < segments.size(); seg_idx++) {
                    const auto& segment = segments[seg_idx];
                    size_t seg_len = segment.infer_end_sample > segment.infer_start_sample
                        ? segment.infer_end_sample - segment.infer_start_sample
                        : 0;
                    if (seg_len == 0) continue;

                    std::vector<float> segment_samples(
                        samples.begin() + static_cast<std::ptrdiff_t>(segment.infer_start_sample),
                        samples.begin() + static_cast<std::ptrdiff_t>(segment.infer_end_sample));

                    funasr::InferenceResult result;
                    {
                        ScopedStdoutToStderr redirect;
                        result = recognizer.transcribe_audio(
                            segment_samples.data(), segment_samples.size(), config);
                    }

                    if (!result.text.empty()) {
                        std::string cleaned_text = cleanup_repeated_text(result.text);
                        if (cleaned_text.empty()) {
                            continue;
                        }
                        SegmentResult seg_result;
                        seg_result.text = cleaned_text;
                        seg_result.start_sec = segment.start_sec;
                        seg_result.end_sec = segment.end_sec;
                        item.segments.push_back(seg_result);
                        item.text = append_without_overlap(item.text, cleaned_text);
                    }

                    print_vad_progress(static_cast<int>(seg_idx + 1),
                                       static_cast<int>(segments.size()),
                                       segment, item.duration_sec);
                }
            }
            std::fprintf(stderr, "\n");
        }
        auto t_end = std::chrono::high_resolution_clock::now();

        item.inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_end - t_start).count();
        item.ok = !item.text.empty();

        if (item.ok) {
            success++;
            total_inference_ms += item.inference_ms;
        } else {
            failed++;
            std::fprintf(stderr, "\nWarning: failed to transcribe %s\n", audio_path.c_str());
        }

        if (!write_to_file && item.ok) {
            if (batch_mode) {
                std::cout << item.filename << "\t" << item.text << "\n";
            } else {
                std::cout << item.text << "\n";
            }
        }

        items.push_back(std::move(item));

        if (batch_mode) {
            print_batch_progress(static_cast<int>(i + 1), static_cast<int>(audio_files.size()),
                                 success, failed, total_inference_ms);
        }
    }

    auto t_all_end = std::chrono::high_resolution_clock::now();
    float total_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_all_end - t_all_start).count() / 1000.0f;

    if (write_to_file) {
        std::ofstream out(opt.output_path);
        if (!out.is_open()) {
            std::fprintf(stderr, "Cannot open output file: %s\n", opt.output_path.c_str());
            return 1;
        }
        write_results(out, opt.format, items, opt.srt_max_chars);
        out.close();
    }

    if (batch_mode) {
        print_summary(static_cast<int>(audio_files.size()), success, failed, total_sec,
                      total_inference_ms, write_to_file ? opt.output_path : "");
    } else if (write_to_file) {
        std::fprintf(stderr, "Output: %s\n", opt.output_path.c_str());
    }

    return failed > 0 ? 1 : 0;
}
