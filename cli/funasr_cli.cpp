// funasr-cli: user-facing command line transcription tool
//
// Usage:
//   funasr-cli -m <model> -f <input> [options]
//
#include "pipeline/recognizer.hpp"
#include "compute/silero_vad.hpp"
#include "miniaudio.h"

#include <algorithm>
#include <chrono>
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
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

enum class OutputFormat {
    Text,
    Srt,
    Tsv,
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
    int n_threads = 4;
    int max_tokens = 100;

    bool use_vad = false;
    std::string vad_model_path;
    int max_segment_sec = 5;
    bool max_segment_sec_set = false;
    int min_silence_ms = 600;
    int segment_pad_ms = 200;
    float energy_threshold = 0.002f;
    funasr::SileroVadParams silero_vad;

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

struct AudioSegment {
    size_t start_sample = 0;
    size_t end_sample = 0;
    size_t infer_start_sample = 0;
    size_t infer_end_sample = 0;
    float start_sec = 0.0f;
    float end_sec = 0.0f;
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
        "\n"
        "Long audio:\n"
        "  --vad                  Enable energy VAD segmentation\n"
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
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    if (opt.model_path.empty() || opt.input_path.empty()) {
        std::fprintf(stderr, "Missing required -m/--model or -f/--file\n");
        return false;
    }
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

void add_audio_segment(std::vector<AudioSegment>& segments,
                       size_t start_sample, size_t end_sample,
                       size_t infer_start_sample, size_t infer_end_sample,
                       size_t total_samples, int sample_rate) {
    constexpr int min_segment_ms = 500;
    start_sample = std::min(start_sample, total_samples);
    end_sample = std::min(end_sample, total_samples);
    infer_start_sample = std::min(infer_start_sample, total_samples);
    infer_end_sample = std::min(infer_end_sample, total_samples);
    size_t min_samples = static_cast<size_t>(sample_rate) * min_segment_ms / 1000;
    if (end_sample <= start_sample || end_sample - start_sample < min_samples) {
        return;
    }
    if (infer_end_sample <= infer_start_sample) {
        infer_start_sample = start_sample;
        infer_end_sample = end_sample;
    }
    AudioSegment seg;
    seg.start_sample = start_sample;
    seg.end_sample = end_sample;
    seg.infer_start_sample = infer_start_sample;
    seg.infer_end_sample = infer_end_sample;
    seg.start_sec = static_cast<float>(start_sample) / sample_rate;
    seg.end_sec = static_cast<float>(end_sample) / sample_rate;
    segments.push_back(seg);
}

std::vector<AudioSegment> split_audio_by_vad(const std::vector<float>& samples,
                                             int sample_rate,
                                             int max_segment_sec,
                                             int min_silence_ms,
                                             int segment_pad_ms,
                                             float threshold) {
    std::vector<AudioSegment> segments;
    if (samples.empty()) {
        return segments;
    }

    constexpr int frame_ms = 10;
    constexpr int min_segment_ms = 500;
    int frame_samples = sample_rate * frame_ms / 1000;
    int pad_frames = std::max(0, segment_pad_ms) / frame_ms;
    int min_silence_frames = std::max(1, min_silence_ms / frame_ms);
    int max_segment_frames = std::max(1, max_segment_sec * 1000 / frame_ms);
    int min_segment_frames = std::max(1, min_segment_ms / frame_ms);

    int n_frames = static_cast<int>((samples.size() + frame_samples - 1) / frame_samples);
    std::vector<float> rms(static_cast<size_t>(n_frames), 0.0f);
    for (int i = 0; i < n_frames; i++) {
        size_t start = static_cast<size_t>(i) * frame_samples;
        size_t end = std::min(samples.size(), start + frame_samples);
        double sum = 0.0;
        for (size_t j = start; j < end; j++) {
            sum += static_cast<double>(samples[j]) * samples[j];
        }
        if (end > start) {
            rms[static_cast<size_t>(i)] = static_cast<float>(
                std::sqrt(sum / static_cast<double>(end - start)));
        }
    }

    bool in_segment = false;
    int seg_start_frame = 0;
    int last_speech_frame = 0;
    int silence_frames = 0;

    auto frame_to_sample = [&](int frame) -> size_t {
        return std::min(samples.size(), static_cast<size_t>(std::max(0, frame)) * frame_samples);
    };

    auto finish_segment = [&](int end_frame) {
        if (end_frame - seg_start_frame + 1 < min_segment_frames) {
            return;
        }
        int core_start = seg_start_frame;
        int core_end = std::min(n_frames, end_frame + 1);
        int infer_start = std::max(0, core_start - pad_frames);
        int infer_end = std::min(n_frames, core_end + pad_frames);
        add_audio_segment(segments,
                          frame_to_sample(core_start),
                          frame_to_sample(core_end),
                          frame_to_sample(infer_start),
                          frame_to_sample(infer_end),
                          samples.size(), sample_rate);
    };

    for (int i = 0; i < n_frames; i++) {
        bool speech = rms[static_cast<size_t>(i)] > threshold;
        if (speech) {
            if (!in_segment) {
                in_segment = true;
                seg_start_frame = i;
                silence_frames = 0;
            }
            last_speech_frame = i;
            silence_frames = 0;
        } else if (in_segment) {
            silence_frames++;
        }

        if (in_segment && i - seg_start_frame + 1 >= max_segment_frames) {
            int search_begin = std::min(i, seg_start_frame + min_segment_frames);
            int split_frame = search_begin;
            float min_rms = rms[static_cast<size_t>(split_frame)];
            for (int j = search_begin; j <= i; j++) {
                if (rms[static_cast<size_t>(j)] < min_rms) {
                    min_rms = rms[static_cast<size_t>(j)];
                    split_frame = j;
                }
            }
            finish_segment(split_frame);
            in_segment = true;
            seg_start_frame = std::min(i, split_frame + 1);
            last_speech_frame = i;
            silence_frames = speech ? 0 : 1;
        } else if (in_segment && silence_frames >= min_silence_frames) {
            finish_segment(last_speech_frame);
            in_segment = false;
            silence_frames = 0;
        }
    }

    if (in_segment) {
        finish_segment(last_speech_frame);
    }

    if (segments.empty()) {
        add_audio_segment(segments, 0, samples.size(), 0, samples.size(),
                          samples.size(), sample_rate);
    }
    return segments;
}

std::vector<AudioSegment> convert_silero_segments(
    const std::vector<funasr::VadSegment>& vad_segments,
    size_t total_samples,
    int sample_rate,
    float samples_overlap
) {
    std::vector<AudioSegment> segments;
    segments.reserve(vad_segments.size());
    size_t overlap_samples = static_cast<size_t>(
        std::max(0.0f, samples_overlap) * sample_rate + 0.5f);
    for (size_t i = 0; i < vad_segments.size(); i++) {
        const auto& vad_segment = vad_segments[i];
        float start_sec = std::max(0.0f, vad_segment.start_sec);
        float end_sec = std::max(start_sec, vad_segment.end_sec);

        size_t start_sample = std::min(
            total_samples,
            static_cast<size_t>(start_sec * sample_rate + 0.5f));
        size_t end_sample = std::min(
            total_samples,
            static_cast<size_t>(end_sec * sample_rate + 0.5f));
        if (end_sample <= start_sample) {
            continue;
        }

        AudioSegment segment;
        segment.start_sample = start_sample;
        segment.end_sample = end_sample;
        segment.infer_start_sample = start_sample;
        segment.infer_end_sample = i + 1 < vad_segments.size()
            ? std::min(total_samples, end_sample + overlap_samples)
            : end_sample;
        segment.start_sec = static_cast<float>(start_sample) / sample_rate;
        segment.end_sec = static_cast<float>(end_sample) / sample_rate;
        segments.push_back(segment);
    }
    return segments;
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

void print_vad_progress(int current, int total, const AudioSegment& segment, float audio_sec) {
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
        opt.use_vad = true;
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
        ScopedStdoutToStderr redirect;
        if (!recognizer.init_gpu(2048, opt.gpu_id)) {
            std::fprintf(stderr, "Warning: GPU init failed, falling back to CPU\n");
            opt.use_gpu = false;
        }
    }
    std::fprintf(stderr, "[2] Mode: %s\n", opt.use_gpu ? "GPU" : "CPU");

    funasr::SileroVAD silero_vad;
    bool use_silero_vad = opt.use_vad && !opt.vad_model_path.empty();
    if (use_silero_vad) {
        std::fprintf(stderr, "[3] Loading Silero VAD: %s\n", opt.vad_model_path.c_str());
        if (!silero_vad.init(opt.vad_model_path)) {
            std::fprintf(stderr, "Failed to init Silero VAD model\n");
            return 1;
        }
    }

    funasr::InferenceConfig config;
    config.use_gpu = opt.use_gpu;
    config.gpu_id = opt.gpu_id;
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

        if (loaded && !opt.use_vad) {
            funasr::InferenceResult result;
            {
                ScopedStdoutToStderr redirect;
                result = recognizer.transcribe_audio(samples.data(), samples.size(), config);
            }
            item.text = result.text;
        } else if (loaded && opt.use_vad) {
            std::vector<AudioSegment> segments;
            if (use_silero_vad) {
                auto vad_segments = silero_vad.detect(
                    samples.data(), samples.size(),
                    static_cast<int>(sample_rate),
                    opt.silero_vad);
                segments = convert_silero_segments(
                    vad_segments, samples.size(), static_cast<int>(sample_rate),
                    opt.silero_vad.samples_overlap);
                std::fprintf(stderr, "%s[Silero VAD] %s: %zu segments\n",
                             batch_mode ? "\n" : "", item.filename.c_str(), segments.size());
            } else {
                segments = split_audio_by_vad(samples, static_cast<int>(sample_rate),
                                              opt.max_segment_sec,
                                              opt.min_silence_ms,
                                              opt.segment_pad_ms,
                                              opt.energy_threshold);
                std::fprintf(stderr, "%s[VAD] %s: %zu segments\n",
                             batch_mode ? "\n" : "", item.filename.c_str(), segments.size());
            }

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
                    SegmentResult seg_result;
                    seg_result.text = result.text;
                    seg_result.start_sec = segment.start_sec;
                    seg_result.end_sec = segment.end_sec;
                    item.segments.push_back(seg_result);
                    item.text = append_without_overlap(item.text, result.text);
                }

                print_vad_progress(static_cast<int>(seg_idx + 1),
                                   static_cast<int>(segments.size()),
                                   segment, item.duration_sec);
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
