// Accuracy/performance benchmark for a directory of independent WAV requests.
// The current paged unified scheduler is used in bounded corpus batches so a
// full AISHELL split does not need to be concatenated into multi-GB host memory.
#include "compute/fbank.hpp"
#include "pipeline/offline_batching.hpp"
#include "pipeline/recognizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string model_path;
    std::string wav_dir;
    std::string output_path;
    std::string expected_ids_path;
    int limit = 0;
    int corpus_batch = 256;
    int batch_size = 48;
    int ctx_size = 4096;
    int max_tokens = 100;
    int n_threads = 4;
    int gpu_id = 0;
    int kv_block_size = 128;
    int kv_num_blocks = 112;
    int max_scheduled_tokens = 1024;
    int max_prefill_chunk_tokens = 512;
    int max_frontend_requests = 4;
    bool frontend_batching = true;
    bool frontend_prefetch = true;
    bool gpu_frontend_overlap = false;
    int frontend_bucket_window = 32;
    int mixed_graph_cache_entries = 16;
};

struct CorpusItem {
    std::string id;
    std::string path;
    int chunk_index = -1;
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s <model.bin> <wav_dir> <hypotheses.tsv> [options]\n\n"
        "Options:\n"
        "  --expected-ids <path>       Output expected IDs (default: <hyp>.ids)\n"
        "  --limit <n>                 Evaluate first N sorted WAVs (default: all)\n"
        "  --corpus-batch <n>          WAVs loaded per scheduler run (default: 256)\n"
        "  --batch-size <n>            Maximum active requests (default: 40)\n"
        "  --ctx-size <n>              Per-request context limit (default: 4096)\n"
        "  --max-tokens <n>            Maximum generated tokens (default: 100)\n"
        "  --threads <n>               CPU frontend threads (default: 4)\n"
        "  --gpu-id <n>                CUDA device (default: 0)\n"
        "  --kv-block-size <n>         Paged KV block size (default: 128)\n"
        "  --kv-num-blocks <n>         Physical KV blocks (default: 192)\n"
        "  --max-scheduled-tokens <n>  Unified step token budget (default: 1024)\n"
        "  --max-prefill-chunk-tokens <n> Prefill chunk cap (default: 512)\n"
        "  --max-frontend-requests <n> Batched frontends per step (default: 4)\n"
        "  --frontend-batching <on|off> Batch Encoder/Adaptor (default: on)\n"
        "  --frontend-prefetch <on|off> Overlap CPU Fbank (default: on)\n"
        "  --gpu-frontend-overlap <on|off> Experimental GPU stream overlap (default: off)\n"
        "  --frontend-bucket-window <n> Length lookahead (default: 32)\n"
        "  --mixed-graph-cache-entries <n> Graph LRU entries (default: 16)\n",
        argv0);
}

bool parse_positive(const char* value, int& out) {
    if (!value) return false;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > 1000000) return false;
    out = static_cast<int>(parsed);
    return true;
}

bool parse_non_negative(const char* value, int& out) {
    if (!value) return false;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed < 0 || parsed > 1000000) return false;
    out = static_cast<int>(parsed);
    return true;
}

bool parse_on_off(const char* value, bool& out) {
    if (!value) return false;
    if (std::string(value) == "on") {
        out = true;
        return true;
    }
    if (std::string(value) == "off") {
        out = false;
        return true;
    }
    return false;
}

bool parse_args(int argc, char** argv, Options& options) {
    if (argc < 4) return false;
    options.model_path = argv[1];
    options.wav_dir = argv[2];
    options.output_path = argv[3];
    options.expected_ids_path = options.output_path + ".ids";

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--expected-ids") {
            const char* parsed = value();
            if (!parsed) return false;
            options.expected_ids_path = parsed;
        } else if (arg == "--limit") {
            if (!parse_positive(value(), options.limit)) return false;
        } else if (arg == "--corpus-batch") {
            if (!parse_positive(value(), options.corpus_batch)) return false;
        } else if (arg == "--batch-size") {
            if (!parse_positive(value(), options.batch_size)) return false;
        } else if (arg == "--ctx-size") {
            if (!parse_positive(value(), options.ctx_size)) return false;
        } else if (arg == "--max-tokens") {
            if (!parse_positive(value(), options.max_tokens)) return false;
        } else if (arg == "--threads") {
            if (!parse_positive(value(), options.n_threads)) return false;
        } else if (arg == "--gpu-id") {
            if (!parse_non_negative(value(), options.gpu_id)) return false;
        } else if (arg == "--kv-block-size") {
            if (!parse_positive(value(), options.kv_block_size)) return false;
        } else if (arg == "--kv-num-blocks") {
            if (!parse_positive(value(), options.kv_num_blocks)) return false;
        } else if (arg == "--max-scheduled-tokens") {
            if (!parse_positive(value(), options.max_scheduled_tokens)) return false;
        } else if (arg == "--max-prefill-chunk-tokens") {
            if (!parse_positive(value(), options.max_prefill_chunk_tokens)) return false;
        } else if (arg == "--max-frontend-requests") {
            if (!parse_positive(value(), options.max_frontend_requests)) return false;
        } else if (arg == "--frontend-batching") {
            if (!parse_on_off(value(), options.frontend_batching)) return false;
        } else if (arg == "--frontend-prefetch") {
            if (!parse_on_off(value(), options.frontend_prefetch)) return false;
        } else if (arg == "--gpu-frontend-overlap") {
            if (!parse_on_off(value(), options.gpu_frontend_overlap)) return false;
        } else if (arg == "--frontend-bucket-window") {
            if (!parse_positive(value(), options.frontend_bucket_window)) return false;
        } else if (arg == "--mixed-graph-cache-entries") {
            if (!parse_positive(value(), options.mixed_graph_cache_entries) ||
                options.mixed_graph_cache_entries > 64) return false;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

std::vector<std::string> collect_wavs(const std::string& root) {
    std::vector<std::string> paths;
    std::error_code error;
    fs::recursive_directory_iterator iterator(root, error);
    fs::recursive_directory_iterator end;
    if (error) return paths;
    for (; iterator != end; iterator.increment(error)) {
        if (error) break;
        if (!iterator->is_regular_file()) continue;
        const fs::path path = iterator->path();
        if (path.extension() == ".wav" || path.extension() == ".WAV") {
            paths.push_back(path.string());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::string sanitize_tsv(std::string text) {
    for (char& ch : text) {
        if (ch == '\t' || ch == '\r' || ch == '\n') ch = ' ';
    }
    return text;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_args(argc, argv, options)) {
        print_usage(argv[0]);
        return 2;
    }

    std::vector<std::string> wav_paths = collect_wavs(options.wav_dir);
    if (options.limit > 0 && static_cast<size_t>(options.limit) < wav_paths.size()) {
        wav_paths.resize(static_cast<size_t>(options.limit));
    }
    if (wav_paths.empty()) {
        std::fprintf(stderr, "[AccuracyBench] no WAV files under %s\n",
                     options.wav_dir.c_str());
        return 2;
    }

    fs::path output_path(options.output_path);
    fs::path ids_path(options.expected_ids_path);
    std::error_code directory_error;
    if (!output_path.parent_path().empty()) {
        fs::create_directories(output_path.parent_path(), directory_error);
    }
    if (!ids_path.parent_path().empty()) {
        fs::create_directories(ids_path.parent_path(), directory_error);
    }
    std::ofstream output(output_path);
    std::ofstream expected_ids(ids_path);
    if (!output || !expected_ids) {
        std::fprintf(stderr, "[AccuracyBench] cannot open output files\n");
        return 2;
    }
    for (const std::string& path : wav_paths) {
        expected_ids << fs::path(path).stem().string() << '\n';
    }
    expected_ids.close();

    std::printf("[AccuracyBench] files=%zu corpus_batch=%d output=%s ids=%s\n",
                wav_paths.size(), options.corpus_batch,
                options.output_path.c_str(), options.expected_ids_path.c_str());

    funasr::OfflineBatchConfig config;
    config.batch_size = options.batch_size;
    config.ctx_size = options.ctx_size;
    config.max_tokens = options.max_tokens;
    config.n_threads = options.n_threads;
    config.use_gpu = true;
    config.gpu_id = options.gpu_id;
    config.use_paged_kv = true;
    config.kv_block_size = options.kv_block_size;
    config.kv_num_blocks = options.kv_num_blocks;
    config.enable_prefix_kv_cache = false;
    config.enable_dynamic_kv_blocks = true;
    config.use_unified_scheduler = true;
    config.max_num_scheduled_tokens = options.max_scheduled_tokens;
    config.max_prefill_chunk_tokens = options.max_prefill_chunk_tokens;
    config.max_frontend_requests_per_step = options.max_frontend_requests;
    config.enable_frontend_batching = options.frontend_batching;
    config.enable_frontend_prefetch = options.frontend_prefetch;
    config.enable_gpu_frontend_overlap = options.gpu_frontend_overlap;
    config.frontend_bucket_window = options.frontend_bucket_window;
    config.mixed_graph_cache_entries = options.mixed_graph_cache_entries;

    funasr::Recognizer recognizer;
    if (!recognizer.init(options.model_path)) {
        std::fprintf(stderr, "[AccuracyBench] model init failed: %s\n",
                     recognizer.last_error().c_str());
        return 2;
    }
    const int physical_rows = funasr::resolve_paged_kv_physical_rows(config);
    const int prompt_slots = options.batch_size + options.max_frontend_requests;
    if (!recognizer.init_gpu(options.ctx_size, options.gpu_id,
                             options.batch_size, physical_rows, prompt_slots,
                             options.gpu_frontend_overlap)) {
        std::fprintf(stderr, "[AccuracyBench] GPU init failed: %s\n",
                     recognizer.last_error().c_str());
        return 2;
    }
    std::printf("[AccuracyBench] scheduler batch=%d blocks=%d block_size=%d "
                "physical_rows=%d token_budget=%d frontend=%d batched=%d "
                "prefetch=%d gpu_overlap=%d bucket_window=%d graph_entries=%d\n",
                options.batch_size, options.kv_num_blocks,
                options.kv_block_size, physical_rows,
                options.max_scheduled_tokens, options.max_frontend_requests,
                options.frontend_batching ? 1 : 0,
                options.frontend_prefetch ? 1 : 0,
                options.gpu_frontend_overlap ? 1 : 0,
                options.frontend_bucket_window,
                options.mixed_graph_cache_entries);

    funasr::FbankProcessor wav_reader(recognizer.config().frontend);
    funasr::OfflineBatchTranscriber transcriber(recognizer);
    const int expected_sample_rate = recognizer.config().frontend.sample_rate;
    int successful = 0;
    int empty = 0;
    int load_failures = 0;
    double audio_seconds = 0.0;
    double load_seconds = 0.0;
    double inference_seconds = 0.0;
    long long frontend_batch_calls = 0;
    long long frontend_batched_requests = 0;
    long long unified_graph_hits = 0;
    long long unified_graph_misses = 0;
    long long frontend_prefetch_launches = 0;
    long long frontend_prefetch_ready_hits = 0;
    double frontend_fbank_ms = 0.0;
    double frontend_prefetch_wait_ms = 0.0;
    long long frontend_input_frames = 0;
    long long frontend_padded_frames = 0;
    long long graph_evictions = 0;
    int graph_entries_peak = 0;
    std::unordered_map<uint64_t, funasr::OfflineMixedGraphShapeStat>
        graph_shapes;
    int peak_blocks = 0;
    int ownership_errors = 0;
    const auto benchmark_begin = std::chrono::steady_clock::now();

    for (size_t group_begin = 0; group_begin < wav_paths.size();
         group_begin += static_cast<size_t>(options.corpus_batch)) {
        const size_t group_end = std::min(
            wav_paths.size(), group_begin + static_cast<size_t>(options.corpus_batch));
        std::vector<CorpusItem> items;
        std::vector<float> samples;
        std::vector<funasr::AudioChunk> chunks;
        items.reserve(group_end - group_begin);
        chunks.reserve(group_end - group_begin);

        const auto load_begin = std::chrono::steady_clock::now();
        for (size_t index = group_begin; index < group_end; ++index) {
            CorpusItem item;
            item.id = fs::path(wav_paths[index]).stem().string();
            item.path = wav_paths[index];
            std::vector<float> audio;
            int sample_rate = 0;
            if (!wav_reader.read_wav(item.path, audio, sample_rate) ||
                audio.empty() || sample_rate != expected_sample_rate) {
                ++load_failures;
                items.push_back(std::move(item));
                continue;
            }

            const size_t start = samples.size();
            samples.insert(samples.end(), audio.begin(), audio.end());
            const size_t finish = samples.size();
            funasr::AudioChunk chunk;
            chunk.start_sample = start;
            chunk.end_sample = finish;
            chunk.infer_start_sample = start;
            chunk.infer_end_sample = finish;
            chunk.start_sec = static_cast<float>(audio_seconds);
            audio_seconds += static_cast<double>(audio.size()) / sample_rate;
            chunk.end_sec = static_cast<float>(audio_seconds);
            item.chunk_index = static_cast<int>(chunks.size());
            chunks.push_back(chunk);
            items.push_back(std::move(item));
        }
        const auto load_end = std::chrono::steady_clock::now();
        load_seconds += std::chrono::duration<double>(load_end - load_begin).count();

        std::vector<funasr::OfflineChunkResult> results;
        if (!chunks.empty()) {
            const auto inference_begin = std::chrono::steady_clock::now();
            results = transcriber.transcribe(
                samples, expected_sample_rate, chunks, config, nullptr);
            const auto inference_end = std::chrono::steady_clock::now();
            inference_seconds += std::chrono::duration<double>(
                inference_end - inference_begin).count();

            const funasr::OfflineBatchStats& stats = transcriber.last_stats();
            frontend_batch_calls += stats.frontend_batch_calls;
            frontend_batched_requests += stats.frontend_batched_requests;
            unified_graph_hits += stats.unified_graph_cache_hits;
            unified_graph_misses += stats.unified_graph_cache_misses;
            frontend_prefetch_launches += stats.frontend_prefetch_launches;
            frontend_prefetch_ready_hits +=
                stats.frontend_prefetch_ready_hits;
            frontend_fbank_ms += stats.frontend_fbank_ms;
            frontend_prefetch_wait_ms +=
                stats.frontend_prefetch_wait_ms;
            frontend_input_frames += stats.frontend_input_frames;
            frontend_padded_frames += stats.frontend_padded_frames;
            graph_evictions = std::max(
                graph_evictions, stats.unified_graph_cache_evictions);
            graph_entries_peak = std::max(
                graph_entries_peak,
                stats.unified_graph_cache_entries_peak);
            for (const auto& [shape_hash, shape] : stats.unified_graph_shapes) {
                auto [it, inserted] = graph_shapes.emplace(shape_hash, shape);
                if (!inserted) {
                    it->second.calls += shape.calls;
                }
            }
            peak_blocks = std::max(peak_blocks, stats.peak_blocks_in_use);
            ownership_errors += stats.kv_ownership_errors;
        }

        std::vector<std::string> texts(chunks.size());
        std::vector<bool> result_ok(chunks.size(), false);
        for (const funasr::OfflineChunkResult& result : results) {
            if (result.id < 0 || static_cast<size_t>(result.id) >= chunks.size()) continue;
            texts[static_cast<size_t>(result.id)] = sanitize_tsv(result.text);
            result_ok[static_cast<size_t>(result.id)] = result.ok && !result.text.empty();
        }
        for (const CorpusItem& item : items) {
            if (item.chunk_index >= 0 &&
                result_ok[static_cast<size_t>(item.chunk_index)]) {
                output << item.id << '\t'
                       << texts[static_cast<size_t>(item.chunk_index)] << '\n';
                ++successful;
            } else {
                output << item.id << "\t[EMPTY]\n";
                ++empty;
            }
        }
        output.flush();
        std::printf("[AccuracyBench] progress=%zu/%zu ok=%d empty=%d load_fail=%d\n",
                    group_end, wav_paths.size(), successful, empty, load_failures);
    }

    const auto benchmark_end = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(
        benchmark_end - benchmark_begin).count();
    const long long graph_calls = unified_graph_hits + unified_graph_misses;
    const double graph_hit_rate = graph_calls > 0
        ? 100.0 * static_cast<double>(unified_graph_hits) / graph_calls
        : 0.0;
    const double prefetch_ready_rate = frontend_prefetch_launches > 0
        ? 100.0 * static_cast<double>(frontend_prefetch_ready_hits) /
            frontend_prefetch_launches
        : 0.0;
    const long long frontend_total_frames =
        frontend_input_frames + frontend_padded_frames;
    const double frontend_padding_rate = frontend_total_frames > 0
        ? 100.0 * static_cast<double>(frontend_padded_frames) /
            frontend_total_frames
        : 0.0;
    std::printf("[AccuracyBench] result total=%zu ok=%d empty=%d load_fail=%d\n",
                wav_paths.size(), successful, empty, load_failures);
    std::printf("[AccuracyBench] timing audio_sec=%.3f wall_sec=%.3f load_sec=%.3f "
                "inference_sec=%.3f rtf=%.6f audio_xrt=%.2f files_s=%.2f\n",
                audio_seconds, wall_seconds, load_seconds, inference_seconds,
                audio_seconds > 0.0 ? wall_seconds / audio_seconds : 0.0,
                wall_seconds > 0.0 ? audio_seconds / wall_seconds : 0.0,
                wall_seconds > 0.0 ? wav_paths.size() / wall_seconds : 0.0);
    std::printf("[AccuracyBench] runtime frontend_calls=%lld frontend_requests=%lld "
                "graph_hits=%lld graph_misses=%lld graph_hit_rate=%.2f%% "
                "graph_entries=%d/%d graph_evictions=%lld graph_shapes=%zu "
                "peak_blocks=%d/%d ownership_errors=%d\n",
                frontend_batch_calls, frontend_batched_requests,
                unified_graph_hits, unified_graph_misses, graph_hit_rate,
                graph_entries_peak, options.mixed_graph_cache_entries,
                graph_evictions, graph_shapes.size(),
                peak_blocks, options.kv_num_blocks, ownership_errors);
    std::printf("[AccuracyBench] frontend prefetch=%lld ready=%lld "
                "ready_rate=%.2f%% fbank_ms=%.2f wait_ms=%.2f "
                "input_frames=%lld padded_frames=%lld padding_rate=%.2f%%\n",
                frontend_prefetch_launches, frontend_prefetch_ready_hits,
                prefetch_ready_rate, frontend_fbank_ms,
                frontend_prefetch_wait_ms, frontend_input_frames,
                frontend_padded_frames, frontend_padding_rate);
    std::vector<funasr::OfflineMixedGraphShapeStat> ranked_shapes;
    ranked_shapes.reserve(graph_shapes.size());
    for (const auto& [shape_hash, shape] : graph_shapes) {
        (void)shape_hash;
        ranked_shapes.push_back(shape);
    }
    std::sort(ranked_shapes.begin(), ranked_shapes.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.calls > rhs.calls;
              });
    const size_t top_shapes = std::min<size_t>(8, ranked_shapes.size());
    for (size_t i = 0; i < top_shapes; ++i) {
        const auto& shape = ranked_shapes[i];
        std::printf("[AccuracyBench] graph_shape rank=%zu calls=%lld "
                    "prefill=%d decode=%d total=%d max_blocks=%d "
                    "max_n_kv=%d prefill_seqs=%d hash=%llu\n",
                    i + 1, shape.calls, shape.prefill_tokens,
                    shape.decode_tokens, shape.total_tokens,
                    shape.max_blocks, shape.max_n_kv,
                    shape.prefill_sequences,
                    static_cast<unsigned long long>(shape.shape_hash));
    }
    return 0;
}
