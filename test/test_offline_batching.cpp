#include "pipeline/recognizer.hpp"
#include "pipeline/chunking.hpp"
#include "pipeline/offline_batching.hpp"
#include "miniaudio.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>

namespace {

enum class TestChunkMode {
    Window,
    Vad,
};

struct TestOptions {
    std::string model_path;
    std::string audio_path;
    TestChunkMode chunk_mode = TestChunkMode::Window;
    int chunk_sec = 30;
    int batch_size = 2;
    int ctx_size = 2048;
    int max_tokens = 80;
    int max_chunks = 0;
    int n_threads = 4;
    bool use_gpu = false;
    int gpu_id = 0;
    bool use_paged_kv = false;
    int kv_block_size = 64;
    int kv_num_blocks = 0;
    bool prefix_kv_cache = false;
    bool dynamic_kv_blocks = false;
    int prefill_chunk_tokens = 0;
    bool unified_scheduler = false;
    int max_scheduled_tokens = 1024;
    int max_prefill_chunk_tokens = 512;
    int max_frontend_requests_per_step = 4;
    bool frontend_batching = true;
    bool frontend_prefetch = true;
    bool gpu_frontend_overlap = false;
    int frontend_bucket_window = 16;
    int mixed_graph_cache_entries = 8;
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s <model.bin> <audio.wav/mp3/flac> [options]\n"
        "\n"
        "Options:\n"
        "  --chunk-mode <window|vad>  Chunk planner (default: window)\n"
        "  --chunk-sec <n>            Window seconds (default: 30)\n"
        "  --batch-size <n>           Offline scheduler slots (default: 2)\n"
        "  --ctx-size <n>             KV context size (default: 2048)\n"
        "  --max-tokens <n>           Max generated tokens (default: 80)\n"
        "  --max-chunks <n>           Limit planned chunks for profiling (default: all)\n"
        "  --threads <n>              CPU threads (default: 4)\n"
        "  --gpu                      Use GPU inference\n"
        "  --kv-mode <continuous|paged> Scheduler KV mode (default: continuous)\n"
        "  --kv-block-size <n>        Paged KV block size (default: 64)\n"
        "  --kv-num-blocks <n>        Paged KV block count (default: derived)\n"
        "  --prefix-kv-cache <on|off> Task-scoped prefix KV cache (default: off)\n"
        "  --dynamic-kv-blocks <on|off> Append blocks on demand (default: off)\n"
        "  --prefill-chunk-tokens <n> Sequential prefill chunk size (default: off)\n"
        "  --unified-scheduler <on|off> Packed prefill/decode scheduler (default: off)\n"
        "  --max-scheduled-tokens <n> Unified token budget per step (default: 1024)\n"
        "  --max-prefill-chunk-tokens <n> Unified prompt chunk cap (default: 512)\n"
        "  --max-frontend-requests <n> Frontends admitted per step (default: 1)\n"
        "  --frontend-batching <on|off> Batch Encoder/Adaptor frontends (default: on)\n"
        "  --frontend-prefetch <on|off> Overlap CPU Fbank work (default: on)\n"
        "  --gpu-frontend-overlap <on|off> Experimental GPU stream overlap (default: off)\n"
        "  --frontend-bucket-window <n> Length-sort lookahead (default: 16)\n"
        "  --mixed-graph-cache-entries <n> Mixed graph LRU entries (default: 8)\n"
        "  --gpu-id <n>               CUDA device id (default: 0)\n",
        argv0);
}

bool parse_int(const std::string& value, int& out) {
    char* end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > 1000000) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

bool parse_non_negative_int(const std::string& value, int& out) {
    char* end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed < 0 || parsed > 1000000) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

bool parse_on_off(const std::string& value, bool& out) {
    if (value == "on") {
        out = true;
        return true;
    }
    if (value == "off") {
        out = false;
        return true;
    }
    return false;
}

bool parse_args(int argc, char** argv, TestOptions& opt) {
    if (argc < 3) {
        return false;
    }

    opt.model_path = argv[1];
    opt.audio_path = argv[2];

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--chunk-mode") {
            const char* value = need_value(arg.c_str());
            if (!value) return false;
            std::string mode = value;
            if (mode == "window") {
                opt.chunk_mode = TestChunkMode::Window;
            } else if (mode == "vad") {
                opt.chunk_mode = TestChunkMode::Vad;
            } else {
                std::fprintf(stderr, "Invalid --chunk-mode: %s\n", value);
                return false;
            }
        } else if (arg == "--chunk-sec") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.chunk_sec)) return false;
        } else if (arg == "--batch-size") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.batch_size)) return false;
        } else if (arg == "--ctx-size") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.ctx_size)) return false;
        } else if (arg == "--max-tokens") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.max_tokens)) return false;
        } else if (arg == "--max-chunks") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.max_chunks)) return false;
        } else if (arg == "--threads") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.n_threads)) return false;
        } else if (arg == "--gpu") {
            opt.use_gpu = true;
        } else if (arg == "--gpu-id") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.gpu_id)) return false;
        } else if (arg == "--kv-mode") {
            const char* value = need_value(arg.c_str());
            if (!value) return false;
            std::string mode = value;
            if (mode == "continuous") {
                opt.use_paged_kv = false;
            } else if (mode == "paged") {
                opt.use_paged_kv = true;
            } else {
                std::fprintf(stderr, "Invalid --kv-mode: %s\n", value);
                return false;
            }
        } else if (arg == "--kv-block-size") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.kv_block_size)) return false;
        } else if (arg == "--kv-num-blocks") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.kv_num_blocks)) return false;
        } else if (arg == "--prefix-kv-cache") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_on_off(value, opt.prefix_kv_cache)) return false;
        } else if (arg == "--dynamic-kv-blocks") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_on_off(value, opt.dynamic_kv_blocks)) return false;
        } else if (arg == "--prefill-chunk-tokens") {
            const char* value = need_value(arg.c_str());
            if (!value ||
                !parse_non_negative_int(value, opt.prefill_chunk_tokens)) return false;
        } else if (arg == "--unified-scheduler") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_on_off(value, opt.unified_scheduler)) return false;
        } else if (arg == "--max-scheduled-tokens") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.max_scheduled_tokens)) return false;
        } else if (arg == "--max-prefill-chunk-tokens") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.max_prefill_chunk_tokens)) return false;
        } else if (arg == "--max-frontend-requests") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(
                    value, opt.max_frontend_requests_per_step)) return false;
        } else if (arg == "--frontend-batching") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_on_off(value, opt.frontend_batching)) return false;
        } else if (arg == "--frontend-prefetch") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_on_off(value, opt.frontend_prefetch)) return false;
        } else if (arg == "--gpu-frontend-overlap") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_on_off(value, opt.gpu_frontend_overlap)) return false;
        } else if (arg == "--frontend-bucket-window") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.frontend_bucket_window)) return false;
        } else if (arg == "--mixed-graph-cache-entries") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_int(value, opt.mixed_graph_cache_entries) ||
                opt.mixed_graph_cache_entries > 64) return false;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    return true;
}

double estimate_kv_cache_mb(const funasr::ModelConfig& model_cfg,
                            int physical_rows) {
    const double bytes_per_f16 = 2.0;
    const double layer_count = static_cast<double>(model_cfg.llm.block_count);
    const double kv_heads = static_cast<double>(model_cfg.llm.head_count_kv);
    const double head_dim = static_cast<double>(model_cfg.llm.head_dim());
    const double rows = static_cast<double>(physical_rows);
    const double k_bytes = layer_count * rows * kv_heads * head_dim * bytes_per_f16;
    const double v_bytes = k_bytes;
    return (k_bytes + v_bytes) / (1024.0 * 1024.0);
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

} // namespace

int main(int argc, char** argv) {
    TestOptions opt;
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }

    funasr::OfflineBatchConfig cfg;
    cfg.batch_size = opt.batch_size;
    cfg.ctx_size = opt.ctx_size;
    cfg.max_tokens = opt.max_tokens;
    cfg.n_threads = opt.n_threads;
    cfg.use_gpu = opt.use_gpu;
    cfg.gpu_id = opt.gpu_id;
    cfg.use_paged_kv = opt.use_paged_kv;
    cfg.kv_block_size = opt.kv_block_size;
    cfg.kv_num_blocks = opt.kv_num_blocks;
    cfg.enable_prefix_kv_cache = opt.use_paged_kv && opt.prefix_kv_cache;
    cfg.enable_dynamic_kv_blocks = opt.use_paged_kv && opt.dynamic_kv_blocks;
    cfg.prefill_chunk_tokens = opt.prefill_chunk_tokens;
    cfg.use_unified_scheduler = opt.unified_scheduler;
    cfg.max_num_scheduled_tokens = opt.max_scheduled_tokens;
    cfg.max_prefill_chunk_tokens = opt.max_prefill_chunk_tokens;
    cfg.max_frontend_requests_per_step =
        opt.max_frontend_requests_per_step;
    cfg.enable_frontend_batching = opt.frontend_batching;
    cfg.enable_frontend_prefetch = opt.frontend_prefetch;
    cfg.enable_gpu_frontend_overlap = opt.gpu_frontend_overlap;
    cfg.frontend_bucket_window = opt.frontend_bucket_window;
    cfg.mixed_graph_cache_entries = opt.mixed_graph_cache_entries;

    funasr::Recognizer recognizer;
    if (!recognizer.init(opt.model_path)) {
        std::fprintf(stderr, "Failed to load model: %s\n",
                     recognizer.last_error().c_str());
        return 1;
    }

    if (opt.use_gpu) {
        const int physical_kv_rows = opt.use_paged_kv
            ? funasr::resolve_paged_kv_physical_rows(cfg)
            : opt.ctx_size * std::max(1, opt.batch_size);
        std::printf("[OfflineTest] init GPU ctx=%d slots=%d physical_rows=%d gpu_id=%d\n",
                    opt.ctx_size, opt.batch_size, physical_kv_rows, opt.gpu_id);
        if (!recognizer.init_gpu(
                opt.ctx_size, opt.gpu_id, opt.batch_size,
                opt.use_paged_kv ? physical_kv_rows : 0,
                opt.batch_size, opt.gpu_frontend_overlap)) {
            std::fprintf(stderr, "Failed to init GPU: %s\n",
                         recognizer.last_error().c_str());
            return 1;
        }
        const double kv_mb = estimate_kv_cache_mb(
            recognizer.config(), physical_kv_rows);
        const double weights_mb = 945.0;
        std::printf("[OfflineTest] gpu budget: ctx=%d slots=%d physical_rows=%d kv_est=%.0fMB "
                    "weights_est=%.0fMB mode=%s\n",
                    opt.ctx_size,
                    opt.batch_size,
                    physical_kv_rows,
                    kv_mb,
                    weights_mb,
                    opt.use_paged_kv ? "paged" : "continuous");
        if (kv_mb > 4800.0 || kv_mb + weights_mb > 6500.0) {
            std::printf("[OfflineTest] WARNING: requested ctx=%d slots=%d may exceed "
                        "the comfortable memory budget on 8GB GPUs; try batch=8/12 "
                        "or ctx=2048.\n",
                        opt.ctx_size,
                        opt.batch_size);
        }
    }

    std::vector<float> samples;
    uint32_t sample_rate = 0;
    if (!load_audio_file(opt.audio_path, samples, sample_rate)) {
        std::fprintf(stderr, "Failed to load audio: %s\n", opt.audio_path.c_str());
        return 1;
    }

    std::vector<funasr::AudioChunk> chunks;
    if (opt.chunk_mode == TestChunkMode::Window) {
        chunks = funasr::split_audio_by_window(
            samples, static_cast<int>(sample_rate), opt.chunk_sec);
    } else {
        chunks = funasr::split_audio_by_energy_vad(
            samples, static_cast<int>(sample_rate),
            opt.chunk_sec, 600, 200, 0.002f);
    }
    if (opt.max_chunks > 0 &&
        static_cast<size_t>(opt.max_chunks) < chunks.size()) {
        chunks.resize(static_cast<size_t>(opt.max_chunks));
    }

    float audio_sec = funasr::selected_audio_duration_seconds(
        chunks, static_cast<int>(sample_rate));
    if (chunks.empty() && sample_rate > 0) {
        audio_sec = static_cast<float>(samples.size()) / static_cast<float>(sample_rate);
    }
    std::printf("[OfflineTest] audio=%.2fs chunks=%zu mode=%s chunk_sec=%d",
                audio_sec, chunks.size(),
                opt.chunk_mode == TestChunkMode::Window ? "window" : "vad",
                opt.chunk_sec);
    if (opt.max_chunks > 0) {
        std::printf(" max_chunks=%d", opt.max_chunks);
    }
    std::printf(" prefill_chunk_tokens=%d unified=%d token_budget=%d "
                "unified_prefill_chunk=%d\n",
                cfg.prefill_chunk_tokens,
                cfg.use_unified_scheduler ? 1 : 0,
                cfg.max_num_scheduled_tokens,
                cfg.max_prefill_chunk_tokens);

    double total_encoder = 0.0;
    double total_prefill = 0.0;
    double total_decode = 0.0;
    double total_infer = 0.0;
    long long total_decode_tokens = 0;

    funasr::OfflineBatchTranscriber transcriber(recognizer);
    auto wall_start = std::chrono::high_resolution_clock::now();
    auto results = transcriber.transcribe(
        samples, static_cast<int>(sample_rate), chunks, cfg,
        [](int completed, int total, const funasr::OfflineChunkResult& result) {
            std::printf("[OfflineTest] %d/%d chunk=%d slot=%d ok=%d %.2f-%.2fs "
                        "prefill_tok=%d decode_tok=%d enc=%.0fms pf=%.0fms dec=%.0fms total=%.0fms "
                        "text='%s'\n",
                        completed, total, result.id, result.slot_id, result.ok ? 1 : 0,
                        result.start_sec, result.end_sec,
                        result.prefill_tokens, result.decode_tokens,
                        result.encoder_ms, result.prefill_ms, result.decode_ms, result.total_ms,
                        result.text.c_str());
            std::printf("[OfflineTestTokens] chunk=%d ids=", result.id);
            for (size_t i = 0; i < result.token_ids.size(); ++i) {
                if (i > 0) {
                    std::printf(",");
                }
                std::printf("%d", result.token_ids[i]);
            }
            std::printf("\n");
        });
    const funasr::OfflineBatchStats& stats = transcriber.last_stats();
    const bool chunked_prefill_enabled =
        cfg.use_gpu && cfg.use_paged_kv && cfg.enable_dynamic_kv_blocks &&
        (cfg.prefill_chunk_tokens > 0 || cfg.use_unified_scheduler);
    const int final_prompt_embedding_slots = chunked_prefill_enabled
        ? recognizer.free_prompt_embedding_slots()
        : -1;
    const int prompt_embedding_capacity = chunked_prefill_enabled
        ? recognizer.prompt_embedding_capacity()
        : -1;
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start).count());

    int ok = 0;
    std::string full_text;
    for (const auto& result : results) {
        if (result.ok) {
            ok++;
            full_text += result.text;
        }
        total_encoder += result.encoder_ms;
        total_prefill += result.prefill_ms;
        total_decode += result.decode_ms;
        total_infer += result.total_ms;
        total_decode_tokens += result.decode_tokens;
    }

    std::printf("[OfflineTest] ok=%d/%zu\n", ok, results.size());
    std::printf("[OfflineTest] wall total=%.0fms rtf=%.4f\n",
                wall_ms,
                audio_sec > 0.0f ? wall_ms / 1000.0 / audio_sec : 0.0);
    std::printf("[OfflineTest] sum per-request enc=%.0fms pf=%.0fms dec=%.0fms total=%.0fms rtf=%.4f\n",
                total_encoder, total_prefill, total_decode, total_infer,
                audio_sec > 0.0f ? total_infer / 1000.0 / audio_sec : 0.0);
    std::printf("[OfflineTest] scheduler: gpu=%d kv=%s batch=%d chunks=%d "
                "decode_steps=%d grouped_calls=%d fallback_calls=%d "
                "avg_active=%.2f blocks_peak=%d/%d block_size=%d\n",
                stats.use_gpu ? 1 : 0,
                stats.use_paged_kv ? "paged" : "continuous",
                stats.batch_size,
                stats.total_chunks,
                stats.decode_steps,
                stats.decode_group_calls,
                stats.decode_fallback_calls,
                stats.average_active_batch(),
                stats.peak_blocks_in_use,
                stats.kv_block_capacity,
                stats.kv_block_size);
    std::printf("[OfflineTest] fallback_reasons: single=%d token_id=%d "
                "host_embed=%d serial_env=%d invalid=%d\n",
                stats.fallback_reasons.single_request,
                stats.fallback_reasons.token_id_fast_path_unavailable,
                stats.fallback_reasons.host_embedding_batch_unavailable,
                stats.fallback_reasons.serial_env_forced,
                stats.fallback_reasons.invalid_paged_input);
    if (stats.use_paged_kv && stats.paged_kv_released_requests > 0) {
        std::printf("[OfflineTest] paged_kv_waste: requests=%lld "
                    "alloc_blocks=%lld used_blocks=%lld wasted_blocks=%lld "
                    "waste_rate=%.2f%% avg_alloc=%.2f avg_used=%.2f "
                    "avg_waste=%.2f max_waste=%d "
                    "alloc_pos=%lld used_pos=%lld wasted_pos=%lld\n",
                    stats.paged_kv_released_requests,
                    stats.kv_allocated_blocks_total,
                    stats.kv_used_blocks_total,
                    stats.kv_wasted_blocks_total,
                    stats.kv_block_waste_rate(),
                    stats.average_allocated_blocks(),
                    stats.average_used_blocks(),
                    stats.average_wasted_blocks(),
                    stats.kv_max_wasted_blocks,
                    stats.kv_allocated_positions_total,
                    stats.kv_used_positions_total,
                    stats.kv_wasted_positions_total);
    }
    if (stats.use_paged_kv) {
        std::printf("[OfflineTest] paged_kv_runtime: prefix_builds=%d "
                    "prefix_hits=%d reused_tokens=%lld saved_prefill_tokens=%lld "
                    "cow_copies=%d prefill_appends=%d decode_appends=%d "
                    "ownership_errors=%d final_free=%d/%d\n",
                    stats.prefix_cache_builds,
                    stats.prefix_cache_hits,
                    stats.prefix_tokens_reused,
                    stats.prefix_prefill_tokens_saved,
                    stats.prefix_cow_copies,
                    stats.dynamic_prefill_blocks,
                    stats.dynamic_decode_appends,
                    stats.kv_ownership_errors,
                    stats.kv_final_free_blocks,
                    stats.kv_block_capacity);
    }
    if (chunked_prefill_enabled) {
        std::printf("[OfflineTest] prompt_embedding_pool: free=%d/%d\n",
                    final_prompt_embedding_slots, prompt_embedding_capacity);
    }
    if (cfg.use_unified_scheduler) {
        std::printf("[OfflineTest] frontend_batch_profile: calls=%d "
                    "batched_requests=%d single_requests=%d fallbacks=%d "
                    "queue_peak=%d batch_wall=%.2fms single_wall=%.2fms "
                    "avg_batch=%.2f prefetch=%d ready=%d wait=%.2fms "
                    "fbank=%.2fms padding=%lld/%lld\n",
                    stats.frontend_batch_calls,
                    stats.frontend_batched_requests,
                    stats.frontend_single_requests,
                    stats.frontend_batch_fallbacks,
                    stats.frontend_prepared_queue_peak,
                    stats.frontend_batch_wall_ms,
                    stats.frontend_single_wall_ms,
                    stats.frontend_batch_calls > 0
                        ? static_cast<double>(stats.frontend_batched_requests) /
                              stats.frontend_batch_calls
                        : 0.0,
                    stats.frontend_prefetch_launches,
                    stats.frontend_prefetch_ready_hits,
                    stats.frontend_prefetch_wait_ms,
                    stats.frontend_fbank_ms,
                    stats.frontend_padded_frames,
                    stats.frontend_input_frames +
                        stats.frontend_padded_frames);
        std::printf("[OfflineTest] unified_profile: steps=%d mixed=%d "
                    "prefill_only=%d decode_only=%d scheduled_tokens=%lld "
                    "selected_rows=%lld staging=%.2fms compute=%.2fms "
                    "graph_hits=%lld graph_misses=%lld hit_rate=%.2f%% "
                    "entries_peak=%d/%d evictions=%lld shapes=%zu\n",
                    stats.unified_steps,
                    stats.unified_mixed_steps,
                    stats.unified_prefill_only_steps,
                    stats.unified_decode_only_steps,
                    stats.unified_scheduled_tokens,
                    stats.unified_selected_rows,
                    stats.unified_staging_ms,
                    stats.unified_compute_ms,
                    stats.unified_graph_cache_hits,
                    stats.unified_graph_cache_misses,
                    stats.unified_graph_cache_hits +
                                stats.unified_graph_cache_misses > 0
                        ? 100.0 * static_cast<double>(
                              stats.unified_graph_cache_hits) /
                              static_cast<double>(
                                  stats.unified_graph_cache_hits +
                                  stats.unified_graph_cache_misses)
                        : 0.0,
                    stats.unified_graph_cache_entries_peak,
                    stats.unified_graph_cache_capacity,
                    stats.unified_graph_cache_evictions,
                    stats.unified_graph_shapes.size());
        std::vector<funasr::OfflineMixedGraphShapeStat> graph_shapes;
        graph_shapes.reserve(stats.unified_graph_shapes.size());
        for (const auto& item : stats.unified_graph_shapes) {
            graph_shapes.push_back(item.second);
        }
        std::sort(graph_shapes.begin(), graph_shapes.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return lhs.calls > rhs.calls;
                  });
        for (size_t i = 0; i < std::min<size_t>(5, graph_shapes.size()); ++i) {
            const auto& shape = graph_shapes[i];
            std::printf("[OfflineTest] mixed_graph_shape: rank=%zu calls=%lld "
                        "hash=%llu prefill=%d decode=%d total=%d "
                        "prefill_seqs=%d max_blocks=%d max_n_kv=%d\n",
                        i + 1, shape.calls,
                        static_cast<unsigned long long>(shape.shape_hash),
                        shape.prefill_tokens, shape.decode_tokens,
                        shape.total_tokens, shape.prefill_sequences,
                        shape.max_blocks, shape.max_n_kv);
        }
    }
    std::printf("[OfflineTest] scheduler_profile: admit=%d/%d no_kv=%d "
                "admit_rounds=%d avg_admit_round=%.2f max_admit_round=%d "
                "prefill_wall=%.0fms avg_prefill=%.2fms "
                "decode_dispatch=%.0fms avg_decode_step=%.2fms idle_steps=%d\n",
                stats.admit_success,
                stats.admit_attempts,
                stats.admit_no_kv_capacity,
                stats.admit_rounds,
                stats.average_admit_round_size(),
                stats.max_admit_round_size,
                stats.prefill_wall_ms,
                stats.average_prefill_wall_ms(),
                stats.decode_dispatch_wall_ms,
                stats.average_decode_dispatch_ms(),
                stats.scheduler_idle_steps);
    const double wall_sec = wall_ms / 1000.0;
    std::printf("[OfflineTest] throughput: audio_sec=%.2f wall_sec=%.2f "
                "audio_sec/s=%.2f rtf=%.4f tokens/s=%.1f\n",
                audio_sec,
                wall_sec,
                wall_sec > 0.0 ? audio_sec / wall_sec : 0.0,
                audio_sec > 0.0f ? wall_sec / audio_sec : 0.0,
                wall_sec > 0.0 ? static_cast<double>(total_decode_tokens) / wall_sec : 0.0);
    if (opt.use_gpu && opt.use_paged_kv) {
        const funasr::PagedDecodeProfile profile = recognizer.gpu_paged_decode_profile();
        if (profile.calls > 0) {
            const double calls = static_cast<double>(profile.calls);
            std::printf("[OfflineTest] paged_profile: calls=%ld build=%.3fms "
                        "alloc=%.3fms set=%.3fms compute=%.3fms get=%.3fms total=%.3fms\n",
                        profile.calls,
                        profile.build_ms / calls,
                        profile.alloc_ms / calls,
                        profile.set_input_ms / calls,
                        profile.compute_ms / calls,
                        profile.get_ms / calls,
                        profile.avg_total_ms());
            if (profile.paged_attn_calls > 0) {
                std::printf("[OfflineTest] paged_attn_profile: calls=%ld total=%.3fms "
                            "avg_kernel=%.6fms graph_compute_share=%.2f%%\n",
                            profile.paged_attn_calls,
                            profile.paged_attn_ms,
                            profile.avg_paged_attn_ms(),
                            profile.compute_ms > 0.0
                                ? 100.0 * profile.paged_attn_ms / profile.compute_ms
                                : 0.0);
            }
            if (profile.graph_cache_probe_calls > 0) {
                std::printf("[OfflineTest] paged_graph_cache_probe: calls=%ld "
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
    }
    std::printf("[OfflineTest] text='%s'\n", full_text.c_str());
    const bool complete = results.size() == chunks.size() &&
                          ok == static_cast<int>(chunks.size());
    const bool ownership_ok = !stats.use_paged_kv ||
                              (stats.kv_ownership_errors == 0 &&
                               stats.kv_final_free_blocks == stats.kv_block_capacity);
    const bool embedding_ownership_ok = !chunked_prefill_enabled ||
        final_prompt_embedding_slots == prompt_embedding_capacity;
    return complete && ownership_ok && embedding_ownership_ok ? 0 : 2;
}
