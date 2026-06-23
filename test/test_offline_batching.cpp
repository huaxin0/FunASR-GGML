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
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    return true;
}

double estimate_kv_cache_mb(const funasr::ModelConfig& model_cfg,
                            int ctx_size,
                            int slots) {
    const double bytes_per_f16 = 2.0;
    const double layer_count = static_cast<double>(model_cfg.llm.block_count);
    const double kv_heads = static_cast<double>(model_cfg.llm.head_count_kv);
    const double head_dim = static_cast<double>(model_cfg.llm.head_dim());
    const double rows = static_cast<double>(ctx_size) * static_cast<double>(slots);
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

    funasr::Recognizer recognizer;
    if (!recognizer.init(opt.model_path)) {
        std::fprintf(stderr, "Failed to load model: %s\n",
                     recognizer.last_error().c_str());
        return 1;
    }

    if (opt.use_gpu) {
        std::printf("[OfflineTest] init GPU ctx=%d slots=%d gpu_id=%d\n",
                    opt.ctx_size, opt.batch_size, opt.gpu_id);
        if (!recognizer.init_gpu(opt.ctx_size, opt.gpu_id, opt.batch_size)) {
            std::fprintf(stderr, "Failed to init GPU: %s\n",
                         recognizer.last_error().c_str());
            return 1;
        }
        const double kv_mb = estimate_kv_cache_mb(
            recognizer.config(), opt.ctx_size, opt.batch_size);
        const double weights_mb = 945.0;
        std::printf("[OfflineTest] gpu budget: ctx=%d slots=%d kv_est=%.0fMB "
                    "weights_est=%.0fMB mode=%s\n",
                    opt.ctx_size,
                    opt.batch_size,
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

    float audio_sec = sample_rate > 0
        ? static_cast<float>(samples.size()) / static_cast<float>(sample_rate)
        : 0.0f;
    std::printf("[OfflineTest] audio=%.2fs chunks=%zu mode=%s chunk_sec=%d",
                audio_sec, chunks.size(),
                opt.chunk_mode == TestChunkMode::Window ? "window" : "vad",
                opt.chunk_sec);
    if (opt.max_chunks > 0) {
        std::printf(" max_chunks=%d", opt.max_chunks);
    }
    std::printf("\n");

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
        });
    const funasr::OfflineBatchStats& stats = transcriber.last_stats();
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
    return ok > 0 ? 0 : 2;
}
