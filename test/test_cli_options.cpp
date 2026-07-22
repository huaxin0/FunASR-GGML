#define FUNASR_CLI_TESTING
#define main funasr_cli_main_for_test
#include "../cli/funasr_cli.cpp"
#undef main

#include <cstdio>
#include <cstring>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::printf("FAIL: %s\n", msg); \
            return false; \
        } \
        std::printf("PASS: %s\n", msg); \
    } while (0)

namespace {

char* arg(char* value) {
    return value;
}

bool parse_with_args(std::initializer_list<char*> raw_args, Options& opt) {
    std::vector<char*> args(raw_args);
    return parse_args(static_cast<int>(args.size()), args.data(), opt);
}

bool test_offline_flags_parse() {
    std::printf("\n--- CLI offline flags parse ---\n");
    Options opt;
    bool ok = parse_with_args({
        arg((char*)"funasr-cli"),
        arg((char*)"-m"), arg((char*)"FunAsr_q8.bin"),
        arg((char*)"-f"), arg((char*)"zh.wav"),
        arg((char*)"--offline-scheduler"),
        arg((char*)"--offline-profile"),
        arg((char*)"--batch-size"), arg((char*)"12"),
        arg((char*)"--kv-mode"), arg((char*)"paged"),
        arg((char*)"--kv-block-size"), arg((char*)"128"),
        arg((char*)"--kv-num-blocks"), arg((char*)"384"),
        arg((char*)"--unified-scheduler"), arg((char*)"on"),
        arg((char*)"--max-scheduled-tokens"), arg((char*)"768"),
        arg((char*)"--max-prefill-chunk-tokens"), arg((char*)"192"),
        arg((char*)"--max-frontend-requests"), arg((char*)"3"),
        arg((char*)"--frontend-batching"), arg((char*)"off"),
        arg((char*)"--frontend-prefetch"), arg((char*)"off"),
        arg((char*)"--gpu-frontend-overlap"), arg((char*)"on"),
        arg((char*)"--frontend-bucket-window"), arg((char*)"24"),
        arg((char*)"--mixed-graph-cache-entries"), arg((char*)"12"),
        arg((char*)"--offline-auto-tune"), arg((char*)"on"),
    }, opt);

    TEST_ASSERT(ok, "offline flags parse successfully");
    TEST_ASSERT(opt.offline_scheduler, "offline scheduler enabled");
    TEST_ASSERT(opt.offline_profile, "offline profile enabled");
    TEST_ASSERT(opt.offline_batch_size == 12, "batch size parsed");
    TEST_ASSERT(opt.offline_kv_mode == OfflineKVMode::Paged, "paged kv mode parsed");
    TEST_ASSERT(opt.offline_kv_block_size == 128, "kv block size parsed");
    TEST_ASSERT(opt.offline_kv_num_blocks == 384, "kv block count parsed");
    TEST_ASSERT(opt.unified_scheduler, "unified scheduler parsed");
    TEST_ASSERT(opt.max_scheduled_tokens == 768,
                "scheduled token budget parsed");
    TEST_ASSERT(opt.max_prefill_chunk_tokens == 192,
                "prefill chunk budget parsed");
    TEST_ASSERT(opt.max_frontend_requests_per_step == 3,
                "frontend request budget parsed");
    TEST_ASSERT(!opt.frontend_batching, "frontend batching flag parsed");
    TEST_ASSERT(!opt.frontend_prefetch, "frontend prefetch flag parsed");
    TEST_ASSERT(opt.gpu_frontend_overlap,
                "GPU frontend overlap flag parsed");
    TEST_ASSERT(opt.frontend_bucket_window == 24,
                "frontend bucket window parsed");
    TEST_ASSERT(opt.mixed_graph_cache_entries == 12,
                "mixed graph cache capacity parsed");
    TEST_ASSERT(opt.offline_auto_tune, "offline auto tune parsed");
    return true;
}

bool test_long_video_preset() {
    std::printf("\n--- CLI long-video preset ---\n");
    Options opt;
    bool ok = parse_with_args({
        arg((char*)"funasr-cli"),
        arg((char*)"-m"), arg((char*)"FunAsr_q8.bin"),
        arg((char*)"-f"), arg((char*)"video.wav"),
        arg((char*)"--gpu"),
        arg((char*)"--offline-preset"), arg((char*)"long-video"),
    }, opt);

    TEST_ASSERT(ok, "long-video preset parses");
    TEST_ASSERT(opt.offline_scheduler, "preset enables offline scheduler");
    TEST_ASSERT(opt.offline_profile, "preset enables offline profile");
    TEST_ASSERT(opt.offline_batch_size == 40, "preset batch size is 40");
    TEST_ASSERT(opt.offline_kv_mode == OfflineKVMode::Paged, "preset uses paged kv");
    TEST_ASSERT(opt.offline_kv_block_size == 128, "preset block size is 128");
    TEST_ASSERT(opt.ctx_size == 4096, "preset ctx size is 4096");
    TEST_ASSERT(opt.chunk_mode == ChunkMode::Window, "preset uses window chunks");
    TEST_ASSERT(opt.chunk_sec == 30, "preset chunk size is 30 seconds");
    TEST_ASSERT(opt.max_tokens == 220, "preset max tokens is 220");
    TEST_ASSERT(opt.offline_kv_num_blocks == 192,
                "preset uses measured physical KV pool");
    TEST_ASSERT(opt.prefix_kv_cache,
                "preset enables task prefix KV reuse");
    TEST_ASSERT(opt.dynamic_kv_blocks, "preset enables dynamic KV blocks");
    TEST_ASSERT(opt.unified_scheduler,
                "preset enables unified scheduler");
    TEST_ASSERT(opt.max_frontend_requests_per_step == 4,
                "preset keeps tuned frontend budget");
    TEST_ASSERT(opt.frontend_batching,
                "preset enables batched frontend by default");
    TEST_ASSERT(opt.frontend_prefetch,
                "preset enables Fbank prefetch");
    TEST_ASSERT(!opt.gpu_frontend_overlap,
                "preset keeps contending GPU overlap disabled");
    TEST_ASSERT(opt.frontend_bucket_window == 32,
                "preset enables bounded length bucketing");
    TEST_ASSERT(opt.mixed_graph_cache_entries == 16,
                "preset enables multi-entry mixed graph cache");
    TEST_ASSERT(opt.offline_auto_tune,
                "preset enables memory-aware auto tuning");
    return true;
}

bool test_long_video_legacy_preset() {
    std::printf("\n--- CLI long-video legacy preset ---\n");
    Options opt;
    const bool ok = parse_with_args({
        arg((char*)"funasr-cli"),
        arg((char*)"-m"), arg((char*)"FunAsr_q8.bin"),
        arg((char*)"-f"), arg((char*)"video.wav"),
        arg((char*)"--offline-preset"), arg((char*)"long-video-legacy"),
    }, opt);
    TEST_ASSERT(ok, "legacy preset parses");
    TEST_ASSERT(opt.offline_batch_size == 12,
                "legacy preset keeps batch 12");
    TEST_ASSERT(!opt.unified_scheduler,
                "legacy preset keeps old scheduler");
    TEST_ASSERT(!opt.prefix_kv_cache,
                "legacy preset keeps prefix reuse off");
    TEST_ASSERT(!opt.frontend_prefetch,
                "legacy preset keeps frontend prefetch off");
    TEST_ASSERT(!opt.gpu_frontend_overlap,
                "legacy preset keeps GPU overlap off");
    TEST_ASSERT(opt.mixed_graph_cache_entries == 1,
                "legacy preset keeps single graph entry");
    TEST_ASSERT(!opt.offline_auto_tune,
                "legacy preset keeps auto tune off");
    return true;
}

bool test_preset_can_be_overridden_afterwards() {
    std::printf("\n--- CLI preset override ---\n");
    Options opt;
    bool ok = parse_with_args({
        arg((char*)"funasr-cli"),
        arg((char*)"-m"), arg((char*)"FunAsr_q8.bin"),
        arg((char*)"-f"), arg((char*)"video.wav"),
        arg((char*)"--offline-preset"), arg((char*)"long-video"),
        arg((char*)"--batch-size"), arg((char*)"8"),
        arg((char*)"--kv-block-size"), arg((char*)"64"),
    }, opt);

    TEST_ASSERT(ok, "preset with overrides parses");
    TEST_ASSERT(opt.offline_batch_size == 8, "following batch size overrides preset");
    TEST_ASSERT(opt.offline_kv_block_size == 64, "following block size overrides preset");
    return true;
}

bool test_invalid_kv_mode_fails() {
    std::printf("\n--- CLI invalid kv mode ---\n");
    Options opt;
    bool ok = parse_with_args({
        arg((char*)"funasr-cli"),
        arg((char*)"-m"), arg((char*)"FunAsr_q8.bin"),
        arg((char*)"-f"), arg((char*)"zh.wav"),
        arg((char*)"--kv-mode"), arg((char*)"bad"),
    }, opt);

    TEST_ASSERT(!ok, "invalid kv mode fails parsing");
    return true;
}

bool test_offline_routing_requires_chunks() {
    std::printf("\n--- CLI offline routing requires chunks ---\n");
    Options opt;

    TEST_ASSERT(!should_use_offline_scheduler(opt, true, 2),
                "offline scheduler disabled by default");

    opt.offline_scheduler = true;
    TEST_ASSERT(!should_use_offline_scheduler(opt, false, 2),
                "offline scheduler requires loaded audio");
    TEST_ASSERT(!should_use_offline_scheduler(opt, true, 0),
                "offline scheduler requires chunks");
    TEST_ASSERT(!should_use_offline_scheduler(opt, true, 1),
                "offline scheduler requires chunk mode");

    opt.chunk_mode = ChunkMode::Window;
    TEST_ASSERT(should_use_offline_scheduler(opt, true, 1),
                "offline scheduler routes loaded chunked audio");
    return true;
}

bool test_make_offline_batch_config() {
    std::printf("\n--- CLI offline batch config ---\n");
    Options opt;
    opt.offline_batch_size = 7;
    opt.ctx_size = 1536;
    opt.max_tokens = 42;
    opt.n_threads = 3;
    opt.use_gpu = true;
    opt.gpu_id = 2;
    opt.offline_kv_mode = OfflineKVMode::Paged;
    opt.offline_kv_block_size = 96;
    opt.offline_kv_num_blocks = 128;
    opt.prefix_kv_cache = true;
    opt.dynamic_kv_blocks = true;
    opt.unified_scheduler = true;
    opt.max_scheduled_tokens = 640;
    opt.max_prefill_chunk_tokens = 160;
    opt.max_frontend_requests_per_step = 2;
    opt.frontend_batching = false;
    opt.frontend_prefetch = false;
    opt.gpu_frontend_overlap = true;
    opt.frontend_bucket_window = 20;
    opt.mixed_graph_cache_entries = 6;

    funasr::OfflineBatchConfig cfg = make_offline_batch_config(opt);

    TEST_ASSERT(cfg.batch_size == 7, "batch size mapped");
    TEST_ASSERT(cfg.ctx_size == 1536, "ctx size mapped");
    TEST_ASSERT(cfg.max_tokens == 42, "max tokens mapped");
    TEST_ASSERT(cfg.n_threads == 3, "thread count mapped");
    TEST_ASSERT(cfg.use_gpu, "gpu flag mapped");
    TEST_ASSERT(cfg.gpu_id == 2, "gpu id mapped");
    TEST_ASSERT(cfg.use_paged_kv, "paged kv mode mapped");
    TEST_ASSERT(cfg.kv_block_size == 96, "kv block size mapped");
    TEST_ASSERT(cfg.kv_num_blocks == 128, "kv block count mapped");
    TEST_ASSERT(cfg.enable_prefix_kv_cache, "prefix KV cache mapped");
    TEST_ASSERT(cfg.enable_dynamic_kv_blocks, "dynamic KV blocks mapped");
    TEST_ASSERT(cfg.use_unified_scheduler, "unified scheduler mapped");
    TEST_ASSERT(cfg.max_num_scheduled_tokens == 640,
                "scheduled token budget mapped");
    TEST_ASSERT(cfg.max_prefill_chunk_tokens == 160,
                "prefill chunk budget mapped");
    TEST_ASSERT(cfg.max_frontend_requests_per_step == 2,
                "frontend request budget mapped");
    TEST_ASSERT(!cfg.enable_frontend_batching,
                "frontend batching flag mapped");
    TEST_ASSERT(!cfg.enable_frontend_prefetch,
                "frontend prefetch flag mapped");
    TEST_ASSERT(cfg.enable_gpu_frontend_overlap,
                "GPU frontend overlap flag mapped");
    TEST_ASSERT(cfg.frontend_bucket_window == 20,
                "frontend bucket window mapped");
    TEST_ASSERT(cfg.mixed_graph_cache_entries == 6,
                "mixed graph cache capacity mapped");

    opt.offline_kv_mode = OfflineKVMode::Continuous;
    cfg = make_offline_batch_config(opt);
    TEST_ASSERT(!cfg.use_paged_kv, "continuous kv mode mapped");
    TEST_ASSERT(!cfg.enable_prefix_kv_cache,
                "continuous mode disables prefix KV cache");
    TEST_ASSERT(!cfg.enable_dynamic_kv_blocks,
                "continuous mode disables dynamic KV blocks");
    return true;
}

bool test_gpu_init_slots() {
    std::printf("\n--- CLI GPU init slots ---\n");
    Options opt;
    opt.use_gpu = true;
    opt.offline_batch_size = 12;

    TEST_ASSERT(gpu_init_slots(opt) == 1,
                "default gpu init uses one slot");

    opt.offline_scheduler = true;
    TEST_ASSERT(gpu_init_slots(opt) == 1,
                "chunk-mode none keeps one gpu slot");

    opt.chunk_mode = ChunkMode::Window;
    TEST_ASSERT(gpu_init_slots(opt) == 12,
                "offline chunked gpu init uses offline batch slots");

    opt.offline_batch_size = 0;
    TEST_ASSERT(gpu_init_slots(opt) == 1,
                "gpu init slot count is clamped positive");
    return true;
}

bool test_gpu_init_physical_kv_rows() {
    std::printf("\n--- CLI GPU physical KV rows ---\n");
    Options opt;
    opt.use_gpu = true;
    opt.offline_scheduler = true;
    opt.chunk_mode = ChunkMode::Window;
    opt.offline_kv_mode = OfflineKVMode::Paged;
    opt.offline_batch_size = 32;
    opt.ctx_size = 4096;
    opt.offline_kv_block_size = 128;
    opt.offline_kv_num_blocks = 160;

    TEST_ASSERT(gpu_init_physical_kv_rows(opt) == 20480,
                "paged init uses explicit global pool");

    opt.offline_kv_mode = OfflineKVMode::Continuous;
    TEST_ASSERT(gpu_init_physical_kv_rows(opt) == 0,
                "continuous init preserves default allocation");
    return true;
}

bool test_gpu_init_prompt_slots() {
    std::printf("\n--- CLI GPU prompt slots ---\n");
    Options opt;
    opt.use_gpu = true;
    opt.offline_batch_size = 12;
    opt.max_frontend_requests_per_step = 4;

    TEST_ASSERT(gpu_init_prompt_slots(opt) == 1,
                "ordinary GPU mode keeps one prompt slot");
    opt.offline_scheduler = true;
    opt.chunk_mode = ChunkMode::Window;
    TEST_ASSERT(gpu_init_prompt_slots(opt) == 12,
                "legacy offline scheduler matches active slots");
    opt.unified_scheduler = true;
    TEST_ASSERT(gpu_init_prompt_slots(opt) == 16,
                "unified scheduler reserves a prepared frontend group");
    return true;
}

bool test_offline_paged_opts_default_and_disable() {
    std::printf("\n--- CLI offline paged optimization defaults ---\n");
    Options opt;

    TEST_ASSERT(!should_enable_offline_paged_opts(opt),
                "paged opts disabled outside offline gpu paged mode");

    opt.use_gpu = true;
    opt.offline_scheduler = true;
    opt.chunk_mode = ChunkMode::Window;
    opt.offline_kv_mode = OfflineKVMode::Paged;
    TEST_ASSERT(should_enable_offline_paged_opts(opt),
                "paged opts enabled for offline gpu paged mode");

    opt.offline_paged_opts = false;
    TEST_ASSERT(!should_enable_offline_paged_opts(opt),
                "paged opts can be disabled");
    return true;
}

bool test_offline_paged_opts_parse_disable() {
    std::printf("\n--- CLI offline paged optimization flag parse ---\n");
    Options opt;
    bool ok = parse_with_args({
        arg((char*)"funasr-cli"),
        arg((char*)"-m"), arg((char*)"FunAsr_q8.bin"),
        arg((char*)"-f"), arg((char*)"video.wav"),
        arg((char*)"--offline-preset"), arg((char*)"long-video"),
        arg((char*)"--no-offline-paged-opts"),
    }, opt);

    TEST_ASSERT(ok, "no-offline-paged-opts parses");
    TEST_ASSERT(!opt.offline_paged_opts, "paged opts disabled by flag");
    return true;
}

bool test_prefix_cache_and_dynamic_block_flags() {
    std::printf("\n--- CLI prefix cache and dynamic block flags ---\n");
    Options opt;
    TEST_ASSERT(!opt.prefix_kv_cache, "prefix KV cache disabled by default");
    TEST_ASSERT(!opt.dynamic_kv_blocks, "dynamic KV blocks disabled by default");

    bool ok = parse_with_args({
        arg((char*)"funasr-cli"),
        arg((char*)"-m"), arg((char*)"FunAsr_q8.bin"),
        arg((char*)"-f"), arg((char*)"video.wav"),
        arg((char*)"--prefix-kv-cache"), arg((char*)"on"),
        arg((char*)"--dynamic-kv-blocks"), arg((char*)"on"),
    }, opt);
    TEST_ASSERT(ok, "runtime KV flags parse");
    TEST_ASSERT(opt.prefix_kv_cache, "prefix KV cache enabled explicitly");
    TEST_ASSERT(opt.dynamic_kv_blocks, "dynamic KV blocks enabled explicitly");

    ok = parse_with_args({
        arg((char*)"funasr-cli"),
        arg((char*)"-m"), arg((char*)"FunAsr_q8.bin"),
        arg((char*)"-f"), arg((char*)"video.wav"),
        arg((char*)"--offline-preset"), arg((char*)"long-video"),
        arg((char*)"--prefix-kv-cache"), arg((char*)"off"),
        arg((char*)"--dynamic-kv-blocks"), arg((char*)"off"),
    }, opt);
    TEST_ASSERT(ok, "runtime KV flags override preset");
    TEST_ASSERT(!opt.prefix_kv_cache, "prefix KV cache can be disabled");
    TEST_ASSERT(!opt.dynamic_kv_blocks, "dynamic KV blocks can be disabled");
    return true;
}

} // namespace

int main() {
    int failed = 0;
    if (!test_offline_flags_parse()) failed++;
    if (!test_long_video_preset()) failed++;
    if (!test_long_video_legacy_preset()) failed++;
    if (!test_preset_can_be_overridden_afterwards()) failed++;
    if (!test_invalid_kv_mode_fails()) failed++;
    if (!test_offline_routing_requires_chunks()) failed++;
    if (!test_make_offline_batch_config()) failed++;
    if (!test_gpu_init_slots()) failed++;
    if (!test_gpu_init_physical_kv_rows()) failed++;
    if (!test_gpu_init_prompt_slots()) failed++;
    if (!test_offline_paged_opts_default_and_disable()) failed++;
    if (!test_offline_paged_opts_parse_disable()) failed++;
    if (!test_prefix_cache_and_dynamic_block_flags()) failed++;

    if (failed == 0) {
        std::printf("\nAll CLI option tests passed!\n");
        return 0;
    }
    std::printf("\n%d CLI option tests failed.\n", failed);
    return 1;
}
