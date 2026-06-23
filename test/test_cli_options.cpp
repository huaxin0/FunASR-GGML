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
    }, opt);

    TEST_ASSERT(ok, "offline flags parse successfully");
    TEST_ASSERT(opt.offline_scheduler, "offline scheduler enabled");
    TEST_ASSERT(opt.offline_profile, "offline profile enabled");
    TEST_ASSERT(opt.offline_batch_size == 12, "batch size parsed");
    TEST_ASSERT(opt.offline_kv_mode == OfflineKVMode::Paged, "paged kv mode parsed");
    TEST_ASSERT(opt.offline_kv_block_size == 128, "kv block size parsed");
    TEST_ASSERT(opt.offline_kv_num_blocks == 384, "kv block count parsed");
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
    TEST_ASSERT(opt.offline_batch_size == 12, "preset batch size is 12");
    TEST_ASSERT(opt.offline_kv_mode == OfflineKVMode::Paged, "preset uses paged kv");
    TEST_ASSERT(opt.offline_kv_block_size == 128, "preset block size is 128");
    TEST_ASSERT(opt.ctx_size == 4096, "preset ctx size is 4096");
    TEST_ASSERT(opt.chunk_mode == ChunkMode::Window, "preset uses window chunks");
    TEST_ASSERT(opt.chunk_sec == 30, "preset chunk size is 30 seconds");
    TEST_ASSERT(opt.max_tokens == 220, "preset max tokens is 220");
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

    opt.offline_kv_mode = OfflineKVMode::Continuous;
    cfg = make_offline_batch_config(opt);
    TEST_ASSERT(!cfg.use_paged_kv, "continuous kv mode mapped");
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

} // namespace

int main() {
    int failed = 0;
    if (!test_offline_flags_parse()) failed++;
    if (!test_long_video_preset()) failed++;
    if (!test_preset_can_be_overridden_afterwards()) failed++;
    if (!test_invalid_kv_mode_fails()) failed++;
    if (!test_offline_routing_requires_chunks()) failed++;
    if (!test_make_offline_batch_config()) failed++;
    if (!test_gpu_init_slots()) failed++;
    if (!test_offline_paged_opts_default_and_disable()) failed++;
    if (!test_offline_paged_opts_parse_disable()) failed++;

    if (failed == 0) {
        std::printf("\nAll CLI option tests passed!\n");
        return 0;
    }
    std::printf("\n%d CLI option tests failed.\n", failed);
    return 1;
}
