#include "pipeline/unified_scheduler.hpp"

#include <cstdio>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(condition, msg) do { \
    if (condition) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        std::printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
    } \
} while (0)

#define TEST_EQ(actual, expected, name) do { \
    if ((actual) == (expected)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        std::printf("  [FAIL] %s: expected %d, got %d (line %d)\n", \
                    name, (int)(expected), (int)(actual), __LINE__); \
    } \
} while (0)

void test_decode_first_mixed_token_budget_plan() {
    std::printf("\n--- Test 1: Decode-first mixed token budget plan ---\n");

    funasr::UnifiedSchedulerConfig config;
    config.max_num_seqs = 4;
    config.max_num_scheduled_tokens = 4;
    config.max_prefill_chunk_tokens = 4;

    funasr::UnifiedRequestProgress decode_request;
    decode_request.request_id = 10;
    decode_request.prompt_tokens = 4;
    decode_request.num_computed_tokens = 4;
    decode_request.max_output_tokens = 8;
    decode_request.output_tokens = {101};

    funasr::UnifiedRequestProgress prefill_request;
    prefill_request.request_id = 20;
    prefill_request.prompt_tokens = 8;
    prefill_request.num_computed_tokens = 0;
    prefill_request.max_output_tokens = 8;

    const funasr::UnifiedTokenScheduler scheduler(config);
    const std::vector<funasr::UnifiedRequestProgress> requests = {
        prefill_request,
        decode_request,
    };
    const funasr::MixedBatchPlan plan = scheduler.build_plan(requests);

    TEST_ASSERT(plan.ok(), "mixed plan should be valid");
    TEST_EQ(plan.total_tokens, 4, "total scheduled tokens");
    TEST_EQ(plan.decode_tokens, 1, "decode scheduled tokens");
    TEST_EQ(plan.prefill_tokens, 3, "prefill scheduled tokens");
    TEST_EQ((int)plan.sequences.size(), 2, "scheduled sequence count");

    if (plan.sequences.size() == 2) {
        const funasr::ScheduledSequence& decode = plan.sequences[0];
        TEST_EQ(decode.request_id, 10, "decode request id");
        TEST_ASSERT(decode.input_kind == funasr::UnifiedInputKind::Decode,
                    "decode request input kind");
        TEST_EQ(decode.request_index, 1, "decode request input index");
        TEST_EQ(decode.token_offset, 4, "decode token offset");
        TEST_EQ(decode.num_tokens, 1, "decode token count");

        const funasr::ScheduledSequence& prefill = plan.sequences[1];
        TEST_EQ(prefill.request_id, 20, "prefill request id");
        TEST_ASSERT(prefill.input_kind == funasr::UnifiedInputKind::Prompt,
                    "prefill request input kind");
        TEST_EQ(prefill.request_index, 0, "prefill request input index");
        TEST_EQ(prefill.token_offset, 0, "prefill token offset");
        TEST_EQ(prefill.num_tokens, 3, "prefill token count");
    }
}

int main() {
    std::printf("========================================\n");
    std::printf("Unified Scheduler Unit Tests\n");
    std::printf("========================================\n");

    test_decode_first_mixed_token_budget_plan();

    std::printf("\n========================================\n");
    std::printf("Tests passed: %d\n", tests_passed);
    std::printf("Tests failed: %d\n", tests_failed);
    std::printf("========================================\n");

    return tests_failed == 0 ? 0 : 1;
}
