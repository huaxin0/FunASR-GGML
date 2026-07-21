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

void test_only_final_prefill_chunk_produces_logits() {
    std::printf("\n--- Test 2: Final prefill chunk produces logits ---\n");

    funasr::UnifiedSchedulerConfig config;
    config.max_num_seqs = 1;
    config.max_num_scheduled_tokens = 128;
    config.max_prefill_chunk_tokens = 128;

    funasr::UnifiedRequestProgress request;
    request.request_id = 30;
    request.prompt_tokens = 522;
    request.num_computed_tokens = 256;
    request.max_output_tokens = 220;
    request.max_schedulable_tokens = 522;
    request.runnable = true;
    request.finished = false;

    const funasr::UnifiedTokenScheduler scheduler(config);
    std::vector<funasr::UnifiedRequestProgress> requests = {request};

    const funasr::MixedBatchPlan middle_plan = scheduler.build_plan(requests);
    TEST_ASSERT(middle_plan.ok(), "middle prefill plan should be valid");
    TEST_EQ((int)middle_plan.sequences.size(), 1,
            "middle prefill sequence count");
    if (middle_plan.sequences.size() == 1) {
        const funasr::ScheduledSequence& middle = middle_plan.sequences[0];
        TEST_ASSERT(middle.input_kind == funasr::UnifiedInputKind::Prompt,
                    "middle prefill input kind");
        TEST_EQ(middle.token_offset, 256, "middle prefill token offset");
        TEST_EQ(middle.num_tokens, 128, "middle prefill token count");
        TEST_ASSERT(!middle.produces_logits,
                    "middle prefill chunk should not produce logits");
    }

    requests[0].num_computed_tokens = 512;
    const funasr::MixedBatchPlan final_plan = scheduler.build_plan(requests);
    TEST_ASSERT(final_plan.ok(), "final prefill plan should be valid");
    TEST_EQ((int)final_plan.sequences.size(), 1,
            "final prefill sequence count");
    if (final_plan.sequences.size() == 1) {
        const funasr::ScheduledSequence& final = final_plan.sequences[0];
        TEST_ASSERT(final.input_kind == funasr::UnifiedInputKind::Prompt,
                    "final prefill input kind");
        TEST_EQ(final.token_offset, 512, "final prefill token offset");
        TEST_EQ(final.num_tokens, 10, "final prefill token count");
        TEST_ASSERT(final.produces_logits,
                    "final prefill chunk should produce logits");
    }
}

void test_commit_prompt_then_schedule_first_decode_input() {
    std::printf("\n--- Test 3: Commit prompt and first decode input ---\n");

    const funasr::UnifiedSchedulerConfig config{4, 128, 128};
    const funasr::UnifiedTokenScheduler scheduler(config);

    funasr::UnifiedRequestProgress request;
    request.request_id = 40;
    request.prompt_tokens = 522;
    request.num_computed_tokens = 512;
    request.max_output_tokens = 2;

    funasr::MixedBatchPlan plan = scheduler.build_plan({request});
    TEST_EQ((int)plan.sequences.size(), 1,
            "final prompt sequence count before commit");
    if (plan.sequences.size() == 1) {
        TEST_ASSERT(funasr::UnifiedTokenScheduler::commit_sequence(
                        request, plan.sequences[0]),
                    "final prompt sequence commits");
    }
    TEST_EQ(request.num_computed_tokens, 522,
            "prompt progress reaches end");

    const funasr::SampleCommitResult first =
        funasr::UnifiedTokenScheduler::commit_sample(request, 1234, 9999);
    TEST_ASSERT(first == funasr::SampleCommitResult::Appended,
                "first output token is appended");
    TEST_EQ((int)request.output_tokens.size(), 1,
            "one sampled token exists");
    TEST_EQ(request.num_computed_tokens, 522,
            "sampled token is not in KV before next iteration");

    plan = scheduler.build_plan({request});
    TEST_EQ((int)plan.sequences.size(), 1, "one decode input is scheduled");
    if (plan.sequences.size() == 1) {
        TEST_ASSERT(plan.sequences[0].input_kind ==
                        funasr::UnifiedInputKind::Decode,
                    "sampled token becomes decode input");
        TEST_EQ(plan.sequences[0].token_offset, 522,
                "first output token uses position 522");
        TEST_ASSERT(funasr::UnifiedTokenScheduler::commit_sequence(
                        request, plan.sequences[0]),
                    "decode sequence commits");
    }

    const funasr::SampleCommitResult second =
        funasr::UnifiedTokenScheduler::commit_sample(request, 5678, 9999);
    TEST_ASSERT(second == funasr::SampleCommitResult::FinishedLimit,
                "second output reaches generation limit");
    TEST_ASSERT(request.finished, "request is finished at output limit");

    funasr::UnifiedRequestProgress eos_request;
    eos_request.request_id = 41;
    eos_request.prompt_tokens = 1;
    eos_request.num_computed_tokens = 1;
    eos_request.max_output_tokens = 4;
    const funasr::SampleCommitResult eos =
        funasr::UnifiedTokenScheduler::commit_sample(eos_request, 99, 99);
    TEST_ASSERT(eos == funasr::SampleCommitResult::FinishedEos,
                "EOS finishes without appending");
    TEST_EQ((int)eos_request.output_tokens.size(), 0,
            "EOS is not included in output tokens");
}

int main() {
    std::printf("========================================\n");
    std::printf("Unified Scheduler Unit Tests\n");
    std::printf("========================================\n");

    test_decode_first_mixed_token_budget_plan();
    test_only_final_prefill_chunk_produces_logits();
    test_commit_prompt_then_schedule_first_decode_input();

    std::printf("\n========================================\n");
    std::printf("Tests passed: %d\n", tests_passed);
    std::printf("Tests failed: %d\n", tests_failed);
    std::printf("========================================\n");

    return tests_failed == 0 ? 0 : 1;
}
