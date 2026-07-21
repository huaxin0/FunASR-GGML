#include "pipeline/unified_scheduler.hpp"

#include <cstdio>
#include <limits>
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

void test_invalid_commit_does_not_advance_request() {
    std::printf("\n--- Test 4: Invalid commit preserves request state ---\n");

    funasr::UnifiedRequestProgress request;
    request.request_id = 45;
    request.prompt_tokens = 4;
    request.num_computed_tokens = 2;
    request.max_output_tokens = 4;

    funasr::ScheduledSequence scheduled;
    scheduled.request_id = 45;
    scheduled.token_offset = 2;
    scheduled.num_tokens = 2;
    scheduled.input_kind = funasr::UnifiedInputKind::Prompt;

    auto candidate = request;
    auto invalid = scheduled;
    invalid.request_id = 46;
    TEST_ASSERT(!funasr::UnifiedTokenScheduler::commit_sequence(
                    candidate, invalid),
                "mismatched request id is rejected");
    TEST_EQ(candidate.num_computed_tokens, 2,
            "request id mismatch preserves progress");

    candidate = request;
    invalid = scheduled;
    invalid.token_offset = 1;
    TEST_ASSERT(!funasr::UnifiedTokenScheduler::commit_sequence(
                    candidate, invalid),
                "mismatched token offset is rejected");
    TEST_EQ(candidate.num_computed_tokens, 2,
            "offset mismatch preserves progress");

    candidate = request;
    invalid = scheduled;
    invalid.num_tokens = 0;
    TEST_ASSERT(!funasr::UnifiedTokenScheduler::commit_sequence(
                    candidate, invalid),
                "zero token sequence is rejected");

    candidate = request;
    invalid = scheduled;
    invalid.num_tokens = 3;
    TEST_ASSERT(!funasr::UnifiedTokenScheduler::commit_sequence(
                    candidate, invalid),
                "sequence beyond available tokens is rejected");
    TEST_EQ(candidate.num_computed_tokens, 2,
            "capacity mismatch preserves progress");

    candidate = request;
    candidate.runnable = false;
    TEST_ASSERT(!funasr::UnifiedTokenScheduler::commit_sequence(
                    candidate, scheduled),
                "non-runnable sequence commit is rejected");
    TEST_EQ(candidate.num_computed_tokens, 2,
            "non-runnable sequence preserves progress");

    candidate = request;
    TEST_ASSERT(funasr::UnifiedTokenScheduler::commit_sample(
                    candidate, 10, 99) ==
                    funasr::SampleCommitResult::Invalid,
                "sample commit requires aligned computed state");
    TEST_EQ((int)candidate.output_tokens.size(), 0,
            "invalid sample does not append a token");

    candidate = request;
    candidate.num_computed_tokens = 4;
    candidate.runnable = false;
    TEST_ASSERT(funasr::UnifiedTokenScheduler::commit_sample(
                    candidate, 10, 99) ==
                    funasr::SampleCommitResult::Invalid,
                "non-runnable sample commit is rejected");
    TEST_EQ((int)candidate.output_tokens.size(), 0,
            "non-runnable sample does not append a token");
}

void test_validation_capacity_and_sequence_limit() {
    std::printf("\n--- Test 5: Validation and capacity limits ---\n");

    const funasr::UnifiedSchedulerConfig config{1, 8, 8};
    const funasr::UnifiedTokenScheduler scheduler(config);

    funasr::UnifiedRequestProgress blocked_decode;
    blocked_decode.request_id = 50;
    blocked_decode.prompt_tokens = 2;
    blocked_decode.num_computed_tokens = 2;
    blocked_decode.max_output_tokens = 4;
    blocked_decode.output_tokens = {7};
    blocked_decode.max_schedulable_tokens = 0;

    funasr::UnifiedRequestProgress capped_prefill;
    capped_prefill.request_id = 51;
    capped_prefill.prompt_tokens = 20;
    capped_prefill.max_output_tokens = 4;
    capped_prefill.max_schedulable_tokens = 3;

    funasr::UnifiedRequestProgress second_prefill = capped_prefill;
    second_prefill.request_id = 52;

    funasr::MixedBatchPlan plan = scheduler.build_plan(
        {blocked_decode, capped_prefill, second_prefill});
    TEST_ASSERT(plan.ok(), "capacity-constrained plan succeeds");
    TEST_EQ((int)plan.sequences.size(), 1,
            "max_num_seqs limits distinct requests");
    if (plan.sequences.size() == 1) {
        TEST_EQ(plan.sequences[0].request_id, 51,
                "first runnable prefill selected");
        TEST_EQ(plan.sequences[0].num_tokens, 3,
                "KV capacity caps prompt chunk");
    }

    funasr::UnifiedRequestProgress invalid = capped_prefill;
    invalid.request_id = 60;
    invalid.prompt_tokens = -1;
    plan = scheduler.build_plan({invalid});
    TEST_ASSERT(plan.error == funasr::UnifiedPlanError::InvalidRequest,
                "invalid request rejects whole plan");
    TEST_EQ((int)plan.sequences.size(), 0,
            "invalid plan schedules no work");

    funasr::UnifiedRequestProgress duplicate = capped_prefill;
    duplicate.request_id = 70;
    plan = scheduler.build_plan({duplicate, duplicate});
    TEST_ASSERT(plan.error == funasr::UnifiedPlanError::DuplicateRequestId,
                "duplicate request id is rejected");

    const funasr::UnifiedSchedulerConfig bad_config{0, 8, 8};
    const funasr::UnifiedTokenScheduler bad_scheduler(bad_config);
    plan = bad_scheduler.build_plan({capped_prefill});
    TEST_ASSERT(plan.error == funasr::UnifiedPlanError::InvalidConfig,
                "invalid scheduler config is rejected");

    const funasr::UnifiedSchedulerConfig starved_decode_config{40, 32, 8};
    const funasr::UnifiedTokenScheduler starved_decode_scheduler(
        starved_decode_config);
    plan = starved_decode_scheduler.build_plan({capped_prefill});
    TEST_ASSERT(plan.error == funasr::UnifiedPlanError::InvalidConfig,
                "token budget must cover every active decode sequence");

    funasr::UnifiedRequestProgress overflowing = capped_prefill;
    overflowing.request_id = 80;
    overflowing.prompt_tokens = std::numeric_limits<int>::max();
    overflowing.output_tokens = {1};
    plan = scheduler.build_plan({overflowing});
    TEST_ASSERT(plan.error == funasr::UnifiedPlanError::InvalidRequest,
                "request token extent must fit the progress type");

    funasr::UnifiedRequestProgress exhausted = capped_prefill;
    exhausted.request_id = 81;
    exhausted.num_computed_tokens = 24;
    exhausted.output_tokens = {1, 2, 3, 4};
    plan = scheduler.build_plan({exhausted});
    TEST_ASSERT(plan.error == funasr::UnifiedPlanError::InvalidRequest,
                "unfinished request cannot already be at output limit");

    funasr::UnifiedRequestProgress over_limit = exhausted;
    over_limit.request_id = 82;
    over_limit.num_computed_tokens = 25;
    over_limit.output_tokens.push_back(5);
    over_limit.finished = true;
    plan = scheduler.build_plan({over_limit});
    TEST_ASSERT(plan.error == funasr::UnifiedPlanError::InvalidRequest,
                "request cannot exceed output limit");
}

int main() {
    std::printf("========================================\n");
    std::printf("Unified Scheduler Unit Tests\n");
    std::printf("========================================\n");

    test_decode_first_mixed_token_budget_plan();
    test_only_final_prefill_chunk_produces_logits();
    test_commit_prompt_then_schedule_first_decode_input();
    test_invalid_commit_does_not_advance_request();
    test_validation_capacity_and_sequence_limit();

    std::printf("\n========================================\n");
    std::printf("Tests passed: %d\n", tests_passed);
    std::printf("Tests failed: %d\n", tests_failed);
    std::printf("========================================\n");

    return tests_failed == 0 ? 0 : 1;
}
