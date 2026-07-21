#ifndef FUNASR_PIPELINE_UNIFIED_SCHEDULER_HPP
#define FUNASR_PIPELINE_UNIFIED_SCHEDULER_HPP

#include <limits>
#include <vector>

namespace funasr {

enum class UnifiedInputKind {
    Prompt,
    Decode,
};

enum class UnifiedPlanError {
    None,
    InvalidConfig,
    InvalidRequest,
    DuplicateRequestId,
};

enum class SampleCommitResult {
    Appended,
    FinishedEos,
    FinishedLimit,
    Invalid,
};

struct UnifiedSchedulerConfig {
    int max_num_seqs = 1;
    int max_num_scheduled_tokens = 1;
    int max_prefill_chunk_tokens = 1;
};

struct UnifiedRequestProgress {
    int request_id = -1;
    int prompt_tokens = 0;
    int num_computed_tokens = 0;
    int max_output_tokens = 0;
    int max_schedulable_tokens = std::numeric_limits<int>::max();
    std::vector<int> output_tokens;
    bool runnable = true;
    bool finished = false;

    int available_tokens() const;
    bool prompt_complete() const;
    bool valid() const;
};

struct ScheduledSequence {
    int request_id = -1;
    int request_index = -1;
    int token_offset = 0;
    int num_tokens = 0;
    UnifiedInputKind input_kind = UnifiedInputKind::Prompt;
    bool produces_logits = false;
};

struct MixedBatchPlan {
    std::vector<ScheduledSequence> sequences;
    int total_tokens = 0;
    int prefill_tokens = 0;
    int decode_tokens = 0;
    UnifiedPlanError error = UnifiedPlanError::None;

    bool ok() const { return error == UnifiedPlanError::None; }
};

class UnifiedTokenScheduler {
public:
    explicit UnifiedTokenScheduler(UnifiedSchedulerConfig config);

    MixedBatchPlan build_plan(
        const std::vector<UnifiedRequestProgress>& requests) const;

    static bool commit_sequence(
        UnifiedRequestProgress& request,
        const ScheduledSequence& scheduled);

    static SampleCommitResult commit_sample(
        UnifiedRequestProgress& request,
        int token_id,
        int eos_id);

private:
    UnifiedSchedulerConfig config_;
};

} // namespace funasr

#endif
