#include "pipeline/unified_scheduler.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_set>

namespace funasr {

UnifiedTokenScheduler::UnifiedTokenScheduler(UnifiedSchedulerConfig config)
    : config_(config) {}

int UnifiedRequestProgress::available_tokens() const {
    const int64_t output_size = std::min<int64_t>(
        static_cast<int64_t>(output_tokens.size()),
        std::numeric_limits<int>::max());
    const int64_t available = static_cast<int64_t>(prompt_tokens) +
                              output_size - num_computed_tokens;
    return static_cast<int>(std::clamp<int64_t>(
        available, std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()));
}

bool UnifiedRequestProgress::prompt_complete() const {
    return num_computed_tokens >= prompt_tokens;
}

bool UnifiedRequestProgress::valid() const {
    if (request_id < 0 || prompt_tokens < 0 || num_computed_tokens < 0 ||
        max_output_tokens <= 0 || max_schedulable_tokens < 0 ||
        output_tokens.size() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    const size_t output_limit = static_cast<size_t>(max_output_tokens);
    if (output_tokens.size() > output_limit ||
        (!finished && output_tokens.size() == output_limit)) {
        return false;
    }

    const int64_t available_end =
        static_cast<int64_t>(prompt_tokens) +
        static_cast<int64_t>(output_tokens.size());
    return available_end <= std::numeric_limits<int>::max() &&
           num_computed_tokens <= available_end;
}

MixedBatchPlan UnifiedTokenScheduler::build_plan(
    const std::vector<UnifiedRequestProgress>& requests) const {
    MixedBatchPlan plan;
    if (config_.max_num_seqs <= 0 ||
        config_.max_num_scheduled_tokens <= 0 ||
        config_.max_prefill_chunk_tokens <= 0 ||
        config_.max_num_scheduled_tokens < config_.max_num_seqs) {
        plan.error = UnifiedPlanError::InvalidConfig;
        return plan;
    }

    std::unordered_set<int> request_ids;
    for (const auto& request : requests) {
        if (!request.valid()) {
            plan.error = UnifiedPlanError::InvalidRequest;
            return plan;
        }
        if (!request_ids.insert(request.request_id).second) {
            plan.error = UnifiedPlanError::DuplicateRequestId;
            return plan;
        }
    }

    int budget = config_.max_num_scheduled_tokens;

    auto schedule = [&](int index, int count, UnifiedInputKind kind) {
        const auto& request = requests[static_cast<size_t>(index)];
        ScheduledSequence sequence;
        sequence.request_id = request.request_id;
        sequence.request_index = index;
        sequence.token_offset = request.num_computed_tokens;
        sequence.num_tokens = count;
        sequence.input_kind = kind;
        sequence.produces_logits = kind == UnifiedInputKind::Decode;
        plan.sequences.push_back(sequence);
        plan.total_tokens += count;
        if (kind == UnifiedInputKind::Decode) {
            plan.decode_tokens += count;
        } else {
            plan.prefill_tokens += count;
        }
        budget -= count;
    };

    for (size_t i = 0;
         i < requests.size() && budget > 0 &&
         static_cast<int>(plan.sequences.size()) < config_.max_num_seqs;
         i++) {
        const auto& request = requests[i];
        if (!request.runnable || request.finished ||
            !request.prompt_complete() || request.available_tokens() <= 0 ||
            request.max_schedulable_tokens <= 0) {
            continue;
        }
        schedule(static_cast<int>(i), 1, UnifiedInputKind::Decode);
    }

    for (size_t i = 0;
         i < requests.size() && budget > 0 &&
         static_cast<int>(plan.sequences.size()) < config_.max_num_seqs;
         i++) {
        const auto& request = requests[i];
        if (!request.runnable || request.finished ||
            request.prompt_complete() || request.available_tokens() <= 0 ||
            request.max_schedulable_tokens <= 0) {
            continue;
        }
        const int count = std::min({request.available_tokens(), budget,
                                    config_.max_prefill_chunk_tokens,
                                    request.max_schedulable_tokens});
        if (count > 0) {
            schedule(static_cast<int>(i), count, UnifiedInputKind::Prompt);
            plan.sequences.back().produces_logits =
                request.num_computed_tokens + count == request.prompt_tokens;
        }
    }

    return plan;
}

bool UnifiedTokenScheduler::commit_sequence(
    UnifiedRequestProgress& request,
    const ScheduledSequence& scheduled) {
    if (!request.valid() || request.finished ||
        scheduled.request_id != request.request_id ||
        scheduled.num_tokens <= 0 ||
        scheduled.token_offset != request.num_computed_tokens ||
        scheduled.num_tokens > request.available_tokens()) {
        return false;
    }

    request.num_computed_tokens += scheduled.num_tokens;
    return true;
}

SampleCommitResult UnifiedTokenScheduler::commit_sample(
    UnifiedRequestProgress& request,
    int token_id,
    int eos_id) {
    if (!request.valid() || request.finished ||
        request.num_computed_tokens !=
            static_cast<int64_t>(request.prompt_tokens) +
                static_cast<int64_t>(request.output_tokens.size())) {
        return SampleCommitResult::Invalid;
    }
    if (token_id == eos_id) {
        request.finished = true;
        return SampleCommitResult::FinishedEos;
    }

    request.output_tokens.push_back(token_id);
    if (static_cast<int>(request.output_tokens.size()) >=
        request.max_output_tokens) {
        request.finished = true;
        return SampleCommitResult::FinishedLimit;
    }
    return SampleCommitResult::Appended;
}

} // namespace funasr
