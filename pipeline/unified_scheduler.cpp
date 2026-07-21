#include "pipeline/unified_scheduler.hpp"

#include <algorithm>

namespace funasr {

UnifiedTokenScheduler::UnifiedTokenScheduler(UnifiedSchedulerConfig config)
    : config_(config) {}

int UnifiedRequestProgress::available_tokens() const {
    return prompt_tokens + static_cast<int>(output_tokens.size()) -
           num_computed_tokens;
}

bool UnifiedRequestProgress::prompt_complete() const {
    return num_computed_tokens >= prompt_tokens;
}

MixedBatchPlan UnifiedTokenScheduler::build_plan(
    const std::vector<UnifiedRequestProgress>& requests) const {
    MixedBatchPlan plan;
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

    for (size_t i = 0; i < requests.size() && budget > 0; i++) {
        const auto& request = requests[i];
        if (!request.runnable || request.finished ||
            !request.prompt_complete() || request.available_tokens() <= 0 ||
            request.max_schedulable_tokens <= 0) {
            continue;
        }
        schedule(static_cast<int>(i), 1, UnifiedInputKind::Decode);
    }

    for (size_t i = 0; i < requests.size() && budget > 0; i++) {
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

} // namespace funasr
