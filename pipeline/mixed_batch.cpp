#include "pipeline/mixed_batch.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <unordered_set>

namespace funasr {
namespace {

bool validate_source(const PackedSequenceSource& source,
                     int block_size) {
    const ScheduledSequence& scheduled = source.scheduled;
    if (scheduled.request_id < 0 || scheduled.request_index < 0 ||
        scheduled.token_offset < 0 || scheduled.num_tokens <= 0 ||
        source.prompt_tokens < 0 || source.cached_prefix_tokens < 0) {
        return false;
    }

    const int64_t token_begin = scheduled.token_offset;
    const int64_t token_end =
        token_begin + static_cast<int64_t>(scheduled.num_tokens);
    if (token_end > std::numeric_limits<int>::max()) {
        return false;
    }

    if (scheduled.input_kind == UnifiedInputKind::Prompt) {
        if (token_begin < source.cached_prefix_tokens ||
            token_end > source.prompt_tokens ||
            source.decode_token_id != -1) {
            return false;
        }
    } else if (scheduled.input_kind == UnifiedInputKind::Decode) {
        if (scheduled.num_tokens != 1 ||
            token_begin < source.prompt_tokens ||
            source.decode_token_id < 0) {
            return false;
        }
    } else {
        return false;
    }

    const int64_t first_block = token_begin / block_size;
    const int64_t last_block = (token_end - 1) / block_size;
    if (last_block >= static_cast<int64_t>(source.block_table.size())) {
        return false;
    }
    for (int64_t block = first_block; block <= last_block; ++block) {
        if (source.block_table[static_cast<size_t>(block)] < 0) {
            return false;
        }
    }
    return true;
}

bool can_add_token_count(int64_t current, int64_t amount) {
    if (current < 0 || amount < 0 ||
        amount > std::numeric_limits<int64_t>::max() - current) {
        return false;
    }
    return amount <= kMaxPackedTokens - current;
}

} // namespace

PackedMixedMetadata build_packed_mixed_metadata(
    const std::vector<PackedSequenceSource>& sources,
    int block_size) {
    PackedMixedMetadata packed;
    if (block_size <= 0 || sources.empty()) {
        return packed;
    }

    int64_t total_tokens = 0;
    int64_t prefill_tokens = 0;
    int64_t decode_tokens = 0;
    int64_t max_blocks = 0;
    std::unordered_set<int> request_ids;

    try {
        for (const auto& source : sources) {
            if (!request_ids.insert(source.scheduled.request_id).second ||
                source.block_table.size() >
                    static_cast<size_t>(std::numeric_limits<int>::max()) ||
                !validate_source(source, block_size)) {
                return packed;
            }

            const int64_t token_count = source.scheduled.num_tokens;
            if (!can_add_token_count(total_tokens, token_count)) {
                return packed;
            }
            total_tokens += token_count;

            if (source.scheduled.input_kind == UnifiedInputKind::Prompt) {
                if (!can_add_token_count(prefill_tokens, token_count)) {
                    return packed;
                }
                prefill_tokens += token_count;
            } else if (source.scheduled.input_kind == UnifiedInputKind::Decode) {
                if (!can_add_token_count(decode_tokens, token_count)) {
                    return packed;
                }
                decode_tokens += token_count;
            }
            max_blocks = std::max<int64_t>(
                max_blocks, static_cast<int64_t>(source.block_table.size()));
        }

        if (max_blocks <= 0 ||
            total_tokens > kMaxExpandedBlockEntries / max_blocks) {
            return packed;
        }
        const int64_t expanded_size = total_tokens * max_blocks;
        if (static_cast<uint64_t>(expanded_size) >
            static_cast<uint64_t>(kMaxExpandedBlockEntries) ||
            static_cast<uint64_t>(expanded_size) >
                std::numeric_limits<size_t>::max()) {
            return packed;
        }

        packed.sequences.reserve(sources.size());
        packed.positions.reserve(static_cast<size_t>(total_tokens));
        packed.sequence_rows.reserve(static_cast<size_t>(total_tokens));
        packed.kv_lens.reserve(static_cast<size_t>(total_tokens));
        packed.selected_rows.reserve(sources.size());
        packed.expanded_block_table.assign(
            static_cast<size_t>(expanded_size), -1);

        int64_t token_row = 0;
        const auto append_kind = [&](UnifiedInputKind kind) {
            for (const auto& source : sources) {
                if (source.scheduled.input_kind != kind) {
                    continue;
                }

                const int sequence_row =
                    static_cast<int>(packed.sequences.size());
                packed.sequences.push_back(source);
                for (int64_t local_token = 0;
                     local_token < source.scheduled.num_tokens;
                     ++local_token, ++token_row) {
                    const int64_t position =
                        static_cast<int64_t>(source.scheduled.token_offset) +
                        local_token;
                    packed.positions.push_back(static_cast<int>(position));
                    packed.sequence_rows.push_back(sequence_row);
                    packed.kv_lens.push_back(static_cast<int>(position + 1));

                    const size_t row_offset = static_cast<size_t>(
                        token_row * max_blocks);
                    std::copy(source.block_table.begin(),
                              source.block_table.end(),
                              packed.expanded_block_table.begin() + row_offset);
                }
                if (source.scheduled.produces_logits) {
                    packed.selected_rows.push_back(
                        static_cast<int>(token_row - 1));
                }
            }
        };

        append_kind(UnifiedInputKind::Prompt);
        append_kind(UnifiedInputKind::Decode);

        packed.max_blocks = static_cast<int>(max_blocks);
        packed.total_tokens = static_cast<int>(total_tokens);
        packed.prefill_tokens = static_cast<int>(prefill_tokens);
        packed.decode_tokens = static_cast<int>(decode_tokens);
        packed.valid = true;
    } catch (const std::bad_alloc&) {
        return {};
    } catch (const std::length_error&) {
        return {};
    }
    return packed;
}

} // namespace funasr
