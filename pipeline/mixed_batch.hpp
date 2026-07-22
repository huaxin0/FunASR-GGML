#ifndef FUNASR_PIPELINE_MIXED_BATCH_HPP
#define FUNASR_PIPELINE_MIXED_BATCH_HPP

#include "pipeline/unified_scheduler.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace funasr {

// Defensive limits for this metadata builder. They protect against malformed
// or adversarial input and are not the scheduler's default budget.
inline constexpr std::int64_t kMaxPackedTokens = 1048576;
inline constexpr std::int64_t kMaxExpandedBlockEntries = 67108864;

struct PackedSequenceSource {
    ScheduledSequence scheduled;
    int prompt_tokens = 0;
    int cached_prefix_tokens = 0;
    int decode_token_id = -1;
    std::vector<int> block_table;
};

struct PackedMixedMetadata {
    std::vector<PackedSequenceSource> sequences;
    std::vector<int> positions;
    std::vector<int> sequence_rows;
    std::vector<int> kv_lens;
    std::vector<int> selected_rows;
    // Flat index = token_row * max_blocks + block_index; GGML shape is
    // [max_blocks, total_tokens], with -1 padding.
    std::vector<int> expanded_block_table;
    int max_blocks = 0;
    int total_tokens = 0;
    int prefill_tokens = 0;
    int decode_tokens = 0;
    bool valid = false;
};

PackedMixedMetadata build_packed_mixed_metadata(
    const std::vector<PackedSequenceSource>& sources,
    int block_size);

inline bool validate_packed_mixed_execution(
    const PackedMixedMetadata& metadata,
    int block_size,
    int physical_rows,
    int vocab_size,
    int staged_prefill_tokens,
    const std::vector<int>& decode_token_ids) {
    const uint64_t expected_table_entries =
        metadata.max_blocks > 0 && metadata.total_tokens > 0
            ? static_cast<uint64_t>(metadata.max_blocks) *
                  static_cast<uint64_t>(metadata.total_tokens)
            : 0;
    if (!metadata.valid || block_size <= 0 || physical_rows <= 0 ||
        vocab_size <= 0 || metadata.total_tokens <= 0 ||
        metadata.prefill_tokens < 0 || metadata.decode_tokens < 0 ||
        metadata.prefill_tokens + metadata.decode_tokens !=
            metadata.total_tokens ||
        metadata.prefill_tokens != staged_prefill_tokens ||
        metadata.decode_tokens != static_cast<int>(decode_token_ids.size()) ||
        metadata.max_blocks <= 0 || metadata.sequences.empty() ||
        metadata.positions.size() !=
            static_cast<size_t>(metadata.total_tokens) ||
        metadata.sequence_rows.size() !=
            static_cast<size_t>(metadata.total_tokens) ||
        metadata.kv_lens.size() !=
            static_cast<size_t>(metadata.total_tokens) ||
        expected_table_entries != metadata.expanded_block_table.size()) {
        return false;
    }

    for (const int token_id : decode_token_ids) {
        if (token_id < 0 || token_id >= vocab_size) {
            return false;
        }
    }

    std::vector<bool> selected(
        static_cast<size_t>(metadata.total_tokens), false);
    for (const int row : metadata.selected_rows) {
        if (row < 0 || row >= metadata.total_tokens ||
            selected[static_cast<size_t>(row)]) {
            return false;
        }
        selected[static_cast<size_t>(row)] = true;
    }

    for (int row = 0; row < metadata.total_tokens; ++row) {
        const int position = metadata.positions[static_cast<size_t>(row)];
        const int kv_len = metadata.kv_lens[static_cast<size_t>(row)];
        const int sequence_row =
            metadata.sequence_rows[static_cast<size_t>(row)];
        const int64_t expected_kv_len =
            static_cast<int64_t>(position) + 1;
        if (position < 0 ||
            expected_kv_len > std::numeric_limits<int>::max() ||
            kv_len != expected_kv_len || sequence_row < 0 ||
            sequence_row >= static_cast<int>(metadata.sequences.size())) {
            return false;
        }

        const int64_t needed_blocks =
            (static_cast<int64_t>(kv_len) + block_size - 1) / block_size;
        if (needed_blocks <= 0 || needed_blocks > metadata.max_blocks) {
            return false;
        }
        const size_t table_offset =
            static_cast<size_t>(row) * metadata.max_blocks;
        for (int64_t logical = 0; logical < needed_blocks; ++logical) {
            const int physical_block = metadata.expanded_block_table[
                table_offset + static_cast<size_t>(logical)];
            const int64_t physical_end =
                (static_cast<int64_t>(physical_block) + 1) * block_size;
            if (physical_block < 0 || physical_end > physical_rows) {
                return false;
            }
        }
    }
    return true;
}

} // namespace funasr

#endif
