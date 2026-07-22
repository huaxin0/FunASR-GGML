#ifndef FUNASR_PIPELINE_GPU_PROMPT_HPP
#define FUNASR_PIPELINE_GPU_PROMPT_HPP

#include "compute/gpu_embedding_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace funasr {

struct GPUPromptLayout {
    int cached_prefix_tokens = 0;
    int stored_prefix_tokens = 0;
    int audio_tokens = 0;
    int suffix_tokens = 0;
    int prompt_tokens = 0;
    int stored_tokens = 0;
    bool valid = false;
};

inline GPUPromptLayout make_gpu_prompt_layout(
        int prefix_tokens,
        int cached_prefix_tokens,
        int audio_tokens,
        int suffix_tokens) {
    GPUPromptLayout layout;
    if (prefix_tokens < 0 || cached_prefix_tokens < 0 ||
        audio_tokens <= 0 || suffix_tokens < 0 ||
        (cached_prefix_tokens != 0 &&
         cached_prefix_tokens != prefix_tokens)) {
        return layout;
    }

    const int64_t prompt_tokens =
        static_cast<int64_t>(prefix_tokens) + audio_tokens + suffix_tokens;
    const int stored_prefix_tokens = cached_prefix_tokens == 0
        ? prefix_tokens
        : 0;
    const int64_t stored_tokens =
        static_cast<int64_t>(stored_prefix_tokens) +
        audio_tokens + suffix_tokens;
    if (prompt_tokens > std::numeric_limits<int>::max() ||
        stored_tokens > std::numeric_limits<int>::max()) {
        return layout;
    }

    layout.cached_prefix_tokens = cached_prefix_tokens;
    layout.stored_prefix_tokens = stored_prefix_tokens;
    layout.audio_tokens = audio_tokens;
    layout.suffix_tokens = suffix_tokens;
    layout.prompt_tokens = static_cast<int>(prompt_tokens);
    layout.stored_tokens = static_cast<int>(stored_tokens);
    layout.valid = true;
    return layout;
}

struct PreparedGPUPrompt {
    GPUEmbeddingHandle embeddings;
    int request_id = -1;
    int prompt_tokens = 0;
    int cached_prefix_tokens = 0;
    int audio_frames = 0;
    float encoder_ms = 0.0f;
    int frontend_batch_size = 1;
    bool ok = false;
};

struct GPUPromptChunkRange {
    int absolute_offset = 0;
    int local_offset = 0;
    int token_count = 0;
    bool produces_logits = false;
    bool valid = false;
};

inline GPUPromptChunkRange make_gpu_prompt_chunk_range(
        const PreparedGPUPrompt& prepared,
        int absolute_offset,
        int requested_tokens) {
    GPUPromptChunkRange range;
    if (!prepared.ok || !prepared.embeddings.valid() ||
        prepared.prompt_tokens <= 0 || prepared.cached_prefix_tokens < 0 ||
        requested_tokens <= 0) {
        return range;
    }

    const int64_t stored_tokens =
        static_cast<int64_t>(prepared.prompt_tokens) -
        prepared.cached_prefix_tokens;
    if (stored_tokens <= 0 ||
        stored_tokens != prepared.embeddings.token_count ||
        absolute_offset < prepared.cached_prefix_tokens ||
        absolute_offset >= prepared.prompt_tokens) {
        return range;
    }

    const int64_t local_offset =
        static_cast<int64_t>(absolute_offset) -
        prepared.cached_prefix_tokens;
    const int64_t remaining =
        static_cast<int64_t>(prepared.prompt_tokens) - absolute_offset;
    const int64_t token_count = std::min<int64_t>(requested_tokens, remaining);
    const int64_t local_end = local_offset + token_count;
    if (local_offset < 0 || token_count <= 0 ||
        local_end > prepared.embeddings.token_count ||
        local_offset > std::numeric_limits<int>::max() ||
        token_count > std::numeric_limits<int>::max()) {
        return range;
    }

    range.absolute_offset = absolute_offset;
    range.local_offset = static_cast<int>(local_offset);
    range.token_count = static_cast<int>(token_count);
    range.produces_logits =
        static_cast<int64_t>(absolute_offset) + token_count ==
        prepared.prompt_tokens;
    range.valid = true;
    return range;
}

}  // namespace funasr

#endif  // FUNASR_PIPELINE_GPU_PROMPT_HPP
