#include "compute/gpu_mixed_runner.hpp"

#include "compute/gpu_context.hpp"
#ifdef FUNASR_USE_CUDA
#include "compute/llm_ops_gpu.hpp"
#endif

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <utility>

namespace funasr {

namespace {

bool packed_embedding_bytes(int token_count, int embedding_dim, size_t& bytes) {
    if (token_count < 0 || embedding_dim <= 0) {
        return false;
    }

    const uint64_t size_limit =
        static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    const uint64_t int64_limit =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    const uint64_t limit = std::min(size_limit, int64_limit);
    const uint64_t tokens = static_cast<uint64_t>(token_count);
    const uint64_t dim = static_cast<uint64_t>(embedding_dim);
    if (tokens > limit / dim) {
        return false;
    }
    const uint64_t elements = tokens * dim;
    if (elements > limit / sizeof(float)) {
        return false;
    }
    bytes = static_cast<size_t>(elements * sizeof(float));
    return true;
}

bool valid_int_range(int extent, int offset, int count) {
    if (extent < 0 || offset < 0 || count <= 0) {
        return false;
    }
    const int64_t end = static_cast<int64_t>(offset) +
                        static_cast<int64_t>(count);
    return end <= static_cast<int64_t>(extent);
}

int graph_cache_capacity_from_env() {
    constexpr int kDefaultCapacity = 8;
    constexpr int kMaxCapacity = 64;
    const char* value = std::getenv("FUNASR_MIXED_GRAPH_CACHE_ENTRIES");
    if (!value || *value == '\0') {
        return kDefaultCapacity;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0) {
        return kDefaultCapacity;
    }
    return static_cast<int>(std::min<long>(parsed, kMaxCapacity));
}

void hash_combine(uint64_t& seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

}  // namespace

GPUMixedRunner::GPUMixedRunner(ggml_backend_t backend, int embedding_dim)
    : backend_(backend), embedding_dim_(embedding_dim),
      graph_cache_capacity_(graph_cache_capacity_from_env()) {
    if (backend_) {
        graph_allocator_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
    }
}

GPUMixedRunner::~GPUMixedRunner() {
    clear_graph_cache();
    if (graph_allocator_) {
        ggml_gallocr_free(graph_allocator_);
    }
    if (staging_buffer_) {
        ggml_backend_buffer_free(staging_buffer_);
    }
    if (staging_ctx_) {
        ggml_free(staging_ctx_);
    }
}

void GPUMixedRunner::clear_graph_cache() {
    if (backend_ && !graph_cache_.empty()) {
        ggml_backend_synchronize(backend_);
    }
    for (CachedGraph& cached : graph_cache_) {
        destroy_cached_graph(cached);
    }
    graph_cache_.clear();
}

bool GPUMixedRunner::GraphSignature::operator==(
        const GraphSignature& other) const {
    return prefill_tokens == other.prefill_tokens &&
           decode_tokens == other.decode_tokens &&
           total_tokens == other.total_tokens &&
           max_blocks == other.max_blocks &&
           block_size == other.block_size &&
           max_n_kv == other.max_n_kv &&
           selected_rows == other.selected_rows &&
           prefill_layout == other.prefill_layout;
}

uint64_t GPUMixedRunner::hash_signature(const GraphSignature& signature) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    hash_combine(hash, static_cast<uint64_t>(signature.prefill_tokens));
    hash_combine(hash, static_cast<uint64_t>(signature.decode_tokens));
    hash_combine(hash, static_cast<uint64_t>(signature.total_tokens));
    hash_combine(hash, static_cast<uint64_t>(signature.max_blocks));
    hash_combine(hash, static_cast<uint64_t>(signature.block_size));
    hash_combine(hash, static_cast<uint64_t>(signature.max_n_kv));
    hash_combine(hash, static_cast<uint64_t>(signature.selected_rows));
    for (const int value : signature.prefill_layout) {
        hash_combine(hash, static_cast<uint64_t>(
            static_cast<uint32_t>(value)));
    }
    return hash;
}

void GPUMixedRunner::destroy_cached_graph(CachedGraph& cached) {
#ifdef FUNASR_USE_CUDA
    if (backend_ && cached.cuda_graph_key) {
        ggml_backend_cuda_graph_reset(backend_, cached.cuda_graph_key);
    }
#endif
    if (cached.ctx) {
        ggml_free(cached.ctx);
    }
    cached = CachedGraph{};
}

void GPUMixedRunner::evict_lru_graph() {
    if (graph_cache_.empty()) {
        return;
    }
    CachedGraph& cached = graph_cache_.back();
    destroy_cached_graph(cached);
    graph_cache_.pop_back();
    ++graph_cache_evictions_;
}

void GPUMixedRunner::set_graph_cache_capacity(int capacity) {
    const int clamped = std::clamp(capacity, 1, 64);
    graph_cache_capacity_ = clamped;
    while (static_cast<int>(graph_cache_.size()) > graph_cache_capacity_) {
        evict_lru_graph();
    }
}

int GPUMixedRunner::graph_cache_capacity() const {
    return graph_cache_capacity_;
}

int GPUMixedRunner::graph_cache_entries() const {
    return static_cast<int>(graph_cache_.size());
}

void GPUMixedRunner::clear_batch_state() {
    ready_ = false;
    prefill_tokens_ = 0;
    decode_tokens_ = 0;
    total_tokens_ = 0;
    decode_token_ids_.clear();
}

bool GPUMixedRunner::validate_inputs(
        const GPUEmbeddingPool& pool,
        const std::vector<PackedPromptSlice>& prompt_slices,
        int prefill_tokens,
        const std::vector<int>& decode_token_ids) const {
    if (!backend_ || embedding_dim_ <= 0 || prefill_tokens < 0 ||
        prefill_tokens > kMaxPackedTokens ||
        prompt_slices.size() > static_cast<size_t>(kMaxPackedTokens) ||
        decode_token_ids.size() > static_cast<size_t>(kMaxPackedTokens)) {
        return false;
    }

    for (const int token_id : decode_token_ids) {
        if (token_id < 0) {
            return false;
        }
    }

    const int64_t total = static_cast<int64_t>(prefill_tokens) +
                          static_cast<int64_t>(decode_token_ids.size());
    if (total <= 0 || total > kMaxPackedTokens ||
        total > std::numeric_limits<int>::max()) {
        return false;
    }

    if (prefill_tokens == 0) {
        return prompt_slices.empty();
    }
    if (prompt_slices.empty()) {
        return false;
    }

    std::vector<std::pair<int64_t, int64_t>> destination_ranges;
    destination_ranges.reserve(prompt_slices.size());
    for (const PackedPromptSlice& slice : prompt_slices) {
        if (!pool.owns(slice.source) ||
            slice.source.embedding_dim != embedding_dim_ ||
            !valid_int_range(
                slice.source.token_count,
                slice.source_token_offset,
                slice.token_count) ||
            !valid_int_range(
                prefill_tokens,
                slice.destination_token_offset,
                slice.token_count)) {
            return false;
        }

        const int64_t begin = static_cast<int64_t>(
            slice.destination_token_offset);
        destination_ranges.emplace_back(
            begin, begin + static_cast<int64_t>(slice.token_count));
    }

    std::sort(destination_ranges.begin(), destination_ranges.end());
    int64_t next_destination = 0;
    for (const auto& range : destination_ranges) {
        if (range.first != next_destination) {
            return false;
        }
        next_destination = range.second;
    }
    return next_destination == static_cast<int64_t>(prefill_tokens);
}

bool GPUMixedRunner::ensure_capacity(int requested_tokens) {
    if (requested_tokens <= 0) {
        return true;
    }
    if (staging_tensor_ && staging_buffer_ && staging_ctx_ &&
        capacity_tokens_ >= requested_tokens) {
        return true;
    }

    size_t tensor_bytes = 0;
    if (requested_tokens > kMaxPackedTokens ||
        !packed_embedding_bytes(
            requested_tokens, embedding_dim_, tensor_bytes)) {
        return false;
    }
    (void)tensor_bytes;

    int new_capacity = requested_tokens;
    if (capacity_tokens_ > 0) {
        new_capacity = capacity_tokens_;
        while (new_capacity < requested_tokens) {
            if (new_capacity > kMaxPackedTokens / 2) {
                new_capacity = kMaxPackedTokens;
            } else {
                new_capacity *= 2;
            }
        }
    }
    if (!packed_embedding_bytes(new_capacity, embedding_dim_, tensor_bytes)) {
        return false;
    }

    const ggml_init_params params = {
        ggml_tensor_overhead() * 2,
        nullptr,
        true,
    };
    ggml_context* new_ctx = ggml_init(params);
    if (!new_ctx) {
        return false;
    }
    ggml_tensor* new_tensor = ggml_new_tensor_2d(
        new_ctx, GGML_TYPE_F32, embedding_dim_, new_capacity);
    if (!new_tensor) {
        ggml_free(new_ctx);
        return false;
    }
    ggml_backend_buffer_t new_buffer =
        ggml_backend_alloc_ctx_tensors(new_ctx, backend_);
    if (!new_buffer) {
        ggml_free(new_ctx);
        return false;
    }

    clear_graph_cache();
    if (staging_buffer_) {
        ggml_backend_buffer_free(staging_buffer_);
    }
    if (staging_ctx_) {
        ggml_free(staging_ctx_);
    }
    staging_ctx_ = new_ctx;
    staging_buffer_ = new_buffer;
    staging_tensor_ = new_tensor;
    capacity_tokens_ = new_capacity;
    return true;
}

bool GPUMixedRunner::stage_inputs(
        const GPUEmbeddingPool& pool,
        const std::vector<PackedPromptSlice>& prompt_slices,
        int prefill_tokens,
        const std::vector<int>& decode_token_ids) {
    clear_batch_state();
    if (!validate_inputs(pool, prompt_slices, prefill_tokens,
                         decode_token_ids)) {
        return false;
    }
    if (!ensure_capacity(prefill_tokens)) {
        return false;
    }

    for (const PackedPromptSlice& slice : prompt_slices) {
        const ggml_init_params params = {
            ggml_tensor_overhead() * 4,
            nullptr,
            true,
        };
        ggml_context* metadata_ctx = ggml_init(params);
        if (!metadata_ctx) {
            return false;
        }

        ggml_tensor* source_view = pool.view(
            metadata_ctx, slice.source, slice.source_token_offset,
            slice.token_count);
        ggml_tensor* destination_view = source_view ? ggml_view_2d(
            metadata_ctx,
            staging_tensor_,
            embedding_dim_,
            slice.token_count,
            staging_tensor_->nb[1],
            static_cast<size_t>(slice.destination_token_offset) *
                staging_tensor_->nb[1]) : nullptr;
        const bool initialized =
            source_view && destination_view &&
            ggml_backend_view_init(destination_view) == GGML_STATUS_SUCCESS;
        if (initialized) {
            ggml_backend_tensor_copy(source_view, destination_view);
        }
        ggml_free(metadata_ctx);
        if (!initialized) {
            return false;
        }
    }

    prefill_tokens_ = prefill_tokens;
    decode_tokens_ = static_cast<int>(decode_token_ids.size());
    total_tokens_ = prefill_tokens_ + decode_tokens_;
    decode_token_ids_ = decode_token_ids;
    ready_ = true;
    return true;
}

bool GPUMixedRunner::execute_paged(
        GPUContext& gpu_context,
        const PackedMixedMetadata& metadata,
        int block_size,
        std::vector<int>& next_tokens,
        MixedGraphExecutionInfo* graph_info) {
    next_tokens.clear();
    if (graph_info) {
        *graph_info = MixedGraphExecutionInfo{};
    }
#ifndef FUNASR_USE_CUDA
    (void)gpu_context;
    (void)metadata;
    (void)block_size;
    return false;
#else
    const LLMConfig& cfg = gpu_context.config();
    GPUKVCache& cache = gpu_context.kv_cache();
    const int physical_rows =
        cache.physical_rows > 0 ? cache.physical_rows : cache.n_ctx;
    if (!ready_ || !graph_allocator_ || backend_ != gpu_context.backend() ||
        !cache.initialized ||
        !validate_packed_mixed_execution(
            metadata, block_size, physical_rows, cfg.vocab_size,
            prefill_tokens_, decode_token_ids_)) {
        return false;
    }

    const int exact_max_n_kv = *std::max_element(
        metadata.kv_lens.begin(), metadata.kv_lens.end());
    const int64_t bucket_n_kv =
        static_cast<int64_t>(metadata.max_blocks) * block_size;
    if (exact_max_n_kv <= 0 || bucket_n_kv < exact_max_n_kv ||
        bucket_n_kv > std::numeric_limits<int>::max()) {
        return false;
    }
    const int max_n_kv = static_cast<int>(bucket_n_kv);

    std::vector<GPUPackedPrefillLayout> prefill_layouts;
    std::vector<const PackedSequenceSource*> prefill_sources;
    int packed_row = 0;
    int packed_prefill_tokens = 0;
    bool saw_decode = false;
    for (const PackedSequenceSource& source : metadata.sequences) {
        const int token_count = source.scheduled.num_tokens;
        if (source.scheduled.input_kind == UnifiedInputKind::Prompt) {
            if (saw_decode) {
                return false;
            }
            GPUPackedPrefillLayout layout;
            layout.row_offset = packed_row;
            layout.token_count = token_count;
            layout.n_past = source.scheduled.token_offset;
            prefill_layouts.push_back(layout);
            prefill_sources.push_back(&source);
            packed_prefill_tokens += token_count;
        } else {
            saw_decode = true;
        }
        packed_row += token_count;
    }
    if (packed_row != metadata.total_tokens ||
        packed_prefill_tokens != metadata.prefill_tokens) {
        return false;
    }

    GraphSignature signature;
    signature.prefill_tokens = prefill_tokens_;
    signature.decode_tokens = decode_tokens_;
    signature.total_tokens = metadata.total_tokens;
    signature.max_blocks = metadata.max_blocks;
    signature.block_size = block_size;
    signature.max_n_kv = max_n_kv;
    signature.selected_rows = static_cast<int>(metadata.selected_rows.size());
    signature.prefill_layout.reserve(prefill_layouts.size() * 3);
    for (const GPUPackedPrefillLayout& layout : prefill_layouts) {
        signature.prefill_layout.push_back(layout.row_offset);
        signature.prefill_layout.push_back(layout.token_count);
        signature.prefill_layout.push_back(layout.n_past);
    }
    const uint64_t signature_hash = hash_signature(signature);
    uint32_t& shape_observations =
        graph_shape_observations_[signature_hash];
    const bool retain_new_graph = shape_observations > 0;
    if (shape_observations < std::numeric_limits<uint32_t>::max()) {
        ++shape_observations;
    }

    ggml_context* ctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor* decode_ids_input = nullptr;
    ggml_tensor* block_table_input = nullptr;
    ggml_tensor* position_input = nullptr;
    ggml_tensor* kv_lens_input = nullptr;
    ggml_tensor* selected_rows_input = nullptr;
    std::vector<ggml_tensor*> prefill_past_row_inputs;
    std::vector<ggml_tensor*> prefill_mask_inputs;
    ggml_tensor* token = nullptr;

    auto cache_it = std::find_if(
        graph_cache_.begin(), graph_cache_.end(),
        [&](const CachedGraph& cached) {
            return cached.valid && cached.signature == signature;
        });
    const bool cache_hit = cache_it != graph_cache_.end();
    bool transient_graph = false;
    if (cache_hit) {
        graph_cache_.splice(graph_cache_.begin(), graph_cache_, cache_it);
        CachedGraph& cached = graph_cache_.front();
        ctx = cached.ctx;
        graph = cached.graph;
        decode_ids_input = cached.decode_ids_input;
        block_table_input = cached.block_table_input;
        position_input = cached.position_input;
        kv_lens_input = cached.kv_lens_input;
        selected_rows_input = cached.selected_rows_input;
        prefill_past_row_inputs = cached.prefill_past_row_inputs;
        prefill_mask_inputs = cached.prefill_mask_inputs;
        token = cached.token;
    } else {
        const ggml_init_params params = {
            32ULL * 1024 * 1024,
            nullptr,
            true,
        };
        ctx = ggml_init(params);
        if (!ctx) {
            return false;
        }

        ggml_tensor* prompt_rows = nullptr;
        if (prefill_tokens_ > 0) {
            prompt_rows = ggml_view_2d(
                ctx, staging_tensor_, embedding_dim_, prefill_tokens_,
                staging_tensor_->nb[1], 0);
            if (ggml_backend_view_init(prompt_rows) != GGML_STATUS_SUCCESS) {
                ggml_free(ctx);
                return false;
            }
        }

        ggml_tensor* decode_rows = nullptr;
        if (decode_tokens_ > 0) {
            decode_ids_input = ggml_new_tensor_1d(
                ctx, GGML_TYPE_I32, decode_tokens_);
            ggml_set_name(decode_ids_input, "packed_decode_token_ids");
            ggml_set_input(decode_ids_input);
            decode_rows = ggml_get_rows(
                ctx, gpu_context.weights().embed_tokens, decode_ids_input);
            decode_rows = ggml_reshape_2d(
                ctx, decode_rows, embedding_dim_, decode_tokens_);
        }

        ggml_tensor* hidden_states = nullptr;
        if (prompt_rows && decode_rows) {
            hidden_states = ggml_concat(ctx, prompt_rows, decode_rows, 1);
        } else {
            hidden_states = prompt_rows ? prompt_rows : decode_rows;
        }
        if (!hidden_states || hidden_states->ne[1] != metadata.total_tokens) {
            ggml_free(ctx);
            return false;
        }

        std::vector<ggml_tensor*> kv_write_ops;
        ggml_tensor* logits = gpu_llm_forward_paged_packed(
            ctx, hidden_states, gpu_context.weights(), cache,
            metadata.max_blocks, block_size, max_n_kv,
            static_cast<int>(metadata.selected_rows.size()),
            prefill_layouts, cfg,
            kv_write_ops, &block_table_input, &position_input,
            &kv_lens_input, &selected_rows_input,
            prefill_past_row_inputs, prefill_mask_inputs);
        ggml_tensor* graph_output = nullptr;
        if (!metadata.selected_rows.empty()) {
            token = ggml_argmax(ctx, logits);
            ggml_set_name(token, "packed_next_tokens");
            graph_output = token;
        } else {
            graph_output = ggml_sum(ctx, logits);
            ggml_set_name(graph_output, "packed_prefill_completion");
        }
        ggml_set_output(graph_output);

        graph = ggml_new_graph_custom(ctx, 65536, false);
        for (ggml_tensor* write : kv_write_ops) {
            ggml_build_forward_expand(graph, write);
        }
        ggml_build_forward_expand(graph, graph_output);
        const size_t generation_before =
            ggml_gallocr_get_buffer_generation(graph_allocator_);
        if (!ggml_gallocr_alloc_graph(graph_allocator_, graph)) {
            ggml_free(ctx);
            return false;
        }
        const size_t generation_after =
            ggml_gallocr_get_buffer_generation(graph_allocator_);
        if (generation_after != generation_before &&
            !graph_cache_.empty()) {
            clear_graph_cache();
        }
        if (retain_new_graph) {
            while (static_cast<int>(graph_cache_.size()) >=
                   graph_cache_capacity_) {
                evict_lru_graph();
            }
        } else {
            transient_graph = true;
        }

        CachedGraph cached;
        cached.ctx = ctx;
        cached.graph = graph;
        cached.decode_ids_input = decode_ids_input;
        cached.block_table_input = block_table_input;
        cached.position_input = position_input;
        cached.kv_lens_input = kv_lens_input;
        cached.selected_rows_input = selected_rows_input;
        cached.prefill_past_row_inputs = prefill_past_row_inputs;
        cached.prefill_mask_inputs = prefill_mask_inputs;
        cached.token = token;
        cached.signature = signature;
        cached.cuda_graph_key = ggml_graph_n_nodes(graph) > 0
            ? ggml_graph_node(graph, 0)
            : nullptr;
        cached.allocator_generation = generation_after;
        cached.valid = true;
        graph_cache_.push_front(std::move(cached));
    }

    if (decode_ids_input) {
        ggml_backend_tensor_set(
            decode_ids_input, decode_token_ids_.data(), 0,
            decode_token_ids_.size() * sizeof(int));
    }
    ggml_backend_tensor_set(
        block_table_input, metadata.expanded_block_table.data(), 0,
        metadata.expanded_block_table.size() * sizeof(int));
    ggml_backend_tensor_set(
        position_input, metadata.positions.data(), 0,
        metadata.positions.size() * sizeof(int));
    if (decode_tokens_ > 0) {
        ggml_backend_tensor_set(
            kv_lens_input, metadata.kv_lens.data(), 0,
            metadata.kv_lens.size() * sizeof(int));
    }
    if (selected_rows_input) {
        ggml_backend_tensor_set(
            selected_rows_input, metadata.selected_rows.data(), 0,
            metadata.selected_rows.size() * sizeof(int));
    }

    if (prefill_past_row_inputs.size() != prefill_layouts.size() ||
        prefill_mask_inputs.size() != prefill_layouts.size() ||
        prefill_sources.size() != prefill_layouts.size()) {
        clear_graph_cache();
        return false;
    }
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (size_t i = 0; i < prefill_layouts.size(); ++i) {
        const GPUPackedPrefillLayout& layout = prefill_layouts[i];
        const PackedSequenceSource& source = *prefill_sources[i];
        if (layout.n_past > 0) {
            ggml_tensor* rows_input = prefill_past_row_inputs[i];
            if (!rows_input ||
                rows_input->ne[0] != layout.n_past) {
                clear_graph_cache();
                return false;
            }
            std::vector<int> physical_rows_data(
                static_cast<size_t>(layout.n_past));
            for (int logical = 0; logical < layout.n_past; ++logical) {
                const int block_index = logical / block_size;
                if (block_index < 0 ||
                    block_index >= static_cast<int>(source.block_table.size())) {
                    clear_graph_cache();
                    return false;
                }
                physical_rows_data[static_cast<size_t>(logical)] =
                    source.block_table[static_cast<size_t>(block_index)] *
                        block_size +
                    logical % block_size;
            }
            ggml_backend_tensor_set(
                rows_input, physical_rows_data.data(), 0,
                physical_rows_data.size() * sizeof(int));
        }

        ggml_tensor* mask_input = prefill_mask_inputs[i];
        const int n_kv = layout.n_past + layout.token_count;
        if (!mask_input || mask_input->ne[0] != n_kv ||
            mask_input->ne[1] != layout.token_count) {
            clear_graph_cache();
            return false;
        }
        std::vector<ggml_fp16_t> mask_data(
            static_cast<size_t>(n_kv) * layout.token_count, zero);
        for (int query = 0; query < layout.token_count; ++query) {
            const int first_masked = layout.n_past + query + 1;
            for (int key = first_masked; key < n_kv; ++key) {
                mask_data[static_cast<size_t>(query) * n_kv + key] =
                    neg_inf;
            }
        }
        ggml_backend_tensor_set(
            mask_input, mask_data.data(), 0,
            mask_data.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_graph_compute(backend_, graph) !=
        GGML_STATUS_SUCCESS) {
        clear_graph_cache();
        return false;
    }

    if (token) {
        next_tokens.resize(metadata.selected_rows.size());
        ggml_backend_tensor_get(
            token, next_tokens.data(), 0,
            next_tokens.size() * sizeof(int));
    }
    if (transient_graph) {
        CachedGraph& transient = graph_cache_.front();
        destroy_cached_graph(transient);
        graph_cache_.pop_front();
    }
    if (graph_info) {
        graph_info->cache_hit = cache_hit;
        graph_info->shape_hash = signature_hash;
        graph_info->prefill_tokens = signature.prefill_tokens;
        graph_info->decode_tokens = signature.decode_tokens;
        graph_info->total_tokens = signature.total_tokens;
        graph_info->max_blocks = signature.max_blocks;
        graph_info->max_n_kv = signature.max_n_kv;
        graph_info->prefill_sequences = static_cast<int>(
            signature.prefill_layout.size() / 3);
        graph_info->cache_entries = graph_cache_entries();
        graph_info->cache_capacity = graph_cache_capacity_;
        graph_info->cache_evictions = graph_cache_evictions_;
    }
    return true;
#endif
}

ggml_tensor* GPUMixedRunner::prompt_staging() const {
    return ready_ && prefill_tokens_ > 0 ? staging_tensor_ : nullptr;
}

int GPUMixedRunner::prefill_tokens() const {
    return prefill_tokens_;
}

int GPUMixedRunner::decode_tokens() const {
    return decode_tokens_;
}

int GPUMixedRunner::total_tokens() const {
    return total_tokens_;
}

const std::vector<int>& GPUMixedRunner::decode_token_ids() const {
    return decode_token_ids_;
}

int GPUMixedRunner::capacity_tokens() const {
    return capacity_tokens_;
}

size_t GPUMixedRunner::reserved_bytes() const {
    return staging_buffer_ ? ggml_backend_buffer_get_size(staging_buffer_) : 0;
}

bool GPUMixedRunner::ready() const {
    return ready_;
}

}  // namespace funasr
