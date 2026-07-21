#include "compute/gpu_embedding_pool.hpp"

#include <ggml-alloc.h>

#include <limits>

namespace funasr {

struct GPUEmbeddingPool::Slot {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor* tensor = nullptr;
    uint64_t generation = 0;
    int capacity_tokens = 0;
    int embedding_dim = 0;
    int owner_token_count = 0;
    bool in_use = false;

    ~Slot() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

GPUEmbeddingPool::GPUEmbeddingPool(ggml_backend_t backend, int max_slots)
    : backend_(backend) {
    if (max_slots <= 0) {
        return;
    }
    slots_.reserve(static_cast<size_t>(max_slots));
    for (int i = 0; i < max_slots; ++i) {
        slots_.push_back(std::make_unique<Slot>());
    }
}

GPUEmbeddingPool::~GPUEmbeddingPool() = default;

bool GPUEmbeddingPool::valid_range(int extent, int offset, int count) {
    if (extent < 0 || offset < 0 || count <= 0) {
        return false;
    }
    const int64_t end = static_cast<int64_t>(offset) + count;
    return end <= static_cast<int64_t>(extent);
}

bool GPUEmbeddingPool::embedding_bytes(
        int token_count, int embedding_dim, size_t& bytes) {
    if (token_count < 0 || embedding_dim <= 0) {
        return false;
    }

    const uint64_t size_limit =
        static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    const uint64_t int64_limit =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    const uint64_t limit = size_limit < int64_limit ? size_limit : int64_limit;
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

bool GPUEmbeddingPool::rebuild_backing(
        Slot& slot, int token_count, int embedding_dim) {
    size_t tensor_bytes = 0;
    if (token_count <= 0 ||
        !embedding_bytes(token_count, embedding_dim, tensor_bytes)) {
        return false;
    }
    (void)tensor_bytes;

    const ggml_init_params params = {
        ggml_tensor_overhead() * 2,
        nullptr,
        true,
    };
    ggml_context* new_ctx = ggml_init(params);
    if (!new_ctx) {
        return false;
    }

    ggml_tensor* new_tensor =
        ggml_new_tensor_2d(new_ctx, GGML_TYPE_F32, embedding_dim, token_count);
    ggml_backend_buffer_t new_buffer =
        ggml_backend_alloc_ctx_tensors(new_ctx, backend_);
    if (!new_buffer) {
        ggml_free(new_ctx);
        return false;
    }

    if (slot.buffer) {
        ggml_backend_buffer_free(slot.buffer);
    }
    if (slot.ctx) {
        ggml_free(slot.ctx);
    }
    slot.ctx = new_ctx;
    slot.buffer = new_buffer;
    slot.tensor = new_tensor;
    slot.capacity_tokens = token_count;
    slot.embedding_dim = embedding_dim;
    return true;
}

GPUEmbeddingHandle GPUEmbeddingPool::claim_slot(
        size_t slot_index, int token_count, int embedding_dim) {
    Slot& slot = *slots_[slot_index];
    ++slot.generation;
    if (slot.generation == 0) {
        ++slot.generation;
    }
    slot.owner_token_count = token_count;
    slot.in_use = true;
    return {
        static_cast<int>(slot_index),
        slot.generation,
        token_count,
        embedding_dim,
    };
}

GPUEmbeddingHandle GPUEmbeddingPool::acquire(
        int token_count, int embedding_dim) {
    size_t tensor_bytes = 0;
    if (!backend_ || token_count <= 0 || embedding_dim <= 0 ||
        !embedding_bytes(token_count, embedding_dim, tensor_bytes)) {
        return {};
    }
    (void)tensor_bytes;

    size_t best_fit = slots_.size();
    for (size_t i = 0; i < slots_.size(); ++i) {
        const Slot& slot = *slots_[i];
        if (!slot.in_use && slot.tensor &&
            slot.embedding_dim == embedding_dim &&
            slot.capacity_tokens >= token_count &&
            (best_fit == slots_.size() ||
             slot.capacity_tokens < slots_[best_fit]->capacity_tokens)) {
            best_fit = i;
        }
    }
    if (best_fit != slots_.size()) {
        return claim_slot(best_fit, token_count, embedding_dim);
    }

    for (size_t i = 0; i < slots_.size(); ++i) {
        Slot& slot = *slots_[i];
        if (!slot.in_use) {
            if (!rebuild_backing(slot, token_count, embedding_dim)) {
                return {};
            }
            return claim_slot(i, token_count, embedding_dim);
        }
    }
    return {};
}

bool GPUEmbeddingPool::owns(const GPUEmbeddingHandle& handle) const {
    if (!handle.valid() || handle.slot >= static_cast<int>(slots_.size())) {
        return false;
    }
    const Slot& slot = *slots_[static_cast<size_t>(handle.slot)];
    return slot.in_use && slot.generation == handle.generation &&
           slot.owner_token_count == handle.token_count &&
           slot.embedding_dim == handle.embedding_dim;
}

bool GPUEmbeddingPool::release(const GPUEmbeddingHandle& handle) {
    if (!owns(handle)) {
        return false;
    }
    Slot& slot = *slots_[static_cast<size_t>(handle.slot)];
    slot.in_use = false;
    slot.owner_token_count = 0;
    return true;
}

int GPUEmbeddingPool::free_count() const {
    int count = 0;
    for (const auto& slot : slots_) {
        if (!slot->in_use) {
            ++count;
        }
    }
    return count;
}

int GPUEmbeddingPool::capacity() const {
    return static_cast<int>(slots_.size());
}

size_t GPUEmbeddingPool::reserved_bytes() const {
    size_t bytes = 0;
    for (const auto& slot : slots_) {
        if (slot->buffer) {
            bytes += ggml_backend_buffer_get_size(slot->buffer);
        }
    }
    return bytes;
}

bool GPUEmbeddingPool::set_host(
        const GPUEmbeddingHandle& handle,
        int token_offset,
        const float* source,
        int token_count) {
    if (!owns(handle) || !source ||
        !valid_range(handle.token_count, token_offset, token_count)) {
        return false;
    }

    const Slot& slot = *slots_[static_cast<size_t>(handle.slot)];
    size_t byte_offset = 0;
    size_t byte_count = 0;
    if (!embedding_bytes(token_offset, handle.embedding_dim, byte_offset) ||
        !embedding_bytes(token_count, handle.embedding_dim, byte_count) ||
        byte_count == 0 ||
        byte_offset > std::numeric_limits<size_t>::max() - byte_count) {
        return false;
    }
    ggml_backend_tensor_set(
        slot.tensor, source, byte_offset, byte_count);
    return true;
}

bool GPUEmbeddingPool::copy_tensor(
        const GPUEmbeddingHandle& handle,
        int destination_offset,
        ggml_tensor* source,
        int source_offset,
        int count) {
    if (!owns(handle) || !source || source->type != GGML_TYPE_F32 ||
        !source->buffer || !source->data || source->ne[0] <= 0 ||
        source->ne[1] <= 0 || source->ne[2] != 1 || source->ne[3] != 1 ||
        source->ne[0] != handle.embedding_dim ||
        !ggml_is_contiguous(source) ||
        !valid_range(handle.token_count, destination_offset, count) ||
        source->ne[1] > std::numeric_limits<int>::max() ||
        !valid_range(static_cast<int>(source->ne[1]), source_offset, count)) {
        return false;
    }

    const size_t metadata_size = ggml_tensor_overhead() * 4;
    const ggml_init_params params = {metadata_size, nullptr, true};
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    Slot& slot = *slots_[static_cast<size_t>(handle.slot)];
    ggml_tensor* source_view = ggml_view_2d(
        ctx, source, handle.embedding_dim, count, source->nb[1],
        static_cast<size_t>(source_offset) * source->nb[1]);
    ggml_tensor* destination_view = ggml_view_2d(
        ctx, slot.tensor, handle.embedding_dim, count, slot.tensor->nb[1],
        static_cast<size_t>(destination_offset) * slot.tensor->nb[1]);
    const bool initialized =
        ggml_backend_view_init(source_view) == GGML_STATUS_SUCCESS &&
        ggml_backend_view_init(destination_view) == GGML_STATUS_SUCCESS;
    if (initialized) {
        ggml_backend_tensor_copy(source_view, destination_view);
    }
    ggml_free(ctx);
    return initialized;
}

ggml_tensor* GPUEmbeddingPool::view(
        ggml_context* ctx,
        const GPUEmbeddingHandle& handle,
        int offset,
        int count) const {
    if (!ctx || !owns(handle) ||
        !valid_range(handle.token_count, offset, count)) {
        return nullptr;
    }

    const Slot& slot = *slots_[static_cast<size_t>(handle.slot)];
    ggml_tensor* result = ggml_view_2d(
        ctx, slot.tensor, handle.embedding_dim, count, slot.tensor->nb[1],
        static_cast<size_t>(offset) * slot.tensor->nb[1]);
    if (ggml_backend_view_init(result) != GGML_STATUS_SUCCESS) {
        return nullptr;
    }
    return result;
}

}  // namespace funasr
