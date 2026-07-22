#ifndef FUNASR_COMPUTE_GPU_EMBEDDING_POOL_HPP
#define FUNASR_COMPUTE_GPU_EMBEDDING_POOL_HPP

#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace funasr {

struct GPUEmbeddingHandle {
    int slot = -1;
    uint64_t generation = 0;
    uint64_t pool_epoch = 0;
    int token_count = 0;
    int embedding_dim = 0;

    bool valid() const {
        return slot >= 0 && generation != 0 && pool_epoch != 0 &&
               token_count > 0 && embedding_dim > 0;
    }
};

class GPUEmbeddingPool {
public:
    GPUEmbeddingPool(ggml_backend_t backend, int max_slots);
    ~GPUEmbeddingPool();

    GPUEmbeddingPool(const GPUEmbeddingPool&) = delete;
    GPUEmbeddingPool& operator=(const GPUEmbeddingPool&) = delete;

    GPUEmbeddingHandle acquire(int token_count, int embedding_dim);
    bool owns(const GPUEmbeddingHandle& handle) const;
    bool release(const GPUEmbeddingHandle& handle);
    bool reserve_slots(int min_capacity);

    int free_count() const;
    int capacity() const;
    size_t reserved_bytes() const;

    bool set_host(
        const GPUEmbeddingHandle& handle,
        int token_offset,
        const float* source,
        int token_count);

    bool copy_tensor(
        const GPUEmbeddingHandle& handle,
        int destination_offset,
        ggml_tensor* source,
        int source_offset,
        int count);

    ggml_tensor* view(
        ggml_context* ctx,
        const GPUEmbeddingHandle& handle,
        int offset,
        int count) const;

private:
    struct Slot;

    bool rebuild_backing(Slot& slot, int token_count, int embedding_dim);
    bool owns_unlocked(const GPUEmbeddingHandle& handle) const;
    GPUEmbeddingHandle claim_slot(
        size_t slot_index, int token_count, int embedding_dim);
    static bool valid_range(int extent, int offset, int count);
    static bool embedding_bytes(
        int token_count, int embedding_dim, size_t& bytes);

    ggml_backend_t backend_ = nullptr;
    uint64_t pool_epoch_ = 0;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Slot>> slots_;
};

}  // namespace funasr

#endif  // FUNASR_COMPUTE_GPU_EMBEDDING_POOL_HPP
