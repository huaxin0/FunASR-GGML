#include "compute/gpu_embedding_pool.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <cstdio>
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
        std::printf("  [FAIL] %s: expected %lld, got %lld (line %d)\n", \
                    name, (long long)(expected), (long long)(actual), __LINE__); \
    } \
} while (0)

namespace {

struct BackendTensor {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor* tensor = nullptr;

    BackendTensor(ggml_backend_t backend, int dim, int tokens) {
        const ggml_init_params params = {
            ggml_tensor_overhead() * 2,
            nullptr,
            true,
        };
        ctx = ggml_init(params);
        if (!ctx) {
            return;
        }
        tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tokens);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }

    ~BackendTensor() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }

    BackendTensor(const BackendTensor&) = delete;
    BackendTensor& operator=(const BackendTensor&) = delete;
};

std::vector<float> read_view(
        funasr::GPUEmbeddingPool& pool,
        const funasr::GPUEmbeddingHandle& handle,
        int offset,
        int count) {
    const ggml_init_params params = {
        ggml_tensor_overhead() * 2,
        nullptr,
        true,
    };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        return {};
    }

    ggml_tensor* tensor = pool.view(ctx, handle, offset, count);
    std::vector<float> values;
    if (tensor) {
        values.resize(static_cast<size_t>(count) * handle.embedding_dim);
        ggml_backend_tensor_get(
            tensor, values.data(), 0, values.size() * sizeof(float));
    }
    ggml_free(ctx);
    return values;
}

void test_capacity_generation_and_stale_handle() {
    std::printf("\n--- Test 1: Capacity, reuse, and stale handles ---\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    TEST_ASSERT(backend != nullptr, "CPU backend initializes");

    {
        funasr::GPUEmbeddingPool pool(backend, 2);
        TEST_EQ(pool.capacity(), 2, "pool capacity");
        TEST_EQ(pool.free_count(), 2, "initial free slots");

        const auto first = pool.acquire(4, 3);
        const auto second = pool.acquire(2, 3);
        const auto blocked = pool.acquire(1, 3);
        TEST_ASSERT(first.valid(), "first acquire succeeds");
        TEST_ASSERT(second.valid(), "second acquire succeeds");
        TEST_ASSERT(!blocked.valid(), "third acquire observes backpressure");
        TEST_EQ(pool.free_count(), 0, "no free slot after two acquires");
        TEST_ASSERT(pool.owns(first), "pool owns first handle");

        const size_t bytes_before_reuse = pool.reserved_bytes();
        TEST_ASSERT(bytes_before_reuse >= 6 * sizeof(float),
                    "reserved bytes cover active tensors");
        TEST_ASSERT(pool.release(first), "first release succeeds");
        TEST_EQ(pool.free_count(), 1, "one slot returns to free list");

        const auto reused = pool.acquire(3, 3);
        TEST_ASSERT(reused.valid(), "released slot can be reused");
        TEST_EQ(reused.slot, first.slot, "same slot is reused");
        TEST_ASSERT(reused.generation != first.generation,
                    "reused slot has a new generation");
        TEST_ASSERT(!pool.owns(first), "old generation is stale");
        TEST_ASSERT(!pool.release(first), "stale release is rejected");
        TEST_ASSERT(pool.owns(reused), "stale release preserves new owner");
        TEST_EQ(pool.free_count(), 0, "stale release does not free slot");

        TEST_ASSERT(pool.release(reused), "reused handle releases");
        TEST_ASSERT(pool.release(second), "second handle releases");
        TEST_EQ(pool.free_count(), 2, "all slots are free again");
        TEST_ASSERT(pool.reserved_bytes() >= bytes_before_reuse,
                    "backing buffers remain reserved for reuse");
    }

    ggml_backend_free(backend);
}

void test_invalid_inputs_and_ranges() {
    std::printf("\n--- Test 2: Invalid inputs and ranges ---\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    {
        funasr::GPUEmbeddingPool pool(backend, 1);
        TEST_ASSERT(!pool.acquire(0, 3).valid(),
                    "zero token acquire is rejected");
        TEST_ASSERT(!pool.acquire(3, 0).valid(),
                    "zero dimension acquire is rejected");
        TEST_EQ(pool.free_count(), 1, "invalid acquire consumes no slot");

        const auto handle = pool.acquire(4, 3);
        const float source[3] = {1.0f, 2.0f, 3.0f};
        TEST_ASSERT(!pool.set_host(handle, -1, source, 1),
                    "negative destination offset is rejected");
        TEST_ASSERT(!pool.set_host(handle, 4, source, 1),
                    "destination overflow is rejected");
        TEST_ASSERT(!pool.set_host(handle, 0, nullptr, 1),
                    "null host source is rejected");
        TEST_ASSERT(!pool.set_host(handle, 0, source, 0),
                    "zero host token count is rejected");

        BackendTensor wrong_dim(backend, 2, 4);
        TEST_ASSERT(!pool.copy_tensor(handle, 0, nullptr, 0, 1),
                    "null tensor source is rejected");
        TEST_ASSERT(!pool.copy_tensor(
                        handle, 0, wrong_dim.tensor, 0, 1),
                    "source dimension mismatch is rejected");

        const ggml_init_params params = {
            ggml_tensor_overhead() * 2, nullptr, true};
        ggml_context* view_ctx = ggml_init(params);
        TEST_ASSERT(pool.view(view_ctx, handle, -1, 1) == nullptr,
                    "negative view offset is rejected");
        TEST_ASSERT(pool.view(view_ctx, handle, 3, 2) == nullptr,
                    "view overflow is rejected");
        TEST_ASSERT(pool.view(nullptr, handle, 0, 1) == nullptr,
                    "null view context is rejected");
        ggml_free(view_ctx);

        TEST_ASSERT(pool.release(handle), "valid handle releases");
        TEST_ASSERT(!pool.set_host(handle, 0, source, 1),
                    "stale handle cannot be written");
    }
    ggml_backend_free(backend);
}

void test_host_write_and_view() {
    std::printf("\n--- Test 3: Host write and backend view ---\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    {
        funasr::GPUEmbeddingPool pool(backend, 1);
        const auto handle = pool.acquire(4, 3);
        const std::vector<float> source = {
             0,  1,  2,
            10, 11, 12,
            20, 21, 22,
            30, 31, 32,
        };
        TEST_ASSERT(pool.set_host(handle, 0, source.data(), 4),
                    "host data upload succeeds");

        const std::vector<float> actual = read_view(pool, handle, 1, 2);
        const std::vector<float> expected = {10, 11, 12, 20, 21, 22};
        TEST_ASSERT(actual == expected,
                    "view reads the requested token-major rows");
    }
    ggml_backend_free(backend);
}

void test_backend_tensor_copy_with_offsets() {
    std::printf("\n--- Test 4: Backend tensor copy with offsets ---\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    {
        funasr::GPUEmbeddingPool pool(backend, 1);
        const auto handle = pool.acquire(5, 3);
        const std::vector<float> zeros(15, 0.0f);
        TEST_ASSERT(pool.set_host(handle, 0, zeros.data(), 5),
                    "destination is initialized");

        BackendTensor source(backend, 3, 4);
        const std::vector<float> source_data = {
             0,  1,  2,
            10, 11, 12,
            20, 21, 22,
            30, 31, 32,
        };
        ggml_backend_tensor_set(
            source.tensor, source_data.data(), 0,
            source_data.size() * sizeof(float));

        TEST_ASSERT(pool.copy_tensor(handle, 2, source.tensor, 1, 2),
                    "backend tensor slice copy succeeds");
        const std::vector<float> actual = read_view(pool, handle, 2, 2);
        const std::vector<float> expected = {10, 11, 12, 20, 21, 22};
        TEST_ASSERT(actual == expected,
                    "copy preserves source and destination offsets");

        TEST_ASSERT(!pool.copy_tensor(handle, 4, source.tensor, 0, 2),
                    "destination copy overflow is rejected");
        TEST_ASSERT(!pool.copy_tensor(handle, 0, source.tensor, 3, 2),
                    "source copy overflow is rejected");
    }
    ggml_backend_free(backend);
}

}  // namespace

int main() {
    std::printf("========================================\n");
    std::printf("GPU Embedding Pool Tests\n");
    std::printf("========================================\n");

    test_capacity_generation_and_stale_handle();
    test_invalid_inputs_and_ranges();
    test_host_write_and_view();
    test_backend_tensor_copy_with_offsets();

    std::printf("\n========================================\n");
    std::printf("Results: %d passed, %d failed\n",
                tests_passed, tests_failed);
    std::printf("========================================\n");
    return tests_failed == 0 ? 0 : 1;
}
