#include "compute/gpu_embedding_pool.hpp"
#include "compute/gpu_mixed_runner.hpp"
#include "pipeline/gpu_prompt.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <cstdio>
#include <climits>
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

std::vector<float> read_staging(const funasr::GPUMixedRunner& runner) {
    ggml_tensor* tensor = runner.prompt_staging();
    if (!tensor || runner.prefill_tokens() <= 0) {
        return {};
    }
    std::vector<float> values(
        static_cast<size_t>(runner.prefill_tokens()) * tensor->ne[0]);
    ggml_backend_tensor_get(
        tensor, values.data(), 0, values.size() * sizeof(float));
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

void test_acquire_uses_best_fit_reusable_slot() {
    std::printf("\n--- Test 5: Best-fit reusable slot selection ---\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    {
        funasr::GPUEmbeddingPool pool(backend, 2);
        const auto small = pool.acquire(2, 3);
        const auto large = pool.acquire(8, 3);
        TEST_ASSERT(small.valid() && large.valid(),
                    "small and large backing slots are allocated");
        TEST_ASSERT(pool.release(small), "small slot releases");
        TEST_ASSERT(pool.release(large), "large slot releases");

        const size_t reserved_before = pool.reserved_bytes();
        const auto medium = pool.acquire(6, 3);
        TEST_ASSERT(medium.valid(), "medium request acquires a slot");
        TEST_EQ(medium.slot, large.slot,
                "medium request reuses the fitting large slot");
        TEST_EQ(pool.reserved_bytes(), reserved_before,
                "best-fit reuse does not allocate more backing memory");
    }
    ggml_backend_free(backend);
}

void test_copy_tensor_rejects_unsafe_sources() {
    std::printf("\n--- Test 6: Unsafe tensor sources are rejected ---\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    {
        funasr::GPUEmbeddingPool pool(backend, 1);
        const auto handle = pool.acquire(4, 3);

        BackendTensor padded_base(backend, 4, 3);
        const ggml_init_params view_params = {
            ggml_tensor_overhead() * 2, nullptr, true};
        ggml_context* view_ctx = ggml_init(view_params);
        ggml_tensor* non_contiguous = ggml_view_2d(
            view_ctx, padded_base.tensor, 3, 2,
            padded_base.tensor->nb[1], 0);
        TEST_ASSERT(
            ggml_backend_view_init(non_contiguous) == GGML_STATUS_SUCCESS,
            "non-contiguous source view is allocated");
        TEST_ASSERT(!ggml_is_contiguous(non_contiguous),
                    "source view is genuinely non-contiguous");
        TEST_ASSERT(!pool.copy_tensor(
                        handle, 0, non_contiguous, 0, 2),
                    "non-contiguous source is rejected without asserting");
        ggml_free(view_ctx);

        const ggml_init_params source_params = {
            ggml_tensor_overhead() * 2, nullptr, true};
        ggml_context* source_ctx = ggml_init(source_params);
        ggml_tensor* unallocated =
            ggml_new_tensor_2d(source_ctx, GGML_TYPE_F32, 3, 2);
        TEST_ASSERT(unallocated->buffer == nullptr,
                    "source tensor has no backend allocation");
        TEST_ASSERT(!pool.copy_tensor(handle, 0, unallocated, 0, 2),
                    "unallocated source is rejected without asserting");
        ggml_free(source_ctx);
    }
    ggml_backend_free(backend);
}

void test_extreme_shape_is_rejected_without_consuming_slot() {
    std::printf("\n--- Test 7: Shape byte overflow is rejected ---\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    {
        funasr::GPUEmbeddingPool pool(backend, 1);
        const int free_before = pool.free_count();
        const auto overflow = pool.acquire(INT_MAX, INT_MAX);
        TEST_ASSERT(!overflow.valid(), "overflowing shape is rejected");
        TEST_EQ(pool.free_count(), free_before,
                "overflowing shape consumes no slot");
        TEST_EQ(pool.reserved_bytes(), 0,
                "overflowing shape reserves no backing memory");
    }
    ggml_backend_free(backend);
}

void test_gpu_mixed_runner_stages_packed_slices() {
    std::printf("\n--- Test 8: GPU mixed runner staging ---\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    {
        funasr::GPUEmbeddingPool pool(backend, 2);
        const auto a = pool.acquire(4, 3);
        const auto b = pool.acquire(3, 3);
        const std::vector<float> a_data = {
             0,  1,  2,
            10, 11, 12,
            20, 21, 22,
            30, 31, 32,
        };
        const std::vector<float> b_data = {
            100, 101, 102,
            110, 111, 112,
            120, 121, 122,
        };
        TEST_ASSERT(pool.set_host(a, 0, a_data.data(), 4),
                    "runner source A uploads");
        TEST_ASSERT(pool.set_host(b, 0, b_data.data(), 3),
                    "runner source B uploads");

        funasr::GPUMixedRunner runner(backend, 3);
        const std::vector<funasr::PackedPromptSlice> slices = {
            {b, 0, 2, 2},
            {a, 1, 2, 0},
        };
        TEST_ASSERT(runner.stage_inputs(pool, slices, 4, {7, 9}),
                    "non-contiguous source slices stage successfully");
        TEST_ASSERT(runner.ready(), "successful staging is ready");
        TEST_EQ(runner.prefill_tokens(), 4, "prefill token count");
        TEST_EQ(runner.decode_tokens(), 2, "decode token count");
        TEST_EQ(runner.total_tokens(), 6, "total token count");
        TEST_ASSERT(runner.decode_token_ids() == std::vector<int>({7, 9}),
                    "decode ids are retained");
        TEST_ASSERT(read_staging(runner) == std::vector<float>({
             10, 11, 12,
             20, 21, 22,
            100,101,102,
            110,111,112,
        }), "staging follows destination offsets");

        const ggml_tensor* reused_tensor = runner.prompt_staging();
        const size_t reserved_before = runner.reserved_bytes();
        TEST_ASSERT(runner.stage_inputs(
                        pool, {{a, 0, 2, 0}}, 2, {}),
                    "smaller batch stages successfully");
        TEST_ASSERT(runner.prompt_staging() == reused_tensor,
                    "smaller batch reuses staging tensor");
        TEST_EQ(runner.reserved_bytes(), reserved_before,
                "smaller batch does not allocate more staging memory");

        TEST_ASSERT(pool.release(b), "B releases for staging growth case");
        const auto larger_b = pool.acquire(6, 3);
        const std::vector<float> larger_b_data = {
            100,101,102, 110,111,112, 120,121,122,
            130,131,132, 140,141,142, 150,151,152,
        };
        TEST_ASSERT(pool.set_host(
                        larger_b, 0, larger_b_data.data(), 6),
                    "larger B uploads");
        TEST_ASSERT(runner.stage_inputs(
                        pool, {{a, 0, 4, 0}, {larger_b, 0, 2, 4}}, 6, {}),
                    "larger batch stages successfully");
        TEST_ASSERT(runner.capacity_tokens() >= 6,
                    "staging capacity grows to requested size");
        TEST_ASSERT(runner.reserved_bytes() >= reserved_before,
                    "staging growth reserves enough memory");

        TEST_ASSERT(runner.stage_inputs(pool, {}, 0, {3, 4}),
                    "decode-only batch stages without prompt slices");
        TEST_ASSERT(runner.ready(), "decode-only batch is ready");
        TEST_ASSERT(runner.prompt_staging() == nullptr,
                    "decode-only batch does not expose stale prompt rows");
        TEST_EQ(runner.prefill_tokens(), 0, "decode-only prefill count");
        TEST_EQ(runner.decode_tokens(), 2, "decode-only decode count");
        TEST_EQ(runner.total_tokens(), 2, "decode-only total count");
    }
    ggml_backend_free(backend);
}

void test_gpu_mixed_runner_rejects_invalid_batches_and_clears_state() {
    std::printf("\n--- Test 9: GPU mixed runner validation ---\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    {
        funasr::GPUEmbeddingPool pool(backend, 2);
        const auto source = pool.acquire(4, 3);
        const auto other_dim = pool.acquire(2, 2);
        const std::vector<float> source_data = {
             0,  1,  2, 10, 11, 12, 20, 21, 22, 30, 31, 32,
        };
        TEST_ASSERT(pool.set_host(source, 0, source_data.data(), 4),
                    "validation source uploads");
        funasr::GPUMixedRunner runner(backend, 3);
        const std::vector<funasr::PackedPromptSlice> valid = {
            {source, 0, 2, 0}, {source, 2, 2, 2},
        };
        TEST_ASSERT(runner.stage_inputs(pool, valid, 4, {8}),
                    "validation baseline stages");
        const size_t reserved_before = runner.reserved_bytes();

        auto expect_failure = [&](const std::vector<funasr::PackedPromptSlice>& slices,
                                  int prefill, const std::vector<int>& decode,
                                  const char* message) {
            TEST_ASSERT(!runner.stage_inputs(pool, slices, prefill, decode),
                        message);
            TEST_ASSERT(!runner.ready(), "failed staging is not ready");
            TEST_EQ(runner.prefill_tokens(), 0, "failed prefill state clears");
            TEST_EQ(runner.decode_tokens(), 0, "failed decode state clears");
            TEST_EQ(runner.total_tokens(), 0, "failed total state clears");
            TEST_ASSERT(runner.decode_token_ids().empty(),
                        "failed decode ids clear");
            TEST_ASSERT(runner.reserved_bytes() == reserved_before,
                        "validation failure does not reallocate staging");
        };

        expect_failure({{source, 0, 3, 1}, {source, 2, 2, 0}}, 4, {8},
                       "overlapping destination slices are rejected");
        TEST_ASSERT(runner.stage_inputs(pool, valid, 4, {8}),
                    "baseline restages after overlap rejection");
        expect_failure({{source, 0, 1, 0}, {source, 2, 1, 2}}, 3, {8},
                       "destination gaps are rejected");
        TEST_ASSERT(runner.stage_inputs(pool, valid, 4, {8}),
                    "baseline restages after gap rejection");
        expect_failure({{source, 0, 1, 0}}, 2, {8},
                       "destination coverage mismatch is rejected");
        expect_failure({{source, 0, 1, 0}}, 1, {-1},
                       "negative decode ids are rejected");
        expect_failure({{other_dim, 0, 1, 0}}, 1, {8},
                       "embedding dimension mismatch is rejected");
        expect_failure({}, 0, {},
                       "empty total batch is rejected");
        expect_failure({}, 1, {8},
                       "nonzero prefill requires slices");

        const auto stale = source;
        TEST_ASSERT(pool.release(source), "source releases for stale test");
        const auto replacement = pool.acquire(4, 3);
        expect_failure({{stale, 0, 1, 0}}, 1, {8},
                       "stale source handle is rejected");
        TEST_ASSERT(pool.owns(replacement),
                    "stale runner input does not damage replacement owner");

        expect_failure({{replacement, INT_MAX, 1, 0}}, 1, {8},
                       "source range overflow is rejected");
        expect_failure({}, INT_MAX, {},
                       "packed token limit is rejected");
    }
    ggml_backend_free(backend);
}

void test_gpu_prompt_layout() {
    std::printf("\n--- Test 8: GPU prompt layout ---\n");

    const auto uncached = funasr::make_gpu_prompt_layout(11, 0, 499, 12);
    TEST_ASSERT(uncached.valid, "uncached prompt layout is valid");
    TEST_EQ(uncached.stored_prefix_tokens, 11,
            "uncached prompt stores the prefix");
    TEST_EQ(uncached.stored_tokens, 522,
            "uncached prompt stores the full sequence");
    TEST_EQ(uncached.prompt_tokens, 522,
            "uncached logical prompt length includes all tokens");
    TEST_EQ(uncached.audio_tokens, 499,
            "uncached layout records audio tokens");
    TEST_EQ(uncached.suffix_tokens, 12,
            "uncached layout records suffix tokens");
    TEST_EQ(uncached.stored_prefix_tokens + uncached.audio_tokens, 510,
            "uncached suffix uses a storage-local offset");

    const auto cached = funasr::make_gpu_prompt_layout(11, 11, 499, 12);
    TEST_ASSERT(cached.valid, "full-prefix cached layout is valid");
    TEST_EQ(cached.stored_prefix_tokens, 0,
            "cached prompt omits the stored prefix");
    TEST_EQ(cached.stored_tokens, 511,
            "cached prompt stores only audio and suffix");
    TEST_EQ(cached.prompt_tokens, 522,
            "cached logical prompt length still includes the prefix");
    TEST_EQ(cached.audio_tokens, 499,
            "cached layout records audio tokens");
    TEST_EQ(cached.suffix_tokens, 12,
            "cached layout records suffix tokens");
    TEST_EQ(cached.stored_prefix_tokens + cached.audio_tokens, 499,
            "cached suffix uses a storage-local offset");

    TEST_ASSERT(!funasr::make_gpu_prompt_layout(11, 5, 499, 12).valid,
                "partial prefix cache is rejected");
    TEST_ASSERT(!funasr::make_gpu_prompt_layout(11, 12, 499, 12).valid,
                "cached prefix beyond the full prefix is rejected");
    TEST_ASSERT(!funasr::make_gpu_prompt_layout(-1, 0, 499, 12).valid,
                "negative prefix length is rejected");
    TEST_ASSERT(!funasr::make_gpu_prompt_layout(11, 0, 0, 12).valid,
                "zero audio length is rejected");
    const auto overflow = funasr::make_gpu_prompt_layout(INT_MAX, 0, 1, 0);
    TEST_ASSERT(!overflow.valid,
                "logical prompt length overflow is rejected");
    TEST_EQ(overflow.cached_prefix_tokens, 0,
            "overflow leaves cached prefix at default");
    TEST_EQ(overflow.stored_prefix_tokens, 0,
            "overflow leaves stored prefix at default");
    TEST_EQ(overflow.audio_tokens, 0,
            "overflow leaves audio tokens at default");
    TEST_EQ(overflow.suffix_tokens, 0,
            "overflow leaves suffix tokens at default");
    TEST_EQ(overflow.prompt_tokens, 0,
            "overflow leaves prompt tokens at default");
    TEST_EQ(overflow.stored_tokens, 0,
            "overflow leaves stored tokens at default");

    const auto invalid = funasr::make_gpu_prompt_layout(11, 5, 499, 12);
    TEST_ASSERT(!invalid.valid, "invalid layout stays invalid");
    TEST_EQ(invalid.cached_prefix_tokens, 0,
            "invalid layout leaves cached prefix at default");
    TEST_EQ(invalid.stored_prefix_tokens, 0,
            "invalid layout leaves stored prefix at default");
    TEST_EQ(invalid.audio_tokens, 0,
            "invalid layout leaves audio tokens at default");
    TEST_EQ(invalid.suffix_tokens, 0,
            "invalid layout leaves suffix tokens at default");
    TEST_EQ(invalid.prompt_tokens, 0,
            "invalid layout leaves prompt tokens at default");
    TEST_EQ(invalid.stored_tokens, 0,
            "invalid layout leaves stored tokens at default");
}

void test_handle_is_rejected_by_a_different_pool_epoch() {
    std::printf("\n--- Test 9: Cross-pool stale handle rejection ---\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    funasr::GPUEmbeddingHandle old_handle;

    {
        funasr::GPUEmbeddingPool first_pool(backend, 1);
        old_handle = first_pool.acquire(4, 3);
        TEST_ASSERT(old_handle.valid(), "first pool returns a valid handle");
        TEST_ASSERT(old_handle.pool_epoch != 0,
                    "first pool handle carries a non-zero epoch");
    }

    {
        funasr::GPUEmbeddingPool second_pool(backend, 1);
        const auto current_handle = second_pool.acquire(4, 3);
        TEST_ASSERT(current_handle.valid(), "second pool returns a valid handle");
        TEST_ASSERT(current_handle.pool_epoch != old_handle.pool_epoch,
                    "successive pools have distinct epochs");
        TEST_EQ(current_handle.slot, old_handle.slot,
                "test reproduces the same slot id");
        TEST_EQ(current_handle.generation, old_handle.generation,
                "test reproduces the same slot generation");
        TEST_ASSERT(!second_pool.release(old_handle),
                    "a handle from another pool epoch is rejected");
        TEST_ASSERT(second_pool.owns(current_handle),
                    "foreign release preserves the current owner");
        TEST_EQ(second_pool.free_count(), 0,
                "foreign release does not free the current slot");
    }

    ggml_backend_free(backend);
}

void test_gpu_prompt_chunk_range() {
    std::printf("\n--- Test 10: GPU prompt chunk ranges ---\n");

    funasr::PreparedGPUPrompt prepared;
    prepared.embeddings = {0, 1, 1, 511, 1024};
    prepared.prompt_tokens = 522;
    prepared.cached_prefix_tokens = 11;
    prepared.ok = true;

    const auto first = funasr::make_gpu_prompt_chunk_range(prepared, 11, 128);
    TEST_ASSERT(first.valid, "first stored prompt chunk is valid");
    TEST_EQ(first.absolute_offset, 11, "first chunk absolute offset");
    TEST_EQ(first.local_offset, 0, "first chunk starts at storage offset zero");
    TEST_EQ(first.token_count, 128, "first chunk uses requested token count");
    TEST_ASSERT(!first.produces_logits,
                "non-final prompt chunk does not produce logits");

    const auto final = funasr::make_gpu_prompt_chunk_range(prepared, 512, 128);
    TEST_ASSERT(final.valid, "final stored prompt chunk is valid");
    TEST_EQ(final.absolute_offset, 512, "final chunk absolute offset");
    TEST_EQ(final.local_offset, 501, "final chunk uses storage-local offset");
    TEST_EQ(final.token_count, 10, "final chunk is clipped at prompt end");
    TEST_ASSERT(final.produces_logits,
                "final prompt chunk produces first-token logits");

    TEST_ASSERT(!funasr::make_gpu_prompt_chunk_range(prepared, 10, 128).valid,
                "chunk before cached prefix is rejected");
    TEST_ASSERT(!funasr::make_gpu_prompt_chunk_range(prepared, 522, 128).valid,
                "chunk at prompt end is rejected");
    TEST_ASSERT(!funasr::make_gpu_prompt_chunk_range(prepared, 11, 0).valid,
                "zero requested tokens are rejected");

    auto invalid_handle = prepared;
    invalid_handle.embeddings.generation = 0;
    TEST_ASSERT(!funasr::make_gpu_prompt_chunk_range(
                    invalid_handle, 11, 128).valid,
                "invalid embedding handle is rejected");

    auto inconsistent_storage = prepared;
    inconsistent_storage.embeddings.token_count = 510;
    TEST_ASSERT(!funasr::make_gpu_prompt_chunk_range(
                    inconsistent_storage, 11, 128).valid,
                "inconsistent stored prompt size is rejected");
}

void test_pool_growth_preserves_live_handles() {
    std::printf("\n--- Test 11: Pool growth preserves live handles ---\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    TEST_ASSERT(backend != nullptr, "CPU backend initializes for pool growth");

    {
        funasr::GPUEmbeddingPool pool(backend, 2);
        const auto handle = pool.acquire(3, 4);
        TEST_ASSERT(handle.valid(), "live handle acquired before growth");
        TEST_ASSERT(pool.reserve_slots(5), "pool grows to requested capacity");
        TEST_EQ(pool.capacity(), 5, "grown pool capacity");
        TEST_EQ(pool.free_count(), 4, "new slots are initially free");
        TEST_ASSERT(pool.owns(handle), "growth preserves existing handle owner");
        TEST_ASSERT(pool.reserve_slots(3), "smaller reserve is a no-op");
        TEST_EQ(pool.capacity(), 5, "no-op reserve does not shrink pool");
        TEST_ASSERT(pool.release(handle), "live handle releases after growth");
        TEST_EQ(pool.free_count(), 5, "all grown slots return free");
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
    test_acquire_uses_best_fit_reusable_slot();
    test_copy_tensor_rejects_unsafe_sources();
    test_extreme_shape_is_rejected_without_consuming_slot();
    test_gpu_mixed_runner_stages_packed_slices();
    test_gpu_mixed_runner_rejects_invalid_batches_and_clears_state();
    test_gpu_prompt_layout();
    test_handle_is_rejected_by_a_different_pool_epoch();
    test_gpu_prompt_chunk_range();
    test_pool_growth_preserves_live_handles();

    std::printf("\n========================================\n");
    std::printf("Results: %d passed, %d failed\n",
                tests_passed, tests_failed);
    std::printf("========================================\n");
    return tests_failed == 0 ? 0 : 1;
}
