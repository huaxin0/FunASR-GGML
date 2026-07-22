#include "pipeline/mixed_batch.hpp"
#include "ggml.h"

#include <cstdio>
#include <limits>
#include <utility>
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
        std::printf("  [FAIL] %s: expected %d, got %d (line %d)\n", \
                    name, (int)(expected), (int)(actual), __LINE__); \
    } \
} while (0)

static funasr::PackedSequenceSource make_source(
    int request_id,
    int request_index,
    funasr::UnifiedInputKind kind,
    int token_offset,
    int num_tokens,
    int prompt_tokens,
    int cached_prefix_tokens,
    int decode_token_id,
    bool produces_logits,
    std::vector<int> block_table) {
    funasr::PackedSequenceSource source;
    source.scheduled.request_id = request_id;
    source.scheduled.request_index = request_index;
    source.scheduled.input_kind = kind;
    source.scheduled.token_offset = token_offset;
    source.scheduled.num_tokens = num_tokens;
    source.scheduled.produces_logits = produces_logits;
    source.prompt_tokens = prompt_tokens;
    source.cached_prefix_tokens = cached_prefix_tokens;
    source.decode_token_id = decode_token_id;
    source.block_table = std::move(block_table);
    return source;
}

template <typename T>
static void expect_vector_eq(const std::vector<T>& actual,
                             const std::vector<T>& expected,
                             const char* name) {
    TEST_EQ(actual.size(), expected.size(), name);
    if (actual.size() != expected.size()) {
        return;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            tests_failed++;
            std::printf("  [FAIL] %s at %zu: expected %d, got %d\n",
                        name, i, (int)expected[i], (int)actual[i]);
            return;
        }
    }
    tests_passed++;
}

static void expect_invalid_and_empty(
    const funasr::PackedMixedMetadata& metadata,
    const char* name) {
    TEST_ASSERT(!metadata.valid, name);
    TEST_ASSERT(metadata.sequences.empty(), "invalid sequences stay empty");
    TEST_ASSERT(metadata.positions.empty(), "invalid positions stay empty");
    TEST_ASSERT(metadata.sequence_rows.empty(), "invalid sequence rows stay empty");
    TEST_ASSERT(metadata.kv_lens.empty(), "invalid KV lengths stay empty");
    TEST_ASSERT(metadata.selected_rows.empty(), "invalid selected rows stay empty");
    TEST_ASSERT(metadata.expanded_block_table.empty(),
                "invalid expanded block table stays empty");
    TEST_EQ(metadata.max_blocks, 0, "invalid max blocks stays zero");
    TEST_EQ(metadata.total_tokens, 0, "invalid total tokens stays zero");
    TEST_EQ(metadata.prefill_tokens, 0, "invalid prefill tokens stays zero");
    TEST_EQ(metadata.decode_tokens, 0, "invalid decode tokens stays zero");
}

void test_prompt_first_stable_mixed_pack() {
    std::printf("\n--- Test 1: Prompt-first stable mixed pack ---\n");
    const std::vector<funasr::PackedSequenceSource> sources = {
        make_source(2, 2, funasr::UnifiedInputKind::Decode,
                    522, 1, 522, 0, 102, true, {30, 31, 32, 33, 34}),
        make_source(0, 0, funasr::UnifiedInputKind::Prompt,
                    11, 3, 14, 11, -1, false, {10, 11}),
        make_source(3, 3, funasr::UnifiedInputKind::Decode,
                    601, 1, 600, 0, 103, true, {40, 41, 42, 43, 44}),
        make_source(1, 1, funasr::UnifiedInputKind::Prompt,
                    20, 2, 22, 10, -1, true, {20, 21, 22}),
    };

    const auto packed = funasr::build_packed_mixed_metadata(sources, 128);

    TEST_ASSERT(packed.valid, "mixed metadata is valid");
    TEST_EQ(packed.total_tokens, 7, "total tokens");
    TEST_EQ(packed.prefill_tokens, 5, "prefill tokens");
    TEST_EQ(packed.decode_tokens, 2, "decode tokens");
    TEST_EQ(packed.max_blocks, 5, "maximum block count");
    TEST_EQ(packed.sequences.size(), 4, "packed sequence count");
    if (packed.sequences.size() == 4) {
        TEST_EQ(packed.sequences[0].scheduled.request_id, 0, "first prompt");
        TEST_EQ(packed.sequences[1].scheduled.request_id, 1, "second prompt");
        TEST_EQ(packed.sequences[2].scheduled.request_id, 2, "first decode");
        TEST_EQ(packed.sequences[3].scheduled.request_id, 3, "second decode");
    }
    expect_vector_eq(packed.positions, {11, 12, 13, 20, 21, 522, 601},
                     "packed positions");
    expect_vector_eq(packed.sequence_rows, {0, 0, 0, 1, 1, 2, 3},
                     "packed sequence rows");
    expect_vector_eq(packed.kv_lens, {12, 13, 14, 21, 22, 523, 602},
                     "packed KV lengths");
    expect_vector_eq(packed.selected_rows, {4, 5, 6}, "selected rows");
    TEST_EQ(packed.positions.size(), packed.total_tokens,
            "position count matches total tokens");
    TEST_EQ(packed.sequence_rows.size(), packed.total_tokens,
            "sequence-row count matches total tokens");
    TEST_EQ(packed.kv_lens.size(), packed.total_tokens,
            "KV-length count matches total tokens");
}

void test_expanded_block_table_padding_and_layout() {
    std::printf("\n--- Test 2: Expanded block-table padding and layout ---\n");
    const std::vector<funasr::PackedSequenceSource> sources = {
        make_source(10, 0, funasr::UnifiedInputKind::Prompt,
                    0, 2, 2, 0, -1, false, {7}),
        make_source(11, 1, funasr::UnifiedInputKind::Decode,
                    4, 1, 4, 0, 88, true, {9, 10, 11}),
    };

    const auto packed = funasr::build_packed_mixed_metadata(sources, 2);
    TEST_ASSERT(packed.valid, "layout metadata is valid");
    TEST_EQ(packed.max_blocks, 3, "layout maximum blocks");
    expect_vector_eq(packed.expanded_block_table,
                     {7, -1, -1, 7, -1, -1, 9, 10, 11},
                     "token-major GGML flat block table");
    TEST_EQ(packed.expanded_block_table.size(),
            packed.max_blocks * packed.total_tokens,
            "expanded table shape matches max_blocks by total_tokens");
}

void test_block_boundary_uses_next_logical_block() {
    std::printf("\n--- Test 3: Block boundary coverage ---\n");
    const auto packed = funasr::build_packed_mixed_metadata({
        make_source(20, 0, funasr::UnifiedInputKind::Prompt,
                    3, 2, 5, 0, -1, true, {70, 80}),
    }, 4);

    TEST_ASSERT(packed.valid, "sequence crossing block boundary is valid");
    expect_vector_eq(packed.positions, {3, 4}, "boundary positions");
    expect_vector_eq(packed.expanded_block_table, {70, 80, 70, 80},
                     "boundary rows copy full block table");
}

void test_duplicate_and_missing_block_are_invalid() {
    std::printf("\n--- Test 4: Duplicate request and missing block ---\n");
    const auto first = make_source(30, 0, funasr::UnifiedInputKind::Prompt,
                                   0, 1, 1, 0, -1, true, {1});
    auto duplicate = make_source(30, 1, funasr::UnifiedInputKind::Decode,
                                 1, 1, 1, 0, 9, true, {1});
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({first, duplicate}, 4),
        "duplicate request is invalid");

    const auto missing = make_source(31, 0, funasr::UnifiedInputKind::Prompt,
                                     4, 1, 5, 0, -1, true, {2});
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({missing}, 4),
        "missing scheduled block is invalid");

    const auto negative = make_source(32, 0, funasr::UnifiedInputKind::Prompt,
                                      0, 1, 1, 0, -1, true, {-1});
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({negative}, 4),
        "negative scheduled block id is invalid");
}

void test_prompt_semantics_are_validated() {
    std::printf("\n--- Test 5: Prompt semantic validation ---\n");
    auto prompt = make_source(40, 0, funasr::UnifiedInputKind::Prompt,
                              2, 2, 4, 1, -1, true, {5});

    auto invalid = prompt;
    invalid.scheduled.token_offset = 0;
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({invalid}, 4),
        "prompt before cached prefix is invalid");

    invalid = prompt;
    invalid.scheduled.num_tokens = 3;
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({invalid}, 4),
        "prompt beyond prompt length is invalid");

    invalid = prompt;
    invalid.decode_token_id = 7;
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({invalid}, 4),
        "prompt decode token id is invalid");
}

void test_decode_semantics_are_validated() {
    std::printf("\n--- Test 6: Decode semantic validation ---\n");
    auto decode = make_source(50, 0, funasr::UnifiedInputKind::Decode,
                              4, 1, 4, 0, 99, true, {6, 7});

    auto invalid = decode;
    invalid.scheduled.num_tokens = 2;
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({invalid}, 4),
        "multi-token decode is invalid");

    invalid = decode;
    invalid.scheduled.token_offset = 3;
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({invalid}, 4),
        "decode before prompt end is invalid");

    invalid = decode;
    invalid.decode_token_id = -1;
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({invalid}, 4),
        "decode without token id is invalid");
}

void test_common_input_validation() {
    std::printf("\n--- Test 7: Common input validation ---\n");
    const auto valid = make_source(60, 0, funasr::UnifiedInputKind::Prompt,
                                   0, 1, 1, 0, -1, true, {8});
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({}, 4),
        "empty sources are invalid");
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({valid}, 0),
        "zero block size is invalid");
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({valid}, -1),
        "negative block size is invalid");

    auto invalid = valid;
    invalid.scheduled.request_id = -1;
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({invalid}, 4),
        "negative request id is invalid");
    invalid = valid;
    invalid.scheduled.request_index = -1;
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({invalid}, 4),
        "negative request index is invalid");
    invalid = valid;
    invalid.scheduled.token_offset = -1;
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({invalid}, 4),
        "negative token offset is invalid");
    invalid = valid;
    invalid.scheduled.num_tokens = 0;
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({invalid}, 4),
        "zero token count is invalid");
}

void test_integer_overflow_is_rejected_before_materialization() {
    std::printf("\n--- Test 8: Integer overflow validation ---\n");
    const int int_max = std::numeric_limits<int>::max();
    const auto huge_prompt = make_source(
        70, 0, funasr::UnifiedInputKind::Prompt,
        0, int_max, int_max, 0, -1, false, {1});
    const auto decode = make_source(
        71, 1, funasr::UnifiedInputKind::Decode,
        0, 1, 0, 0, 10, true, {2});
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata(
            {huge_prompt, decode}, int_max),
        "total token overflow is invalid");

    const auto kv_len_overflow = make_source(
        72, 0, funasr::UnifiedInputKind::Decode,
        int_max, 1, int_max, 0, 11, true, {3, 4});
    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({kv_len_overflow}, int_max),
        "KV length overflow is invalid");
}

void test_single_huge_prompt_is_rejected_before_allocation() {
    std::printf("\n--- Test 9: Single huge prompt is rejected before allocation ---\n");
    const int int_max = std::numeric_limits<int>::max();
    const auto huge_prompt = make_source(
        73, 0, funasr::UnifiedInputKind::Prompt,
        0, int_max, int_max, 0, -1, false, {1});

    expect_invalid_and_empty(
        funasr::build_packed_mixed_metadata({huge_prompt}, int_max),
        "single huge prompt is invalid before allocation");
}

void test_paged_attention_tracks_kv_write_dependency() {
    std::printf("\n--- Test 10: Paged attention KV-write dependency ---\n");

    constexpr int64_t head_dim = 8;
    constexpr int64_t n_heads = 4;
    constexpr int64_t n_head_kv = 2;
    constexpr int64_t batch = 2;
    constexpr int64_t kv_dim = head_dim * n_head_kv;
    constexpr int64_t block_size = 4;
    constexpr int64_t max_blocks = 2;
    constexpr int64_t physical_rows = max_blocks * block_size;
    constexpr int64_t max_n_kv = max_blocks * block_size;

    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "no-alloc GGML context is created");
    if (ctx == nullptr) {
        return;
    }

    ggml_tensor* q = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, head_dim, 1, n_heads, batch);
    ggml_tensor* k_cache = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, kv_dim, physical_rows);
    ggml_tensor* v_cache = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, kv_dim, physical_rows);
    ggml_tensor* k_cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_dim, batch);
    ggml_tensor* v_cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_dim, batch);
    ggml_tensor* block_table = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, max_blocks, batch);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch);
    ggml_tensor* kv_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch);

    ggml_tensor* write = ggml_paged_kv_write_ext(
        ctx, k_cur, v_cur, k_cache, v_cache, block_table, positions, block_size);
    ggml_tensor* attention = ggml_paged_attn_ext_v_dep(
        ctx, q, k_cache, v_cache, block_table, kv_lens, write,
        1.0f, block_size, max_n_kv, n_head_kv);

    TEST_ASSERT(attention->src[5] == write,
                "paged attention dependency source is the KV write");

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, write);
    ggml_build_forward_expand(graph, attention);

    int write_count = 0;
    int attention_count = 0;
    int write_index = -1;
    int attention_index = -1;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor* node = ggml_graph_node(graph, i);
        if (node == write) {
            ++write_count;
            write_index = i;
        }
        if (node == attention) {
            ++attention_count;
            attention_index = i;
        }
    }

    TEST_EQ(write_count, 1, "KV write appears once in graph");
    TEST_EQ(attention_count, 1, "paged attention appears once in graph");
    TEST_ASSERT(write_index >= 0 && write_index < attention_index,
                "KV write precedes paged attention");

    ggml_cgraph* attention_root_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(attention_root_graph, attention);

    write_count = 0;
    attention_count = 0;
    write_index = -1;
    attention_index = -1;
    for (int i = 0; i < ggml_graph_n_nodes(attention_root_graph); ++i) {
        ggml_tensor* node = ggml_graph_node(attention_root_graph, i);
        if (node == write) {
            ++write_count;
            write_index = i;
        }
        if (node == attention) {
            ++attention_count;
            attention_index = i;
        }
    }

    TEST_EQ(write_count, 1, "single-root graph includes KV write once");
    TEST_EQ(attention_count, 1, "single-root graph includes paged attention once");
    TEST_ASSERT(write_index >= 0 && write_index < attention_index,
                "single-root graph keeps KV write before paged attention");

    ggml_tensor* legacy_attention = ggml_paged_attn_ext_v(
        ctx, q, k_cache, v_cache, block_table, kv_lens,
        1.0f, block_size, max_n_kv, n_head_kv);
    TEST_ASSERT(legacy_attention->src[5] == nullptr,
                "legacy paged attention has no KV-write dependency");

    ggml_free(ctx);
}

void test_execution_metadata_validation() {
    std::printf("\n--- Test 11: Packed execution validation ---\n");
    const auto packed = funasr::build_packed_mixed_metadata({
        make_source(80, 0, funasr::UnifiedInputKind::Prompt,
                    0, 2, 2, 0, -1, true, {1}),
        make_source(81, 1, funasr::UnifiedInputKind::Decode,
                    4, 1, 4, 0, 9, true, {2, 3, 4}),
    }, 2);
    TEST_ASSERT(funasr::validate_packed_mixed_execution(
                    packed, 2, 16, 100, 2, {9}),
                "valid packed execution is accepted");

    auto invalid = packed;
    invalid.kv_lens[0] = 2;
    TEST_ASSERT(!funasr::validate_packed_mixed_execution(
                    invalid, 2, 16, 100, 2, {9}),
                "KV length must end at the current position");

    invalid = packed;
    invalid.selected_rows.push_back(invalid.selected_rows.front());
    TEST_ASSERT(!funasr::validate_packed_mixed_execution(
                    invalid, 2, 16, 100, 2, {9}),
                "duplicate selected rows are rejected");

    invalid = packed;
    invalid.expanded_block_table[6] = 8;
    TEST_ASSERT(!funasr::validate_packed_mixed_execution(
                    invalid, 2, 16, 100, 2, {9}),
                "physical blocks outside the KV pool are rejected");

    TEST_ASSERT(!funasr::validate_packed_mixed_execution(
                    packed, 2, 16, 100, 1, {9}),
                "staged prompt count must match metadata");
    TEST_ASSERT(!funasr::validate_packed_mixed_execution(
                    packed, 2, 16, 9, 2, {9}),
                "decode token ids must be inside the vocabulary");

    const auto intermediate = funasr::build_packed_mixed_metadata({
        make_source(82, 0, funasr::UnifiedInputKind::Prompt,
                    0, 2, 4, 0, -1, false, {1}),
    }, 2);
    TEST_ASSERT(intermediate.selected_rows.empty(),
                "intermediate prefill produces no logits rows");
    TEST_ASSERT(funasr::validate_packed_mixed_execution(
                    intermediate, 2, 16, 100, 2, {}),
                "intermediate prefill without selected rows is executable");
}

int main() {
    std::printf("========================================\n");
    std::printf("Packed Mixed Metadata Unit Tests\n");
    std::printf("========================================\n");

    test_prompt_first_stable_mixed_pack();
    test_expanded_block_table_padding_and_layout();
    test_block_boundary_uses_next_logical_block();
    test_duplicate_and_missing_block_are_invalid();
    test_prompt_semantics_are_validated();
    test_decode_semantics_are_validated();
    test_common_input_validation();
    test_integer_overflow_is_rejected_before_materialization();
    test_single_huge_prompt_is_rejected_before_allocation();
    test_paged_attention_tracks_kv_write_dependency();
    test_execution_metadata_validation();

    std::printf("\n========================================\n");
    std::printf("Tests passed: %d\n", tests_passed);
    std::printf("Tests failed: %d\n", tests_failed);
    std::printf("========================================\n");
    return tests_failed == 0 ? 0 : 1;
}
