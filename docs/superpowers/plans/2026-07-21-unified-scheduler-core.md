# Unified Scheduler Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个与 CUDA 无关、可单元测试的 token-budget scheduler core，确定 Decode-first、Chunked Prefill、首 token 边界和 prepare/execute/commit 状态语义。

**Architecture:** 新建独立的 `pipeline/unified_scheduler.hpp/.cpp`，只接收 request token progress 并输出 `MixedBatchPlan`，不直接持有音频、GPU tensor 或 KV Pool。后续 GPU 阶段通过 `max_schedulable_tokens` 将 KV 容量约束传给 planner，并在执行成功后调用显式 commit API。

**Tech Stack:** C++17、现有手写测试框架、CMake、Git

---

## 范围拆分

完整设计包含四个连续实施阶段：

1. **本计划：Unified Scheduler Core**
   - request token progress；
   - Decode-first token budget；
   - Prompt chunk；
   - 计划校验与 commit 语义。
2. **后续计划：Owning GPU Embedding 与 Chunked Prefill**
   - Prompt tail GPU pool；
   - Prefix KV 起点；
   - 单 request 分块 Prefill 正确性。
3. **后续计划：Packed Mixed Runner 与 Varlen Paged Attention**
   - Packed input；
   - Prefill/Decode 专用 Attention dispatch；
   - Request/KV 隔离。
4. **后续计划：Selective LM Head、CUDA Graph 与 Benchmark**
   - Hidden row gather；
   - Graph bucket；
   - 204 chunk 正确性及三轮性能门槛。

本计划不添加尚不能工作的 CLI `--scheduler-mode unified`，也不改变当前默认
离线转写行为。它交付后，GPU 接入可以直接复用已经测试通过的 planner。

## 文件结构

- Create: `pipeline/unified_scheduler.hpp`
  - 定义 request progress、scheduler config、scheduled sequence、mixed plan 和
    commit API。
- Create: `pipeline/unified_scheduler.cpp`
  - 实现纯 token 调度与状态提交，不依赖 `Recognizer`、GGML 或 CUDA。
- Create: `test/test_unified_scheduler.cpp`
  - 独立测试 Decode-first、Prompt chunk、首 token、容量限制和错误输入。
- Modify: `CMakeLists.txt`
  - 将实现加入 `funasr_pipeline`，并增加 `test_unified_scheduler` target。

### Task 1: 建立 Decode-first Token Budget Planner

**Files:**
- Create: `pipeline/unified_scheduler.hpp`
- Create: `pipeline/unified_scheduler.cpp`
- Create: `test/test_unified_scheduler.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 在 CMake 中声明新实现与测试 target**

在 `funasr_pipeline` source list 的 `pipeline/offline_batching.cpp` 后加入：

```cmake
    pipeline/unified_scheduler.cpp
```

在 `test_offline_scheduler` target 后加入：

```cmake
    add_executable(test_unified_scheduler test/test_unified_scheduler.cpp)
    target_link_libraries(test_unified_scheduler PRIVATE funasr_pipeline)
```

- [ ] **Step 2: 写第一个失败测试，要求 Decode 优先并用 Prefill 填满预算**

创建 `test/test_unified_scheduler.cpp`：

```cpp
#include "pipeline/unified_scheduler.hpp"

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

#define TEST_EQ(actual, expected, msg) do { \
    if ((actual) == (expected)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        std::printf("  [FAIL] %s: expected %d, got %d (line %d)\n", \
                    msg, static_cast<int>(expected), \
                    static_cast<int>(actual), __LINE__); \
    } \
} while (0)

void test_decode_first_and_prefill_fills_remaining_budget() {
    std::printf("\n--- Decode-first token budget ---\n");

    funasr::UnifiedSchedulerConfig config;
    config.max_num_seqs = 4;
    config.max_num_scheduled_tokens = 4;
    config.max_prefill_chunk_tokens = 4;
    funasr::UnifiedTokenScheduler scheduler(config);

    funasr::UnifiedRequestProgress decode;
    decode.request_id = 10;
    decode.prompt_tokens = 4;
    decode.num_computed_tokens = 4;
    decode.max_output_tokens = 8;
    decode.output_tokens = {101};

    funasr::UnifiedRequestProgress prefill;
    prefill.request_id = 20;
    prefill.prompt_tokens = 8;
    prefill.max_output_tokens = 8;

    const auto plan = scheduler.build_plan({decode, prefill});

    TEST_ASSERT(plan.ok(), "plan succeeds");
    TEST_EQ(plan.total_tokens, 4, "plan fills token budget");
    TEST_EQ(plan.decode_tokens, 1, "decode consumes one token first");
    TEST_EQ(plan.prefill_tokens, 3, "prefill consumes remaining budget");
    TEST_EQ(plan.sequences.size(), 2, "two requests are scheduled");
    TEST_EQ(plan.sequences[0].request_id, 10, "decode request is first");
    TEST_ASSERT(plan.sequences[0].input_kind == funasr::UnifiedInputKind::Decode,
                "first sequence is decode");
    TEST_EQ(plan.sequences[0].token_offset, 4, "decode absolute position");
    TEST_EQ(plan.sequences[1].request_id, 20, "prefill request follows");
    TEST_EQ(plan.sequences[1].num_tokens, 3, "prefill uses remaining tokens");
}

int main() {
    std::printf("========================================\n");
    std::printf("Unified Scheduler Unit Tests\n");
    std::printf("========================================\n");

    test_decode_first_and_prefill_fills_remaining_budget();

    std::printf("\nTests passed: %d\n", tests_passed);
    std::printf("Tests failed: %d\n", tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
```

- [ ] **Step 3: 构建并确认测试因实现缺失而失败**

Run:

```bash
cmake --build build-cuda --target test_unified_scheduler -j 8
```

Expected: FAIL，提示缺少 `pipeline/unified_scheduler.cpp` 或
`pipeline/unified_scheduler.hpp`。

- [ ] **Step 4: 创建最终稳定的数据结构和最小 Planner 实现**

创建 `pipeline/unified_scheduler.hpp`：

```cpp
#ifndef FUNASR_PIPELINE_UNIFIED_SCHEDULER_HPP
#define FUNASR_PIPELINE_UNIFIED_SCHEDULER_HPP

#include <limits>
#include <vector>

namespace funasr {

enum class UnifiedInputKind {
    Prompt,
    Decode,
};

enum class UnifiedPlanError {
    None,
    InvalidConfig,
    InvalidRequest,
    DuplicateRequestId,
};

enum class SampleCommitResult {
    Appended,
    FinishedEos,
    FinishedLimit,
    Invalid,
};

struct UnifiedSchedulerConfig {
    int max_num_seqs = 1;
    int max_num_scheduled_tokens = 1;
    int max_prefill_chunk_tokens = 1;
};

struct UnifiedRequestProgress {
    int request_id = -1;
    int prompt_tokens = 0;
    int num_computed_tokens = 0;
    int max_output_tokens = 0;
    int max_schedulable_tokens = std::numeric_limits<int>::max();
    std::vector<int> output_tokens;
    bool runnable = true;
    bool finished = false;

    int available_tokens() const;
    bool prompt_complete() const;
    bool valid() const;
};

struct ScheduledSequence {
    int request_id = -1;
    int request_index = -1;
    int token_offset = 0;
    int num_tokens = 0;
    UnifiedInputKind input_kind = UnifiedInputKind::Prompt;
    bool produces_logits = false;
};

struct MixedBatchPlan {
    std::vector<ScheduledSequence> sequences;
    int total_tokens = 0;
    int prefill_tokens = 0;
    int decode_tokens = 0;
    UnifiedPlanError error = UnifiedPlanError::None;

    bool ok() const { return error == UnifiedPlanError::None; }
};

class UnifiedTokenScheduler {
public:
    explicit UnifiedTokenScheduler(UnifiedSchedulerConfig config);

    MixedBatchPlan build_plan(
        const std::vector<UnifiedRequestProgress>& requests) const;

    static bool commit_sequence(
        UnifiedRequestProgress& request,
        const ScheduledSequence& scheduled);

    static SampleCommitResult commit_sample(
        UnifiedRequestProgress& request,
        int token_id,
        int eos_id);

private:
    UnifiedSchedulerConfig config_;
};

} // namespace funasr

#endif
```

创建 `pipeline/unified_scheduler.cpp`，先实现当前测试所需行为：

```cpp
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
        }
    }

    return plan;
}

} // namespace funasr
```

本 Task 中 `valid()`、`commit_sequence()` 和 `commit_sample()` 只声明、不调用，
将在后续 Task 通过新失败测试驱动实现。

- [ ] **Step 5: 构建并运行测试**

Run:

```bash
cmake --build build-cuda --target test_unified_scheduler -j 8
./build-cuda/test_unified_scheduler
```

Expected:

```text
Tests failed: 0
```

- [ ] **Step 6: 提交 Task 1**

```bash
git add CMakeLists.txt pipeline/unified_scheduler.hpp \
    pipeline/unified_scheduler.cpp test/test_unified_scheduler.cpp
git commit -m "feat: add unified token budget planner"
```

### Task 2: 固化 Chunked Prefill 与首 Token 边界

**Files:**
- Modify: `pipeline/unified_scheduler.cpp`
- Modify: `test/test_unified_scheduler.cpp`

- [ ] **Step 1: 增加 Prompt 分块与最终 chunk 测试**

在 `test/test_unified_scheduler.cpp` 的 `main()` 前加入：

```cpp
void test_prompt_chunks_and_first_logits_boundary() {
    std::printf("\n--- Prompt chunk and first-token boundary ---\n");

    funasr::UnifiedSchedulerConfig config;
    config.max_num_seqs = 4;
    config.max_num_scheduled_tokens = 128;
    config.max_prefill_chunk_tokens = 128;
    funasr::UnifiedTokenScheduler scheduler(config);

    funasr::UnifiedRequestProgress request;
    request.request_id = 30;
    request.prompt_tokens = 522;
    request.num_computed_tokens = 256;
    request.max_output_tokens = 220;

    auto plan = scheduler.build_plan({request});
    TEST_EQ(plan.sequences[0].token_offset, 256, "chunk starts at progress");
    TEST_EQ(plan.sequences[0].num_tokens, 128, "chunk respects cap");
    TEST_ASSERT(!plan.sequences[0].produces_logits,
                "intermediate prompt chunk skips logits");

    request.num_computed_tokens = 512;
    plan = scheduler.build_plan({request});
    TEST_EQ(plan.sequences[0].num_tokens, 10, "final chunk uses exact remainder");
    TEST_ASSERT(plan.sequences[0].produces_logits,
                "final prompt position produces first-token logits");
}
```

在 `main()` 中调用：

```cpp
    test_prompt_chunks_and_first_logits_boundary();
```

- [ ] **Step 2: 运行测试并确认最终 Prompt chunk 的 logits 断言失败**

Run:

```bash
cmake --build build-cuda --target test_unified_scheduler -j 8
./build-cuda/test_unified_scheduler
```

Expected: FAIL，`final prompt position produces first-token logits` 不成立。

- [ ] **Step 3: 仅在 Prompt 最终 chunk 标记 `produces_logits`**

将 `build_plan()` Prompt loop 中原来的 `if (count > 0)` 代码块替换为：

```cpp
            schedule(static_cast<int>(i), count, UnifiedInputKind::Prompt);
            plan.sequences.back().produces_logits =
                request.num_computed_tokens + count == request.prompt_tokens;
```

- [ ] **Step 4: 运行测试并确认通过**

```bash
cmake --build build-cuda --target test_unified_scheduler -j 8
./build-cuda/test_unified_scheduler
```

Expected: `Tests failed: 0`。

- [ ] **Step 5: 提交 Task 2**

```bash
git add pipeline/unified_scheduler.cpp test/test_unified_scheduler.cpp
git commit -m "test: define chunked prefill logits boundary"
```

### Task 3: 实现 Execute 成功后的 Token 与 Sample Commit

**Files:**
- Modify: `pipeline/unified_scheduler.cpp`
- Modify: `test/test_unified_scheduler.cpp`

- [ ] **Step 1: 增加 Prompt 完成、首 token 与下一轮 Decode 测试**

在测试文件加入：

```cpp
void test_commit_prompt_then_schedule_first_decode_input() {
    std::printf("\n--- Commit prompt and first decode input ---\n");

    funasr::UnifiedSchedulerConfig config{4, 128, 128};
    funasr::UnifiedTokenScheduler scheduler(config);

    funasr::UnifiedRequestProgress request;
    request.request_id = 40;
    request.prompt_tokens = 522;
    request.num_computed_tokens = 512;
    request.max_output_tokens = 2;

    auto plan = scheduler.build_plan({request});
    TEST_ASSERT(funasr::UnifiedTokenScheduler::commit_sequence(
                    request, plan.sequences[0]),
                "final prompt sequence commits");
    TEST_EQ(request.num_computed_tokens, 522, "prompt progress reaches end");

    const auto first = funasr::UnifiedTokenScheduler::commit_sample(
        request, 1234, 9999);
    TEST_ASSERT(first == funasr::SampleCommitResult::Appended,
                "first output token is appended");
    TEST_EQ(request.output_tokens.size(), 1, "one sampled token exists");
    TEST_EQ(request.num_computed_tokens, 522,
            "sampled token is not in KV before next iteration");

    plan = scheduler.build_plan({request});
    TEST_EQ(plan.sequences.size(), 1, "one decode input is scheduled");
    TEST_ASSERT(plan.sequences[0].input_kind == funasr::UnifiedInputKind::Decode,
                "sampled token becomes decode input");
    TEST_EQ(plan.sequences[0].token_offset, 522,
            "first output token uses position 522");

    TEST_ASSERT(funasr::UnifiedTokenScheduler::commit_sequence(
                    request, plan.sequences[0]),
                "decode sequence commits");
    const auto second = funasr::UnifiedTokenScheduler::commit_sample(
        request, 5678, 9999);
    TEST_ASSERT(second == funasr::SampleCommitResult::FinishedLimit,
                "second output reaches generation limit");
    TEST_ASSERT(request.finished, "request is finished at output limit");

    funasr::UnifiedRequestProgress eos_request;
    eos_request.request_id = 41;
    eos_request.prompt_tokens = 1;
    eos_request.num_computed_tokens = 1;
    eos_request.max_output_tokens = 4;
    const auto eos = funasr::UnifiedTokenScheduler::commit_sample(
        eos_request, 99, 99);
    TEST_ASSERT(eos == funasr::SampleCommitResult::FinishedEos,
                "EOS finishes without appending");
    TEST_EQ(eos_request.output_tokens.size(), 0,
            "EOS is not included in output tokens");
}
```

在 `main()` 中调用该测试。

- [ ] **Step 2: 构建并确认链接失败**

```bash
cmake --build build-cuda --target test_unified_scheduler -j 8
```

Expected: FAIL，缺少 `commit_sequence` 和 `commit_sample` 定义。

- [ ] **Step 3: 实现事务提交 API**

在 `pipeline/unified_scheduler.cpp` 中加入：

```cpp
bool UnifiedTokenScheduler::commit_sequence(
    UnifiedRequestProgress& request,
    const ScheduledSequence& scheduled) {
    if (request.finished || scheduled.request_id != request.request_id ||
        scheduled.num_tokens <= 0 ||
        scheduled.token_offset != request.num_computed_tokens ||
        scheduled.num_tokens > request.available_tokens()) {
        return false;
    }
    request.num_computed_tokens += scheduled.num_tokens;
    return true;
}

SampleCommitResult UnifiedTokenScheduler::commit_sample(
    UnifiedRequestProgress& request,
    int token_id,
    int eos_id) {
    if (request.finished || request.max_output_tokens <= 0 ||
        request.num_computed_tokens !=
            request.prompt_tokens + static_cast<int>(request.output_tokens.size())) {
        return SampleCommitResult::Invalid;
    }
    if (token_id == eos_id) {
        request.finished = true;
        return SampleCommitResult::FinishedEos;
    }

    request.output_tokens.push_back(token_id);
    if (static_cast<int>(request.output_tokens.size()) >=
        request.max_output_tokens) {
        request.finished = true;
        return SampleCommitResult::FinishedLimit;
    }
    return SampleCommitResult::Appended;
}
```

- [ ] **Step 4: 运行测试并确认通过**

```bash
cmake --build build-cuda --target test_unified_scheduler -j 8
./build-cuda/test_unified_scheduler
```

Expected: `Tests failed: 0`。

- [ ] **Step 5: 提交 Task 3**

```bash
git add pipeline/unified_scheduler.cpp test/test_unified_scheduler.cpp
git commit -m "feat: add unified scheduler commit semantics"
```

### Task 4: 加入输入校验、KV 容量上限与 Sequence 上限

**Files:**
- Modify: `pipeline/unified_scheduler.cpp`
- Modify: `test/test_unified_scheduler.cpp`

- [ ] **Step 1: 增加错误输入和容量限制测试**

在测试文件加入：

```cpp
void test_validation_capacity_and_sequence_limit() {
    std::printf("\n--- Validation and capacity limits ---\n");

    funasr::UnifiedSchedulerConfig config{1, 8, 8};
    funasr::UnifiedTokenScheduler scheduler(config);

    funasr::UnifiedRequestProgress blocked_decode;
    blocked_decode.request_id = 50;
    blocked_decode.prompt_tokens = 2;
    blocked_decode.num_computed_tokens = 2;
    blocked_decode.max_output_tokens = 4;
    blocked_decode.output_tokens = {7};
    blocked_decode.max_schedulable_tokens = 0;

    funasr::UnifiedRequestProgress capped_prefill;
    capped_prefill.request_id = 51;
    capped_prefill.prompt_tokens = 20;
    capped_prefill.max_output_tokens = 4;
    capped_prefill.max_schedulable_tokens = 3;

    funasr::UnifiedRequestProgress second_prefill = capped_prefill;
    second_prefill.request_id = 52;

    auto plan = scheduler.build_plan(
        {blocked_decode, capped_prefill, second_prefill});
    TEST_ASSERT(plan.ok(), "capacity-constrained plan succeeds");
    TEST_EQ(plan.sequences.size(), 1, "max_num_seqs limits distinct requests");
    TEST_EQ(plan.sequences[0].request_id, 51, "first runnable prefill selected");
    TEST_EQ(plan.sequences[0].num_tokens, 3, "KV capacity caps prompt chunk");

    funasr::UnifiedRequestProgress invalid = capped_prefill;
    invalid.request_id = 60;
    invalid.prompt_tokens = -1;
    plan = scheduler.build_plan({invalid});
    TEST_ASSERT(plan.error == funasr::UnifiedPlanError::InvalidRequest,
                "invalid request rejects whole plan");
    TEST_EQ(plan.sequences.size(), 0, "invalid plan schedules no work");

    funasr::UnifiedRequestProgress duplicate = capped_prefill;
    duplicate.request_id = 70;
    plan = scheduler.build_plan({duplicate, duplicate});
    TEST_ASSERT(plan.error == funasr::UnifiedPlanError::DuplicateRequestId,
                "duplicate request id is rejected");

    funasr::UnifiedSchedulerConfig bad_config{0, 8, 8};
    funasr::UnifiedTokenScheduler bad_scheduler(bad_config);
    plan = bad_scheduler.build_plan({capped_prefill});
    TEST_ASSERT(plan.error == funasr::UnifiedPlanError::InvalidConfig,
                "invalid scheduler config is rejected");

    funasr::UnifiedSchedulerConfig starved_decode_config{40, 32, 8};
    funasr::UnifiedTokenScheduler starved_decode_scheduler(
        starved_decode_config);
    plan = starved_decode_scheduler.build_plan({capped_prefill});
    TEST_ASSERT(plan.error == funasr::UnifiedPlanError::InvalidConfig,
                "token budget must cover every active decode sequence");
}
```

在 `main()` 中调用该测试。

- [ ] **Step 2: 运行并确认测试失败**

```bash
cmake --build build-cuda --target test_unified_scheduler -j 8
./build-cuda/test_unified_scheduler
```

Expected: FAIL，sequence 上限或 validation 断言失败。

- [ ] **Step 3: 实现 request 校验与计划预检**

在 `pipeline/unified_scheduler.cpp` 增加：

```cpp
#include <unordered_set>

bool UnifiedRequestProgress::valid() const {
    if (request_id < 0 || prompt_tokens < 0 || num_computed_tokens < 0 ||
        max_output_tokens <= 0 || max_schedulable_tokens < 0) {
        return false;
    }
    const int available_end =
        prompt_tokens + static_cast<int>(output_tokens.size());
    return num_computed_tokens <= available_end;
}
```

在 `build_plan()` 开始处加入：

```cpp
    if (config_.max_num_seqs <= 0 ||
        config_.max_num_scheduled_tokens <= 0 ||
        config_.max_prefill_chunk_tokens <= 0 ||
        config_.max_num_scheduled_tokens < config_.max_num_seqs) {
        plan.error = UnifiedPlanError::InvalidConfig;
        return plan;
    }

    std::unordered_set<int> request_ids;
    for (const auto& request : requests) {
        if (!request.valid()) {
            plan.error = UnifiedPlanError::InvalidRequest;
            return plan;
        }
        if (!request_ids.insert(request.request_id).second) {
            plan.error = UnifiedPlanError::DuplicateRequestId;
            return plan;
        }
    }
```

在两个 scheduling loop 条件中都加入：

```cpp
static_cast<int>(plan.sequences.size()) < config_.max_num_seqs
```

这里每个 request 单轮最多出现一次，因此 `plan.sequences.size()` 就是本轮
sequence 数；后续支持单 request 多段计划时再改成 request-id set。

- [ ] **Step 4: 运行新测试和现有 Scheduler 回归测试**

```bash
cmake --build build-cuda --target \
    test_unified_scheduler test_offline_scheduler -j 8
./build-cuda/test_unified_scheduler
./build-cuda/test_offline_scheduler
```

Expected: 两个测试程序均输出 `Tests failed: 0`。

- [ ] **Step 5: 检查格式与工作区差异**

```bash
git diff --check
git diff -- CMakeLists.txt pipeline/unified_scheduler.hpp \
    pipeline/unified_scheduler.cpp test/test_unified_scheduler.cpp
```

Expected: `git diff --check` 无输出；diff 只包含本计划涉及的 scheduler core。

- [ ] **Step 6: 提交 Task 4**

```bash
git add pipeline/unified_scheduler.cpp test/test_unified_scheduler.cpp
git commit -m "test: harden unified scheduler planning invariants"
```

## 阶段完成标准

- `test_unified_scheduler` 全部通过。
- `test_offline_scheduler` 保持通过。
- 当前 Decode-only Runtime 行为完全不变。
- Planner 能稳定表达：
  - Decode-first；
  - 剩余 budget 填充 Prompt chunk；
  - Prefix Cache 可通过非零 `num_computed_tokens` 起步；
  - 最后 Prompt chunk 产生首 token logits；
  - 首 token 下一轮才作为 Decode 输入进入 KV；
  - 调用方通过 `max_schedulable_tokens` 施加 KV backpressure；
  - GPU 成功前不提交 request token 进度。
- 完成后编写第二阶段 Owning GPU Embedding 与 Chunked Prefill 实施计划。
