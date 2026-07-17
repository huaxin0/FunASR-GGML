# Paged KV Physical Pool Decoupling Design

日期：2026-07-17

## 1. 背景

当前离线 GPU paged KV scheduler 已支持按需 append block，并在完整长音频测试中把逻辑 block 浪费降到零：

```text
paged_kv_waste: waste_rate=0.00%
blocks_peak=61/384
```

但底层 `GPUContext` 仍使用以下公式分配物理 KV buffer：

```text
physical_rows = batch_size * ctx_size
              = 12 * 4096
              = 49152
```

该配置为 K 和 V 分别分配约 2818.6 MB，总计约 5.64 GB。实际峰值只使用 61 个 128-token block，即 7808 rows，物理容量利用率约 15.9%。

因此，目前动态 block 只解决了 scheduler 的逻辑分配浪费，没有解决 GPU 物理 KV pool 按每个 slot 最坏上下文预留的问题。这也限制了 batch size 和有效并发继续扩大。

## 2. 目标

将三个概念解耦：

```text
ctx_size        单个 request 的上下文上限
batch_size      scheduler 允许的最大活跃 request 数
physical_rows   GPU 全局 KV block pool 的物理容量
```

在 paged KV 模式下使用：

```text
physical_rows = global_num_blocks * block_size
```

完成后使用固定物理 KV pool 测试 `batch=12/16/24/32`，验证更高并发能否缩小与 FunASR-vLLM 的吞吐差距。

## 3. 非目标

本阶段不包含：

- 修改 realtime 推理路径；
- 修改 continuous KV 的 slot 语义；
- 自动探测剩余 GPU 显存并决定 block 数；
- 重写 scheduler 为完整 token-budget scheduler；
- 修改 Q8 matmul、paged attention 或其他 CUDA kernel；
- 声称仅靠该改动一定超过 vLLM。

## 4. 方案选择

### 方案 A：显式物理容量参数，推荐

扩展 GPU 初始化链路，使其可选接收 `physical_kv_rows`。值为零时保持旧行为；paged 模式显式传入 `kv_num_blocks * kv_block_size`。

优点：边界清晰、向后兼容、适合做受控 benchmark。缺点：第一阶段需要用户或 preset 明确 block 数。

### 方案 B：启动时按可用显存自动分配

参考 vLLM，在加载权重和评估 workspace 后，把剩余显存分配给 KV pool。

优点：使用方便。缺点：需要处理 CUDA 显存查询、graph workspace、不同 GPU 和后台占用，第一阶段变量过多。

### 方案 C：直接实现 token-budget scheduler

同时取消固定活跃 request 上限，由每轮 token/block budget 决定 batch。

优点：最接近 vLLM。缺点：调度和物理内存两个变量同时改变，不利于定位收益来源。

本阶段采用方案 A。方案 B 和 C 根据 benchmark 结果后续推进。

## 5. API 与兼容规则

GPU 初始化链路增加一个可选物理容量参数：

```cpp
bool init_gpu(int n_ctx = 2048,
              int gpu_id = 0,
              int n_slots = 1,
              int physical_kv_rows = 0);
```

该参数从 `Recognizer` 传递至 `Pipeline` 和 `GPUContext`。

容量解析规则：

```text
physical_kv_rows > 0:
    GPU KV buffer 使用 physical_kv_rows

physical_kv_rows == 0:
    GPU KV buffer 继续使用 n_ctx * n_slots
```

因此已有 SDK、realtime 和 continuous KV 调用不需要修改，行为保持不变。

离线 paged KV 调用方使用统一 helper 解析物理容量：

```text
kv_num_blocks > 0:
    num_blocks = kv_num_blocks

kv_num_blocks == 0:
    num_blocks = batch_size * ceil(ctx_size / block_size)

physical_kv_rows = num_blocks * block_size
```

同一 helper 同时供 GPU 初始化和 `PagedKVBlockPool` 使用，避免 scheduler 容量与真实 GPU buffer 不一致。

## 6. 数据流

```text
CLI/Test options
  batch_size
  ctx_size
  kv_block_size
  kv_num_blocks
        |
        v
resolve_paged_kv_num_blocks(config)
        |
        +--> physical_kv_rows = blocks * block_size
        |       |
        |       v
        |   Recognizer::init_gpu
        |       -> Pipeline::init_gpu
        |       -> GPUContext::init
        |       -> GPUContext::init_kv_cache
        |
        +--> PagedKVBlockPool(blocks, block_size)
```

`ctx_size` 仍用于检查单个 request 的 `n_past` 上限，不再隐式决定 paged 模式的全局物理容量。

## 7. 校验与错误处理

- `physical_kv_rows <= 0` 使用兼容路径，不视为错误。
- 显式 paged block 数必须大于零，block size 必须大于零。
- scheduler 的最大物理 block id 必须落在 GPU buffer 对应范围内。
- block 不足时，scheduler 暂停 admission 或等待 append，不允许越界或退化为重新分配整块 GPU KV buffer。
- 运行结束必须满足 `final_free == capacity` 且 `ownership_errors == 0`。
- GPU 分配失败沿用现有初始化失败流程，不静默回退到不同容量。

## 8. 测试策略

### 单元测试

1. 默认初始化容量仍解析为 `n_ctx * n_slots`。
2. 显式 `physical_kv_rows` 覆盖默认容量。
3. paged block 数显式配置时，GPU rows 与 scheduler capacity 使用同一计算结果。
4. paged block 数为零时，保持旧的 worst-case 自动计算。
5. continuous/realtime 配置不传物理 rows。
6. block pool 耗尽后拒绝 admission，释放后可重新分配。

### 回归测试

- 运行现有 C++ 测试集。
- 运行 benchmark Python 工具测试集。
- 构建 CUDA 目标 `test_offline_batching`。

### GPU benchmark

第一轮固定 `160` 个 block、block size `128`：

```text
batch=12, blocks=160
batch=16, blocks=160
batch=24, blocks=160
batch=32, blocks=160
```

第二轮根据第一轮结果补充：

```text
batch=24, blocks=128/192
batch=32, blocks=192
```

每组至少预热一次、正式运行三次，记录中位数。

## 9. 验收标准

- continuous KV、realtime 和已有三参数 `init_gpu` 行为不变。
- paged scheduler capacity 与 GPU physical rows 完全一致。
- `batch=12, blocks=160` 的峰值显存显著低于当前 7691 MiB。
- `batch=24/32` 在 8 GiB GPU 上不会因 `batch * ctx_size` 预分配而 OOM。
- block 不足通过 admission backpressure 处理。
- 所有 request 完成，KV block 全部回收，无 ownership error。
- 转写文本通过逐 chunk 一致性检查。
- 吞吐若未提高，则保留该内存架构修复，并把下一优化点转向 Q8 GEMM、decode graph kernel 数量和 scheduler token budget。

## 10. 性能判断边界

该改动确定会减少不必要的物理 KV 分配，但更大 batch 是否提高吞吐取决于 Q8 matmul 和 GPU 是否已在 batch 12 饱和。因此本阶段的成功分为两层：

1. 内存架构成功：物理容量与 slot/ctx 解耦，显存下降且行为正确。
2. 性能实验成功：更大有效 batch 进一步降低 wall time。

只有第二层达到并通过文本质量门槛后，才能将结果用于与 vLLM 的最终性能比较。
