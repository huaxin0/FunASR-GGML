# Unified Chunked Prefill Scheduler 设计

**日期：** 2026-07-21

**状态：** 已确认

## 1. 背景与目的

当前离线长音频 Runtime 实际采用两套调度方式：

```text
Encoder / Adaptor / Prefill：一次完整执行一个 request
Decode：                      多 request continuous batching
```

在完成 scheduler 并发与物理 Paged KV Pool 解耦后，6119.29 秒音频在
`batch=40, blocks=192` 下的最佳稳定结果为：

```text
wall 中位数：          40.812 s
prefill wall：         27.910 s
decode dispatch wall： 12.859 s
平均活跃 batch：       32.95
```

Prefill 已经占端到端 wall time 的约 68%。因此下一版 Runtime 将
admission 阶段的整请求 Prefill 改为类似 vLLM 的 token-budget scheduler，
使 Prompt chunk 和 Decode token 共享每轮执行预算，并统一表示为一个混合执行计划。

本设计明确区分两个概念：

1. Scheduler 与输入打包需要统一。
2. Prefill 和 Decode 的 Attention kernel 仍然应针对不同负载专门优化。

本项目不追求一个同时处理所有形状的“万能 Attention kernel”。正确方向是保留
现有高效 Decode kernel，同时在统一 executor 接口下增加 Chunked Prefill 路径。

## 2. 目标

- 用 token-budget scheduling 替换整请求 Prefill admission。
- 使用 `num_computed_tokens` 表示进度，不在 scheduler 中硬编码
  Prefill/Decode 阶段边界。
- 将 request 特有的 Prompt embedding 切分为有上限的 chunk。
- 将本轮 Prompt token 和 Decode token 打包为无 padding 输入。
- 通过每个 request 独立的 position 和 block table 保持请求隔离。
- 只为本轮真正要执行的 token position 分配 Paged KV block。
- 保留任务级 Prefix KV 复用与共享末块的 Copy-on-Write。
- `q_len == 1` 时继续使用现有 Paged Decode Fast Path。
- 为 `q_len > 1` 增加多 query 的 Paged Prefill Attention 路径。
- 只对需要采样的位置执行 LM Head。
- 保留当前 scheduler 作为正确性参考和 A/B benchmark 基线。
- 为未来多 GPU PD 架构保留明确的 KV 传输接口边界。

## 3. 非目标

本阶段不包含：

- 多节点请求路由；
- Prefill/Decode 分离部署或远程 KV 传输；
- Tensor Parallel；
- Speculative Decoding；
- 请求抢占及 Host KV Swap；
- Acoustic Encoder 和 Adaptor 的自动批处理；
- 用 TensorRT、Triton、PyTorch 或其他 Runtime 替换 GGML；
- 在完成三轮 benchmark 之前声称性能已经提高。

Encoder/Adaptor batching 和多 GPU 执行属于后续独立项目，但本次引入的接口
不能阻碍这些方向。

## 4. 主流框架参考

本设计对应当前主流推理 Runtime 的以下思路：

- vLLM V1 使用每轮最大 scheduled-token budget，优先安排 Decode request，
  再用剩余预算执行 Chunked Prefill。
- TensorRT-LLM In-flight Batching 将 Context 和 Generation sequence
  无 padding 地打包，并通过 `max_num_tokens` 限制单轮 token 总数。
- SGLang 支持 Chunked Prefill，并提供可选 Mixed Chunk 模式。这说明统一调度
  并不意味着所有 backend 必须使用同一个 Attention kernel。

参考资料：

- https://docs.vllm.ai/en/stable/configuration/optimization/
- https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html
- https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md

## 5. Request 状态模型

### 5.1 生命周期状态

Scheduler 可见的生命周期调整为：

```cpp
enum class OfflineRequestState {
    WaitingFrontend,
    Ready,
    Running,
    Finished,
    Failed,
};
```

`Prefilling` 和 `Decoding` 不再是独立生命周期状态，而是通过 token 进度推导。

### 5.2 Token 进度

每个 request 保存：

```cpp
struct UnifiedOfflineRequest {
    int request_id = -1;
    OfflineRequestState state = OfflineRequestState::WaitingFrontend;

    int prompt_tokens = 0;
    int cached_prefix_tokens = 0;
    int num_computed_tokens = 0;
    int max_output_tokens = 0;

    GPUEmbeddingHandle prompt_tail;
    KVHandle kv;
    std::vector<int> output_tokens;

    bool waiting_for_kv = false;
};
```

当前可供模型计算的新 token 数为：

```text
num_tokens_with_output = prompt_tokens + output_tokens.size()
num_new_tokens = num_tokens_with_output - num_computed_tokens
```

示例一，Prompt 尚未完成：

```text
prompt_tokens=522
output_tokens=0
num_computed_tokens=256

num_new_tokens=266
```

Scheduler 可以从剩余 266 个 Prompt token 中安排一个 chunk。

示例二，Prompt 刚完成：

```text
prompt_tokens=522
output_tokens=1
num_computed_tokens=522

num_new_tokens=1
```

最新采样出的 token 尚未进入 KV，因此下一轮安排一个 Decode token。

当 Prompt 最后一个 position 完成计算后，该位置的 LM Head 输出第一个生成
token。这个 token 会先加入 `output_tokens`，直到下一轮作为 Decoder 输入时
才会写入 KV。

EOS 一旦被采样，request 立即结束，不再把 EOS 作为下一轮 Decoder 输入。
`max_output_tokens` 只限制输出长度，不参与 KV 容量预估。

### 5.3 Prefix Cache 初始状态

当 request 命中任务级 Prefix KV Cache 时：

```text
num_computed_tokens = cached_prefix_tokens
Prompt tail 从 position=cached_prefix_tokens 开始
```

Request retain 共享 Prefix block。若共享 Prefix 结束于一个未写满的 block，
必须沿用现有 COW 逻辑，在写入 request 特有 Prompt token 前创建私有末块。

## 6. 持有型 GPU Prompt Embedding Pool

当前 `PreparedGPUAudio` 只是 Encoder/Adaptor 可复用 staging tensor 的非持有
视图。下一次 frontend 调用会覆盖上一次结果，因此它不能跨多个 scheduler
iteration 保存未完成的 Prompt。

新增有容量上限的 `GPUEmbeddingPool`：

```cpp
struct GPUEmbeddingHandle {
    int slot = -1;
    int token_count = 0;
    int embedding_dim = 0;
};
```

Embedding Pool 为每个 in-flight request 保存稳定的 GPU 数据：

```text
[audio embeddings][ChatML suffix embeddings]
```

固定 ChatML Prefix 已由任务级 Prefix KV Cache 表示，不再复制到每个 request
的 embedding slot 中。

生命周期规则：

- Frontend 完成后，request 获取一个 embedding slot。
- Adaptor 输出和 suffix embedding 被复制到该 slot。
- 在所有 Prompt chunk 完成之前，slot 地址始终有效。
- 完整 Prompt 计算完成后立即释放 slot。
- Pool 耗尽时暂停新的 frontend，而不是覆盖仍在使用的数据。
- 失败和取消路径必须同时释放 embedding 与 KV 所有权。

按当前形状估计，40 个 request 的 tail 约占 80-90 MiB。这个开销相对 KV Pool
较小，但仍必须由 benchmark 实测。

## 7. Scheduler 策略

### 7.1 配置项

新增明确的 scheduler 配置：

```cpp
int max_num_seqs;
int max_num_scheduled_tokens;
int max_prefill_chunk_tokens;
int max_frontend_requests_per_step;
```

RTX 4070 Laptop GPU 的第一轮实验矩阵：

```text
max_num_seqs:              40
max_num_scheduled_tokens:  256 / 512 / 1024 / 2048
max_prefill_chunk_tokens:  128 / 256 / 512
frontend requests/step:    1
```

这些参数用于 benchmark 调优，不作为代码中的固定假设。

### 7.2 单轮调度流程

每个 scheduler iteration 执行：

1. 回收 Finished/Failed request。
2. 当 embedding slot 和 request 容量允许时，执行有限数量的 frontend。
3. 初始化 `token_budget = max_num_scheduled_tokens`。
4. 为每个可运行 Decode request 安排一个 token，优先保护 Decode ITL。
5. 使用剩余 budget 安排已有 Partial Prefill 和新 Ready request。
6. 每个 Prompt chunk 不超过 `max_prefill_chunk_tokens`。
7. 只为本轮 scheduled token range 分配所需 KV block。
8. 生成并执行一个 `MixedBatchPlan`。
9. 只有 GPU 执行成功后才提交 token 进度和采样结果。

在不浪费剩余 token budget 的前提下，中间 Prompt chunk 优先在 KV block 边界
结束；最后一个 Prompt chunk 可以短于 block size。

### 7.3 KV Backpressure

Request 加入执行计划之前计算：

```text
required_positions = num_computed_tokens + num_scheduled_tokens
required_blocks = ceil(required_positions / block_size)
```

缺少的 block 采用事务式追加。如果 KV Pool 无法满足完整 Prompt chunk，
scheduler 首先根据可用 block 缩小 chunk；Decode token 必须完整 admission，
否则延后到下一轮。

任何 token 在执行前都必须已经拥有对应的物理 KV row。分配失败绝不能退化为
写入另一个 request 的 block。

## 8. Mixed Batch Plan

Scheduler 输出与执行 backend 解耦的计划：

```cpp
struct ScheduledSequence {
    int request_id = -1;
    int token_offset = 0;
    int num_tokens = 0;
    int num_computed_tokens = 0;
    bool produces_logits = false;
    const KVHandle* kv = nullptr;
};

struct MixedBatchPlan {
    std::vector<ScheduledSequence> sequences;
    int total_tokens = 0;
    int logits_rows = 0;
};
```

Model Runner 将其组织为 packed tensor：

```text
input_embeds       [embedding_dim, total_tokens]
positions          [total_tokens]
token_request_ids  [total_tokens]
query_start        [num_sequences + 1]
context_lens       [num_sequences]
block_table        [num_sequences, max_blocks]
selected_rows      [logits_rows]
```

同一 request 的 token 在 packed input 中保持连续。为了简化 Attention dispatch，
Prompt chunk 在内存布局中位于 Decode-only sequence 之前，但 scheduler 层仍然
保持 Decode-first 的预算优先级。

## 9. Packed Model Runner

### 9.1 输入组装

对于每个 scheduled absolute position：

```text
position < prompt_tokens:
    从 prompt_tail[position - cached_prefix_tokens] 复制 embedding

position >= prompt_tokens:
    根据对应 generated token id 获取 token embedding
```

小于 `cached_prefix_tokens` 的 position 已经存在于共享 Prefix KV 中，不会再次
从 request 的 Prompt tail 调度。

输入组装保留在 GPU 上。Generated token ID 不能导致 GPU→CPU→GPU 的 embedding
往返。

### 9.2 共享 Transformer 计算

Projection 和 MLP 将 packed token 维度作为矩阵列数：

```text
[hidden, total_tokens]
```

这会让 Q8 MatMul 看到比当前单请求 Prefill 或单 token Decode 更大的有效
`ncols`，提高 GPU 利用率的可能性。

### 9.3 专用 Attention Dispatch

Mixed Attention backend 接收统一的 Paged KV metadata，但保留两个优化子路径：

```text
query length == 1:
    现有 Paged Decode Attention Fast Path

query length > 1:
    新的 Causal Varlen Paged Prefill Attention
```

对于每个 query token，Attention backend 根据 request metadata 选择该 request
自己的 block table，只允许读取：

```text
[0, absolute_query_position]
```

当前 chunk 的 K/V 必须先写入，Attention 才能读取它们。GGML graph 必须显式
表达这个依赖关系，不能仅依赖“先添加 KV output、后添加 logits output”的偶然
执行顺序保证正确性。

第一版允许 mixed backend 在内部 dispatch 两种 Attention kernel；对外仍然是
一次 `MixedBatchPlan` 执行。不要求也不推荐编写一个万能 Attention CUDA kernel。

### 9.4 Selective LM Head

中间 Prompt token 不需要词表 logits。最后一层 Transformer 完成后，只 gather：

- Decode 输入 token 对应的 hidden row；
- 刚好完成某个 request 完整 Prompt 的最后一个 hidden row。

Final RMSNorm 和 LM Head 只处理：

```text
[hidden, logits_rows]
```

这样可以避免为不参与采样的 Prompt token 构造巨大的
`[vocab_size, prompt_chunk_tokens]` 输出。

## 10. CUDA Graph 策略

新 Runner 首先在关闭 CUDA Graph 复用的条件下验证正确性，Packed execution
稳定后再恢复 graph cache。

候选 graph signature：

```text
total_token_bucket
sequence_count_bucket
logits_row_bucket
max_kv_bucket
```

Position、request ID、block table、context length 和 selected row 都必须作为
运行时 tensor input，不能捕获为不可变化的 host 参数。

在新 cache 被证明更快之前，Decode-only mixed iteration 仍可使用现有 Decode
Graph Cache；旧 scheduler 也继续保留原有 graph 路径。

## 11. 错误处理与 Commit 语义

Scheduler 状态更新遵循 prepare/execute/commit：

1. 构建临时执行计划并预留所需 block。
2. 执行 GPU plan。
3. 成功后更新 `num_computed_tokens` 并追加 sampled token。
4. 失败时回滚本轮新分配的 block，token 进度保持不变。

必须始终满足：

```text
0 <= num_computed_tokens <= prompt_tokens + output_tokens.size()
每个 scheduled position 都有且只有一个已预留物理 KV row
私有 block 的 ref_count == 1
共享 Prefix block 在 COW 前不可写
任务结束：final_free == capacity
任务结束：ownership_errors == 0
```

如果所有可运行 request 都在等待 KV，且没有任何 request 能结束并释放 block，
系统必须报告确定性的 KV Pool exhaustion，而不是空转死循环。

## 12. 可观测性

新增 scheduler 与 executor 指标：

```text
unified_steps
mixed_steps
decode_only_steps
prefill_only_steps
scheduled_prefill_tokens
scheduled_decode_tokens
average_tokens_per_step
average_prefill_chunk_tokens
frontend_wait_steps
embedding_pool_peak
kv_backpressure_steps
packed_build_ms
packed_compute_ms
attention_prefill_ms
attention_decode_ms
selected_lm_head_rows
```

现有 wall、RTF、tokens/s、active batch、block ownership、graph hit 和逐 request
结果指标全部保留。

## 13. 兼容性与实施顺序

旧 scheduler 保持可选择：

```text
--scheduler-mode decode-only
--scheduler-mode unified
```

最终目标是完整 Unified Scheduler，但实现按可审查的阶段推进：

1. 引入 request token progress、纯 token-budget scheduler 和单元测试。
2. 增加持有型 GPU Embedding Pool 及生命周期测试。
3. 通过新 plan 接口打通 Chunked Paged Prefill 正确性。
4. 增加 Packed Mixed Execution 和专用 Attention Dispatch。
5. 增加 Selective LM Head。
6. 恢复 Graph Bucketing 并执行调优矩阵。
7. 只有正确性和性能门槛全部通过后，才将 Unified 模式设为默认。

中间阶段不能被描述成最终完整 Unified Runtime。

## 14. 测试策略

### 14.1 Scheduler 单元测试

- Decode request 在预算中优先于 Prefill。
- 剩余预算能由一个或多个 Prompt chunk 填充。
- 大于预算的 Prompt 会跨多个 iteration 执行。
- Prompt 完成时只生成一次首 token。
- 首 token 只在下一轮作为输入时写入 KV。
- Block 只在越过逻辑块边界时追加。
- KV 不足会缩小或延后计划，不破坏 request 进度。
- Prefix Cache 命中后从 `cached_prefix_tokens` 开始计算。
- 共享的未写满 Prefix block 在私有写入前执行 COW。
- 完成和失败路径只释放一次 embedding 与 KV。

### 14.2 Kernel 与 Runner 测试

- 单 request Chunked Prefill 与 Whole Prefill 的最终 logits 一致。
- 覆盖 chunk 边界 `1`、`127`、`128`、`129` 和最后一个短 chunk。
- Mixed batch 不得读取或覆盖其他 request 的 KV。
- Packed Prefill + Decode 与各自独立执行的参考结果一致。
- Decode-only plan 保持现有输出 token 不变。
- Selective LM Head 与 Full LM Head 对应行一致。

### 14.3 端到端验证

- 先执行 24 chunk smoke test。
- 再执行全部 204 chunk 的 greedy decoding。
- 逐 chunk 比较 token ID 与当前 C++ reference 文本。
- 一次 warmup 后执行三轮正式性能测试。
- 记录峰值显存、最终 free block 和 ownership error。

## 15. 验收标准

正确性门槛：

- `204/204` request 完成。
- 每个 chunk 的输出 token ID 与当前 C++ reference 一致。
- 三轮 Unified 执行得到完全相同的输出哈希。
- `final_free == capacity` 且 `ownership_errors == 0`。
- 任一 request 的 frontend 都不会重复执行。
- 中间 Prompt chunk 不进行 token sampling。

Unified 模式成为默认之前的性能门槛：

- 同一机器、同一音频下，三轮 wall 中位数至少比当前 `40.812 s` 快 5%，
  即不高于 `38.771 s`。
- Prefill 相关 wall time 必须真正降低，不能只是被隐藏到另一个统计项。
- 峰值显存不得超过 8 GiB，并必须记录实际值。
- Decode 输出质量保持不变。

如果正确性通过但性能未通过，Unified 模式保留为 experimental，并继续通过
profile 判断瓶颈位于 Packed Q8 MatMul、Prefill Attention、Input Assembly、
LM Head 还是 Graph Replay。

## 16. 双 GPU 与 PD 扩展

拥有两张 GPU 不意味着必须采用 Prefill/Decode 分离。

### 16.1 离线 Data Parallel

当前工作负载的 204 个音频 chunk 彼此独立，并且完整模型可以放入单张 GPU。
因此双 GPU 的第一基线应是两个完整 Runtime Replica，各自处理一部分 chunk。

优点：

- 不需要跨 GPU 搬运 KV；
- 不存在 P/D Pipeline 比例失衡；
- 故障隔离简单；
- 当 CPU frontend 和 I/O 不成为瓶颈时，吞吐有机会接近线性扩展。

### 16.2 Prefill/Decode Disaggregation

对于在线服务，如果 Prompt 较长、Decode 流量持续存在，或 P/D Worker 需要按
不同吞吐比例独立扩缩容，PD 才更有吸引力。

在本模型中，P GPU 负责：

```text
Fbank → Encoder → Adaptor → Prompt Prefill
```

D GPU 负责自回归 Decode。生成开始前，P Worker 必须将所有 Decoder layer 的
Prompt K/V 传到 D Worker。

物理 block ID 只在当前 GPU 的 KV Pool 内有意义。PD 实现不能直接把 P Worker
的 block table 发给 D Worker 使用。D Worker 必须：

1. 在本地分配目标 block；
2. 创建自己的 block table；
3. 将 K/V row 接收到这些本地 block；
4. 传输完成并确认后，才 admission 到 Decode。

为后续实现保留接口边界：

```cpp
struct KVTransferDescriptor {
    int request_id;
    int token_count;
    int block_size;
};

class KVTransferBackend {
public:
    virtual bool send(const KVTransferDescriptor&, const KVHandle&) = 0;
    virtual bool receive(const KVTransferDescriptor&, KVHandle&) = 0;
};
```

`KVTransferDescriptor` 传递逻辑 metadata，不传递源设备物理指针。

根据当前约 `28 s` Prefill 和 `13 s` Decode 的总耗时，固定 `1P + 1D` 会明显
偏向 Prefill，D GPU 容易空闲。Unified Scheduler 完成后必须重新测量这个比例，
才能决定合理的 PD 拓扑。

在恰好两张 GPU 的离线场景中，Data Parallel 预计是更强的第一基线；PD 作为
面向 Serving 的对比方案，不包含在本次实现范围内。

## 17. 项目表述边界

全部验收门槛通过后，可以准确描述为：

> 面向离线 ASR 实现 token-budget unified scheduler，将 Chunked Prefill 与
> Continuous Decode 组织为 Mixed Packed Batch，按需分配 Paged KV，并保留
> 专用 Decode Attention Fast Path；通过 Selective LM Head 避免中间 Prompt
> token 的无效词表投影。

在验收完成前，准确描述仍然是：

> 已实现以 Decode 为核心的 Continuous Batching，并通过 profile 定位出串行
> Prefill 是当前主要瓶颈，在此基础上完成 Unified Chunked Prefill Scheduler
> 的架构设计与实施规划。
