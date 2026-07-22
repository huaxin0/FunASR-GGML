# 统一离线 ASR Benchmark 结果与审计

日期：2026-07-17

## 1. 测试目标

在同一台 RTX 4070 Laptop GPU、同一份 6119.294 秒音频和相同的 30 秒固定切片上，对比以下离线推理路径：

- 本项目 C++ Q8_0，单请求连续 KV（`cpp-b1`）
- 本项目 C++ Q8_0，`batch=12` 的 paged KV 调度（`cpp-b12`）
- FunASR 官方 PyTorch 路径
- FunASR 官方 vLLM 路径

每条路径预热一次、正式运行三次，性能表采用中位数。模型加载时间不计入推理 wall time。

原始产物：

- 四引擎主测试：`outputs/benchmark_suite/20260717_204951`
- vLLM 独立复测：`outputs/benchmark_suite/20260717_220724`
- C++ 固定物理池并发扫描：`outputs/bench_paged_kv_pool_concurrency/20260717_225412`
- C++ `batch=40, blocks=192` 最终复测：`outputs/bench_paged_kv_pool_concurrency/20260717_232744`

## 2. 性能结果

| Engine | 精度/后端 | Median wall | RTFx | RTF | Peak VRAM | 完成数 |
|---|---|---:|---:|---:|---:|---:|
| C++ `batch=1` | Q8_0 | 321.009 s | 19.06x | 0.0525 | 3031 MiB | 204/204 |
| C++ `batch=12` | Q8_0 + paged KV | 56.774 s | 107.78x | 0.0093 | 7691 MiB | 204/204 |
| C++ `batch=40` | Q8_0 + 192-block paged KV | **40.812 s** | **149.94x** | **0.0067** | 5592 MiB | 204/204 |
| C++ Unified `batch=40` | Q8_0 + mixed prefill/decode | **38.015 s** | **160.97x** | **0.0062** | 未采样 | 204/204 |
| FunASR PyTorch | framework default | 812.275 s | 7.53x | 0.1327 | 5126 MiB | 204/204 |
| FunASR vLLM（复测） | BF16/default | 42.770 s | 143.08x | 0.0070 | 7880 MiB | 204/204 |

由此得到：

- C++ `batch=12` 相对 C++ `batch=1`：`5.65x` 加速。
- C++ `batch=12` 相对 FunASR PyTorch：`14.31x` 加速。
- 物理池解耦和并发调优后，C++ `batch=40` 相对原 C++ `batch=12` 再加速 `1.39x`。
- C++ `batch=40` 的中位 wall 比该次 vLLM 低 `4.6%`，对应吞吐高约 `4.8%`。
- experimental Unified `batch=40` 相对旧 C++ `batch=40` 再降低 `6.85%` wall，
  相对单请求总加速达到 `8.44x`。
- Unified 中位 wall 比该次 vLLM BF16/default 低 `11.12%`，但仍不是同精度结论。
- 这不是同精度结论：C++ 是 Q8_0，vLLM 是 BF16/default，且 vLLM 输出存在下文记录的稳定性问题。

## 3. C++ 路径的有效性

`cpp-b12` 的调度指标与此前长音频测试一致：

```text
avg_active       11.47
fallback_calls   2
graph_hit_rate   80.5%
peak_blocks      61
```

这说明性能提升不是来自少跑任务，而是来自以下机制共同生效：

- scheduler 将多个 request 的单 token decode 合成 GPU batch；
- paged KV 允许 request 动态占用和释放 KV block；
- decode bucket 稳定计算 shape；
- CUDA graph cache 复用约 80% 的 decode graph；
- batch 长时间接近满载，快速路径只在收尾阶段少量 fallback。

## 4. 固定物理 KV 池与并发解耦

原实现把 scheduler 并发和物理 KV 容量绑在一起：

```text
physical_rows = batch_size * ctx_size
batch=12, ctx=4096 -> 49152 rows -> 384 blocks
```

但本工作负载中，每个 30 秒 chunk 的实际 prefill 约为 522 token，动态分配通常只需要 5 个 128-token block。按每个 slot 预留完整 4096-token context 会浪费显存，也限制了可运行的 batch。

修改后：

```text
physical_rows = global_num_blocks * block_size
batch_size     = scheduler 最大并发
num_blocks     = 独立的全局 KV budget
```

固定 160-block 池的并发扫描结果如下。每个配置均为三次中位数：

| Batch | Blocks | Median wall | Avg active | Peak blocks |
|---:|---:|---:|---:|---:|
| 12 | 160 | 53.468 s | 11.47 | 61 |
| 16 | 160 | 48.716 s | 15.18 | 81 |
| 24 | 160 | 44.685 s | 22.24 | 121 |
| 32 | 160 | 43.371 s | 27.75 | 160 |

把池扩到 192 blocks 后，`batch=40` 能保持更高 active batch。最终三次结果为：

```text
40.483 s
40.812 s
41.097 s
median = 40.812 s
avg_active = 32.95
peak_blocks = 192/192
peak_vram = 5518--5592 MiB
```

这说明本轮最大收益并不是单个 CUDA kernel 变快，而是把原来浪费在每-slot 最大 context 预留上的显存换成了更高的有效并发。

## 5. `batch=40` 退化的定位与修复

第一次运行 `batch=40, blocks=192` 时 wall 反而达到 `90.158 s`。关键指标是：

```text
no_kv=465
peak_blocks=192/192
```

block pool 的 backpressure 本身是正确的，真正的问题在 admission 顺序：scheduler 先调用 `prepare_audio_gpu()` 完成 encoder/adaptor，再计算实际 block 需求；如果此时容量不足，请求 admission 失败，下一轮又会为同一个 chunk 重跑 encoder。

修复采用按 `sample_count` 学习 prefill block 需求的轻量 hint：

1. 第一次遇到某种 chunk 长度时，正常执行 frontend 并记录实际 `extra_blocks`。
2. 后续同长度请求在已知容量不足时，只做 block 数检查并等待。
3. 容量恢复后，才重新进入 frontend 和 prefill。

修复后 `no_kv=465` 仍然保留，它表示 scheduler 确实经历了容量等待；但这些等待不再重复 encoder。完整 wall 从 `90.158 s` 降到稳定的 `40.812 s` 中位数。

三轮最终输出和旧 `batch=12` 基线的完整转写哈希均为：

```text
c8bdfe8f76567676f4c69853350df89324f67ab8d593720e62dfaeb82803ab5a
```

因此本次调度与内存优化没有改变转写输出。

## 6. vLLM 质量审计

仅看 `204/204` 非空输出不等于识别正确。对三轮文本做去空白和标点后的相似度检查，发现官方 vLLM 路径存在明显的跨轮不稳定和重复退化。

第二次 vLLM 测试中：

```text
r1 vs r2: mean=0.961816, min=0.564756, below_0.90=28/204
r2 vs r3: mean=0.961050, min=0.569593, below_0.90=25/204
```

异常片段出现了“吧吧吧……”“因为在医院里……”等重复生成。它不是单纯的标点差异，因此当前 vLLM 数字只能作为吞吐上界参考，不能作为通过质量门槛后的最终对比结果。

作为辅助检查，PyTorch 第一轮与 vLLM 复测第一轮的平均相似度为 `0.989093`，但仍有 7 个 chunk 低于 `0.95`、1 个低于 `0.90`。由于没有标注文本，这只是跨引擎一致性检查，不是 CER/WER。

检查 FunASR 1.3.15 源码后确认，音频 encoder 和 adaptor 已显式进入 `eval()`；因此不能把异常归因于 dropout。当前更合理的待验证方向是 vLLM prompt-embeds 路径、批调度下的数值非确定性，以及生成参数/停止条件。

## 7. 公平性限制

这次结果适合说明工程进展，但还不是论文级横评：

- C++ 使用 Q8_0，vLLM 使用框架默认 BF16，精度和显存口径不同。
- PyTorch 环境为 FunASR 1.3.1，vLLM 环境为 FunASR 1.3.15。
- 峰值显存来自设备级采样，接近 8 GiB 上限时会受后台上下文影响。
- 目前没有人工标注，不能报告 CER/WER，也不能证明四条路径质量等价。
- vLLM 首次主测试的三轮 wall 为 28.10--43.21 秒，复测为 41.58--45.99 秒，说明预热/图捕获口径还需继续收紧。
- C++ `batch=40` 与 vLLM 使用不同权重精度，因此“wall 更低”不能外推成同精度、同质量下全面领先。

## 8. 当前可对外表述

可以严谨地说：

> 面向 102 分钟长音频实现 C++ 离线推理 runtime，通过 paged KV、动态 block、continuous batching、decode bucket 与 CUDA graph cache，将 Q8_0 推理从单请求 321.0 秒降低到 40.8 秒；进一步实现共享 token budget 的 unified prefill/decode scheduler，将 Prompt 与 Decode 打包进同一次 Transformer forward，最终把三轮中位 wall 降至 38.0 秒、总加速 8.44 倍。在本机该配置比官方 vLLM BF16/default 复测中位 wall 低 11.1%，三轮 204-chunk token hash 完全一致。同时建立统一 benchmark 和逐 chunk 文本审计，避免只以非空输出和吞吐下结论。

在补齐标注集 CER/WER 前，不应写“精度无损”或“全面超过 vLLM”。

## 9. 下一步

1. 给 benchmark suite 增加自动 repeat-consistency 报告和退化检测，使质量门槛成为正式指标。
2. 用小规模 A/B 隔离 vLLM 的 eager/CUDA graph、batch size、prompt embeds 和停止条件，定位重复生成来源。
3. 建立带人工标注的固定子集，统一计算 CER，并给 C++、PyTorch、vLLM 设置相同质量门槛。
4. 在质量门槛通过后，再比较 P50/P95、吞吐、峰值显存和冷启动时间。
5. 将当前按音频长度学习的 admission hint 升级为显式的 prefill-cost estimator 或 pending prepared-request cache，覆盖长度高度离散的 VAD chunk。
6. 继续优化高并发下 graph shape 命中率；`batch=40` 的 graph hit rate 为 `67.53%`，低于 `batch=12` 的约 80%。

## 10. Unified Prefill/Decode Scheduler 进展（2026-07-21）

这一轮把此前“逐请求 Prefill、批量 Decode”的两段式执行升级为 experimental unified
scheduler。它在每轮共享一个 token budget，可同时调度 Prompt chunk 和 Decode token，
并且只在 GPU forward 成功后提交 request 进度。

当前 mixed runner 的执行结构是：

1. Prompt embedding 保持 GPU resident，按本轮 plan 复制到 packed staging；Decode 只上传 token id。
2. Q8 Q/K/V projection、O projection 和 MLP 对 `[hidden, total_packed_tokens]` 合批执行。
3. Prompt Attention 按 sequence 使用 FlashAttention；历史 K/V 根据运行时 block table 从物理 Paged KV 行 gather，当前 chunk K/V 直接参与 attention。
4. Decode suffix 继续走原有 Paged Attention，并用显式 graph dependency 保证 KV write 先于 attention read。
5. 仅对完成 Prompt 的最后一行和 Decode 行执行 LM Head，避免为中间 Prompt token 计算词表 logits。
6. graph signature 覆盖 Prompt/Decode shape、block bucket 和 selected rows，运行时更新 position、block table、KV length、token id 和 physical-row inputs。

固定 RTX 4070 Laptop GPU、前 24 个 30 秒 chunk、`batch=12`、`max_tokens=220`、
`block_size=128`、`num_blocks=128`，单轮 A/B 结果如下：

| 路径 | Frontend/step | Wall | Audio xRT | Graph hit | Packed compute |
|---|---:|---:|---:|---:|---:|
| 旧 decode-only scheduler | 旧 admission | 6.206 s | 116.02x | 88.60% | N/A |
| Unified | 1 | 6.021 s | 119.58x | 75.25% | 3.634 s |
| Unified | **4** | **5.921 s** | **121.60x** | **81.41%** | **3.551 s** |
| Unified | 12 | 6.046 s | 119.09x | 81.41% | 3.574 s |

`frontend_per_step=4` 的 wall 比旧路径低 `4.59%`，同时保持：

```text
ok=24/24
blocks_peak=61/128
ownership_errors=0
final_free=128/128
prompt_embedding_pool=12/12 free
prefix_hits=24/24
saved_prefill_tokens=432
```

`frontend=4` 优于 1 的原因是能让更多 Prompt row 共享 packed Q8 GEMM，并减少 graph
shape 抖动；继续增到 12 会先串行准备过多 frontend，请求首轮 GPU 执行被推迟，因此
wall 反而回升。

### 当前质量边界

同一 `4-chunk/frontend=1` 配置重复两次，生成 token 完全一致，说明新路径本身具有
重复确定性。不同 scheduler/admission batch shape 之间仍可能在少量位置产生不同 token；
原因是 Q8 MatMul 和 Attention 的归约 shape 改变后，接近的 top-1 logit 可能交换顺序。
24-chunk 各配置均完成转写且文本整体一致，但在补齐标注集 CER 和逐 chunk logits margin
审计前，不能把这组数据表述成“精度无损”。

### 204-chunk 三轮结果

随后使用此前最佳资源配置 `batch=40 / 192 blocks` 跑完整 6119.29 秒音频。第一组
`token_budget=1024 / prefill_chunk=256 / frontend=4` 的三轮中位数为 `38.956s`。
观察到每个约 504-token 的非共享 Prompt 被拆成 `256+248` 两个 shape 后，将
`prefill_chunk` 提升为 512，使单请求 Prompt 在一个 scheduler step 内完成。最终三轮为：

```text
round 1: 37.973 s
round 2: 38.015 s
round 3: 38.164 s
median : 38.015 s
RTFx   : 160.97x（按中位数）
```

三轮逐 chunk token 按 chunk id 排序后的 SHA-256 完全一致：

```text
cef39f7eab92d84474874f2182b407652b2f9c93b2684f252a194d693d173d18
```

每轮关键资源指标也完全一致：

```text
ok=204/204
steps=601, mixed=170, prefill_only=1, decode_only=430
avg_active=32.03
blocks_peak=192/192
prefix_hits=204/204, saved_prefill_tokens=3672
ownership_errors=0, final_free=192/192
prompt_embedding_pool=40/40 free
graph_hit_rate=59.57%
```

相对旧 `batch=40` 的 `40.812s` 中位数，本轮 wall 下降 `6.85%`，总加速达到
`321.009 / 38.015 = 8.44x`。相对该次官方 vLLM BF16/default 的 `42.770s`，
wall 低 `11.12%`；后一个数字仍受 Q8 与 BF16 精度口径不同的限制，不能表述成
同精度全面领先。

因此当前结论是：unified runtime 已完成功能闭环，在完整负载上稳定超过旧 C++ 路径，
且三轮输出完全一致；`6.85%` 已通过计划中的 5% 性能门槛。由于还没有标注集 CER/WER，
unified 总开关仍保持 experimental 默认关闭，但其内部默认参数更新为
`prefill_chunk=512 / frontend=4`。下一步优先降低仍只有 59.57% 的 graph hit rate，
并补齐跨旧路径的 CER/logits-margin 审计。

## 2026-07-22：批量 Acoustic Frontend 与确定性准备队列

### 问题

此前 Unified Scheduler 已经能够合并 Prompt/Decode 的 LLM forward，但声学前端仍有
两个串行点：

1. 每个 request 的 Fbank、Encoder、Adaptor 逐个执行。
2. active batch 满载后，每次通常只空出一个槽，admission 因而退化成 batch=1 的
   frontend；其分组还会受请求完成时序影响。

### 实现

- 新增 `[dim, frames, batch]` Encoder/Adaptor 图，QKV、FFN 和 Attention 以显式 batch
  维执行。
- 扩展 fused FSMN CUDA op，使时间卷积以 sequence slice 为边界，禁止跨 request 读取。
- 对变长音频按 `max_frames` padding，并同时使用 attention key mask 和 FSMN valid mask；
  只复制每个 request 的有效输出帧。
- 同一个 frontend group 的 Fbank 在 CPU 上并行执行，并由 `n_threads` 限制并发数。
- 引入 FIFO prepared queue。frontend 固定按输入顺序每 4 个 request 准备，request
  进入 active 时才 retain prefix KV、执行 partial-block COW 和申请后续动态 block。
- prompt embedding pool 与 active batch 解耦，并由 runtime 自动扩到
  `batch_size + max_frontend_requests`；外部 C++ 调用不再依赖 CLI 初始化细节。
- 新增 `--frontend-batching on|off`，使新旧前端可以在同一 scheduler 下直接 A/B。

### 完整负载 A/B

固定 RTX 4070 Laptop GPU、6119.29 秒音频、204 chunks、`batch=40`、192 KV blocks、
`frontend=4`：

| 路径 | Wall | Audio xRT | Frontend wall | Graph hit |
|---|---:|---:|---:|---:|
| 同一 FIFO + 逐请求 frontend | 40.213 s | 152.17x | 20.927 s | 59.57% |
| 批量 frontend round 1 | 27.471 s | 222.75x | 8.312 s | 59.14% |
| 批量 frontend round 2 | 27.910 s | 219.25x | 8.512 s | 59.14% |
| 批量 frontend 中位数 | **27.691 s** | **220.99x** | **8.412 s** | 59.14% |

相对同调度器逐请求前端，端到端 wall 下降 `31.14%`；相对上一阶段 38.015 秒的
Unified 中位数下降 `27.16%`。相对最初单请求 321.009 秒基线，总加速为 `11.59x`。

两轮批量前端的逐 chunk token hash 完全一致：

```text
727d1a0857818dcfa8e463ea461ec6a66b3b257df3f5208b4e01578d6ccf79ca
```

资源和调度指标：

```text
ok=204/204
frontend_calls=51, avg_batch=4.00, single=0, fallback=0
prepared_queue_peak=4
blocks_peak=192/192, ownership_errors=0, final_free=192/192
prompt_embedding_pool=44/44 free
```

日志位于 `outputs/bench_batched_frontend/20260722/`。

### 质量边界与下一步

关闭 frontend batching 的对照 hash 仍为旧基线的
`cef39f7eab92d84474874f2182b407652b2f9c93b2684f252a194d693d173d18`，证明 FIFO 本身
不改变结果。批量前端与逐请求前端有 62/204 个 chunk 的 token 序列不完全相同；原因是
batched GEMM/Attention 使用不同归约 shape，少量接近的 top-1 logit 会交换并在自回归中
放大。新路径跨运行完全确定，但在标注集 CER/WER 通过前仍不能写成“精度无损”。

本轮已经关闭声学前端的串行瓶颈，但没有改善 mixed LLM graph 的 shape 命中率；约 59%
的 graph hit 是独立的后续优化项。下一步应先补 CER/WER 门禁，再按
Prompt/Decode token bucket 分析 246 次 graph miss，而不是继续盲目改 kernel。

## 2026-07-22：AISHELL-1 CER 门禁与全量精度结果

### 门禁实现

- 新增 unified scheduler 多 WAV accuracy runner。7176 个独立 utterance 按固定顺序、
  每组最多 256 条加载，避免把约 1.1GB WAV 全部展开成 float 音频常驻内存。
- 每条待评音频先写入 `expected_ids.txt`；缺失转写按空输出和全删除计分，不允许只取
  reference/hypothesis 交集得到虚假的低 CER。
- evaluator 输出 corpus-level S/D/I/N、coverage、empty、duplicate、extra、malformed、
  worst cases、JSON 和 Markdown；门禁失败返回非零退出码。
- 工具支持按真实空格词边界计算 WER，但 AISHELL 中文 hypothesis 没有可靠词边界，
  因此本数据集只以 CER 作为 release metric。

一键命令：

```bash
tools/bench_asr_accuracy.sh
```

默认严格条件为 `coverage=100%`、missing/empty/duplicate/extra/malformed 全为 0、
`CER <= 2.2%`。该阈值在首轮用 3.0% 完成有效性验证后，根据全量实测的 1.9902%
收紧，保留约 0.21 个百分点的跨环境余量。

### 全量结果

固定 Q8 模型、RTX 4070 Laptop GPU、`batch=40`、192 个 128-token KV blocks、
unified token budget 1024、frontend batch 4：

```text
utterances=7176/7176
audio=36108.919s
wall=179.684s
audio_xrt=200.96x
empty=0, load_fail=0, ownership_errors=0
CER=1.9902%
S=1726, D=248, I=111, N=104765
perfect_utterances=5818
gate=PASS
```

该 CER 与项目此前逐请求链路约 1.98% 的历史结果一致，说明批量声学前端产生的少量
token 分歧没有造成 corpus quality regression，可以把“新路径精度通过标注集门禁”作为
当前结论。完整产物位于：

```text
outputs/accuracy_benchmark/20260722_165600/
```

### 新发现

短音频 accuracy workload 的 graph cache 命中为 `95 / 2240 = 4.24%`，明显低于固定
30 秒长视频 workload 的约 59%。两者不是相互矛盾：AISHELL utterance 长度、prompt
长度和 decode 完成时序更分散，mixed batch shape 的组合更多。后续 graph 优化应分别
报告长视频和短请求 corpus，不能只针对单一 shape 分布调参。

## 2026-07-22：Runtime 收敛与产品预设

### 本轮实现

1. Mixed CUDA Graph 从“仅保留上一次”改成可配置的多 entry LRU；signature 覆盖
   Prefill/Decode token 数、KV block bucket、selected rows 与逐序列 Prefill layout。
2. 记录所有 mixed shape 的调用频率；新 shape 第一次只临时执行，第二次出现才进入
   LRU，避免大量一次性变长 Prefill graph 污染热点缓存。
3. Frontend 按有限 lookahead window 做长度排序；每组仍保持确定性顺序，最终结果按原
   chunk id 还原。AISHELL padding 从 FIFO 对照的约 20.55% 降到 5.57%。
4. 增加异步 Fbank job 与 prepared queue，在当前 LLM mixed step 执行时准备下一组
   Fbank。完整 AISHELL 中 98.38% 的 frontend job 在消费时已经 ready。
5. 增加独立 CUDA stream 的 Encoder/Adaptor overlap 实验开关。RTX 4070 Laptop 上
   Encoder 与 Decoder 争用 SM/带宽，24-chunk wall 从约 3.24 秒退化到约 3.54 秒，
   因此保留实现但默认关闭。
6. 修复 KV backpressure：prepared request 在拿不到首个 private block 时继续排队，
   不再被错误标记失败；同时避免 `fill_active` 在队首等待 KV 时空转。
7. `long-video` preset 启用 unified scheduler、prefix KV、dynamic block、Fbank prefetch、
   frontend bucket、16-entry graph cache 和显存自动调优；`long-video-legacy` 保留旧路径。

### 并发与 KV 自动调优结论

固定 6119.29 秒音频、204 个 30 秒 chunk，三轮结果为：

| 配置 | 三轮 wall | 中位数 | 峰值 KV | 逐 chunk 重复一致性 |
|---|---|---:|---:|---|
| batch 48 / 224 blocks | 23.340 / 23.080 / 23.097 s | **23.097 s** | 224 | 三轮完全一致 |
| batch 56 / 256 blocks | 23.174 / 23.263 / 23.422 s | 23.263 s | 256 | 首轮与后两轮相差 12/204 |

batch 56 已进入收益平台：平均 active 只从 36.98 增到 38.22，额外 32 个 block 没有换来
更高的中位吞吐。因此 8GB 档自动默认收敛为 `batch=48, blocks=224`；56/256 继续作为
显式实验档，不作为产品默认。

相对单-entry、无 prefetch 的同代 `graph1` 对照（28.543 秒），48/224 路径中位数下降
约 19.1%。相对上一阶段 batched frontend 中位数 27.691 秒，下降约 16.6%。相对最初
单请求 321.009 秒基线，端到端加速约 **13.90x**。

### 全量精度门禁

长视频配置收敛后，又针对短请求做了独立矩阵。512 条样本的 inference 时间为：

```text
batch 40 /  96 blocks: 9.441s
batch 48 / 112 blocks: 9.291s
batch 56 / 128 blocks: 9.433s
```

因此完整 AISHELL 使用 `batch=48, blocks=112` 复验，最终结果为：

```text
utterances=7176/7176
audio=36108.919s
wall=151.778s
inference=137.650s
audio_xrt=237.91x
CER=2.0035% (S=1745, D=246, I=108, N=104765)
missing=0, empty=0, ownership_errors=0
blocks_peak=85/112
graph_hit_rate=6.22%, graph_entries=16/16
frontend_padding=5.57%, prefetch_ready=98.38%
gate=PASS
```

与本页前一轮 179.684 秒结果相比，端到端 wall 下降 15.53%；相对本轮先跑的
`batch=40, blocks=192`（166.235 秒）再下降 8.70%。物理 KV pool 从 192 blocks 缩到
112 blocks，下降 41.67%，同时允许 scheduler 并发从 40 提升到 48。CER 从 1.9902%
变化到 2.0035%，增加 0.0133 个百分点，仍通过 2.2% 严格门禁。结果目录：

```text
outputs/accuracy_benchmark/20260722_183326/
outputs/bench_unified_runtime/20260722_181720/
```

### 当前边界

- LRU 已解决“只缓存最后一个 graph”和一次性 shape 污染，但 AISHELL 仍有 1862 种
  mixed shape；仅靠扩大 cache 不会把 6.22% 命中率变成高命中。
- 尚未对任意 mixed Prefill tensor 做伪 padding。直接伪造 signature 不安全；真正的
  padding 需要 dummy sequence、独立 scratch KV block、输出过滤和 CER 回归。
- CPU Fbank 已与 LLM 重叠；单卡 GPU Encoder/Adaptor 与 LLM 的强行并发在本机负收益。
  多 GPU 场景应在独立设备上做 pipeline/PD 风格资源隔离，而不是复用这个单卡开关。
