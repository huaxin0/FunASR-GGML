# vLLM-style Offline ASR Optimization Retrospective

Date: 2026-05-27

## 一句话总结

这次优化把 FunASR-GGML 的长视频离线识别路径，从“单请求或弱 batching 的 GPU 推理”推进到了一个 vLLM-style 的离线 continuous batching MVP：音频切块可以持续进入调度器，LLM decode 可以共享 paged KV cache，并以 token-id batch decode 的方式连续推进多个请求。

最终在 RTX 4070 Laptop 8GB 上，针对约 6119 秒音频、204 个 30 秒窗口 chunk，当前最佳结果为：

```text
wall total=52116ms
rtf=0.0085
audio_sec/s=117.42
tokens/s=368.1
batch=12
block_size=128
fallback_calls=2
```

从早期单路 continuous decode 的约 308 秒，到当前约 52 秒，整体 wall time 约 5.9 倍提升；从实时系数看，从约 `rtf=0.0504` 到 `rtf=0.0085`。如果从最初普通离线流程或 CPU/弱 GPU 路径对比，提升可以达到十几倍到二十多倍量级。更重要的是，这次优化不是偶然调参，而是把 vLLM 的核心思想迁移到了 ASR 长视频离线场景。

## 为什么要走 vLLM-style 路线

FunASR 的离线长视频任务和在线 LLM serving 有一个共同点：大量请求不是一次性独立跑完，而是由很多 token decode step 组成。每个 chunk 在 prefill 之后，都会进入逐 token decode。单个请求 decode 时 GPU 利用率不高，但如果把多个 chunk 的 decode step 合并，就可以提高吞吐。

vLLM 的几个核心思想很适合这个问题：

- Continuous batching：请求不需要等一整个 batch 都完成才开始下一批，谁空出来就补谁。
- Paged KV cache：每个请求的 KV cache 不需要一整段连续显存，而是按 block 分配。
- Block table：逻辑 token position 通过 block table 映射到物理 KV cache。
- Decode step batching：每一步把活跃请求合并成一个 batch decode。
- 稳定图和稳定 kernel：尽量减少每步重新构图、重新分配和 host/device 同步。

在 ASR 场景里，请求不是用户 prompt，而是音频窗口 chunk。整体流程可以抽象成：

```text
audio
  -> split into chunks
  -> GPU encoder/adaptor
  -> LLM prefill
  -> paged KV admission
  -> continuous batched decode
  -> token output
  -> final transcript
```

## 最开始的问题

一开始项目已经有 GPU 计算、KV cache 和一些 batching 雏形，但离真正的 vLLM-style scheduler 还差几个关键点：

- batch 看起来存在，但 token decode 并没有稳定走 paged batch fast path。
- paged decode 频繁 fallback，日志里反复出现类似：

```text
GPU scheduler paged token-id decode unavailable; falling back
```

- batch 增大以后不一定更快，甚至 batch=16 会严重反弹。
- 缺少足够的 profile，很难判断瓶颈是在 prefill、decode、KV 分配、graph build、kernel，还是 fallback。

这时不能靠猜。第一步是建立可观测性。

## 如何定位问题

我们先加了几类统计，而不是直接改 kernel：

- `fallback_reasons`
  - 区分 single fallback、token-id fallback、host-embed fallback、serial env fallback、invalid fallback。
- `scheduler_profile`
  - 统计 admission 次数、prefill wall time、decode dispatch wall time、idle step。
- `throughput`
  - 输出 audio_sec/s、rtf、tokens/s。
- `paged_profile`
  - 拆分每个 paged decode step 的 build、alloc、set、compute、get 时间。
- `paged_attn_profile`
  - 统计 paged attention kernel 的调用次数和平均 kernel 时间。
- `paged_graph_cache_probe`
  - 判断 graph shape、参数和完整 graph 是否具备复用潜力。

这些 profile 让优化从“感觉哪里慢”变成“看数字做决策”。

## 第一阶段：把 fallback 打干净

早期 paged batch 路径有大量 fallback，说明调度器虽然在组织 batch，但真正 decode 还经常掉回单路或 host embedding 路径。

修复目标是让 token-id paged batch decode 成为主路径。完成后，关键指标变成：

```text
fallback_calls ~= 1~7
token_id=0
host_embed=0
invalid=0
avg_active ~= batch size
```

这一步的意义很大：只有 fast path 稳定，后面的 kernel、graph cache、KV write 优化才有意义。否则优化的是旁路，吞吐不会稳定提升。

## 第二阶段：找到 batch sweet spot

长视频离线吞吐优先，但 batch 不是越大越好。用户在 RTX 4070 Laptop 8GB 上测试过：

```text
batch=4   wall ~= 105s
batch=6   wall ~= 88s
batch=8   wall ~= 80s
batch=12  wall ~= 73s, 后续优化到 52s
batch=16  wall ~= 316s, 明显反弹
```

batch=16 反弹的原因不是调度器逻辑错误，而是显存压力、KV cache 规模、graph/compute 开销共同导致的性能拐点。最后选择：

```text
--batch-size 12
--ctx-size 4096
--kv-block-size 128
```

这是当前 8GB 4070 Laptop 上的吞吐甜点。

## 第三阶段：block size 从 16 调到 128

最开始 paged KV block size 是 16。这个粒度比较细，适合减少碎片，但对长视频离线场景不一定最优。每个 30 秒 chunk 的 token 数相对稳定，使用更大的 block 可以减少 block table 规模和调度开销。

测试后，`block_size=128` 在当前场景更好：

```text
blocks_peak=108/384
block_size=128
```

相比 `block_size=16`，它降低了 block table 和 attention 侧的一部分管理开销，同时碎片浪费仍然可接受。

## 第四阶段：优化 paged attention kernel

在 fallback 清理和 block size 调整之后，paged attention kernel 成为明显热点之一。我们尝试了多个 kernel 版本，其中 v3 最稳定：

```text
v2 avg_kernel ~= 0.208ms
v3 avg_kernel ~= 0.149ms
v4 avg_kernel ~= 0.238ms
```

最后删掉 v4，避免后续混淆，把 v3 提升为默认 paged attention kernel。

这一步的收益很直接：

```text
wall ~= 66s -> 61s
tokens/s ~= 288 -> 310
```

但继续 profile 后发现，paged attention 不是唯一瓶颈。

## 第五阶段：发现 graph build 和 KV copy 问题

加入 `paged_graph_cache_probe` 后，我们发现 graph shape 有复用潜力：

```text
shape_hit_rate=80.74%
```

但完整 graph 不能复用：

```text
param_hit_rate=0.00%
full_hit_rate=0.00%
```

继续拆分后发现，阻碍复用的主要因素有两个：

1. `max_n_kv` 每步变化，导致 graph 参数不稳定。
2. KV 写入使用 per-row `ggml_view_2d + ggml_cpy`，具体 physical row 每步变化，导致 graph 节点和参数不稳定。

第一个问题可以通过 bucket 化 `max_n_kv` 解决：

```text
graph_max_n_kv = max_blocks * block_size
```

这样 graph 不再因为当前最大 KV 长度轻微变化而变化。

加上 `FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1` 后：

```text
param_hit_rate=80.74%
```

但 full graph 仍然不能命中，因为 KV copy rows 仍然每步变化。

## 第六阶段：dynamic paged KV write

这是这次优化最关键的一步。

旧路径里，每层、每个 batch item 都会创建具体 row 的 view 和 copy：

```text
k_src -> k_dst physical row
v_src -> v_dst physical row
```

这会导致：

- graph 节点很多。
- graph 依赖具体 physical row。
- 每一步 copy rows 变化，完整 graph 很难复用。
- build 和 compute 都被额外拖慢。

新的设计是增加一个动态 KV write op：

```text
ggml_paged_kv_write_ext(
  k_cur,
  v_cur,
  k_cache,
  v_cache,
  block_table,
  positions,
  block_size
)
```

它把“根据 block table 和 position 找 physical row 并写 KV”的逻辑放进 CUDA kernel：

```text
block_idx = position / block_size
block_off = position % block_size
block_id  = block_table[request, block_idx]
row       = block_id * block_size + block_off
write k_cur/v_cur -> k_cache/v_cache[row]
```

这样 graph 不再关心每个请求这一步具体写哪个 physical row。graph 只依赖 batch、max_blocks、block_size、max_n_kv 这类稳定参数。

隔离测试证明，dynamic KV write 单独就是大头：

```text
dynamic KV write only:
wall total=53321ms
tokens/s=359.8
paged_profile total=15.842ms
build=0.649ms
compute=14.858ms
param_hit_rate=0.00%
full_hit_rate=0.00%
```

这里 `param/full=0` 但性能已经大幅提升，说明收益不是 graph cache 造成的，而是 dynamic KV write 本身减少了图节点和执行负担。

## 第七阶段：真正 graph cache

在 dynamic KV write 和 bucketed `max_n_kv` 都打开后，完整 graph 签名变稳定：

```text
shape_hit_rate=80.74%
param_hit_rate=80.74%
full_hit_rate=80.74%
```

于是实现了一个最小可行 graph cache：

- 只缓存最近一个 paged token-id decode graph。
- 只在 dynamic KV write 和 bucketed `max_n_kv` 同时开启时启用。
- 用 `FUNASR_PAGED_DECODE_GRAPH_CACHE=1` 控制。
- 如果其他 decode 路径复用同一个 `allocr_decode_`，先 invalidate cache，避免内存布局错乱。

最终结果：

```text
wall total=52116ms
rtf=0.0085
audio_sec/s=117.42
tokens/s=368.1

paged_profile:
build=0.159ms
alloc=0.019ms
set=0.141ms
compute=14.842ms
get=0.082ms
total=15.242ms

paged_graph_cache_probe:
shape_hit_rate=80.74%
param_hit_rate=80.74%
full_hit_rate=80.74%
cache_hit_rate=80.74%
```

graph cache 的收益比 dynamic KV write 小，但方向正确：它主要压低了 build 和 alloc：

```text
build: 0.649ms -> 0.159ms
alloc: 0.096ms -> 0.019ms
wall:  53.32s  -> 52.12s
```

这是合理的，因为主耗时已经转移到 compute。

## 性能演进表

下面是这次优化过程中最有代表性的几个节点：

| 阶段 | 关键状态 | wall | tokens/s | 备注 |
|---|---:|---:|---:|---|
| 单路 continuous | batch=1 | 308.63s | - | decode 单路，avg_active=1 |
| paged batch 初版 | batch=4 | 105.78s | - | fallback 基本打干净 |
| batch sweep | batch=8 | 80.00s | - | batch 继续增大有效 |
| batch 过大 | batch=16 | 316.93s | - | 8GB 上反弹严重 |
| v3 paged attention | batch=12, block=128 | 61.77s | 310.5 | kernel 优化有效 |
| dynamic KV write | batch=12, block=128 | 53.32s | 359.8 | 最大单点收益 |
| graph cache | batch=12, block=128 | 52.12s | 368.1 | build/alloc 进一步下降 |

## 测试方法

这次优化的测试分成三层。

第一层是单元测试，保证统计逻辑和 helper 行为正确：

```bash
cmake --build build-cuda --target test_offline_scheduler -j$(nproc)
./build-cuda/test_offline_scheduler
```

覆盖内容包括：

- scheduler profile 平均值。
- admission round 统计。
- fallback total。
- graph cache probe hit rate。
- bucketed `max_n_kv` helper。

第二层是 CPU smoke test，保证 offline batching 基础路径没有被破坏：

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin zh.wav \
  --kv-mode paged --batch-size 2 --ctx-size 256 \
  --chunk-mode window --chunk-sec 3 --max-tokens 3
```

第三层是用户本地 GPU benchmark，因为沙盒没有 GPU：

```bash
FUNASR_PAGED_KV_WRITE_OP=1 \
FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1 \
FUNASR_PAGED_DECODE_GRAPH_CACHE=1 \
./build-cuda/test_offline_batching FunAsr_q8.bin outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 12 --ctx-size 4096 \
  --kv-block-size 128 --chunk-mode window --chunk-sec 30 --max-tokens 220
```

每轮重点看这些指标：

- `ok=204/204`：正确性基本门槛。
- `fallback_reasons`：确认没有回到慢路径。
- `scheduler_profile`：确认 prefill 和 decode 时间分布。
- `throughput`：看 wall、rtf、tokens/s。
- `paged_profile`：看 build、alloc、compute 是否下降。
- `paged_graph_cache_probe`：看 probe 命中和真实 cache 命中是否一致。

## 这是不是一劳永逸

不是。

这次解决的是当前长视频离线吞吐路径里最明显、最可控的一组瓶颈：

- paged batch fast path 不稳定。
- fallback 太多。
- block size 不适合当前 workload。
- paged attention kernel 有明显优化空间。
- KV copy 方式让 graph 过重、难复用。
- graph build/alloc 每步重复。

这些问题解决后，系统进入了一个新的阶段：主瓶颈已经更接近真正的 GPU compute。也就是说，接下来继续优化会更难，收益也会更接近硬件和 kernel 上限。

但是，这套流程是可以复用的：

```text
建立 profile
  -> 清理 fallback
  -> 找吞吐 sweet spot
  -> 分解 build/alloc/set/compute/get
  -> 用 probe 判断缓存可行性
  -> 做隔离实验
  -> 把有效实验收敛为默认路径
  -> 保留 fallback 开关
```

真正“一劳永逸”的不是某个 kernel，而是这个定位和收敛方法。

## 当前离完整 vLLM 还差什么

现在已经完成了 vLLM-style offline ASR 的第一阶段闭环，但还不是完整 vLLM。

已经具备：

- Continuous batching。
- Paged KV block pool。
- Block table。
- Batched token-id decode。
- Dynamic paged KV write。
- Graph cache experiment。
- Scheduler/profile/fallback observability。
- 长视频离线吞吐 benchmark。

仍然可以继续做：

- 多 shape、多 batch bucket 的 graph cache，而不是只缓存最近一个 graph。
- 更成熟的 admission policy，例如按剩余 token、KV 压力、音频窗口长度调度。
- Prefix/prompt cache，对 ASR 来说可能是 system prompt、固定语言提示或重复开头 token。
- 更强的 paged attention kernel，当前 compute 仍是主耗时。
- 更完整的 benchmark matrix，例如 30min/1h/2h、不同显存、不同 batch。
- 服务化接口，例如任务队列、取消、优先级、流式输出。
- 默认策略收敛，例如 dynamic KV write 默认启用，graph cache 继续作为实验开关观察。

## 面试时可以怎么讲

可以按这个结构讲：

1. 背景：长视频 ASR 离线识别本质上有大量独立 chunk，每个 chunk 都会进行 LLM token decode，单路 GPU 利用率低。
2. 目标：借鉴 vLLM continuous batching 和 paged KV，让多个 chunk 的 decode step 合并执行，提高吞吐。
3. 第一问题：batch 路径不稳定，paged token decode fallback 多，所以先做 fallback reasons 和 scheduler profile。
4. 第一修复：打通 token-id paged batch fast path，让 fallback 从大量下降到个位数。
5. 第二问题：batch 不是越大越好，通过 benchmark 找到 8GB 机器上 batch=12 是甜点。
6. 第三问题：paged attention kernel 是热点，优化 kernel 后 wall 从 66s 左右降到 61s 左右。
7. 第四问题：profile 显示 graph build/compute 仍高，cache probe 显示 shape 可复用但 full graph 不可复用。
8. 关键设计：把 per-row KV copy 改成 dynamic paged KV write CUDA op，让 graph 不依赖具体 physical row。
9. 结果：dynamic KV write 把 wall 从 61s 左右降到 53s 左右。
10. 收尾：基于稳定签名做 graph cache，把 build/alloc 继续压低，最终到 52s。

可以强调的一点是：这不是简单“调大 batch”，而是围绕 vLLM 的核心约束做系统化工程：

```text
调度稳定性
KV 内存管理
fast path 覆盖率
kernel 热点
graph 稳定性
profile-driven iteration
```

这也是这个项目最有价值的部分。
