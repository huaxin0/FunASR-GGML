# Offline ASR vLLM-Style Optimization Summary

Date: 2026-06-17

## 一句话总结

这个项目的核心路线，是把 FunASR-GGML 的长视频离线识别从普通 GPU
推理，推进成一个 vLLM-style 的离线 continuous batching 推理系统。

这里的 vLLM-style 指的是借鉴 vLLM 的推理系统思想：

```text
continuous batching
paged KV cache
block table
batched token-id decode
stable decode shapes
graph/cache reuse
profile-driven kernel optimization
```

不是简单把 batch 调大，也不是只写一个 CUDA kernel，而是围绕“长视频被切成
很多 chunk，每个 chunk 都要进入 LLM decode”这个 workload，重新组织调度、
KV 内存、decode fast path、graph 构建和 kernel 执行。

当前最好的完整长视频配置在 RTX 4070 Laptop 8GB 上，可以把约 6119 秒音频、
204 个 30 秒 chunk 的离线 ASR 推到大约：

```text
wall ~= 52-56s
rtf ~= 0.0085-0.0092
audio_sec/s > 100
batch=12
avg_active ~= 11.5
fallback_calls ~= 2
graph_cache_hit_rate ~= 80.74%
```

从早期单路 continuous decode 的约 308 秒，到现在约 52-56 秒，长视频离线
decode 这条路径已经提升了约 5.5-6 倍。

## 为什么一开始选择 vLLM-Style 路线

长视频 ASR 的离线流程天然适合借鉴 LLM serving 的优化思想。

传统看法里，ASR 是：

```text
audio -> encoder -> decoder -> transcript
```

但在这个项目里，长视频会被切成很多 30 秒窗口 chunk。每个 chunk 的前半段是
encoder/adaptor/prefill，后半段是 LLM decoder 的逐 token decode：

```text
audio
  -> chunks
  -> encoder/adaptor
  -> LLM prefill
  -> token-by-token decode
  -> transcript
```

单个 chunk decode 时 GPU 利用率不高，尤其是 decode 阶段每一步只生成一个
token。如果把多个 chunk 的 decode step 合并起来，就可以提高吞吐。这和
vLLM 在线服务里的核心问题非常像：

```text
很多请求都在 decode
每个请求长度不同
KV cache 要动态增长
请求会不断完成和补位
GPU 需要保持高利用率
```

所以一开始选择 vLLM-style 路线，是因为它正好对应这个问题的本质：

```text
用 continuous batching 提高 decode 并行度
用 paged KV cache 管理不同 chunk 的 KV
用 block table 解耦逻辑 position 和物理 KV row
用稳定 shape 和 graph cache 减少每步构图开销
```

## 最开始的问题

项目早期已经有 GPU 推理和一些 batching 雏形，但离真正稳定的 vLLM-style
离线 scheduler 还有距离。

主要问题是：

```text
batch 看起来存在，但 decode fast path 不稳定
paged token-id decode 经常 fallback
batch 变大后不一定更快
block size、ctx size、batch size 的甜点不清楚
graph build / alloc / compute 的占比不清楚
缺少按阶段拆分的 profile
```

典型现象是日志里会出现：

```text
GPU scheduler paged token-id decode unavailable; falling back
```

这说明 scheduler 组织了 batch，但真正执行时并没有稳定走到最想要的 paged
batched decode 路径。这个阶段最重要的不是急着改 kernel，而是先建立可观测性。

## 第一阶段：先建立可观测性

最开始补的是 profile 和统计，而不是直接做大改。

增加的核心观测指标包括：

```text
fallback_reasons
scheduler_profile
throughput
paged_profile
paged_attn_profile
paged_graph_cache_probe
```

这些指标分别回答不同问题：

```text
fallback_reasons:
  到底为什么掉慢路径，是 single、token_id、host_embed 还是 invalid？

scheduler_profile:
  admission、prefill、decode dispatch、idle step 分别花多少？

throughput:
  wall、rtf、audio_sec/s、tokens/s 到底是多少？

paged_profile:
  每步 paged decode 的 build、alloc、set、compute、get 各占多少？

paged_graph_cache_probe:
  graph 的 shape、参数、完整签名有没有复用可能？
```

这一步的意义是把问题从“感觉慢”变成“知道慢在哪里”。

## 第二阶段：把 Paged Batch Fast Path 打稳

有了 fallback 统计后，第一目标是让 token-id paged batch decode 成为主路径。

优化后的目标状态是：

```text
fallback_calls ~= 1-7
token_id=0
host_embed=0
invalid=0
avg_active 接近 batch size
```

这一步是后面所有优化的前提。因为如果 batch decode 仍然频繁 fallback，那么
后续做 paged attention、dynamic KV write、graph cache 都是在优化旁路，实际
吞吐不会稳定。

## 第三阶段：找 Batch / KV 的甜点

batch 不是越大越好。RTX 4070 Laptop 8GB 上做过 batch sweep：

```text
batch=4   wall ~= 105s
batch=6   wall ~= 88s
batch=8   wall ~= 80s
batch=12  wall ~= 73s, 后续优化到 52s
batch=16  wall ~= 316s, 明显反弹
```

batch=16 变慢不是 scheduler 逻辑错误，而是显存、KV cache、graph 和 kernel
开销一起触发了性能拐点。最终当前机器上比较稳的配置是：

```text
--batch-size 12
--ctx-size 4096
--kv-block-size 128
```

KV block size 也从早期的 16 调到 128。对于这个长视频离线 workload，30 秒
chunk 的 token 数比较稳定，大 block 能减少 block table 和调度管理开销，
同时碎片浪费仍然可接受。

## 第四阶段：Paged Attention Kernel

当 fallback 清掉、batch 和 block size 稳定后，paged attention 成为明显热点。

不同 paged attention kernel 版本的结果大致是：

```text
v2 avg_kernel ~= 0.208ms
v3 avg_kernel ~= 0.149ms
v4 avg_kernel ~= 0.238ms
```

最后选择 v3 作为默认，删掉表现不稳定的 v4，避免后续混淆。

这一步把 wall 从约 66 秒推进到约 61 秒，说明 kernel 本身确实是有效优化点。
但继续 profile 后发现，paged attention 不是唯一瓶颈。

## 第五阶段：发现 Graph 和 KV Copy 问题

`paged_graph_cache_probe` 显示 graph shape 有复用潜力：

```text
shape_hit_rate=80.74%
```

但完整 graph 一开始不能复用：

```text
param_hit_rate=0.00%
full_hit_rate=0.00%
```

主要原因有两个：

```text
max_n_kv 每步变化
KV 写入依赖具体 physical row
```

第一个问题用 bucket 化 `max_n_kv` 解决：

```text
graph_max_n_kv = max_blocks * block_size
```

这样 graph 不再因为当前最大 KV 长度轻微变化而变化。

第二个问题更关键。旧路径里，每层、每个 batch item 都会建具体 row 的 view
和 copy：

```text
k_src -> k_dst physical row
v_src -> v_dst physical row
```

这导致 graph 依赖具体 physical row，每步都不稳定。

## 第六阶段：Dynamic Paged KV Write

这是目前收益最大的单点优化。

新设计是增加一个动态 KV write op：

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

它把“根据 block table 和 position 找 physical row 并写 KV”的逻辑放进 CUDA
kernel：

```text
block_idx = position / block_size
block_off = position % block_size
block_id  = block_table[request, block_idx]
row       = block_id * block_size + block_off
write k_cur/v_cur -> k_cache/v_cache[row]
```

这样 graph 不再关心每个请求具体写哪个 physical row，只依赖 batch、
max_blocks、block_size、max_n_kv 这些稳定参数。

隔离测试显示，dynamic KV write 本身就能把 wall 大幅拉下来：

```text
dynamic KV write:
wall ~= 53.32s
tokens/s ~= 359.8
paged_profile total ~= 15.842ms
```

这一步的意义是：优化的不是一个小 kernel，而是把 graph 结构从“每步生成很多
具体 copy view”改成“一个稳定动态写入 op”。

## 第七阶段：Graph Cache

dynamic KV write 和 bucketed `max_n_kv` 都打开后，完整 graph 签名稳定：

```text
shape_hit_rate=80.74%
param_hit_rate=80.74%
full_hit_rate=80.74%
cache_hit_rate=80.74%
```

于是实现了最小可行 graph cache：

```text
只缓存最近一个 paged token-id decode graph
只在 dynamic KV write 和 bucketed max_n_kv 同时开启时使用
其他 decode 路径复用 allocr_decode_ 前先 invalidate
```

收益比 dynamic KV write 小，但方向正确：

```text
build: 0.649ms -> 0.159ms
alloc: 0.096ms -> 0.019ms
wall:  53.32s  -> 52.12s
```

这说明系统瓶颈已经从 graph build/alloc 转移到了真正的 GPU compute。

## 当前阶段：Scheduler 已经健康，瓶颈转到 Decode Kernel

最近的 profiling 说明，当前 scheduler 层已经比较健康：

```text
batch=12
avg_active ~= 11.47
fallback_calls ~= 2
graph_cache_hit_rate ~= 80.74%
paged_profile compute 占 decode step 约 97%
```

graph cache 和 bucket A/B：

```text
all on:          wall ~= 56.28s, tokens/s ~= 340.9
graph cache off: wall ~= 57.63s, tokens/s ~= 332.9
bucket off:      wall ~= 59.34s, tokens/s ~= 323.3
```

结论是：

```text
graph cache 有收益
bucketed shape 有收益
但现在主耗时已经不是 build/alloc，而是 compute kernel
```

NCU 看到的关键 kernel：

```text
paged_kv_write_f32_to_f16_kernel:
  ~= 3.7us，很小，不是主瓶颈

paged_attn_decode_f32_f16_warp_qk_cached_rows:
  ~= 176us
  SM throughput ~= 51%
  DRAM throughput ~= 57%
  混合 compute / memory bottleneck

mul_mat_q:
  ~= 40us / 66us / 88us / 96us 多个 band

mul_mat_q_stream_k_fixup:
  ~= 17-21us
```

一个 decode step 里有 28 层，每层都有 attention、Q/K/V/O projection、MLP
gate/up/down、norm、rope、copy/fixup 等 kernel。所以当前瓶颈不是单独一个
paged attention，而是一组 decoder compute kernels。

## 最新实验：MMQ Stream-K 为什么没有继续

看到 `mul_mat_q_stream_k_fixup` 后，尝试过关闭 MMQ stream-k。

全局关闭的结果：

```text
baseline:
wall=56.45s
prefill=29.38s
decode_dispatch=27.03s
paged_total=16.15ms
tokens/s=339.8

disable_all:
wall=70.25s
prefill=45.57s
decode_dispatch=24.63s
paged_total=14.67ms
tokens/s=273.3
```

它证明了一件很重要的事：

```text
关掉 stream-k 后，decode 局部变快了；
但 prefill 大幅变慢，所以完整长视频 pipeline 变慢。
```

后面又试了 `FUNASR_MMQ_STREAM_K_MIN_NCOLS` 阈值策略，希望只影响小 decode
shape，保留 prefill 的收益。但结果仍然比 baseline 慢：

```text
baseline       wall=56.45s  prefill=29.38s  decode=27.03s  tokens/s=339.8
min_ncols_8    wall=60.39s  prefill=31.88s  decode=28.46s  tokens/s=317.7
min_ncols_16   wall=60.78s  prefill=32.14s  decode=28.59s  tokens/s=315.6
min_ncols_32   wall=62.95s  prefill=33.50s  decode=29.40s  tokens/s=304.7
```

所以这个方向暂时收掉。这个负结果很有价值，因为它说明：

```text
stream-k 不是绝对好或坏；
它对 prefill 和 decode 的影响不同；
简单用 ncols 阈值不能干净分离 workload；
不能因为看到 fixup kernel 就盲目关掉。
```

这就是当前问题的本质：现在已经不是 scheduler 级别的问题，而是 kernel 级别
的 workload-shape 问题。

## 当前还存在的问题

现在主要问题有四类。

第一，decode kernel 组成还没有完整分类统计。我们知道 paged attention 和 Q8
matmul 都重要，但还需要按类别统计每个 decode step 的总时间：

```text
paged attention
MMQ matmul
MMQ / flash attention fixup
KV write
quantize / copy / cast
RMSNorm
RoPE
other small kernels
```

第二，paged attention 还有优化空间。当前 NCU 显示它不是纯 memory-bound，也
不是纯 compute-bound，而是 QK、softmax、V aggregation、KV 读取共同作用的
混合瓶颈。

第三，Q8 matmul 的 decode-batch 小 shape 可能还有专门路径空间。但这必须避免
伤害 prefill，不能再用全局策略。

第四，decode graph 里小 kernel 数量仍然很多。即使 graph cache 命中，实际
compute 里仍然存在大量小 kernel launch 和中间 copy/cast/fixup。

## 下一步怎么做

下一步不要继续猜某个 kernel 名字，而是先做 decode-step kernel category
summary。

目标是把一次 decode step 的 kernel 时间按类别汇总：

```text
paged_attn_decode
paged_kv_write
mul_mat_q
mul_mat_q_stream_k_fixup
flash_attn_stream_k_fixup
quantize_mmq_q8_1
cpy / cast
rms_norm
rope
other
```

然后用这个表决定下一步：

```text
如果 paged attention 最大：
  继续优化 KV 读取、QK、softmax、V 聚合

如果 MMQ matmul 最大：
  看 decode batch 下 Q8 matmul 是否需要专门路径

如果 copy/cast/fixup 太多：
  优先做 kernel fusion 或减少中间张量

如果 launch 数太多：
  继续做 graph 合并和小 kernel 消除
```

当前推荐优先级：

1. 保持默认 MMQ stream-k，不启用阈值策略。
2. 做 decode kernel 分类统计。
3. 继续优化 paged attention。
4. 再回头研究 decode-batch Q8 matmul 专用路径。
5. 减少 decode graph 小 kernel 数量。
6. 保持 bucket + graph cache，因为它们已经证明有效。

## 面试或项目介绍怎么讲

可以按这个脉络讲：

```text
我做的是长视频离线 ASR 推理优化。这个场景不是单个音频一次跑完，而是把长视频
切成很多 30 秒 chunk，每个 chunk 都会进入 LLM decoder 的逐 token decode。

单路 decode GPU 利用率低，所以我借鉴 vLLM 的 continuous batching 和 paged
KV cache，把多个 chunk 的 decode step 合并执行。这里我实现和优化了 scheduler、
block table、paged KV、batched token-id decode、dynamic KV write 和 graph
cache。

一开始最大的问题是 batch fast path 不稳定，paged decode 经常 fallback。所以
我先加了 fallback reasons、scheduler profile、paged profile、graph cache
probe，把问题量化。之后清掉 fallback，找到 batch=12、block_size=128 是当前
4070 Laptop 8GB 的甜点。

接着优化 paged attention kernel，又发现 graph 由于 per-row KV copy 不稳定，
无法复用。于是我把 KV 写入改成 dynamic paged KV write CUDA op，让 graph 不
再依赖具体 physical row。这一步把 wall 从 61s 降到 53s 左右。随后加 bucketed
max_n_kv 和 graph cache，把 build/alloc 继续压低，最终到 52s 左右。

最近的 profiling 说明 scheduler 已经健康，fallback 只有 2 次，avg_active 约
11.5，graph cache hit 约 80%。当前瓶颈已经转到 decoder compute kernels。
我用 NCU 看到 paged attention、Q8 matmul、stream-k fixup 都有明显占比。我们
还试过关闭 MMQ stream-k，发现 decode 局部会变快，但 prefill 大幅变慢，所以
没有盲目合入。下一步是按 kernel category 统计 decode step，再决定是优化
paged attention、Q8 matmul，还是减少小 kernel。
```

这个项目最有价值的点不是某个单独优化，而是完整方法论：

```text
profile first
clean fallback
find workload sweet spot
stabilize KV/graph shape
isolate each optimization
keep only full-pipeline wins
move from scheduler optimization to kernel optimization
```

这条路线也和真实推理系统优化很接近：前期靠调度和内存组织拿大收益，后期进入
kernel 和硬件效率阶段，优化难度变高，但技术含金量也更高。

## 2026-07-17：动态 Paged KV 分配与任务级 Prefix KV 实验

这一轮重新审视了两个不合理点：

```text
admit 时用 seconds * 25 + 128 + max_tokens 预留全部 KV block
每个 chunk 都重新计算完全相同的 ChatML prefix
```

原实现对 30 秒 chunk 通常固定分配 9 个 128-token block，但实际 prefill 是 522
token，decode 通常结束在 80-120 token 左右。完整 204-chunk 测量显示：

```text
alloc_blocks=1836
used_blocks=1024
wasted_blocks=812
waste_rate=44.23%
blocks_peak=108/384
```

### 动态 block 实现

新的动态路径改为：

```text
frontend 先得到真实 audio_frames
按 prefix + audio_frames + suffix 精确分配 prefill blocks
decode 写入 n_past 前检查是否跨 block 边界
只有跨边界时 append 一个新 block
request 完成后逐块归还
```

`PagedKVBlockPool` 同时增加了 refcount、重复释放检测、原子 append 和 COW
ownership transfer。完整 204-chunk 结果：

```text
ok=204/204
wall=57.679s
blocks_peak=61/384
alloc_blocks=1024
used_blocks=1024
wasted_blocks=0
waste_rate=0.00%
prefill_appends=1019
decode_appends=5
ownership_errors=0
final_free=384/384
```

因此，这一轮确定性的收益是 KV 逻辑块峰值从 108 降到 61，消除了 44.23% 的
整块过量预留。wall 落在既有约 55-59 秒波动区间，当前只能判断性能基本持平，
不能宣称完整 workload 加速。

这也修正了短 benchmark 的统计问题：使用 `--max-chunks` 后，RTF 和
`audio_sec/s` 必须按选中的 chunk 音频时长计算，不能继续使用完整源文件时长。

### Prefix KV cache 与 partial-block COW

任务级 prefix cache 实现了以下完整生命周期：

```text
固定 ChatML prefix 只做一次 paged prefill
cache handle 持有一个 block reference
request attach 时 retain
prefix 最后一个 block 未对齐时先做 GPU D2D COW
tail 仅 prefill [audio embeddings + suffix]
request 与 task cache 分别 release
```

当前默认 prompt 的 prefix 只有 18 token，因此它全部落在一个 partial block 中。
每个 request 在追加 audio KV 前都必须复制 28 层的 K/V 有效行。功能验证结果是：

```text
prefix_builds=1
prefix_hits=24
cow_copies=24
ownership_errors=0
final_free=384/384
```

但 24-chunk 四象限实验表明，prefix cache 当前不适合作为默认性能优化：

```text
variant                 wall_ms  peak_blocks  waste_rate
static_no_cache             7842      108/384      44.44%
dynamic_no_cache            6717       60/384       0.00%
static_prefix_cache         6959      109/384      44.44%
dynamic_prefix_cache        6995       61/384       0.00%
```

cache-on 与 cache-off 在 24 个 chunk 中有 6 个出现单字或标点级差异，例如
“倍后/背后”“他/它”。这是 full prefill 与 split prefill 使用不同 GPU shape 后的
浮点数值路径差异，不是 block ownership 或 COW 数据损坏：cache-on 两组输出一致，
cache-off 两组输出一致，长度也保持一致。

综合判断：

```text
dynamic KV blocks: long-video preset 默认开启
prefix KV cache:   保留 --prefix-kv-cache on 实验开关，默认关闭
```

这个负结果同样重要。prefix caching 不是看到固定 prompt 就一定有收益；收益取决于
prefix 长度、可共享的完整 block 数量、COW 成本和 split-prefill 的数值稳定性。当前
18-token prefix 太短，COW 的逐层同步成本抵消了省下的计算。后续若要继续，应先把
多层 K/V copy 合成单个 CUDA graph/kernel，或者在更长 hotword/system prompt 上
重新评估。

四象限测试脚本：

```bash
tools/bench_prefix_kv_dynamic_blocks.sh --max-chunks 24 --repeat 1
```
