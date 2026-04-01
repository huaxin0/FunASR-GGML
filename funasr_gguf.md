# FunAsr.bin - GGUF Internal File Dump

- Endian: BIG endian

## Key Value Metadata Store

There are 38 key-value pairs in this file

| POS | TYPE     |  Count | Key                             | Value                                                               |
|----:|:---------|-------:|:--------------------------------|:--------------------------------------------------------------------|
|   1 | UINT32   |      1 | GGUF.version                    | 3                                                                   |
|   2 | UINT64   |      1 | GGUF.tensor_count               | 1541                                                                |
|   3 | UINT64   |      1 | GGUF.kv_count                   | 35                                                                  |
|   4 | STRING   |      1 | general.architecture            | `FunAsr`                                                            |
|   5 | INT32    |      1 | frontend.sample_rate            | 16000                                                               |
|   6 | STRING   |      1 | frontend.window                 | `hamming`                                                           |
|   7 | INT32    |      1 | frontend.num_mels               | 80                                                                  |
|   8 | INT32    |      1 | frontend.frame_length           | 25                                                                  |
|   9 | INT32    |      1 | frontend.frame_shift            | 10                                                                  |
|  10 | INT32    |      1 | frontend.lfr_m                  | 7                                                                   |
|  11 | INT32    |      1 | frontend.lfr_n                  | 6                                                                   |
|  12 | INT32    |      1 | encoder.output_size             | 512                                                                 |
|  13 | INT32    |      1 | encoder.attention_heads         | 4                                                                   |
|  14 | INT32    |      1 | encoder.linear_units            | 2048                                                                |
|  15 | INT32    |      1 | encoder.num_blocks              | 50                                                                  |
|  16 | INT32    |      1 | encoder.tp_blocks               | 20                                                                  |
|  17 | INT32    |      1 | encoder.kernel_size             | 11                                                                  |
|  18 | INT32    |      1 | encoder.sanm_shift              | 0                                                                   |
|  19 | UINT32   |      1 | FunAsr.block_count              | 28                                                                  |
|  20 | UINT32   |      1 | FunAsr.embedding_length         | 1024                                                                |
|  21 | UINT32   |      1 | FunAsr.context_length           | 40960                                                               |
|  22 | UINT32   |      1 | FunAsr.feed_forward_length      | 3072                                                                |
|  23 | UINT32   |      1 | FunAsr.attention.head_count     | 16                                                                  |
|  24 | UINT32   |      1 | FunAsr.attention.head_count_kv  | 8                                                                   |
|  25 | FLOAT32  |      1 | FunAsr.rope.freq_base           | 1000000.0                                                           |
|  26 | UINT32   |      1 | FunAsr.attention.key_length     | 128                                                                 |
|  27 | UINT32   |      1 | FunAsr.attention.value_length   | 128                                                                 |
|  28 | UINT32   |      1 | general.file_type               | 1                                                                   |
|  29 | STRING   |      1 | tokenizer.ggml.model            | `gpt2`                                                              |
|  30 | STRING   |      1 | tokenizer.ggml.pre              | `qwen2`                                                             |
|  31 | [STRING] | 151936 | tokenizer.ggml.tokens           | [ `!`, `"`, `#`, `$`, `%`, ... ]                                    |
|  32 | [INT32]  | 151936 | tokenizer.ggml.token_type       | [ 1, 1, 1, 1, 1, 1, 1, ... ]                                        |
|  33 | [STRING] | 151387 | tokenizer.ggml.merges           | [ `Ġ Ġ`, `ĠĠ ĠĠ`, `i n`, `Ġ t`, `ĠĠĠĠ ĠĠĠĠ`, ... ]                  |
|  34 | UINT32   |      1 | tokenizer.ggml.eos_token_id     | 151645                                                              |
|  35 | UINT32   |      1 | tokenizer.ggml.padding_token_id | 151643                                                              |
|  36 | UINT32   |      1 | tokenizer.ggml.bos_token_id     | 151643                                                              |
|  37 | BOOL     |      1 | tokenizer.ggml.add_bos_token    | False                                                               |
|  38 | STRING   |      1 | tokenizer.chat_template         | `{%- if tools %}
    {{- '<|im_`...`
    {%- endif %}
{%- endif %}` |

## Tensors Overview ~985M Elements

Total number of elements in all tensors: 985374304 Elements

- [Base Tensor Group - ~985M Elements](#base)

### Tensor Data Offset

This table contains the offset and data segment relative to start of file

| T_ID | Tensor Layer Name                                        |  Data Offset (B) |    Data Size (B) |
|-----:|:---------------------------------------------------------|-----------------:|-----------------:|
|    0 | audio_encoder.encoders0.0.self_attn.linear_out.weight    |         0x5c7700 |          0x80000 |
|    1 | audio_encoder.encoders0.0.self_attn.linear_out.bias      |         0x647700 |            0x400 |
|    2 | audio_encoder.encoders0.0.self_attn.linear_q.weight      |         0x647b00 |          0x8c000 |
|    3 | audio_encoder.encoders0.0.self_attn.linear_k.weight      |         0x6d3b00 |          0x8c000 |
|    4 | audio_encoder.encoders0.0.self_attn.linear_v.weight      |         0x75fb00 |          0x8c000 |
|    5 | audio_encoder.encoders0.0.self_attn.linear_q.bias        |         0x7ebb00 |            0x400 |
|    6 | audio_encoder.encoders0.0.self_attn.linear_k.bias        |         0x7ebf00 |            0x400 |
|    7 | audio_encoder.encoders0.0.self_attn.linear_v.bias        |         0x7ec300 |            0x400 |
|    8 | audio_encoder.encoders0.0.self_attn.fsmn_block.weight    |         0x7ec700 |           0x2c00 |
|    9 | audio_encoder.encoders0.0.feed_forward.w_1.weight        |         0x7ef300 |         0x200000 |
|   10 | audio_encoder.encoders0.0.feed_forward.w_1.bias          |         0x9ef300 |           0x1000 |
|   11 | audio_encoder.encoders0.0.feed_forward.w_2.weight        |         0x9f0300 |         0x200000 |
|   12 | audio_encoder.encoders0.0.feed_forward.w_2.bias          |         0xbf0300 |            0x400 |
|   13 | audio_encoder.encoders0.0.norm1.weight                   |         0xbf0700 |            0x460 |
|   14 | audio_encoder.encoders0.0.norm1.bias                     |         0xbf0b60 |            0x460 |
|   15 | audio_encoder.encoders0.0.norm2.weight                   |         0xbf0fc0 |            0x400 |
|   16 | audio_encoder.encoders0.0.norm2.bias                     |         0xbf13c0 |            0x400 |
|   17 | audio_encoder.encoders.0.self_attn.linear_out.weight     |         0xbf17c0 |          0x80000 |
|   18 | audio_encoder.encoders.0.self_attn.linear_out.bias       |         0xc717c0 |            0x400 |
|   19 | audio_encoder.encoders.0.self_attn.linear_q.weight       |         0xc71bc0 |          0x80000 |
|   20 | audio_encoder.encoders.0.self_attn.linear_k.weight       |         0xcf1bc0 |          0x80000 |
|   21 | audio_encoder.encoders.0.self_attn.linear_v.weight       |         0xd71bc0 |          0x80000 |
|   22 | audio_encoder.encoders.0.self_attn.linear_q.bias         |         0xdf1bc0 |            0x400 |
|   23 | audio_encoder.encoders.0.self_attn.linear_k.bias         |         0xdf1fc0 |            0x400 |
|   24 | audio_encoder.encoders.0.self_attn.linear_v.bias         |         0xdf23c0 |            0x400 |
|   25 | audio_encoder.encoders.0.self_attn.fsmn_block.weight     |         0xdf27c0 |           0x2c00 |
|   26 | audio_encoder.encoders.0.feed_forward.w_1.weight         |         0xdf53c0 |         0x200000 |
|   27 | audio_encoder.encoders.0.feed_forward.w_1.bias           |         0xff53c0 |           0x1000 |
|   28 | audio_encoder.encoders.0.feed_forward.w_2.weight         |         0xff63c0 |         0x200000 |
|   29 | audio_encoder.encoders.0.feed_forward.w_2.bias           |        0x11f63c0 |            0x400 |
|   30 | audio_encoder.encoders.0.norm1.weight                    |        0x11f67c0 |            0x400 |
|   31 | audio_encoder.encoders.0.norm1.bias                      |        0x11f6bc0 |            0x400 |
|   32 | audio_encoder.encoders.0.norm2.weight                    |        0x11f6fc0 |            0x400 |
|   33 | audio_encoder.encoders.0.norm2.bias                      |        0x11f73c0 |            0x400 |
|   34 | audio_encoder.encoders.1.self_attn.linear_out.weight     |        0x11f77c0 |          0x80000 |
|   35 | audio_encoder.encoders.1.self_attn.linear_out.bias       |        0x12777c0 |            0x400 |
|   36 | audio_encoder.encoders.1.self_attn.linear_q.weight       |        0x1277bc0 |          0x80000 |
|   37 | audio_encoder.encoders.1.self_attn.linear_k.weight       |        0x12f7bc0 |          0x80000 |
|   38 | audio_encoder.encoders.1.self_attn.linear_v.weight       |        0x1377bc0 |          0x80000 |
|   39 | audio_encoder.encoders.1.self_attn.linear_q.bias         |        0x13f7bc0 |            0x400 |
|   40 | audio_encoder.encoders.1.self_attn.linear_k.bias         |        0x13f7fc0 |            0x400 |
|   41 | audio_encoder.encoders.1.self_attn.linear_v.bias         |        0x13f83c0 |            0x400 |
|   42 | audio_encoder.encoders.1.self_attn.fsmn_block.weight     |        0x13f87c0 |           0x2c00 |
|   43 | audio_encoder.encoders.1.feed_forward.w_1.weight         |        0x13fb3c0 |         0x200000 |
|   44 | audio_encoder.encoders.1.feed_forward.w_1.bias           |        0x15fb3c0 |           0x1000 |
|   45 | audio_encoder.encoders.1.feed_forward.w_2.weight         |        0x15fc3c0 |         0x200000 |
|   46 | audio_encoder.encoders.1.feed_forward.w_2.bias           |        0x17fc3c0 |            0x400 |
|   47 | audio_encoder.encoders.1.norm1.weight                    |        0x17fc7c0 |            0x400 |
|   48 | audio_encoder.encoders.1.norm1.bias                      |        0x17fcbc0 |            0x400 |
|   49 | audio_encoder.encoders.1.norm2.weight                    |        0x17fcfc0 |            0x400 |
|   50 | audio_encoder.encoders.1.norm2.bias                      |        0x17fd3c0 |            0x400 |
|   51 | audio_encoder.encoders.2.self_attn.linear_out.weight     |        0x17fd7c0 |          0x80000 |
|   52 | audio_encoder.encoders.2.self_attn.linear_out.bias       |        0x187d7c0 |            0x400 |
|   53 | audio_encoder.encoders.2.self_attn.linear_q.weight       |        0x187dbc0 |          0x80000 |
|   54 | audio_encoder.encoders.2.self_attn.linear_k.weight       |        0x18fdbc0 |          0x80000 |
|   55 | audio_encoder.encoders.2.self_attn.linear_v.weight       |        0x197dbc0 |          0x80000 |
|   56 | audio_encoder.encoders.2.self_attn.linear_q.bias         |        0x19fdbc0 |            0x400 |
|   57 | audio_encoder.encoders.2.self_attn.linear_k.bias         |        0x19fdfc0 |            0x400 |
|   58 | audio_encoder.encoders.2.self_attn.linear_v.bias         |        0x19fe3c0 |            0x400 |
|   59 | audio_encoder.encoders.2.self_attn.fsmn_block.weight     |        0x19fe7c0 |           0x2c00 |
|   60 | audio_encoder.encoders.2.feed_forward.w_1.weight         |        0x1a013c0 |         0x200000 |
|   61 | audio_encoder.encoders.2.feed_forward.w_1.bias           |        0x1c013c0 |           0x1000 |
|   62 | audio_encoder.encoders.2.feed_forward.w_2.weight         |        0x1c023c0 |         0x200000 |
|   63 | audio_encoder.encoders.2.feed_forward.w_2.bias           |        0x1e023c0 |            0x400 |
|   64 | audio_encoder.encoders.2.norm1.weight                    |        0x1e027c0 |            0x400 |
|   65 | audio_encoder.encoders.2.norm1.bias                      |        0x1e02bc0 |            0x400 |
|   66 | audio_encoder.encoders.2.norm2.weight                    |        0x1e02fc0 |            0x400 |
|   67 | audio_encoder.encoders.2.norm2.bias                      |        0x1e033c0 |            0x400 |
|   68 | audio_encoder.encoders.3.self_attn.linear_out.weight     |        0x1e037c0 |          0x80000 |
|   69 | audio_encoder.encoders.3.self_attn.linear_out.bias       |        0x1e837c0 |            0x400 |
|   70 | audio_encoder.encoders.3.self_attn.linear_q.weight       |        0x1e83bc0 |          0x80000 |
|   71 | audio_encoder.encoders.3.self_attn.linear_k.weight       |        0x1f03bc0 |          0x80000 |
|   72 | audio_encoder.encoders.3.self_attn.linear_v.weight       |        0x1f83bc0 |          0x80000 |
|   73 | audio_encoder.encoders.3.self_attn.linear_q.bias         |        0x2003bc0 |            0x400 |
|   74 | audio_encoder.encoders.3.self_attn.linear_k.bias         |        0x2003fc0 |            0x400 |
|   75 | audio_encoder.encoders.3.self_attn.linear_v.bias         |        0x20043c0 |            0x400 |
|   76 | audio_encoder.encoders.3.self_attn.fsmn_block.weight     |        0x20047c0 |           0x2c00 |
|   77 | audio_encoder.encoders.3.feed_forward.w_1.weight         |        0x20073c0 |         0x200000 |
|   78 | audio_encoder.encoders.3.feed_forward.w_1.bias           |        0x22073c0 |           0x1000 |
|   79 | audio_encoder.encoders.3.feed_forward.w_2.weight         |        0x22083c0 |         0x200000 |
|   80 | audio_encoder.encoders.3.feed_forward.w_2.bias           |        0x24083c0 |            0x400 |
|   81 | audio_encoder.encoders.3.norm1.weight                    |        0x24087c0 |            0x400 |
|   82 | audio_encoder.encoders.3.norm1.bias                      |        0x2408bc0 |            0x400 |
|   83 | audio_encoder.encoders.3.norm2.weight                    |        0x2408fc0 |            0x400 |
|   84 | audio_encoder.encoders.3.norm2.bias                      |        0x24093c0 |            0x400 |
|   85 | audio_encoder.encoders.4.self_attn.linear_out.weight     |        0x24097c0 |          0x80000 |
|   86 | audio_encoder.encoders.4.self_attn.linear_out.bias       |        0x24897c0 |            0x400 |
|   87 | audio_encoder.encoders.4.self_attn.linear_q.weight       |        0x2489bc0 |          0x80000 |
|   88 | audio_encoder.encoders.4.self_attn.linear_k.weight       |        0x2509bc0 |          0x80000 |
|   89 | audio_encoder.encoders.4.self_attn.linear_v.weight       |        0x2589bc0 |          0x80000 |
|   90 | audio_encoder.encoders.4.self_attn.linear_q.bias         |        0x2609bc0 |            0x400 |
|   91 | audio_encoder.encoders.4.self_attn.linear_k.bias         |        0x2609fc0 |            0x400 |
|   92 | audio_encoder.encoders.4.self_attn.linear_v.bias         |        0x260a3c0 |            0x400 |
|   93 | audio_encoder.encoders.4.self_attn.fsmn_block.weight     |        0x260a7c0 |           0x2c00 |
|   94 | audio_encoder.encoders.4.feed_forward.w_1.weight         |        0x260d3c0 |         0x200000 |
|   95 | audio_encoder.encoders.4.feed_forward.w_1.bias           |        0x280d3c0 |           0x1000 |
|   96 | audio_encoder.encoders.4.feed_forward.w_2.weight         |        0x280e3c0 |         0x200000 |
|   97 | audio_encoder.encoders.4.feed_forward.w_2.bias           |        0x2a0e3c0 |            0x400 |
|   98 | audio_encoder.encoders.4.norm1.weight                    |        0x2a0e7c0 |            0x400 |
|   99 | audio_encoder.encoders.4.norm1.bias                      |        0x2a0ebc0 |            0x400 |
|  100 | audio_encoder.encoders.4.norm2.weight                    |        0x2a0efc0 |            0x400 |
|  101 | audio_encoder.encoders.4.norm2.bias                      |        0x2a0f3c0 |            0x400 |
|  102 | audio_encoder.encoders.5.self_attn.linear_out.weight     |        0x2a0f7c0 |          0x80000 |
|  103 | audio_encoder.encoders.5.self_attn.linear_out.bias       |        0x2a8f7c0 |            0x400 |
|  104 | audio_encoder.encoders.5.self_attn.linear_q.weight       |        0x2a8fbc0 |          0x80000 |
|  105 | audio_encoder.encoders.5.self_attn.linear_k.weight       |        0x2b0fbc0 |          0x80000 |
|  106 | audio_encoder.encoders.5.self_attn.linear_v.weight       |        0x2b8fbc0 |          0x80000 |
|  107 | audio_encoder.encoders.5.self_attn.linear_q.bias         |        0x2c0fbc0 |            0x400 |
|  108 | audio_encoder.encoders.5.self_attn.linear_k.bias         |        0x2c0ffc0 |            0x400 |
|  109 | audio_encoder.encoders.5.self_attn.linear_v.bias         |        0x2c103c0 |            0x400 |
|  110 | audio_encoder.encoders.5.self_attn.fsmn_block.weight     |        0x2c107c0 |           0x2c00 |
|  111 | audio_encoder.encoders.5.feed_forward.w_1.weight         |        0x2c133c0 |         0x200000 |
|  112 | audio_encoder.encoders.5.feed_forward.w_1.bias           |        0x2e133c0 |           0x1000 |
|  113 | audio_encoder.encoders.5.feed_forward.w_2.weight         |        0x2e143c0 |         0x200000 |
|  114 | audio_encoder.encoders.5.feed_forward.w_2.bias           |        0x30143c0 |            0x400 |
|  115 | audio_encoder.encoders.5.norm1.weight                    |        0x30147c0 |            0x400 |
|  116 | audio_encoder.encoders.5.norm1.bias                      |        0x3014bc0 |            0x400 |
|  117 | audio_encoder.encoders.5.norm2.weight                    |        0x3014fc0 |            0x400 |
|  118 | audio_encoder.encoders.5.norm2.bias                      |        0x30153c0 |            0x400 |
|  119 | audio_encoder.encoders.6.self_attn.linear_out.weight     |        0x30157c0 |          0x80000 |
|  120 | audio_encoder.encoders.6.self_attn.linear_out.bias       |        0x30957c0 |            0x400 |
|  121 | audio_encoder.encoders.6.self_attn.linear_q.weight       |        0x3095bc0 |          0x80000 |
|  122 | audio_encoder.encoders.6.self_attn.linear_k.weight       |        0x3115bc0 |          0x80000 |
|  123 | audio_encoder.encoders.6.self_attn.linear_v.weight       |        0x3195bc0 |          0x80000 |
|  124 | audio_encoder.encoders.6.self_attn.linear_q.bias         |        0x3215bc0 |            0x400 |
|  125 | audio_encoder.encoders.6.self_attn.linear_k.bias         |        0x3215fc0 |            0x400 |
|  126 | audio_encoder.encoders.6.self_attn.linear_v.bias         |        0x32163c0 |            0x400 |
|  127 | audio_encoder.encoders.6.self_attn.fsmn_block.weight     |        0x32167c0 |           0x2c00 |
|  128 | audio_encoder.encoders.6.feed_forward.w_1.weight         |        0x32193c0 |         0x200000 |
|  129 | audio_encoder.encoders.6.feed_forward.w_1.bias           |        0x34193c0 |           0x1000 |
|  130 | audio_encoder.encoders.6.feed_forward.w_2.weight         |        0x341a3c0 |         0x200000 |
|  131 | audio_encoder.encoders.6.feed_forward.w_2.bias           |        0x361a3c0 |            0x400 |
|  132 | audio_encoder.encoders.6.norm1.weight                    |        0x361a7c0 |            0x400 |
|  133 | audio_encoder.encoders.6.norm1.bias                      |        0x361abc0 |            0x400 |
|  134 | audio_encoder.encoders.6.norm2.weight                    |        0x361afc0 |            0x400 |
|  135 | audio_encoder.encoders.6.norm2.bias                      |        0x361b3c0 |            0x400 |
|  136 | audio_encoder.encoders.7.self_attn.linear_out.weight     |        0x361b7c0 |          0x80000 |
|  137 | audio_encoder.encoders.7.self_attn.linear_out.bias       |        0x369b7c0 |            0x400 |
|  138 | audio_encoder.encoders.7.self_attn.linear_q.weight       |        0x369bbc0 |          0x80000 |
|  139 | audio_encoder.encoders.7.self_attn.linear_k.weight       |        0x371bbc0 |          0x80000 |
|  140 | audio_encoder.encoders.7.self_attn.linear_v.weight       |        0x379bbc0 |          0x80000 |
|  141 | audio_encoder.encoders.7.self_attn.linear_q.bias         |        0x381bbc0 |            0x400 |
|  142 | audio_encoder.encoders.7.self_attn.linear_k.bias         |        0x381bfc0 |            0x400 |
|  143 | audio_encoder.encoders.7.self_attn.linear_v.bias         |        0x381c3c0 |            0x400 |
|  144 | audio_encoder.encoders.7.self_attn.fsmn_block.weight     |        0x381c7c0 |           0x2c00 |
|  145 | audio_encoder.encoders.7.feed_forward.w_1.weight         |        0x381f3c0 |         0x200000 |
|  146 | audio_encoder.encoders.7.feed_forward.w_1.bias           |        0x3a1f3c0 |           0x1000 |
|  147 | audio_encoder.encoders.7.feed_forward.w_2.weight         |        0x3a203c0 |         0x200000 |
|  148 | audio_encoder.encoders.7.feed_forward.w_2.bias           |        0x3c203c0 |            0x400 |
|  149 | audio_encoder.encoders.7.norm1.weight                    |        0x3c207c0 |            0x400 |
|  150 | audio_encoder.encoders.7.norm1.bias                      |        0x3c20bc0 |            0x400 |
|  151 | audio_encoder.encoders.7.norm2.weight                    |        0x3c20fc0 |            0x400 |
|  152 | audio_encoder.encoders.7.norm2.bias                      |        0x3c213c0 |            0x400 |
|  153 | audio_encoder.encoders.8.self_attn.linear_out.weight     |        0x3c217c0 |          0x80000 |
|  154 | audio_encoder.encoders.8.self_attn.linear_out.bias       |        0x3ca17c0 |            0x400 |
|  155 | audio_encoder.encoders.8.self_attn.linear_q.weight       |        0x3ca1bc0 |          0x80000 |
|  156 | audio_encoder.encoders.8.self_attn.linear_k.weight       |        0x3d21bc0 |          0x80000 |
|  157 | audio_encoder.encoders.8.self_attn.linear_v.weight       |        0x3da1bc0 |          0x80000 |
|  158 | audio_encoder.encoders.8.self_attn.linear_q.bias         |        0x3e21bc0 |            0x400 |
|  159 | audio_encoder.encoders.8.self_attn.linear_k.bias         |        0x3e21fc0 |            0x400 |
|  160 | audio_encoder.encoders.8.self_attn.linear_v.bias         |        0x3e223c0 |            0x400 |
|  161 | audio_encoder.encoders.8.self_attn.fsmn_block.weight     |        0x3e227c0 |           0x2c00 |
|  162 | audio_encoder.encoders.8.feed_forward.w_1.weight         |        0x3e253c0 |         0x200000 |
|  163 | audio_encoder.encoders.8.feed_forward.w_1.bias           |        0x40253c0 |           0x1000 |
|  164 | audio_encoder.encoders.8.feed_forward.w_2.weight         |        0x40263c0 |         0x200000 |
|  165 | audio_encoder.encoders.8.feed_forward.w_2.bias           |        0x42263c0 |            0x400 |
|  166 | audio_encoder.encoders.8.norm1.weight                    |        0x42267c0 |            0x400 |
|  167 | audio_encoder.encoders.8.norm1.bias                      |        0x4226bc0 |            0x400 |
|  168 | audio_encoder.encoders.8.norm2.weight                    |        0x4226fc0 |            0x400 |
|  169 | audio_encoder.encoders.8.norm2.bias                      |        0x42273c0 |            0x400 |
|  170 | audio_encoder.encoders.9.self_attn.linear_out.weight     |        0x42277c0 |          0x80000 |
|  171 | audio_encoder.encoders.9.self_attn.linear_out.bias       |        0x42a77c0 |            0x400 |
|  172 | audio_encoder.encoders.9.self_attn.linear_q.weight       |        0x42a7bc0 |          0x80000 |
|  173 | audio_encoder.encoders.9.self_attn.linear_k.weight       |        0x4327bc0 |          0x80000 |
|  174 | audio_encoder.encoders.9.self_attn.linear_v.weight       |        0x43a7bc0 |          0x80000 |
|  175 | audio_encoder.encoders.9.self_attn.linear_q.bias         |        0x4427bc0 |            0x400 |
|  176 | audio_encoder.encoders.9.self_attn.linear_k.bias         |        0x4427fc0 |            0x400 |
|  177 | audio_encoder.encoders.9.self_attn.linear_v.bias         |        0x44283c0 |            0x400 |
|  178 | audio_encoder.encoders.9.self_attn.fsmn_block.weight     |        0x44287c0 |           0x2c00 |
|  179 | audio_encoder.encoders.9.feed_forward.w_1.weight         |        0x442b3c0 |         0x200000 |
|  180 | audio_encoder.encoders.9.feed_forward.w_1.bias           |        0x462b3c0 |           0x1000 |
|  181 | audio_encoder.encoders.9.feed_forward.w_2.weight         |        0x462c3c0 |         0x200000 |
|  182 | audio_encoder.encoders.9.feed_forward.w_2.bias           |        0x482c3c0 |            0x400 |
|  183 | audio_encoder.encoders.9.norm1.weight                    |        0x482c7c0 |            0x400 |
|  184 | audio_encoder.encoders.9.norm1.bias                      |        0x482cbc0 |            0x400 |
|  185 | audio_encoder.encoders.9.norm2.weight                    |        0x482cfc0 |            0x400 |
|  186 | audio_encoder.encoders.9.norm2.bias                      |        0x482d3c0 |            0x400 |
|  187 | audio_encoder.encoders.10.self_attn.linear_out.weight    |        0x482d7c0 |          0x80000 |
|  188 | audio_encoder.encoders.10.self_attn.linear_out.bias      |        0x48ad7c0 |            0x400 |
|  189 | audio_encoder.encoders.10.self_attn.linear_q.weight      |        0x48adbc0 |          0x80000 |
|  190 | audio_encoder.encoders.10.self_attn.linear_k.weight      |        0x492dbc0 |          0x80000 |
|  191 | audio_encoder.encoders.10.self_attn.linear_v.weight      |        0x49adbc0 |          0x80000 |
|  192 | audio_encoder.encoders.10.self_attn.linear_q.bias        |        0x4a2dbc0 |            0x400 |
|  193 | audio_encoder.encoders.10.self_attn.linear_k.bias        |        0x4a2dfc0 |            0x400 |
|  194 | audio_encoder.encoders.10.self_attn.linear_v.bias        |        0x4a2e3c0 |            0x400 |
|  195 | audio_encoder.encoders.10.self_attn.fsmn_block.weight    |        0x4a2e7c0 |           0x2c00 |
|  196 | audio_encoder.encoders.10.feed_forward.w_1.weight        |        0x4a313c0 |         0x200000 |
|  197 | audio_encoder.encoders.10.feed_forward.w_1.bias          |        0x4c313c0 |           0x1000 |
|  198 | audio_encoder.encoders.10.feed_forward.w_2.weight        |        0x4c323c0 |         0x200000 |
|  199 | audio_encoder.encoders.10.feed_forward.w_2.bias          |        0x4e323c0 |            0x400 |
|  200 | audio_encoder.encoders.10.norm1.weight                   |        0x4e327c0 |            0x400 |
|  201 | audio_encoder.encoders.10.norm1.bias                     |        0x4e32bc0 |            0x400 |
|  202 | audio_encoder.encoders.10.norm2.weight                   |        0x4e32fc0 |            0x400 |
|  203 | audio_encoder.encoders.10.norm2.bias                     |        0x4e333c0 |            0x400 |
|  204 | audio_encoder.encoders.11.self_attn.linear_out.weight    |        0x4e337c0 |          0x80000 |
|  205 | audio_encoder.encoders.11.self_attn.linear_out.bias      |        0x4eb37c0 |            0x400 |
|  206 | audio_encoder.encoders.11.self_attn.linear_q.weight      |        0x4eb3bc0 |          0x80000 |
|  207 | audio_encoder.encoders.11.self_attn.linear_k.weight      |        0x4f33bc0 |          0x80000 |
|  208 | audio_encoder.encoders.11.self_attn.linear_v.weight      |        0x4fb3bc0 |          0x80000 |
|  209 | audio_encoder.encoders.11.self_attn.linear_q.bias        |        0x5033bc0 |            0x400 |
|  210 | audio_encoder.encoders.11.self_attn.linear_k.bias        |        0x5033fc0 |            0x400 |
|  211 | audio_encoder.encoders.11.self_attn.linear_v.bias        |        0x50343c0 |            0x400 |
|  212 | audio_encoder.encoders.11.self_attn.fsmn_block.weight    |        0x50347c0 |           0x2c00 |
|  213 | audio_encoder.encoders.11.feed_forward.w_1.weight        |        0x50373c0 |         0x200000 |
|  214 | audio_encoder.encoders.11.feed_forward.w_1.bias          |        0x52373c0 |           0x1000 |
|  215 | audio_encoder.encoders.11.feed_forward.w_2.weight        |        0x52383c0 |         0x200000 |
|  216 | audio_encoder.encoders.11.feed_forward.w_2.bias          |        0x54383c0 |            0x400 |
|  217 | audio_encoder.encoders.11.norm1.weight                   |        0x54387c0 |            0x400 |
|  218 | audio_encoder.encoders.11.norm1.bias                     |        0x5438bc0 |            0x400 |
|  219 | audio_encoder.encoders.11.norm2.weight                   |        0x5438fc0 |            0x400 |
|  220 | audio_encoder.encoders.11.norm2.bias                     |        0x54393c0 |            0x400 |
|  221 | audio_encoder.encoders.12.self_attn.linear_out.weight    |        0x54397c0 |          0x80000 |
|  222 | audio_encoder.encoders.12.self_attn.linear_out.bias      |        0x54b97c0 |            0x400 |
|  223 | audio_encoder.encoders.12.self_attn.linear_q.weight      |        0x54b9bc0 |          0x80000 |
|  224 | audio_encoder.encoders.12.self_attn.linear_k.weight      |        0x5539bc0 |          0x80000 |
|  225 | audio_encoder.encoders.12.self_attn.linear_v.weight      |        0x55b9bc0 |          0x80000 |
|  226 | audio_encoder.encoders.12.self_attn.linear_q.bias        |        0x5639bc0 |            0x400 |
|  227 | audio_encoder.encoders.12.self_attn.linear_k.bias        |        0x5639fc0 |            0x400 |
|  228 | audio_encoder.encoders.12.self_attn.linear_v.bias        |        0x563a3c0 |            0x400 |
|  229 | audio_encoder.encoders.12.self_attn.fsmn_block.weight    |        0x563a7c0 |           0x2c00 |
|  230 | audio_encoder.encoders.12.feed_forward.w_1.weight        |        0x563d3c0 |         0x200000 |
|  231 | audio_encoder.encoders.12.feed_forward.w_1.bias          |        0x583d3c0 |           0x1000 |
|  232 | audio_encoder.encoders.12.feed_forward.w_2.weight        |        0x583e3c0 |         0x200000 |
|  233 | audio_encoder.encoders.12.feed_forward.w_2.bias          |        0x5a3e3c0 |            0x400 |
|  234 | audio_encoder.encoders.12.norm1.weight                   |        0x5a3e7c0 |            0x400 |
|  235 | audio_encoder.encoders.12.norm1.bias                     |        0x5a3ebc0 |            0x400 |
|  236 | audio_encoder.encoders.12.norm2.weight                   |        0x5a3efc0 |            0x400 |
|  237 | audio_encoder.encoders.12.norm2.bias                     |        0x5a3f3c0 |            0x400 |
|  238 | audio_encoder.encoders.13.self_attn.linear_out.weight    |        0x5a3f7c0 |          0x80000 |
|  239 | audio_encoder.encoders.13.self_attn.linear_out.bias      |        0x5abf7c0 |            0x400 |
|  240 | audio_encoder.encoders.13.self_attn.linear_q.weight      |        0x5abfbc0 |          0x80000 |
|  241 | audio_encoder.encoders.13.self_attn.linear_k.weight      |        0x5b3fbc0 |          0x80000 |
|  242 | audio_encoder.encoders.13.self_attn.linear_v.weight      |        0x5bbfbc0 |          0x80000 |
|  243 | audio_encoder.encoders.13.self_attn.linear_q.bias        |        0x5c3fbc0 |            0x400 |
|  244 | audio_encoder.encoders.13.self_attn.linear_k.bias        |        0x5c3ffc0 |            0x400 |
|  245 | audio_encoder.encoders.13.self_attn.linear_v.bias        |        0x5c403c0 |            0x400 |
|  246 | audio_encoder.encoders.13.self_attn.fsmn_block.weight    |        0x5c407c0 |           0x2c00 |
|  247 | audio_encoder.encoders.13.feed_forward.w_1.weight        |        0x5c433c0 |         0x200000 |
|  248 | audio_encoder.encoders.13.feed_forward.w_1.bias          |        0x5e433c0 |           0x1000 |
|  249 | audio_encoder.encoders.13.feed_forward.w_2.weight        |        0x5e443c0 |         0x200000 |
|  250 | audio_encoder.encoders.13.feed_forward.w_2.bias          |        0x60443c0 |            0x400 |
|  251 | audio_encoder.encoders.13.norm1.weight                   |        0x60447c0 |            0x400 |
|  252 | audio_encoder.encoders.13.norm1.bias                     |        0x6044bc0 |            0x400 |
|  253 | audio_encoder.encoders.13.norm2.weight                   |        0x6044fc0 |            0x400 |
|  254 | audio_encoder.encoders.13.norm2.bias                     |        0x60453c0 |            0x400 |
|  255 | audio_encoder.encoders.14.self_attn.linear_out.weight    |        0x60457c0 |          0x80000 |
|  256 | audio_encoder.encoders.14.self_attn.linear_out.bias      |        0x60c57c0 |            0x400 |
|  257 | audio_encoder.encoders.14.self_attn.linear_q.weight      |        0x60c5bc0 |          0x80000 |
|  258 | audio_encoder.encoders.14.self_attn.linear_k.weight      |        0x6145bc0 |          0x80000 |
|  259 | audio_encoder.encoders.14.self_attn.linear_v.weight      |        0x61c5bc0 |          0x80000 |
|  260 | audio_encoder.encoders.14.self_attn.linear_q.bias        |        0x6245bc0 |            0x400 |
|  261 | audio_encoder.encoders.14.self_attn.linear_k.bias        |        0x6245fc0 |            0x400 |
|  262 | audio_encoder.encoders.14.self_attn.linear_v.bias        |        0x62463c0 |            0x400 |
|  263 | audio_encoder.encoders.14.self_attn.fsmn_block.weight    |        0x62467c0 |           0x2c00 |
|  264 | audio_encoder.encoders.14.feed_forward.w_1.weight        |        0x62493c0 |         0x200000 |
|  265 | audio_encoder.encoders.14.feed_forward.w_1.bias          |        0x64493c0 |           0x1000 |
|  266 | audio_encoder.encoders.14.feed_forward.w_2.weight        |        0x644a3c0 |         0x200000 |
|  267 | audio_encoder.encoders.14.feed_forward.w_2.bias          |        0x664a3c0 |            0x400 |
|  268 | audio_encoder.encoders.14.norm1.weight                   |        0x664a7c0 |            0x400 |
|  269 | audio_encoder.encoders.14.norm1.bias                     |        0x664abc0 |            0x400 |
|  270 | audio_encoder.encoders.14.norm2.weight                   |        0x664afc0 |            0x400 |
|  271 | audio_encoder.encoders.14.norm2.bias                     |        0x664b3c0 |            0x400 |
|  272 | audio_encoder.encoders.15.self_attn.linear_out.weight    |        0x664b7c0 |          0x80000 |
|  273 | audio_encoder.encoders.15.self_attn.linear_out.bias      |        0x66cb7c0 |            0x400 |
|  274 | audio_encoder.encoders.15.self_attn.linear_q.weight      |        0x66cbbc0 |          0x80000 |
|  275 | audio_encoder.encoders.15.self_attn.linear_k.weight      |        0x674bbc0 |          0x80000 |
|  276 | audio_encoder.encoders.15.self_attn.linear_v.weight      |        0x67cbbc0 |          0x80000 |
|  277 | audio_encoder.encoders.15.self_attn.linear_q.bias        |        0x684bbc0 |            0x400 |
|  278 | audio_encoder.encoders.15.self_attn.linear_k.bias        |        0x684bfc0 |            0x400 |
|  279 | audio_encoder.encoders.15.self_attn.linear_v.bias        |        0x684c3c0 |            0x400 |
|  280 | audio_encoder.encoders.15.self_attn.fsmn_block.weight    |        0x684c7c0 |           0x2c00 |
|  281 | audio_encoder.encoders.15.feed_forward.w_1.weight        |        0x684f3c0 |         0x200000 |
|  282 | audio_encoder.encoders.15.feed_forward.w_1.bias          |        0x6a4f3c0 |           0x1000 |
|  283 | audio_encoder.encoders.15.feed_forward.w_2.weight        |        0x6a503c0 |         0x200000 |
|  284 | audio_encoder.encoders.15.feed_forward.w_2.bias          |        0x6c503c0 |            0x400 |
|  285 | audio_encoder.encoders.15.norm1.weight                   |        0x6c507c0 |            0x400 |
|  286 | audio_encoder.encoders.15.norm1.bias                     |        0x6c50bc0 |            0x400 |
|  287 | audio_encoder.encoders.15.norm2.weight                   |        0x6c50fc0 |            0x400 |
|  288 | audio_encoder.encoders.15.norm2.bias                     |        0x6c513c0 |            0x400 |
|  289 | audio_encoder.encoders.16.self_attn.linear_out.weight    |        0x6c517c0 |          0x80000 |
|  290 | audio_encoder.encoders.16.self_attn.linear_out.bias      |        0x6cd17c0 |            0x400 |
|  291 | audio_encoder.encoders.16.self_attn.linear_q.weight      |        0x6cd1bc0 |          0x80000 |
|  292 | audio_encoder.encoders.16.self_attn.linear_k.weight      |        0x6d51bc0 |          0x80000 |
|  293 | audio_encoder.encoders.16.self_attn.linear_v.weight      |        0x6dd1bc0 |          0x80000 |
|  294 | audio_encoder.encoders.16.self_attn.linear_q.bias        |        0x6e51bc0 |            0x400 |
|  295 | audio_encoder.encoders.16.self_attn.linear_k.bias        |        0x6e51fc0 |            0x400 |
|  296 | audio_encoder.encoders.16.self_attn.linear_v.bias        |        0x6e523c0 |            0x400 |
|  297 | audio_encoder.encoders.16.self_attn.fsmn_block.weight    |        0x6e527c0 |           0x2c00 |
|  298 | audio_encoder.encoders.16.feed_forward.w_1.weight        |        0x6e553c0 |         0x200000 |
|  299 | audio_encoder.encoders.16.feed_forward.w_1.bias          |        0x70553c0 |           0x1000 |
|  300 | audio_encoder.encoders.16.feed_forward.w_2.weight        |        0x70563c0 |         0x200000 |
|  301 | audio_encoder.encoders.16.feed_forward.w_2.bias          |        0x72563c0 |            0x400 |
|  302 | audio_encoder.encoders.16.norm1.weight                   |        0x72567c0 |            0x400 |
|  303 | audio_encoder.encoders.16.norm1.bias                     |        0x7256bc0 |            0x400 |
|  304 | audio_encoder.encoders.16.norm2.weight                   |        0x7256fc0 |            0x400 |
|  305 | audio_encoder.encoders.16.norm2.bias                     |        0x72573c0 |            0x400 |
|  306 | audio_encoder.encoders.17.self_attn.linear_out.weight    |        0x72577c0 |          0x80000 |
|  307 | audio_encoder.encoders.17.self_attn.linear_out.bias      |        0x72d77c0 |            0x400 |
|  308 | audio_encoder.encoders.17.self_attn.linear_q.weight      |        0x72d7bc0 |          0x80000 |
|  309 | audio_encoder.encoders.17.self_attn.linear_k.weight      |        0x7357bc0 |          0x80000 |
|  310 | audio_encoder.encoders.17.self_attn.linear_v.weight      |        0x73d7bc0 |          0x80000 |
|  311 | audio_encoder.encoders.17.self_attn.linear_q.bias        |        0x7457bc0 |            0x400 |
|  312 | audio_encoder.encoders.17.self_attn.linear_k.bias        |        0x7457fc0 |            0x400 |
|  313 | audio_encoder.encoders.17.self_attn.linear_v.bias        |        0x74583c0 |            0x400 |
|  314 | audio_encoder.encoders.17.self_attn.fsmn_block.weight    |        0x74587c0 |           0x2c00 |
|  315 | audio_encoder.encoders.17.feed_forward.w_1.weight        |        0x745b3c0 |         0x200000 |
|  316 | audio_encoder.encoders.17.feed_forward.w_1.bias          |        0x765b3c0 |           0x1000 |
|  317 | audio_encoder.encoders.17.feed_forward.w_2.weight        |        0x765c3c0 |         0x200000 |
|  318 | audio_encoder.encoders.17.feed_forward.w_2.bias          |        0x785c3c0 |            0x400 |
|  319 | audio_encoder.encoders.17.norm1.weight                   |        0x785c7c0 |            0x400 |
|  320 | audio_encoder.encoders.17.norm1.bias                     |        0x785cbc0 |            0x400 |
|  321 | audio_encoder.encoders.17.norm2.weight                   |        0x785cfc0 |            0x400 |
|  322 | audio_encoder.encoders.17.norm2.bias                     |        0x785d3c0 |            0x400 |
|  323 | audio_encoder.encoders.18.self_attn.linear_out.weight    |        0x785d7c0 |          0x80000 |
|  324 | audio_encoder.encoders.18.self_attn.linear_out.bias      |        0x78dd7c0 |            0x400 |
|  325 | audio_encoder.encoders.18.self_attn.linear_q.weight      |        0x78ddbc0 |          0x80000 |
|  326 | audio_encoder.encoders.18.self_attn.linear_k.weight      |        0x795dbc0 |          0x80000 |
|  327 | audio_encoder.encoders.18.self_attn.linear_v.weight      |        0x79ddbc0 |          0x80000 |
|  328 | audio_encoder.encoders.18.self_attn.linear_q.bias        |        0x7a5dbc0 |            0x400 |
|  329 | audio_encoder.encoders.18.self_attn.linear_k.bias        |        0x7a5dfc0 |            0x400 |
|  330 | audio_encoder.encoders.18.self_attn.linear_v.bias        |        0x7a5e3c0 |            0x400 |
|  331 | audio_encoder.encoders.18.self_attn.fsmn_block.weight    |        0x7a5e7c0 |           0x2c00 |
|  332 | audio_encoder.encoders.18.feed_forward.w_1.weight        |        0x7a613c0 |         0x200000 |
|  333 | audio_encoder.encoders.18.feed_forward.w_1.bias          |        0x7c613c0 |           0x1000 |
|  334 | audio_encoder.encoders.18.feed_forward.w_2.weight        |        0x7c623c0 |         0x200000 |
|  335 | audio_encoder.encoders.18.feed_forward.w_2.bias          |        0x7e623c0 |            0x400 |
|  336 | audio_encoder.encoders.18.norm1.weight                   |        0x7e627c0 |            0x400 |
|  337 | audio_encoder.encoders.18.norm1.bias                     |        0x7e62bc0 |            0x400 |
|  338 | audio_encoder.encoders.18.norm2.weight                   |        0x7e62fc0 |            0x400 |
|  339 | audio_encoder.encoders.18.norm2.bias                     |        0x7e633c0 |            0x400 |
|  340 | audio_encoder.encoders.19.self_attn.linear_out.weight    |        0x7e637c0 |          0x80000 |
|  341 | audio_encoder.encoders.19.self_attn.linear_out.bias      |        0x7ee37c0 |            0x400 |
|  342 | audio_encoder.encoders.19.self_attn.linear_q.weight      |        0x7ee3bc0 |          0x80000 |
|  343 | audio_encoder.encoders.19.self_attn.linear_k.weight      |        0x7f63bc0 |          0x80000 |
|  344 | audio_encoder.encoders.19.self_attn.linear_v.weight      |        0x7fe3bc0 |          0x80000 |
|  345 | audio_encoder.encoders.19.self_attn.linear_q.bias        |        0x8063bc0 |            0x400 |
|  346 | audio_encoder.encoders.19.self_attn.linear_k.bias        |        0x8063fc0 |            0x400 |
|  347 | audio_encoder.encoders.19.self_attn.linear_v.bias        |        0x80643c0 |            0x400 |
|  348 | audio_encoder.encoders.19.self_attn.fsmn_block.weight    |        0x80647c0 |           0x2c00 |
|  349 | audio_encoder.encoders.19.feed_forward.w_1.weight        |        0x80673c0 |         0x200000 |
|  350 | audio_encoder.encoders.19.feed_forward.w_1.bias          |        0x82673c0 |           0x1000 |
|  351 | audio_encoder.encoders.19.feed_forward.w_2.weight        |        0x82683c0 |         0x200000 |
|  352 | audio_encoder.encoders.19.feed_forward.w_2.bias          |        0x84683c0 |            0x400 |
|  353 | audio_encoder.encoders.19.norm1.weight                   |        0x84687c0 |            0x400 |
|  354 | audio_encoder.encoders.19.norm1.bias                     |        0x8468bc0 |            0x400 |
|  355 | audio_encoder.encoders.19.norm2.weight                   |        0x8468fc0 |            0x400 |
|  356 | audio_encoder.encoders.19.norm2.bias                     |        0x84693c0 |            0x400 |
|  357 | audio_encoder.encoders.20.self_attn.linear_out.weight    |        0x84697c0 |          0x80000 |
|  358 | audio_encoder.encoders.20.self_attn.linear_out.bias      |        0x84e97c0 |            0x400 |
|  359 | audio_encoder.encoders.20.self_attn.linear_q.weight      |        0x84e9bc0 |          0x80000 |
|  360 | audio_encoder.encoders.20.self_attn.linear_k.weight      |        0x8569bc0 |          0x80000 |
|  361 | audio_encoder.encoders.20.self_attn.linear_v.weight      |        0x85e9bc0 |          0x80000 |
|  362 | audio_encoder.encoders.20.self_attn.linear_q.bias        |        0x8669bc0 |            0x400 |
|  363 | audio_encoder.encoders.20.self_attn.linear_k.bias        |        0x8669fc0 |            0x400 |
|  364 | audio_encoder.encoders.20.self_attn.linear_v.bias        |        0x866a3c0 |            0x400 |
|  365 | audio_encoder.encoders.20.self_attn.fsmn_block.weight    |        0x866a7c0 |           0x2c00 |
|  366 | audio_encoder.encoders.20.feed_forward.w_1.weight        |        0x866d3c0 |         0x200000 |
|  367 | audio_encoder.encoders.20.feed_forward.w_1.bias          |        0x886d3c0 |           0x1000 |
|  368 | audio_encoder.encoders.20.feed_forward.w_2.weight        |        0x886e3c0 |         0x200000 |
|  369 | audio_encoder.encoders.20.feed_forward.w_2.bias          |        0x8a6e3c0 |            0x400 |
|  370 | audio_encoder.encoders.20.norm1.weight                   |        0x8a6e7c0 |            0x400 |
|  371 | audio_encoder.encoders.20.norm1.bias                     |        0x8a6ebc0 |            0x400 |
|  372 | audio_encoder.encoders.20.norm2.weight                   |        0x8a6efc0 |            0x400 |
|  373 | audio_encoder.encoders.20.norm2.bias                     |        0x8a6f3c0 |            0x400 |
|  374 | audio_encoder.encoders.21.self_attn.linear_out.weight    |        0x8a6f7c0 |          0x80000 |
|  375 | audio_encoder.encoders.21.self_attn.linear_out.bias      |        0x8aef7c0 |            0x400 |
|  376 | audio_encoder.encoders.21.self_attn.linear_q.weight      |        0x8aefbc0 |          0x80000 |
|  377 | audio_encoder.encoders.21.self_attn.linear_k.weight      |        0x8b6fbc0 |          0x80000 |
|  378 | audio_encoder.encoders.21.self_attn.linear_v.weight      |        0x8befbc0 |          0x80000 |
|  379 | audio_encoder.encoders.21.self_attn.linear_q.bias        |        0x8c6fbc0 |            0x400 |
|  380 | audio_encoder.encoders.21.self_attn.linear_k.bias        |        0x8c6ffc0 |            0x400 |
|  381 | audio_encoder.encoders.21.self_attn.linear_v.bias        |        0x8c703c0 |            0x400 |
|  382 | audio_encoder.encoders.21.self_attn.fsmn_block.weight    |        0x8c707c0 |           0x2c00 |
|  383 | audio_encoder.encoders.21.feed_forward.w_1.weight        |        0x8c733c0 |         0x200000 |
|  384 | audio_encoder.encoders.21.feed_forward.w_1.bias          |        0x8e733c0 |           0x1000 |
|  385 | audio_encoder.encoders.21.feed_forward.w_2.weight        |        0x8e743c0 |         0x200000 |
|  386 | audio_encoder.encoders.21.feed_forward.w_2.bias          |        0x90743c0 |            0x400 |
|  387 | audio_encoder.encoders.21.norm1.weight                   |        0x90747c0 |            0x400 |
|  388 | audio_encoder.encoders.21.norm1.bias                     |        0x9074bc0 |            0x400 |
|  389 | audio_encoder.encoders.21.norm2.weight                   |        0x9074fc0 |            0x400 |
|  390 | audio_encoder.encoders.21.norm2.bias                     |        0x90753c0 |            0x400 |
|  391 | audio_encoder.encoders.22.self_attn.linear_out.weight    |        0x90757c0 |          0x80000 |
|  392 | audio_encoder.encoders.22.self_attn.linear_out.bias      |        0x90f57c0 |            0x400 |
|  393 | audio_encoder.encoders.22.self_attn.linear_q.weight      |        0x90f5bc0 |          0x80000 |
|  394 | audio_encoder.encoders.22.self_attn.linear_k.weight      |        0x9175bc0 |          0x80000 |
|  395 | audio_encoder.encoders.22.self_attn.linear_v.weight      |        0x91f5bc0 |          0x80000 |
|  396 | audio_encoder.encoders.22.self_attn.linear_q.bias        |        0x9275bc0 |            0x400 |
|  397 | audio_encoder.encoders.22.self_attn.linear_k.bias        |        0x9275fc0 |            0x400 |
|  398 | audio_encoder.encoders.22.self_attn.linear_v.bias        |        0x92763c0 |            0x400 |
|  399 | audio_encoder.encoders.22.self_attn.fsmn_block.weight    |        0x92767c0 |           0x2c00 |
|  400 | audio_encoder.encoders.22.feed_forward.w_1.weight        |        0x92793c0 |         0x200000 |
|  401 | audio_encoder.encoders.22.feed_forward.w_1.bias          |        0x94793c0 |           0x1000 |
|  402 | audio_encoder.encoders.22.feed_forward.w_2.weight        |        0x947a3c0 |         0x200000 |
|  403 | audio_encoder.encoders.22.feed_forward.w_2.bias          |        0x967a3c0 |            0x400 |
|  404 | audio_encoder.encoders.22.norm1.weight                   |        0x967a7c0 |            0x400 |
|  405 | audio_encoder.encoders.22.norm1.bias                     |        0x967abc0 |            0x400 |
|  406 | audio_encoder.encoders.22.norm2.weight                   |        0x967afc0 |            0x400 |
|  407 | audio_encoder.encoders.22.norm2.bias                     |        0x967b3c0 |            0x400 |
|  408 | audio_encoder.encoders.23.self_attn.linear_out.weight    |        0x967b7c0 |          0x80000 |
|  409 | audio_encoder.encoders.23.self_attn.linear_out.bias      |        0x96fb7c0 |            0x400 |
|  410 | audio_encoder.encoders.23.self_attn.linear_q.weight      |        0x96fbbc0 |          0x80000 |
|  411 | audio_encoder.encoders.23.self_attn.linear_k.weight      |        0x977bbc0 |          0x80000 |
|  412 | audio_encoder.encoders.23.self_attn.linear_v.weight      |        0x97fbbc0 |          0x80000 |
|  413 | audio_encoder.encoders.23.self_attn.linear_q.bias        |        0x987bbc0 |            0x400 |
|  414 | audio_encoder.encoders.23.self_attn.linear_k.bias        |        0x987bfc0 |            0x400 |
|  415 | audio_encoder.encoders.23.self_attn.linear_v.bias        |        0x987c3c0 |            0x400 |
|  416 | audio_encoder.encoders.23.self_attn.fsmn_block.weight    |        0x987c7c0 |           0x2c00 |
|  417 | audio_encoder.encoders.23.feed_forward.w_1.weight        |        0x987f3c0 |         0x200000 |
|  418 | audio_encoder.encoders.23.feed_forward.w_1.bias          |        0x9a7f3c0 |           0x1000 |
|  419 | audio_encoder.encoders.23.feed_forward.w_2.weight        |        0x9a803c0 |         0x200000 |
|  420 | audio_encoder.encoders.23.feed_forward.w_2.bias          |        0x9c803c0 |            0x400 |
|  421 | audio_encoder.encoders.23.norm1.weight                   |        0x9c807c0 |            0x400 |
|  422 | audio_encoder.encoders.23.norm1.bias                     |        0x9c80bc0 |            0x400 |
|  423 | audio_encoder.encoders.23.norm2.weight                   |        0x9c80fc0 |            0x400 |
|  424 | audio_encoder.encoders.23.norm2.bias                     |        0x9c813c0 |            0x400 |
|  425 | audio_encoder.encoders.24.self_attn.linear_out.weight    |        0x9c817c0 |          0x80000 |
|  426 | audio_encoder.encoders.24.self_attn.linear_out.bias      |        0x9d017c0 |            0x400 |
|  427 | audio_encoder.encoders.24.self_attn.linear_q.weight      |        0x9d01bc0 |          0x80000 |
|  428 | audio_encoder.encoders.24.self_attn.linear_k.weight      |        0x9d81bc0 |          0x80000 |
|  429 | audio_encoder.encoders.24.self_attn.linear_v.weight      |        0x9e01bc0 |          0x80000 |
|  430 | audio_encoder.encoders.24.self_attn.linear_q.bias        |        0x9e81bc0 |            0x400 |
|  431 | audio_encoder.encoders.24.self_attn.linear_k.bias        |        0x9e81fc0 |            0x400 |
|  432 | audio_encoder.encoders.24.self_attn.linear_v.bias        |        0x9e823c0 |            0x400 |
|  433 | audio_encoder.encoders.24.self_attn.fsmn_block.weight    |        0x9e827c0 |           0x2c00 |
|  434 | audio_encoder.encoders.24.feed_forward.w_1.weight        |        0x9e853c0 |         0x200000 |
|  435 | audio_encoder.encoders.24.feed_forward.w_1.bias          |        0xa0853c0 |           0x1000 |
|  436 | audio_encoder.encoders.24.feed_forward.w_2.weight        |        0xa0863c0 |         0x200000 |
|  437 | audio_encoder.encoders.24.feed_forward.w_2.bias          |        0xa2863c0 |            0x400 |
|  438 | audio_encoder.encoders.24.norm1.weight                   |        0xa2867c0 |            0x400 |
|  439 | audio_encoder.encoders.24.norm1.bias                     |        0xa286bc0 |            0x400 |
|  440 | audio_encoder.encoders.24.norm2.weight                   |        0xa286fc0 |            0x400 |
|  441 | audio_encoder.encoders.24.norm2.bias                     |        0xa2873c0 |            0x400 |
|  442 | audio_encoder.encoders.25.self_attn.linear_out.weight    |        0xa2877c0 |          0x80000 |
|  443 | audio_encoder.encoders.25.self_attn.linear_out.bias      |        0xa3077c0 |            0x400 |
|  444 | audio_encoder.encoders.25.self_attn.linear_q.weight      |        0xa307bc0 |          0x80000 |
|  445 | audio_encoder.encoders.25.self_attn.linear_k.weight      |        0xa387bc0 |          0x80000 |
|  446 | audio_encoder.encoders.25.self_attn.linear_v.weight      |        0xa407bc0 |          0x80000 |
|  447 | audio_encoder.encoders.25.self_attn.linear_q.bias        |        0xa487bc0 |            0x400 |
|  448 | audio_encoder.encoders.25.self_attn.linear_k.bias        |        0xa487fc0 |            0x400 |
|  449 | audio_encoder.encoders.25.self_attn.linear_v.bias        |        0xa4883c0 |            0x400 |
|  450 | audio_encoder.encoders.25.self_attn.fsmn_block.weight    |        0xa4887c0 |           0x2c00 |
|  451 | audio_encoder.encoders.25.feed_forward.w_1.weight        |        0xa48b3c0 |         0x200000 |
|  452 | audio_encoder.encoders.25.feed_forward.w_1.bias          |        0xa68b3c0 |           0x1000 |
|  453 | audio_encoder.encoders.25.feed_forward.w_2.weight        |        0xa68c3c0 |         0x200000 |
|  454 | audio_encoder.encoders.25.feed_forward.w_2.bias          |        0xa88c3c0 |            0x400 |
|  455 | audio_encoder.encoders.25.norm1.weight                   |        0xa88c7c0 |            0x400 |
|  456 | audio_encoder.encoders.25.norm1.bias                     |        0xa88cbc0 |            0x400 |
|  457 | audio_encoder.encoders.25.norm2.weight                   |        0xa88cfc0 |            0x400 |
|  458 | audio_encoder.encoders.25.norm2.bias                     |        0xa88d3c0 |            0x400 |
|  459 | audio_encoder.encoders.26.self_attn.linear_out.weight    |        0xa88d7c0 |          0x80000 |
|  460 | audio_encoder.encoders.26.self_attn.linear_out.bias      |        0xa90d7c0 |            0x400 |
|  461 | audio_encoder.encoders.26.self_attn.linear_q.weight      |        0xa90dbc0 |          0x80000 |
|  462 | audio_encoder.encoders.26.self_attn.linear_k.weight      |        0xa98dbc0 |          0x80000 |
|  463 | audio_encoder.encoders.26.self_attn.linear_v.weight      |        0xaa0dbc0 |          0x80000 |
|  464 | audio_encoder.encoders.26.self_attn.linear_q.bias        |        0xaa8dbc0 |            0x400 |
|  465 | audio_encoder.encoders.26.self_attn.linear_k.bias        |        0xaa8dfc0 |            0x400 |
|  466 | audio_encoder.encoders.26.self_attn.linear_v.bias        |        0xaa8e3c0 |            0x400 |
|  467 | audio_encoder.encoders.26.self_attn.fsmn_block.weight    |        0xaa8e7c0 |           0x2c00 |
|  468 | audio_encoder.encoders.26.feed_forward.w_1.weight        |        0xaa913c0 |         0x200000 |
|  469 | audio_encoder.encoders.26.feed_forward.w_1.bias          |        0xac913c0 |           0x1000 |
|  470 | audio_encoder.encoders.26.feed_forward.w_2.weight        |        0xac923c0 |         0x200000 |
|  471 | audio_encoder.encoders.26.feed_forward.w_2.bias          |        0xae923c0 |            0x400 |
|  472 | audio_encoder.encoders.26.norm1.weight                   |        0xae927c0 |            0x400 |
|  473 | audio_encoder.encoders.26.norm1.bias                     |        0xae92bc0 |            0x400 |
|  474 | audio_encoder.encoders.26.norm2.weight                   |        0xae92fc0 |            0x400 |
|  475 | audio_encoder.encoders.26.norm2.bias                     |        0xae933c0 |            0x400 |
|  476 | audio_encoder.encoders.27.self_attn.linear_out.weight    |        0xae937c0 |          0x80000 |
|  477 | audio_encoder.encoders.27.self_attn.linear_out.bias      |        0xaf137c0 |            0x400 |
|  478 | audio_encoder.encoders.27.self_attn.linear_q.weight      |        0xaf13bc0 |          0x80000 |
|  479 | audio_encoder.encoders.27.self_attn.linear_k.weight      |        0xaf93bc0 |          0x80000 |
|  480 | audio_encoder.encoders.27.self_attn.linear_v.weight      |        0xb013bc0 |          0x80000 |
|  481 | audio_encoder.encoders.27.self_attn.linear_q.bias        |        0xb093bc0 |            0x400 |
|  482 | audio_encoder.encoders.27.self_attn.linear_k.bias        |        0xb093fc0 |            0x400 |
|  483 | audio_encoder.encoders.27.self_attn.linear_v.bias        |        0xb0943c0 |            0x400 |
|  484 | audio_encoder.encoders.27.self_attn.fsmn_block.weight    |        0xb0947c0 |           0x2c00 |
|  485 | audio_encoder.encoders.27.feed_forward.w_1.weight        |        0xb0973c0 |         0x200000 |
|  486 | audio_encoder.encoders.27.feed_forward.w_1.bias          |        0xb2973c0 |           0x1000 |
|  487 | audio_encoder.encoders.27.feed_forward.w_2.weight        |        0xb2983c0 |         0x200000 |
|  488 | audio_encoder.encoders.27.feed_forward.w_2.bias          |        0xb4983c0 |            0x400 |
|  489 | audio_encoder.encoders.27.norm1.weight                   |        0xb4987c0 |            0x400 |
|  490 | audio_encoder.encoders.27.norm1.bias                     |        0xb498bc0 |            0x400 |
|  491 | audio_encoder.encoders.27.norm2.weight                   |        0xb498fc0 |            0x400 |
|  492 | audio_encoder.encoders.27.norm2.bias                     |        0xb4993c0 |            0x400 |
|  493 | audio_encoder.encoders.28.self_attn.linear_out.weight    |        0xb4997c0 |          0x80000 |
|  494 | audio_encoder.encoders.28.self_attn.linear_out.bias      |        0xb5197c0 |            0x400 |
|  495 | audio_encoder.encoders.28.self_attn.linear_q.weight      |        0xb519bc0 |          0x80000 |
|  496 | audio_encoder.encoders.28.self_attn.linear_k.weight      |        0xb599bc0 |          0x80000 |
|  497 | audio_encoder.encoders.28.self_attn.linear_v.weight      |        0xb619bc0 |          0x80000 |
|  498 | audio_encoder.encoders.28.self_attn.linear_q.bias        |        0xb699bc0 |            0x400 |
|  499 | audio_encoder.encoders.28.self_attn.linear_k.bias        |        0xb699fc0 |            0x400 |
|  500 | audio_encoder.encoders.28.self_attn.linear_v.bias        |        0xb69a3c0 |            0x400 |
|  501 | audio_encoder.encoders.28.self_attn.fsmn_block.weight    |        0xb69a7c0 |           0x2c00 |
|  502 | audio_encoder.encoders.28.feed_forward.w_1.weight        |        0xb69d3c0 |         0x200000 |
|  503 | audio_encoder.encoders.28.feed_forward.w_1.bias          |        0xb89d3c0 |           0x1000 |
|  504 | audio_encoder.encoders.28.feed_forward.w_2.weight        |        0xb89e3c0 |         0x200000 |
|  505 | audio_encoder.encoders.28.feed_forward.w_2.bias          |        0xba9e3c0 |            0x400 |
|  506 | audio_encoder.encoders.28.norm1.weight                   |        0xba9e7c0 |            0x400 |
|  507 | audio_encoder.encoders.28.norm1.bias                     |        0xba9ebc0 |            0x400 |
|  508 | audio_encoder.encoders.28.norm2.weight                   |        0xba9efc0 |            0x400 |
|  509 | audio_encoder.encoders.28.norm2.bias                     |        0xba9f3c0 |            0x400 |
|  510 | audio_encoder.encoders.29.self_attn.linear_out.weight    |        0xba9f7c0 |          0x80000 |
|  511 | audio_encoder.encoders.29.self_attn.linear_out.bias      |        0xbb1f7c0 |            0x400 |
|  512 | audio_encoder.encoders.29.self_attn.linear_q.weight      |        0xbb1fbc0 |          0x80000 |
|  513 | audio_encoder.encoders.29.self_attn.linear_k.weight      |        0xbb9fbc0 |          0x80000 |
|  514 | audio_encoder.encoders.29.self_attn.linear_v.weight      |        0xbc1fbc0 |          0x80000 |
|  515 | audio_encoder.encoders.29.self_attn.linear_q.bias        |        0xbc9fbc0 |            0x400 |
|  516 | audio_encoder.encoders.29.self_attn.linear_k.bias        |        0xbc9ffc0 |            0x400 |
|  517 | audio_encoder.encoders.29.self_attn.linear_v.bias        |        0xbca03c0 |            0x400 |
|  518 | audio_encoder.encoders.29.self_attn.fsmn_block.weight    |        0xbca07c0 |           0x2c00 |
|  519 | audio_encoder.encoders.29.feed_forward.w_1.weight        |        0xbca33c0 |         0x200000 |
|  520 | audio_encoder.encoders.29.feed_forward.w_1.bias          |        0xbea33c0 |           0x1000 |
|  521 | audio_encoder.encoders.29.feed_forward.w_2.weight        |        0xbea43c0 |         0x200000 |
|  522 | audio_encoder.encoders.29.feed_forward.w_2.bias          |        0xc0a43c0 |            0x400 |
|  523 | audio_encoder.encoders.29.norm1.weight                   |        0xc0a47c0 |            0x400 |
|  524 | audio_encoder.encoders.29.norm1.bias                     |        0xc0a4bc0 |            0x400 |
|  525 | audio_encoder.encoders.29.norm2.weight                   |        0xc0a4fc0 |            0x400 |
|  526 | audio_encoder.encoders.29.norm2.bias                     |        0xc0a53c0 |            0x400 |
|  527 | audio_encoder.encoders.30.self_attn.linear_out.weight    |        0xc0a57c0 |          0x80000 |
|  528 | audio_encoder.encoders.30.self_attn.linear_out.bias      |        0xc1257c0 |            0x400 |
|  529 | audio_encoder.encoders.30.self_attn.linear_q.weight      |        0xc125bc0 |          0x80000 |
|  530 | audio_encoder.encoders.30.self_attn.linear_k.weight      |        0xc1a5bc0 |          0x80000 |
|  531 | audio_encoder.encoders.30.self_attn.linear_v.weight      |        0xc225bc0 |          0x80000 |
|  532 | audio_encoder.encoders.30.self_attn.linear_q.bias        |        0xc2a5bc0 |            0x400 |
|  533 | audio_encoder.encoders.30.self_attn.linear_k.bias        |        0xc2a5fc0 |            0x400 |
|  534 | audio_encoder.encoders.30.self_attn.linear_v.bias        |        0xc2a63c0 |            0x400 |
|  535 | audio_encoder.encoders.30.self_attn.fsmn_block.weight    |        0xc2a67c0 |           0x2c00 |
|  536 | audio_encoder.encoders.30.feed_forward.w_1.weight        |        0xc2a93c0 |         0x200000 |
|  537 | audio_encoder.encoders.30.feed_forward.w_1.bias          |        0xc4a93c0 |           0x1000 |
|  538 | audio_encoder.encoders.30.feed_forward.w_2.weight        |        0xc4aa3c0 |         0x200000 |
|  539 | audio_encoder.encoders.30.feed_forward.w_2.bias          |        0xc6aa3c0 |            0x400 |
|  540 | audio_encoder.encoders.30.norm1.weight                   |        0xc6aa7c0 |            0x400 |
|  541 | audio_encoder.encoders.30.norm1.bias                     |        0xc6aabc0 |            0x400 |
|  542 | audio_encoder.encoders.30.norm2.weight                   |        0xc6aafc0 |            0x400 |
|  543 | audio_encoder.encoders.30.norm2.bias                     |        0xc6ab3c0 |            0x400 |
|  544 | audio_encoder.encoders.31.self_attn.linear_out.weight    |        0xc6ab7c0 |          0x80000 |
|  545 | audio_encoder.encoders.31.self_attn.linear_out.bias      |        0xc72b7c0 |            0x400 |
|  546 | audio_encoder.encoders.31.self_attn.linear_q.weight      |        0xc72bbc0 |          0x80000 |
|  547 | audio_encoder.encoders.31.self_attn.linear_k.weight      |        0xc7abbc0 |          0x80000 |
|  548 | audio_encoder.encoders.31.self_attn.linear_v.weight      |        0xc82bbc0 |          0x80000 |
|  549 | audio_encoder.encoders.31.self_attn.linear_q.bias        |        0xc8abbc0 |            0x400 |
|  550 | audio_encoder.encoders.31.self_attn.linear_k.bias        |        0xc8abfc0 |            0x400 |
|  551 | audio_encoder.encoders.31.self_attn.linear_v.bias        |        0xc8ac3c0 |            0x400 |
|  552 | audio_encoder.encoders.31.self_attn.fsmn_block.weight    |        0xc8ac7c0 |           0x2c00 |
|  553 | audio_encoder.encoders.31.feed_forward.w_1.weight        |        0xc8af3c0 |         0x200000 |
|  554 | audio_encoder.encoders.31.feed_forward.w_1.bias          |        0xcaaf3c0 |           0x1000 |
|  555 | audio_encoder.encoders.31.feed_forward.w_2.weight        |        0xcab03c0 |         0x200000 |
|  556 | audio_encoder.encoders.31.feed_forward.w_2.bias          |        0xccb03c0 |            0x400 |
|  557 | audio_encoder.encoders.31.norm1.weight                   |        0xccb07c0 |            0x400 |
|  558 | audio_encoder.encoders.31.norm1.bias                     |        0xccb0bc0 |            0x400 |
|  559 | audio_encoder.encoders.31.norm2.weight                   |        0xccb0fc0 |            0x400 |
|  560 | audio_encoder.encoders.31.norm2.bias                     |        0xccb13c0 |            0x400 |
|  561 | audio_encoder.encoders.32.self_attn.linear_out.weight    |        0xccb17c0 |          0x80000 |
|  562 | audio_encoder.encoders.32.self_attn.linear_out.bias      |        0xcd317c0 |            0x400 |
|  563 | audio_encoder.encoders.32.self_attn.linear_q.weight      |        0xcd31bc0 |          0x80000 |
|  564 | audio_encoder.encoders.32.self_attn.linear_k.weight      |        0xcdb1bc0 |          0x80000 |
|  565 | audio_encoder.encoders.32.self_attn.linear_v.weight      |        0xce31bc0 |          0x80000 |
|  566 | audio_encoder.encoders.32.self_attn.linear_q.bias        |        0xceb1bc0 |            0x400 |
|  567 | audio_encoder.encoders.32.self_attn.linear_k.bias        |        0xceb1fc0 |            0x400 |
|  568 | audio_encoder.encoders.32.self_attn.linear_v.bias        |        0xceb23c0 |            0x400 |
|  569 | audio_encoder.encoders.32.self_attn.fsmn_block.weight    |        0xceb27c0 |           0x2c00 |
|  570 | audio_encoder.encoders.32.feed_forward.w_1.weight        |        0xceb53c0 |         0x200000 |
|  571 | audio_encoder.encoders.32.feed_forward.w_1.bias          |        0xd0b53c0 |           0x1000 |
|  572 | audio_encoder.encoders.32.feed_forward.w_2.weight        |        0xd0b63c0 |         0x200000 |
|  573 | audio_encoder.encoders.32.feed_forward.w_2.bias          |        0xd2b63c0 |            0x400 |
|  574 | audio_encoder.encoders.32.norm1.weight                   |        0xd2b67c0 |            0x400 |
|  575 | audio_encoder.encoders.32.norm1.bias                     |        0xd2b6bc0 |            0x400 |
|  576 | audio_encoder.encoders.32.norm2.weight                   |        0xd2b6fc0 |            0x400 |
|  577 | audio_encoder.encoders.32.norm2.bias                     |        0xd2b73c0 |            0x400 |
|  578 | audio_encoder.encoders.33.self_attn.linear_out.weight    |        0xd2b77c0 |          0x80000 |
|  579 | audio_encoder.encoders.33.self_attn.linear_out.bias      |        0xd3377c0 |            0x400 |
|  580 | audio_encoder.encoders.33.self_attn.linear_q.weight      |        0xd337bc0 |          0x80000 |
|  581 | audio_encoder.encoders.33.self_attn.linear_k.weight      |        0xd3b7bc0 |          0x80000 |
|  582 | audio_encoder.encoders.33.self_attn.linear_v.weight      |        0xd437bc0 |          0x80000 |
|  583 | audio_encoder.encoders.33.self_attn.linear_q.bias        |        0xd4b7bc0 |            0x400 |
|  584 | audio_encoder.encoders.33.self_attn.linear_k.bias        |        0xd4b7fc0 |            0x400 |
|  585 | audio_encoder.encoders.33.self_attn.linear_v.bias        |        0xd4b83c0 |            0x400 |
|  586 | audio_encoder.encoders.33.self_attn.fsmn_block.weight    |        0xd4b87c0 |           0x2c00 |
|  587 | audio_encoder.encoders.33.feed_forward.w_1.weight        |        0xd4bb3c0 |         0x200000 |
|  588 | audio_encoder.encoders.33.feed_forward.w_1.bias          |        0xd6bb3c0 |           0x1000 |
|  589 | audio_encoder.encoders.33.feed_forward.w_2.weight        |        0xd6bc3c0 |         0x200000 |
|  590 | audio_encoder.encoders.33.feed_forward.w_2.bias          |        0xd8bc3c0 |            0x400 |
|  591 | audio_encoder.encoders.33.norm1.weight                   |        0xd8bc7c0 |            0x400 |
|  592 | audio_encoder.encoders.33.norm1.bias                     |        0xd8bcbc0 |            0x400 |
|  593 | audio_encoder.encoders.33.norm2.weight                   |        0xd8bcfc0 |            0x400 |
|  594 | audio_encoder.encoders.33.norm2.bias                     |        0xd8bd3c0 |            0x400 |
|  595 | audio_encoder.encoders.34.self_attn.linear_out.weight    |        0xd8bd7c0 |          0x80000 |
|  596 | audio_encoder.encoders.34.self_attn.linear_out.bias      |        0xd93d7c0 |            0x400 |
|  597 | audio_encoder.encoders.34.self_attn.linear_q.weight      |        0xd93dbc0 |          0x80000 |
|  598 | audio_encoder.encoders.34.self_attn.linear_k.weight      |        0xd9bdbc0 |          0x80000 |
|  599 | audio_encoder.encoders.34.self_attn.linear_v.weight      |        0xda3dbc0 |          0x80000 |
|  600 | audio_encoder.encoders.34.self_attn.linear_q.bias        |        0xdabdbc0 |            0x400 |
|  601 | audio_encoder.encoders.34.self_attn.linear_k.bias        |        0xdabdfc0 |            0x400 |
|  602 | audio_encoder.encoders.34.self_attn.linear_v.bias        |        0xdabe3c0 |            0x400 |
|  603 | audio_encoder.encoders.34.self_attn.fsmn_block.weight    |        0xdabe7c0 |           0x2c00 |
|  604 | audio_encoder.encoders.34.feed_forward.w_1.weight        |        0xdac13c0 |         0x200000 |
|  605 | audio_encoder.encoders.34.feed_forward.w_1.bias          |        0xdcc13c0 |           0x1000 |
|  606 | audio_encoder.encoders.34.feed_forward.w_2.weight        |        0xdcc23c0 |         0x200000 |
|  607 | audio_encoder.encoders.34.feed_forward.w_2.bias          |        0xdec23c0 |            0x400 |
|  608 | audio_encoder.encoders.34.norm1.weight                   |        0xdec27c0 |            0x400 |
|  609 | audio_encoder.encoders.34.norm1.bias                     |        0xdec2bc0 |            0x400 |
|  610 | audio_encoder.encoders.34.norm2.weight                   |        0xdec2fc0 |            0x400 |
|  611 | audio_encoder.encoders.34.norm2.bias                     |        0xdec33c0 |            0x400 |
|  612 | audio_encoder.encoders.35.self_attn.linear_out.weight    |        0xdec37c0 |          0x80000 |
|  613 | audio_encoder.encoders.35.self_attn.linear_out.bias      |        0xdf437c0 |            0x400 |
|  614 | audio_encoder.encoders.35.self_attn.linear_q.weight      |        0xdf43bc0 |          0x80000 |
|  615 | audio_encoder.encoders.35.self_attn.linear_k.weight      |        0xdfc3bc0 |          0x80000 |
|  616 | audio_encoder.encoders.35.self_attn.linear_v.weight      |        0xe043bc0 |          0x80000 |
|  617 | audio_encoder.encoders.35.self_attn.linear_q.bias        |        0xe0c3bc0 |            0x400 |
|  618 | audio_encoder.encoders.35.self_attn.linear_k.bias        |        0xe0c3fc0 |            0x400 |
|  619 | audio_encoder.encoders.35.self_attn.linear_v.bias        |        0xe0c43c0 |            0x400 |
|  620 | audio_encoder.encoders.35.self_attn.fsmn_block.weight    |        0xe0c47c0 |           0x2c00 |
|  621 | audio_encoder.encoders.35.feed_forward.w_1.weight        |        0xe0c73c0 |         0x200000 |
|  622 | audio_encoder.encoders.35.feed_forward.w_1.bias          |        0xe2c73c0 |           0x1000 |
|  623 | audio_encoder.encoders.35.feed_forward.w_2.weight        |        0xe2c83c0 |         0x200000 |
|  624 | audio_encoder.encoders.35.feed_forward.w_2.bias          |        0xe4c83c0 |            0x400 |
|  625 | audio_encoder.encoders.35.norm1.weight                   |        0xe4c87c0 |            0x400 |
|  626 | audio_encoder.encoders.35.norm1.bias                     |        0xe4c8bc0 |            0x400 |
|  627 | audio_encoder.encoders.35.norm2.weight                   |        0xe4c8fc0 |            0x400 |
|  628 | audio_encoder.encoders.35.norm2.bias                     |        0xe4c93c0 |            0x400 |
|  629 | audio_encoder.encoders.36.self_attn.linear_out.weight    |        0xe4c97c0 |          0x80000 |
|  630 | audio_encoder.encoders.36.self_attn.linear_out.bias      |        0xe5497c0 |            0x400 |
|  631 | audio_encoder.encoders.36.self_attn.linear_q.weight      |        0xe549bc0 |          0x80000 |
|  632 | audio_encoder.encoders.36.self_attn.linear_k.weight      |        0xe5c9bc0 |          0x80000 |
|  633 | audio_encoder.encoders.36.self_attn.linear_v.weight      |        0xe649bc0 |          0x80000 |
|  634 | audio_encoder.encoders.36.self_attn.linear_q.bias        |        0xe6c9bc0 |            0x400 |
|  635 | audio_encoder.encoders.36.self_attn.linear_k.bias        |        0xe6c9fc0 |            0x400 |
|  636 | audio_encoder.encoders.36.self_attn.linear_v.bias        |        0xe6ca3c0 |            0x400 |
|  637 | audio_encoder.encoders.36.self_attn.fsmn_block.weight    |        0xe6ca7c0 |           0x2c00 |
|  638 | audio_encoder.encoders.36.feed_forward.w_1.weight        |        0xe6cd3c0 |         0x200000 |
|  639 | audio_encoder.encoders.36.feed_forward.w_1.bias          |        0xe8cd3c0 |           0x1000 |
|  640 | audio_encoder.encoders.36.feed_forward.w_2.weight        |        0xe8ce3c0 |         0x200000 |
|  641 | audio_encoder.encoders.36.feed_forward.w_2.bias          |        0xeace3c0 |            0x400 |
|  642 | audio_encoder.encoders.36.norm1.weight                   |        0xeace7c0 |            0x400 |
|  643 | audio_encoder.encoders.36.norm1.bias                     |        0xeacebc0 |            0x400 |
|  644 | audio_encoder.encoders.36.norm2.weight                   |        0xeacefc0 |            0x400 |
|  645 | audio_encoder.encoders.36.norm2.bias                     |        0xeacf3c0 |            0x400 |
|  646 | audio_encoder.encoders.37.self_attn.linear_out.weight    |        0xeacf7c0 |          0x80000 |
|  647 | audio_encoder.encoders.37.self_attn.linear_out.bias      |        0xeb4f7c0 |            0x400 |
|  648 | audio_encoder.encoders.37.self_attn.linear_q.weight      |        0xeb4fbc0 |          0x80000 |
|  649 | audio_encoder.encoders.37.self_attn.linear_k.weight      |        0xebcfbc0 |          0x80000 |
|  650 | audio_encoder.encoders.37.self_attn.linear_v.weight      |        0xec4fbc0 |          0x80000 |
|  651 | audio_encoder.encoders.37.self_attn.linear_q.bias        |        0xeccfbc0 |            0x400 |
|  652 | audio_encoder.encoders.37.self_attn.linear_k.bias        |        0xeccffc0 |            0x400 |
|  653 | audio_encoder.encoders.37.self_attn.linear_v.bias        |        0xecd03c0 |            0x400 |
|  654 | audio_encoder.encoders.37.self_attn.fsmn_block.weight    |        0xecd07c0 |           0x2c00 |
|  655 | audio_encoder.encoders.37.feed_forward.w_1.weight        |        0xecd33c0 |         0x200000 |
|  656 | audio_encoder.encoders.37.feed_forward.w_1.bias          |        0xeed33c0 |           0x1000 |
|  657 | audio_encoder.encoders.37.feed_forward.w_2.weight        |        0xeed43c0 |         0x200000 |
|  658 | audio_encoder.encoders.37.feed_forward.w_2.bias          |        0xf0d43c0 |            0x400 |
|  659 | audio_encoder.encoders.37.norm1.weight                   |        0xf0d47c0 |            0x400 |
|  660 | audio_encoder.encoders.37.norm1.bias                     |        0xf0d4bc0 |            0x400 |
|  661 | audio_encoder.encoders.37.norm2.weight                   |        0xf0d4fc0 |            0x400 |
|  662 | audio_encoder.encoders.37.norm2.bias                     |        0xf0d53c0 |            0x400 |
|  663 | audio_encoder.encoders.38.self_attn.linear_out.weight    |        0xf0d57c0 |          0x80000 |
|  664 | audio_encoder.encoders.38.self_attn.linear_out.bias      |        0xf1557c0 |            0x400 |
|  665 | audio_encoder.encoders.38.self_attn.linear_q.weight      |        0xf155bc0 |          0x80000 |
|  666 | audio_encoder.encoders.38.self_attn.linear_k.weight      |        0xf1d5bc0 |          0x80000 |
|  667 | audio_encoder.encoders.38.self_attn.linear_v.weight      |        0xf255bc0 |          0x80000 |
|  668 | audio_encoder.encoders.38.self_attn.linear_q.bias        |        0xf2d5bc0 |            0x400 |
|  669 | audio_encoder.encoders.38.self_attn.linear_k.bias        |        0xf2d5fc0 |            0x400 |
|  670 | audio_encoder.encoders.38.self_attn.linear_v.bias        |        0xf2d63c0 |            0x400 |
|  671 | audio_encoder.encoders.38.self_attn.fsmn_block.weight    |        0xf2d67c0 |           0x2c00 |
|  672 | audio_encoder.encoders.38.feed_forward.w_1.weight        |        0xf2d93c0 |         0x200000 |
|  673 | audio_encoder.encoders.38.feed_forward.w_1.bias          |        0xf4d93c0 |           0x1000 |
|  674 | audio_encoder.encoders.38.feed_forward.w_2.weight        |        0xf4da3c0 |         0x200000 |
|  675 | audio_encoder.encoders.38.feed_forward.w_2.bias          |        0xf6da3c0 |            0x400 |
|  676 | audio_encoder.encoders.38.norm1.weight                   |        0xf6da7c0 |            0x400 |
|  677 | audio_encoder.encoders.38.norm1.bias                     |        0xf6dabc0 |            0x400 |
|  678 | audio_encoder.encoders.38.norm2.weight                   |        0xf6dafc0 |            0x400 |
|  679 | audio_encoder.encoders.38.norm2.bias                     |        0xf6db3c0 |            0x400 |
|  680 | audio_encoder.encoders.39.self_attn.linear_out.weight    |        0xf6db7c0 |          0x80000 |
|  681 | audio_encoder.encoders.39.self_attn.linear_out.bias      |        0xf75b7c0 |            0x400 |
|  682 | audio_encoder.encoders.39.self_attn.linear_q.weight      |        0xf75bbc0 |          0x80000 |
|  683 | audio_encoder.encoders.39.self_attn.linear_k.weight      |        0xf7dbbc0 |          0x80000 |
|  684 | audio_encoder.encoders.39.self_attn.linear_v.weight      |        0xf85bbc0 |          0x80000 |
|  685 | audio_encoder.encoders.39.self_attn.linear_q.bias        |        0xf8dbbc0 |            0x400 |
|  686 | audio_encoder.encoders.39.self_attn.linear_k.bias        |        0xf8dbfc0 |            0x400 |
|  687 | audio_encoder.encoders.39.self_attn.linear_v.bias        |        0xf8dc3c0 |            0x400 |
|  688 | audio_encoder.encoders.39.self_attn.fsmn_block.weight    |        0xf8dc7c0 |           0x2c00 |
|  689 | audio_encoder.encoders.39.feed_forward.w_1.weight        |        0xf8df3c0 |         0x200000 |
|  690 | audio_encoder.encoders.39.feed_forward.w_1.bias          |        0xfadf3c0 |           0x1000 |
|  691 | audio_encoder.encoders.39.feed_forward.w_2.weight        |        0xfae03c0 |         0x200000 |
|  692 | audio_encoder.encoders.39.feed_forward.w_2.bias          |        0xfce03c0 |            0x400 |
|  693 | audio_encoder.encoders.39.norm1.weight                   |        0xfce07c0 |            0x400 |
|  694 | audio_encoder.encoders.39.norm1.bias                     |        0xfce0bc0 |            0x400 |
|  695 | audio_encoder.encoders.39.norm2.weight                   |        0xfce0fc0 |            0x400 |
|  696 | audio_encoder.encoders.39.norm2.bias                     |        0xfce13c0 |            0x400 |
|  697 | audio_encoder.encoders.40.self_attn.linear_out.weight    |        0xfce17c0 |          0x80000 |
|  698 | audio_encoder.encoders.40.self_attn.linear_out.bias      |        0xfd617c0 |            0x400 |
|  699 | audio_encoder.encoders.40.self_attn.linear_q.weight      |        0xfd61bc0 |          0x80000 |
|  700 | audio_encoder.encoders.40.self_attn.linear_k.weight      |        0xfde1bc0 |          0x80000 |
|  701 | audio_encoder.encoders.40.self_attn.linear_v.weight      |        0xfe61bc0 |          0x80000 |
|  702 | audio_encoder.encoders.40.self_attn.linear_q.bias        |        0xfee1bc0 |            0x400 |
|  703 | audio_encoder.encoders.40.self_attn.linear_k.bias        |        0xfee1fc0 |            0x400 |
|  704 | audio_encoder.encoders.40.self_attn.linear_v.bias        |        0xfee23c0 |            0x400 |
|  705 | audio_encoder.encoders.40.self_attn.fsmn_block.weight    |        0xfee27c0 |           0x2c00 |
|  706 | audio_encoder.encoders.40.feed_forward.w_1.weight        |        0xfee53c0 |         0x200000 |
|  707 | audio_encoder.encoders.40.feed_forward.w_1.bias          |       0x100e53c0 |           0x1000 |
|  708 | audio_encoder.encoders.40.feed_forward.w_2.weight        |       0x100e63c0 |         0x200000 |
|  709 | audio_encoder.encoders.40.feed_forward.w_2.bias          |       0x102e63c0 |            0x400 |
|  710 | audio_encoder.encoders.40.norm1.weight                   |       0x102e67c0 |            0x400 |
|  711 | audio_encoder.encoders.40.norm1.bias                     |       0x102e6bc0 |            0x400 |
|  712 | audio_encoder.encoders.40.norm2.weight                   |       0x102e6fc0 |            0x400 |
|  713 | audio_encoder.encoders.40.norm2.bias                     |       0x102e73c0 |            0x400 |
|  714 | audio_encoder.encoders.41.self_attn.linear_out.weight    |       0x102e77c0 |          0x80000 |
|  715 | audio_encoder.encoders.41.self_attn.linear_out.bias      |       0x103677c0 |            0x400 |
|  716 | audio_encoder.encoders.41.self_attn.linear_q.weight      |       0x10367bc0 |          0x80000 |
|  717 | audio_encoder.encoders.41.self_attn.linear_k.weight      |       0x103e7bc0 |          0x80000 |
|  718 | audio_encoder.encoders.41.self_attn.linear_v.weight      |       0x10467bc0 |          0x80000 |
|  719 | audio_encoder.encoders.41.self_attn.linear_q.bias        |       0x104e7bc0 |            0x400 |
|  720 | audio_encoder.encoders.41.self_attn.linear_k.bias        |       0x104e7fc0 |            0x400 |
|  721 | audio_encoder.encoders.41.self_attn.linear_v.bias        |       0x104e83c0 |            0x400 |
|  722 | audio_encoder.encoders.41.self_attn.fsmn_block.weight    |       0x104e87c0 |           0x2c00 |
|  723 | audio_encoder.encoders.41.feed_forward.w_1.weight        |       0x104eb3c0 |         0x200000 |
|  724 | audio_encoder.encoders.41.feed_forward.w_1.bias          |       0x106eb3c0 |           0x1000 |
|  725 | audio_encoder.encoders.41.feed_forward.w_2.weight        |       0x106ec3c0 |         0x200000 |
|  726 | audio_encoder.encoders.41.feed_forward.w_2.bias          |       0x108ec3c0 |            0x400 |
|  727 | audio_encoder.encoders.41.norm1.weight                   |       0x108ec7c0 |            0x400 |
|  728 | audio_encoder.encoders.41.norm1.bias                     |       0x108ecbc0 |            0x400 |
|  729 | audio_encoder.encoders.41.norm2.weight                   |       0x108ecfc0 |            0x400 |
|  730 | audio_encoder.encoders.41.norm2.bias                     |       0x108ed3c0 |            0x400 |
|  731 | audio_encoder.encoders.42.self_attn.linear_out.weight    |       0x108ed7c0 |          0x80000 |
|  732 | audio_encoder.encoders.42.self_attn.linear_out.bias      |       0x1096d7c0 |            0x400 |
|  733 | audio_encoder.encoders.42.self_attn.linear_q.weight      |       0x1096dbc0 |          0x80000 |
|  734 | audio_encoder.encoders.42.self_attn.linear_k.weight      |       0x109edbc0 |          0x80000 |
|  735 | audio_encoder.encoders.42.self_attn.linear_v.weight      |       0x10a6dbc0 |          0x80000 |
|  736 | audio_encoder.encoders.42.self_attn.linear_q.bias        |       0x10aedbc0 |            0x400 |
|  737 | audio_encoder.encoders.42.self_attn.linear_k.bias        |       0x10aedfc0 |            0x400 |
|  738 | audio_encoder.encoders.42.self_attn.linear_v.bias        |       0x10aee3c0 |            0x400 |
|  739 | audio_encoder.encoders.42.self_attn.fsmn_block.weight    |       0x10aee7c0 |           0x2c00 |
|  740 | audio_encoder.encoders.42.feed_forward.w_1.weight        |       0x10af13c0 |         0x200000 |
|  741 | audio_encoder.encoders.42.feed_forward.w_1.bias          |       0x10cf13c0 |           0x1000 |
|  742 | audio_encoder.encoders.42.feed_forward.w_2.weight        |       0x10cf23c0 |         0x200000 |
|  743 | audio_encoder.encoders.42.feed_forward.w_2.bias          |       0x10ef23c0 |            0x400 |
|  744 | audio_encoder.encoders.42.norm1.weight                   |       0x10ef27c0 |            0x400 |
|  745 | audio_encoder.encoders.42.norm1.bias                     |       0x10ef2bc0 |            0x400 |
|  746 | audio_encoder.encoders.42.norm2.weight                   |       0x10ef2fc0 |            0x400 |
|  747 | audio_encoder.encoders.42.norm2.bias                     |       0x10ef33c0 |            0x400 |
|  748 | audio_encoder.encoders.43.self_attn.linear_out.weight    |       0x10ef37c0 |          0x80000 |
|  749 | audio_encoder.encoders.43.self_attn.linear_out.bias      |       0x10f737c0 |            0x400 |
|  750 | audio_encoder.encoders.43.self_attn.linear_q.weight      |       0x10f73bc0 |          0x80000 |
|  751 | audio_encoder.encoders.43.self_attn.linear_k.weight      |       0x10ff3bc0 |          0x80000 |
|  752 | audio_encoder.encoders.43.self_attn.linear_v.weight      |       0x11073bc0 |          0x80000 |
|  753 | audio_encoder.encoders.43.self_attn.linear_q.bias        |       0x110f3bc0 |            0x400 |
|  754 | audio_encoder.encoders.43.self_attn.linear_k.bias        |       0x110f3fc0 |            0x400 |
|  755 | audio_encoder.encoders.43.self_attn.linear_v.bias        |       0x110f43c0 |            0x400 |
|  756 | audio_encoder.encoders.43.self_attn.fsmn_block.weight    |       0x110f47c0 |           0x2c00 |
|  757 | audio_encoder.encoders.43.feed_forward.w_1.weight        |       0x110f73c0 |         0x200000 |
|  758 | audio_encoder.encoders.43.feed_forward.w_1.bias          |       0x112f73c0 |           0x1000 |
|  759 | audio_encoder.encoders.43.feed_forward.w_2.weight        |       0x112f83c0 |         0x200000 |
|  760 | audio_encoder.encoders.43.feed_forward.w_2.bias          |       0x114f83c0 |            0x400 |
|  761 | audio_encoder.encoders.43.norm1.weight                   |       0x114f87c0 |            0x400 |
|  762 | audio_encoder.encoders.43.norm1.bias                     |       0x114f8bc0 |            0x400 |
|  763 | audio_encoder.encoders.43.norm2.weight                   |       0x114f8fc0 |            0x400 |
|  764 | audio_encoder.encoders.43.norm2.bias                     |       0x114f93c0 |            0x400 |
|  765 | audio_encoder.encoders.44.self_attn.linear_out.weight    |       0x114f97c0 |          0x80000 |
|  766 | audio_encoder.encoders.44.self_attn.linear_out.bias      |       0x115797c0 |            0x400 |
|  767 | audio_encoder.encoders.44.self_attn.linear_q.weight      |       0x11579bc0 |          0x80000 |
|  768 | audio_encoder.encoders.44.self_attn.linear_k.weight      |       0x115f9bc0 |          0x80000 |
|  769 | audio_encoder.encoders.44.self_attn.linear_v.weight      |       0x11679bc0 |          0x80000 |
|  770 | audio_encoder.encoders.44.self_attn.linear_q.bias        |       0x116f9bc0 |            0x400 |
|  771 | audio_encoder.encoders.44.self_attn.linear_k.bias        |       0x116f9fc0 |            0x400 |
|  772 | audio_encoder.encoders.44.self_attn.linear_v.bias        |       0x116fa3c0 |            0x400 |
|  773 | audio_encoder.encoders.44.self_attn.fsmn_block.weight    |       0x116fa7c0 |           0x2c00 |
|  774 | audio_encoder.encoders.44.feed_forward.w_1.weight        |       0x116fd3c0 |         0x200000 |
|  775 | audio_encoder.encoders.44.feed_forward.w_1.bias          |       0x118fd3c0 |           0x1000 |
|  776 | audio_encoder.encoders.44.feed_forward.w_2.weight        |       0x118fe3c0 |         0x200000 |
|  777 | audio_encoder.encoders.44.feed_forward.w_2.bias          |       0x11afe3c0 |            0x400 |
|  778 | audio_encoder.encoders.44.norm1.weight                   |       0x11afe7c0 |            0x400 |
|  779 | audio_encoder.encoders.44.norm1.bias                     |       0x11afebc0 |            0x400 |
|  780 | audio_encoder.encoders.44.norm2.weight                   |       0x11afefc0 |            0x400 |
|  781 | audio_encoder.encoders.44.norm2.bias                     |       0x11aff3c0 |            0x400 |
|  782 | audio_encoder.encoders.45.self_attn.linear_out.weight    |       0x11aff7c0 |          0x80000 |
|  783 | audio_encoder.encoders.45.self_attn.linear_out.bias      |       0x11b7f7c0 |            0x400 |
|  784 | audio_encoder.encoders.45.self_attn.linear_q.weight      |       0x11b7fbc0 |          0x80000 |
|  785 | audio_encoder.encoders.45.self_attn.linear_k.weight      |       0x11bffbc0 |          0x80000 |
|  786 | audio_encoder.encoders.45.self_attn.linear_v.weight      |       0x11c7fbc0 |          0x80000 |
|  787 | audio_encoder.encoders.45.self_attn.linear_q.bias        |       0x11cffbc0 |            0x400 |
|  788 | audio_encoder.encoders.45.self_attn.linear_k.bias        |       0x11cfffc0 |            0x400 |
|  789 | audio_encoder.encoders.45.self_attn.linear_v.bias        |       0x11d003c0 |            0x400 |
|  790 | audio_encoder.encoders.45.self_attn.fsmn_block.weight    |       0x11d007c0 |           0x2c00 |
|  791 | audio_encoder.encoders.45.feed_forward.w_1.weight        |       0x11d033c0 |         0x200000 |
|  792 | audio_encoder.encoders.45.feed_forward.w_1.bias          |       0x11f033c0 |           0x1000 |
|  793 | audio_encoder.encoders.45.feed_forward.w_2.weight        |       0x11f043c0 |         0x200000 |
|  794 | audio_encoder.encoders.45.feed_forward.w_2.bias          |       0x121043c0 |            0x400 |
|  795 | audio_encoder.encoders.45.norm1.weight                   |       0x121047c0 |            0x400 |
|  796 | audio_encoder.encoders.45.norm1.bias                     |       0x12104bc0 |            0x400 |
|  797 | audio_encoder.encoders.45.norm2.weight                   |       0x12104fc0 |            0x400 |
|  798 | audio_encoder.encoders.45.norm2.bias                     |       0x121053c0 |            0x400 |
|  799 | audio_encoder.encoders.46.self_attn.linear_out.weight    |       0x121057c0 |          0x80000 |
|  800 | audio_encoder.encoders.46.self_attn.linear_out.bias      |       0x121857c0 |            0x400 |
|  801 | audio_encoder.encoders.46.self_attn.linear_q.weight      |       0x12185bc0 |          0x80000 |
|  802 | audio_encoder.encoders.46.self_attn.linear_k.weight      |       0x12205bc0 |          0x80000 |
|  803 | audio_encoder.encoders.46.self_attn.linear_v.weight      |       0x12285bc0 |          0x80000 |
|  804 | audio_encoder.encoders.46.self_attn.linear_q.bias        |       0x12305bc0 |            0x400 |
|  805 | audio_encoder.encoders.46.self_attn.linear_k.bias        |       0x12305fc0 |            0x400 |
|  806 | audio_encoder.encoders.46.self_attn.linear_v.bias        |       0x123063c0 |            0x400 |
|  807 | audio_encoder.encoders.46.self_attn.fsmn_block.weight    |       0x123067c0 |           0x2c00 |
|  808 | audio_encoder.encoders.46.feed_forward.w_1.weight        |       0x123093c0 |         0x200000 |
|  809 | audio_encoder.encoders.46.feed_forward.w_1.bias          |       0x125093c0 |           0x1000 |
|  810 | audio_encoder.encoders.46.feed_forward.w_2.weight        |       0x1250a3c0 |         0x200000 |
|  811 | audio_encoder.encoders.46.feed_forward.w_2.bias          |       0x1270a3c0 |            0x400 |
|  812 | audio_encoder.encoders.46.norm1.weight                   |       0x1270a7c0 |            0x400 |
|  813 | audio_encoder.encoders.46.norm1.bias                     |       0x1270abc0 |            0x400 |
|  814 | audio_encoder.encoders.46.norm2.weight                   |       0x1270afc0 |            0x400 |
|  815 | audio_encoder.encoders.46.norm2.bias                     |       0x1270b3c0 |            0x400 |
|  816 | audio_encoder.encoders.47.self_attn.linear_out.weight    |       0x1270b7c0 |          0x80000 |
|  817 | audio_encoder.encoders.47.self_attn.linear_out.bias      |       0x1278b7c0 |            0x400 |
|  818 | audio_encoder.encoders.47.self_attn.linear_q.weight      |       0x1278bbc0 |          0x80000 |
|  819 | audio_encoder.encoders.47.self_attn.linear_k.weight      |       0x1280bbc0 |          0x80000 |
|  820 | audio_encoder.encoders.47.self_attn.linear_v.weight      |       0x1288bbc0 |          0x80000 |
|  821 | audio_encoder.encoders.47.self_attn.linear_q.bias        |       0x1290bbc0 |            0x400 |
|  822 | audio_encoder.encoders.47.self_attn.linear_k.bias        |       0x1290bfc0 |            0x400 |
|  823 | audio_encoder.encoders.47.self_attn.linear_v.bias        |       0x1290c3c0 |            0x400 |
|  824 | audio_encoder.encoders.47.self_attn.fsmn_block.weight    |       0x1290c7c0 |           0x2c00 |
|  825 | audio_encoder.encoders.47.feed_forward.w_1.weight        |       0x1290f3c0 |         0x200000 |
|  826 | audio_encoder.encoders.47.feed_forward.w_1.bias          |       0x12b0f3c0 |           0x1000 |
|  827 | audio_encoder.encoders.47.feed_forward.w_2.weight        |       0x12b103c0 |         0x200000 |
|  828 | audio_encoder.encoders.47.feed_forward.w_2.bias          |       0x12d103c0 |            0x400 |
|  829 | audio_encoder.encoders.47.norm1.weight                   |       0x12d107c0 |            0x400 |
|  830 | audio_encoder.encoders.47.norm1.bias                     |       0x12d10bc0 |            0x400 |
|  831 | audio_encoder.encoders.47.norm2.weight                   |       0x12d10fc0 |            0x400 |
|  832 | audio_encoder.encoders.47.norm2.bias                     |       0x12d113c0 |            0x400 |
|  833 | audio_encoder.encoders.48.self_attn.linear_out.weight    |       0x12d117c0 |          0x80000 |
|  834 | audio_encoder.encoders.48.self_attn.linear_out.bias      |       0x12d917c0 |            0x400 |
|  835 | audio_encoder.encoders.48.self_attn.linear_q.weight      |       0x12d91bc0 |          0x80000 |
|  836 | audio_encoder.encoders.48.self_attn.linear_k.weight      |       0x12e11bc0 |          0x80000 |
|  837 | audio_encoder.encoders.48.self_attn.linear_v.weight      |       0x12e91bc0 |          0x80000 |
|  838 | audio_encoder.encoders.48.self_attn.linear_q.bias        |       0x12f11bc0 |            0x400 |
|  839 | audio_encoder.encoders.48.self_attn.linear_k.bias        |       0x12f11fc0 |            0x400 |
|  840 | audio_encoder.encoders.48.self_attn.linear_v.bias        |       0x12f123c0 |            0x400 |
|  841 | audio_encoder.encoders.48.self_attn.fsmn_block.weight    |       0x12f127c0 |           0x2c00 |
|  842 | audio_encoder.encoders.48.feed_forward.w_1.weight        |       0x12f153c0 |         0x200000 |
|  843 | audio_encoder.encoders.48.feed_forward.w_1.bias          |       0x131153c0 |           0x1000 |
|  844 | audio_encoder.encoders.48.feed_forward.w_2.weight        |       0x131163c0 |         0x200000 |
|  845 | audio_encoder.encoders.48.feed_forward.w_2.bias          |       0x133163c0 |            0x400 |
|  846 | audio_encoder.encoders.48.norm1.weight                   |       0x133167c0 |            0x400 |
|  847 | audio_encoder.encoders.48.norm1.bias                     |       0x13316bc0 |            0x400 |
|  848 | audio_encoder.encoders.48.norm2.weight                   |       0x13316fc0 |            0x400 |
|  849 | audio_encoder.encoders.48.norm2.bias                     |       0x133173c0 |            0x400 |
|  850 | audio_encoder.tp_encoders.0.self_attn.linear_out.weight  |       0x133177c0 |          0x80000 |
|  851 | audio_encoder.tp_encoders.0.self_attn.linear_out.bias    |       0x133977c0 |            0x400 |
|  852 | audio_encoder.tp_encoders.0.self_attn.linear_q.weight    |       0x13397bc0 |          0x80000 |
|  853 | audio_encoder.tp_encoders.0.self_attn.linear_k.weight    |       0x13417bc0 |          0x80000 |
|  854 | audio_encoder.tp_encoders.0.self_attn.linear_v.weight    |       0x13497bc0 |          0x80000 |
|  855 | audio_encoder.tp_encoders.0.self_attn.linear_q.bias      |       0x13517bc0 |            0x400 |
|  856 | audio_encoder.tp_encoders.0.self_attn.linear_k.bias      |       0x13517fc0 |            0x400 |
|  857 | audio_encoder.tp_encoders.0.self_attn.linear_v.bias      |       0x135183c0 |            0x400 |
|  858 | audio_encoder.tp_encoders.0.self_attn.fsmn_block.weight  |       0x135187c0 |           0x2c00 |
|  859 | audio_encoder.tp_encoders.0.feed_forward.w_1.weight      |       0x1351b3c0 |         0x200000 |
|  860 | audio_encoder.tp_encoders.0.feed_forward.w_1.bias        |       0x1371b3c0 |           0x1000 |
|  861 | audio_encoder.tp_encoders.0.feed_forward.w_2.weight      |       0x1371c3c0 |         0x200000 |
|  862 | audio_encoder.tp_encoders.0.feed_forward.w_2.bias        |       0x1391c3c0 |            0x400 |
|  863 | audio_encoder.tp_encoders.0.norm1.weight                 |       0x1391c7c0 |            0x400 |
|  864 | audio_encoder.tp_encoders.0.norm1.bias                   |       0x1391cbc0 |            0x400 |
|  865 | audio_encoder.tp_encoders.0.norm2.weight                 |       0x1391cfc0 |            0x400 |
|  866 | audio_encoder.tp_encoders.0.norm2.bias                   |       0x1391d3c0 |            0x400 |
|  867 | audio_encoder.tp_encoders.1.self_attn.linear_out.weight  |       0x1391d7c0 |          0x80000 |
|  868 | audio_encoder.tp_encoders.1.self_attn.linear_out.bias    |       0x1399d7c0 |            0x400 |
|  869 | audio_encoder.tp_encoders.1.self_attn.linear_q.weight    |       0x1399dbc0 |          0x80000 |
|  870 | audio_encoder.tp_encoders.1.self_attn.linear_k.weight    |       0x13a1dbc0 |          0x80000 |
|  871 | audio_encoder.tp_encoders.1.self_attn.linear_v.weight    |       0x13a9dbc0 |          0x80000 |
|  872 | audio_encoder.tp_encoders.1.self_attn.linear_q.bias      |       0x13b1dbc0 |            0x400 |
|  873 | audio_encoder.tp_encoders.1.self_attn.linear_k.bias      |       0x13b1dfc0 |            0x400 |
|  874 | audio_encoder.tp_encoders.1.self_attn.linear_v.bias      |       0x13b1e3c0 |            0x400 |
|  875 | audio_encoder.tp_encoders.1.self_attn.fsmn_block.weight  |       0x13b1e7c0 |           0x2c00 |
|  876 | audio_encoder.tp_encoders.1.feed_forward.w_1.weight      |       0x13b213c0 |         0x200000 |
|  877 | audio_encoder.tp_encoders.1.feed_forward.w_1.bias        |       0x13d213c0 |           0x1000 |
|  878 | audio_encoder.tp_encoders.1.feed_forward.w_2.weight      |       0x13d223c0 |         0x200000 |
|  879 | audio_encoder.tp_encoders.1.feed_forward.w_2.bias        |       0x13f223c0 |            0x400 |
|  880 | audio_encoder.tp_encoders.1.norm1.weight                 |       0x13f227c0 |            0x400 |
|  881 | audio_encoder.tp_encoders.1.norm1.bias                   |       0x13f22bc0 |            0x400 |
|  882 | audio_encoder.tp_encoders.1.norm2.weight                 |       0x13f22fc0 |            0x400 |
|  883 | audio_encoder.tp_encoders.1.norm2.bias                   |       0x13f233c0 |            0x400 |
|  884 | audio_encoder.tp_encoders.2.self_attn.linear_out.weight  |       0x13f237c0 |          0x80000 |
|  885 | audio_encoder.tp_encoders.2.self_attn.linear_out.bias    |       0x13fa37c0 |            0x400 |
|  886 | audio_encoder.tp_encoders.2.self_attn.linear_q.weight    |       0x13fa3bc0 |          0x80000 |
|  887 | audio_encoder.tp_encoders.2.self_attn.linear_k.weight    |       0x14023bc0 |          0x80000 |
|  888 | audio_encoder.tp_encoders.2.self_attn.linear_v.weight    |       0x140a3bc0 |          0x80000 |
|  889 | audio_encoder.tp_encoders.2.self_attn.linear_q.bias      |       0x14123bc0 |            0x400 |
|  890 | audio_encoder.tp_encoders.2.self_attn.linear_k.bias      |       0x14123fc0 |            0x400 |
|  891 | audio_encoder.tp_encoders.2.self_attn.linear_v.bias      |       0x141243c0 |            0x400 |
|  892 | audio_encoder.tp_encoders.2.self_attn.fsmn_block.weight  |       0x141247c0 |           0x2c00 |
|  893 | audio_encoder.tp_encoders.2.feed_forward.w_1.weight      |       0x141273c0 |         0x200000 |
|  894 | audio_encoder.tp_encoders.2.feed_forward.w_1.bias        |       0x143273c0 |           0x1000 |
|  895 | audio_encoder.tp_encoders.2.feed_forward.w_2.weight      |       0x143283c0 |         0x200000 |
|  896 | audio_encoder.tp_encoders.2.feed_forward.w_2.bias        |       0x145283c0 |            0x400 |
|  897 | audio_encoder.tp_encoders.2.norm1.weight                 |       0x145287c0 |            0x400 |
|  898 | audio_encoder.tp_encoders.2.norm1.bias                   |       0x14528bc0 |            0x400 |
|  899 | audio_encoder.tp_encoders.2.norm2.weight                 |       0x14528fc0 |            0x400 |
|  900 | audio_encoder.tp_encoders.2.norm2.bias                   |       0x145293c0 |            0x400 |
|  901 | audio_encoder.tp_encoders.3.self_attn.linear_out.weight  |       0x145297c0 |          0x80000 |
|  902 | audio_encoder.tp_encoders.3.self_attn.linear_out.bias    |       0x145a97c0 |            0x400 |
|  903 | audio_encoder.tp_encoders.3.self_attn.linear_q.weight    |       0x145a9bc0 |          0x80000 |
|  904 | audio_encoder.tp_encoders.3.self_attn.linear_k.weight    |       0x14629bc0 |          0x80000 |
|  905 | audio_encoder.tp_encoders.3.self_attn.linear_v.weight    |       0x146a9bc0 |          0x80000 |
|  906 | audio_encoder.tp_encoders.3.self_attn.linear_q.bias      |       0x14729bc0 |            0x400 |
|  907 | audio_encoder.tp_encoders.3.self_attn.linear_k.bias      |       0x14729fc0 |            0x400 |
|  908 | audio_encoder.tp_encoders.3.self_attn.linear_v.bias      |       0x1472a3c0 |            0x400 |
|  909 | audio_encoder.tp_encoders.3.self_attn.fsmn_block.weight  |       0x1472a7c0 |           0x2c00 |
|  910 | audio_encoder.tp_encoders.3.feed_forward.w_1.weight      |       0x1472d3c0 |         0x200000 |
|  911 | audio_encoder.tp_encoders.3.feed_forward.w_1.bias        |       0x1492d3c0 |           0x1000 |
|  912 | audio_encoder.tp_encoders.3.feed_forward.w_2.weight      |       0x1492e3c0 |         0x200000 |
|  913 | audio_encoder.tp_encoders.3.feed_forward.w_2.bias        |       0x14b2e3c0 |            0x400 |
|  914 | audio_encoder.tp_encoders.3.norm1.weight                 |       0x14b2e7c0 |            0x400 |
|  915 | audio_encoder.tp_encoders.3.norm1.bias                   |       0x14b2ebc0 |            0x400 |
|  916 | audio_encoder.tp_encoders.3.norm2.weight                 |       0x14b2efc0 |            0x400 |
|  917 | audio_encoder.tp_encoders.3.norm2.bias                   |       0x14b2f3c0 |            0x400 |
|  918 | audio_encoder.tp_encoders.4.self_attn.linear_out.weight  |       0x14b2f7c0 |          0x80000 |
|  919 | audio_encoder.tp_encoders.4.self_attn.linear_out.bias    |       0x14baf7c0 |            0x400 |
|  920 | audio_encoder.tp_encoders.4.self_attn.linear_q.weight    |       0x14bafbc0 |          0x80000 |
|  921 | audio_encoder.tp_encoders.4.self_attn.linear_k.weight    |       0x14c2fbc0 |          0x80000 |
|  922 | audio_encoder.tp_encoders.4.self_attn.linear_v.weight    |       0x14cafbc0 |          0x80000 |
|  923 | audio_encoder.tp_encoders.4.self_attn.linear_q.bias      |       0x14d2fbc0 |            0x400 |
|  924 | audio_encoder.tp_encoders.4.self_attn.linear_k.bias      |       0x14d2ffc0 |            0x400 |
|  925 | audio_encoder.tp_encoders.4.self_attn.linear_v.bias      |       0x14d303c0 |            0x400 |
|  926 | audio_encoder.tp_encoders.4.self_attn.fsmn_block.weight  |       0x14d307c0 |           0x2c00 |
|  927 | audio_encoder.tp_encoders.4.feed_forward.w_1.weight      |       0x14d333c0 |         0x200000 |
|  928 | audio_encoder.tp_encoders.4.feed_forward.w_1.bias        |       0x14f333c0 |           0x1000 |
|  929 | audio_encoder.tp_encoders.4.feed_forward.w_2.weight      |       0x14f343c0 |         0x200000 |
|  930 | audio_encoder.tp_encoders.4.feed_forward.w_2.bias        |       0x151343c0 |            0x400 |
|  931 | audio_encoder.tp_encoders.4.norm1.weight                 |       0x151347c0 |            0x400 |
|  932 | audio_encoder.tp_encoders.4.norm1.bias                   |       0x15134bc0 |            0x400 |
|  933 | audio_encoder.tp_encoders.4.norm2.weight                 |       0x15134fc0 |            0x400 |
|  934 | audio_encoder.tp_encoders.4.norm2.bias                   |       0x151353c0 |            0x400 |
|  935 | audio_encoder.tp_encoders.5.self_attn.linear_out.weight  |       0x151357c0 |          0x80000 |
|  936 | audio_encoder.tp_encoders.5.self_attn.linear_out.bias    |       0x151b57c0 |            0x400 |
|  937 | audio_encoder.tp_encoders.5.self_attn.linear_q.weight    |       0x151b5bc0 |          0x80000 |
|  938 | audio_encoder.tp_encoders.5.self_attn.linear_k.weight    |       0x15235bc0 |          0x80000 |
|  939 | audio_encoder.tp_encoders.5.self_attn.linear_v.weight    |       0x152b5bc0 |          0x80000 |
|  940 | audio_encoder.tp_encoders.5.self_attn.linear_q.bias      |       0x15335bc0 |            0x400 |
|  941 | audio_encoder.tp_encoders.5.self_attn.linear_k.bias      |       0x15335fc0 |            0x400 |
|  942 | audio_encoder.tp_encoders.5.self_attn.linear_v.bias      |       0x153363c0 |            0x400 |
|  943 | audio_encoder.tp_encoders.5.self_attn.fsmn_block.weight  |       0x153367c0 |           0x2c00 |
|  944 | audio_encoder.tp_encoders.5.feed_forward.w_1.weight      |       0x153393c0 |         0x200000 |
|  945 | audio_encoder.tp_encoders.5.feed_forward.w_1.bias        |       0x155393c0 |           0x1000 |
|  946 | audio_encoder.tp_encoders.5.feed_forward.w_2.weight      |       0x1553a3c0 |         0x200000 |
|  947 | audio_encoder.tp_encoders.5.feed_forward.w_2.bias        |       0x1573a3c0 |            0x400 |
|  948 | audio_encoder.tp_encoders.5.norm1.weight                 |       0x1573a7c0 |            0x400 |
|  949 | audio_encoder.tp_encoders.5.norm1.bias                   |       0x1573abc0 |            0x400 |
|  950 | audio_encoder.tp_encoders.5.norm2.weight                 |       0x1573afc0 |            0x400 |
|  951 | audio_encoder.tp_encoders.5.norm2.bias                   |       0x1573b3c0 |            0x400 |
|  952 | audio_encoder.tp_encoders.6.self_attn.linear_out.weight  |       0x1573b7c0 |          0x80000 |
|  953 | audio_encoder.tp_encoders.6.self_attn.linear_out.bias    |       0x157bb7c0 |            0x400 |
|  954 | audio_encoder.tp_encoders.6.self_attn.linear_q.weight    |       0x157bbbc0 |          0x80000 |
|  955 | audio_encoder.tp_encoders.6.self_attn.linear_k.weight    |       0x1583bbc0 |          0x80000 |
|  956 | audio_encoder.tp_encoders.6.self_attn.linear_v.weight    |       0x158bbbc0 |          0x80000 |
|  957 | audio_encoder.tp_encoders.6.self_attn.linear_q.bias      |       0x1593bbc0 |            0x400 |
|  958 | audio_encoder.tp_encoders.6.self_attn.linear_k.bias      |       0x1593bfc0 |            0x400 |
|  959 | audio_encoder.tp_encoders.6.self_attn.linear_v.bias      |       0x1593c3c0 |            0x400 |
|  960 | audio_encoder.tp_encoders.6.self_attn.fsmn_block.weight  |       0x1593c7c0 |           0x2c00 |
|  961 | audio_encoder.tp_encoders.6.feed_forward.w_1.weight      |       0x1593f3c0 |         0x200000 |
|  962 | audio_encoder.tp_encoders.6.feed_forward.w_1.bias        |       0x15b3f3c0 |           0x1000 |
|  963 | audio_encoder.tp_encoders.6.feed_forward.w_2.weight      |       0x15b403c0 |         0x200000 |
|  964 | audio_encoder.tp_encoders.6.feed_forward.w_2.bias        |       0x15d403c0 |            0x400 |
|  965 | audio_encoder.tp_encoders.6.norm1.weight                 |       0x15d407c0 |            0x400 |
|  966 | audio_encoder.tp_encoders.6.norm1.bias                   |       0x15d40bc0 |            0x400 |
|  967 | audio_encoder.tp_encoders.6.norm2.weight                 |       0x15d40fc0 |            0x400 |
|  968 | audio_encoder.tp_encoders.6.norm2.bias                   |       0x15d413c0 |            0x400 |
|  969 | audio_encoder.tp_encoders.7.self_attn.linear_out.weight  |       0x15d417c0 |          0x80000 |
|  970 | audio_encoder.tp_encoders.7.self_attn.linear_out.bias    |       0x15dc17c0 |            0x400 |
|  971 | audio_encoder.tp_encoders.7.self_attn.linear_q.weight    |       0x15dc1bc0 |          0x80000 |
|  972 | audio_encoder.tp_encoders.7.self_attn.linear_k.weight    |       0x15e41bc0 |          0x80000 |
|  973 | audio_encoder.tp_encoders.7.self_attn.linear_v.weight    |       0x15ec1bc0 |          0x80000 |
|  974 | audio_encoder.tp_encoders.7.self_attn.linear_q.bias      |       0x15f41bc0 |            0x400 |
|  975 | audio_encoder.tp_encoders.7.self_attn.linear_k.bias      |       0x15f41fc0 |            0x400 |
|  976 | audio_encoder.tp_encoders.7.self_attn.linear_v.bias      |       0x15f423c0 |            0x400 |
|  977 | audio_encoder.tp_encoders.7.self_attn.fsmn_block.weight  |       0x15f427c0 |           0x2c00 |
|  978 | audio_encoder.tp_encoders.7.feed_forward.w_1.weight      |       0x15f453c0 |         0x200000 |
|  979 | audio_encoder.tp_encoders.7.feed_forward.w_1.bias        |       0x161453c0 |           0x1000 |
|  980 | audio_encoder.tp_encoders.7.feed_forward.w_2.weight      |       0x161463c0 |         0x200000 |
|  981 | audio_encoder.tp_encoders.7.feed_forward.w_2.bias        |       0x163463c0 |            0x400 |
|  982 | audio_encoder.tp_encoders.7.norm1.weight                 |       0x163467c0 |            0x400 |
|  983 | audio_encoder.tp_encoders.7.norm1.bias                   |       0x16346bc0 |            0x400 |
|  984 | audio_encoder.tp_encoders.7.norm2.weight                 |       0x16346fc0 |            0x400 |
|  985 | audio_encoder.tp_encoders.7.norm2.bias                   |       0x163473c0 |            0x400 |
|  986 | audio_encoder.tp_encoders.8.self_attn.linear_out.weight  |       0x163477c0 |          0x80000 |
|  987 | audio_encoder.tp_encoders.8.self_attn.linear_out.bias    |       0x163c77c0 |            0x400 |
|  988 | audio_encoder.tp_encoders.8.self_attn.linear_q.weight    |       0x163c7bc0 |          0x80000 |
|  989 | audio_encoder.tp_encoders.8.self_attn.linear_k.weight    |       0x16447bc0 |          0x80000 |
|  990 | audio_encoder.tp_encoders.8.self_attn.linear_v.weight    |       0x164c7bc0 |          0x80000 |
|  991 | audio_encoder.tp_encoders.8.self_attn.linear_q.bias      |       0x16547bc0 |            0x400 |
|  992 | audio_encoder.tp_encoders.8.self_attn.linear_k.bias      |       0x16547fc0 |            0x400 |
|  993 | audio_encoder.tp_encoders.8.self_attn.linear_v.bias      |       0x165483c0 |            0x400 |
|  994 | audio_encoder.tp_encoders.8.self_attn.fsmn_block.weight  |       0x165487c0 |           0x2c00 |
|  995 | audio_encoder.tp_encoders.8.feed_forward.w_1.weight      |       0x1654b3c0 |         0x200000 |
|  996 | audio_encoder.tp_encoders.8.feed_forward.w_1.bias        |       0x1674b3c0 |           0x1000 |
|  997 | audio_encoder.tp_encoders.8.feed_forward.w_2.weight      |       0x1674c3c0 |         0x200000 |
|  998 | audio_encoder.tp_encoders.8.feed_forward.w_2.bias        |       0x1694c3c0 |            0x400 |
|  999 | audio_encoder.tp_encoders.8.norm1.weight                 |       0x1694c7c0 |            0x400 |
| 1000 | audio_encoder.tp_encoders.8.norm1.bias                   |       0x1694cbc0 |            0x400 |
| 1001 | audio_encoder.tp_encoders.8.norm2.weight                 |       0x1694cfc0 |            0x400 |
| 1002 | audio_encoder.tp_encoders.8.norm2.bias                   |       0x1694d3c0 |            0x400 |
| 1003 | audio_encoder.tp_encoders.9.self_attn.linear_out.weight  |       0x1694d7c0 |          0x80000 |
| 1004 | audio_encoder.tp_encoders.9.self_attn.linear_out.bias    |       0x169cd7c0 |            0x400 |
| 1005 | audio_encoder.tp_encoders.9.self_attn.linear_q.weight    |       0x169cdbc0 |          0x80000 |
| 1006 | audio_encoder.tp_encoders.9.self_attn.linear_k.weight    |       0x16a4dbc0 |          0x80000 |
| 1007 | audio_encoder.tp_encoders.9.self_attn.linear_v.weight    |       0x16acdbc0 |          0x80000 |
| 1008 | audio_encoder.tp_encoders.9.self_attn.linear_q.bias      |       0x16b4dbc0 |            0x400 |
| 1009 | audio_encoder.tp_encoders.9.self_attn.linear_k.bias      |       0x16b4dfc0 |            0x400 |
| 1010 | audio_encoder.tp_encoders.9.self_attn.linear_v.bias      |       0x16b4e3c0 |            0x400 |
| 1011 | audio_encoder.tp_encoders.9.self_attn.fsmn_block.weight  |       0x16b4e7c0 |           0x2c00 |
| 1012 | audio_encoder.tp_encoders.9.feed_forward.w_1.weight      |       0x16b513c0 |         0x200000 |
| 1013 | audio_encoder.tp_encoders.9.feed_forward.w_1.bias        |       0x16d513c0 |           0x1000 |
| 1014 | audio_encoder.tp_encoders.9.feed_forward.w_2.weight      |       0x16d523c0 |         0x200000 |
| 1015 | audio_encoder.tp_encoders.9.feed_forward.w_2.bias        |       0x16f523c0 |            0x400 |
| 1016 | audio_encoder.tp_encoders.9.norm1.weight                 |       0x16f527c0 |            0x400 |
| 1017 | audio_encoder.tp_encoders.9.norm1.bias                   |       0x16f52bc0 |            0x400 |
| 1018 | audio_encoder.tp_encoders.9.norm2.weight                 |       0x16f52fc0 |            0x400 |
| 1019 | audio_encoder.tp_encoders.9.norm2.bias                   |       0x16f533c0 |            0x400 |
| 1020 | audio_encoder.tp_encoders.10.self_attn.linear_out.weight |       0x16f537c0 |          0x80000 |
| 1021 | audio_encoder.tp_encoders.10.self_attn.linear_out.bias   |       0x16fd37c0 |            0x400 |
| 1022 | audio_encoder.tp_encoders.10.self_attn.linear_q.weight   |       0x16fd3bc0 |          0x80000 |
| 1023 | audio_encoder.tp_encoders.10.self_attn.linear_k.weight   |       0x17053bc0 |          0x80000 |
| 1024 | audio_encoder.tp_encoders.10.self_attn.linear_v.weight   |       0x170d3bc0 |          0x80000 |
| 1025 | audio_encoder.tp_encoders.10.self_attn.linear_q.bias     |       0x17153bc0 |            0x400 |
| 1026 | audio_encoder.tp_encoders.10.self_attn.linear_k.bias     |       0x17153fc0 |            0x400 |
| 1027 | audio_encoder.tp_encoders.10.self_attn.linear_v.bias     |       0x171543c0 |            0x400 |
| 1028 | audio_encoder.tp_encoders.10.self_attn.fsmn_block.weight |       0x171547c0 |           0x2c00 |
| 1029 | audio_encoder.tp_encoders.10.feed_forward.w_1.weight     |       0x171573c0 |         0x200000 |
| 1030 | audio_encoder.tp_encoders.10.feed_forward.w_1.bias       |       0x173573c0 |           0x1000 |
| 1031 | audio_encoder.tp_encoders.10.feed_forward.w_2.weight     |       0x173583c0 |         0x200000 |
| 1032 | audio_encoder.tp_encoders.10.feed_forward.w_2.bias       |       0x175583c0 |            0x400 |
| 1033 | audio_encoder.tp_encoders.10.norm1.weight                |       0x175587c0 |            0x400 |
| 1034 | audio_encoder.tp_encoders.10.norm1.bias                  |       0x17558bc0 |            0x400 |
| 1035 | audio_encoder.tp_encoders.10.norm2.weight                |       0x17558fc0 |            0x400 |
| 1036 | audio_encoder.tp_encoders.10.norm2.bias                  |       0x175593c0 |            0x400 |
| 1037 | audio_encoder.tp_encoders.11.self_attn.linear_out.weight |       0x175597c0 |          0x80000 |
| 1038 | audio_encoder.tp_encoders.11.self_attn.linear_out.bias   |       0x175d97c0 |            0x400 |
| 1039 | audio_encoder.tp_encoders.11.self_attn.linear_q.weight   |       0x175d9bc0 |          0x80000 |
| 1040 | audio_encoder.tp_encoders.11.self_attn.linear_k.weight   |       0x17659bc0 |          0x80000 |
| 1041 | audio_encoder.tp_encoders.11.self_attn.linear_v.weight   |       0x176d9bc0 |          0x80000 |
| 1042 | audio_encoder.tp_encoders.11.self_attn.linear_q.bias     |       0x17759bc0 |            0x400 |
| 1043 | audio_encoder.tp_encoders.11.self_attn.linear_k.bias     |       0x17759fc0 |            0x400 |
| 1044 | audio_encoder.tp_encoders.11.self_attn.linear_v.bias     |       0x1775a3c0 |            0x400 |
| 1045 | audio_encoder.tp_encoders.11.self_attn.fsmn_block.weight |       0x1775a7c0 |           0x2c00 |
| 1046 | audio_encoder.tp_encoders.11.feed_forward.w_1.weight     |       0x1775d3c0 |         0x200000 |
| 1047 | audio_encoder.tp_encoders.11.feed_forward.w_1.bias       |       0x1795d3c0 |           0x1000 |
| 1048 | audio_encoder.tp_encoders.11.feed_forward.w_2.weight     |       0x1795e3c0 |         0x200000 |
| 1049 | audio_encoder.tp_encoders.11.feed_forward.w_2.bias       |       0x17b5e3c0 |            0x400 |
| 1050 | audio_encoder.tp_encoders.11.norm1.weight                |       0x17b5e7c0 |            0x400 |
| 1051 | audio_encoder.tp_encoders.11.norm1.bias                  |       0x17b5ebc0 |            0x400 |
| 1052 | audio_encoder.tp_encoders.11.norm2.weight                |       0x17b5efc0 |            0x400 |
| 1053 | audio_encoder.tp_encoders.11.norm2.bias                  |       0x17b5f3c0 |            0x400 |
| 1054 | audio_encoder.tp_encoders.12.self_attn.linear_out.weight |       0x17b5f7c0 |          0x80000 |
| 1055 | audio_encoder.tp_encoders.12.self_attn.linear_out.bias   |       0x17bdf7c0 |            0x400 |
| 1056 | audio_encoder.tp_encoders.12.self_attn.linear_q.weight   |       0x17bdfbc0 |          0x80000 |
| 1057 | audio_encoder.tp_encoders.12.self_attn.linear_k.weight   |       0x17c5fbc0 |          0x80000 |
| 1058 | audio_encoder.tp_encoders.12.self_attn.linear_v.weight   |       0x17cdfbc0 |          0x80000 |
| 1059 | audio_encoder.tp_encoders.12.self_attn.linear_q.bias     |       0x17d5fbc0 |            0x400 |
| 1060 | audio_encoder.tp_encoders.12.self_attn.linear_k.bias     |       0x17d5ffc0 |            0x400 |
| 1061 | audio_encoder.tp_encoders.12.self_attn.linear_v.bias     |       0x17d603c0 |            0x400 |
| 1062 | audio_encoder.tp_encoders.12.self_attn.fsmn_block.weight |       0x17d607c0 |           0x2c00 |
| 1063 | audio_encoder.tp_encoders.12.feed_forward.w_1.weight     |       0x17d633c0 |         0x200000 |
| 1064 | audio_encoder.tp_encoders.12.feed_forward.w_1.bias       |       0x17f633c0 |           0x1000 |
| 1065 | audio_encoder.tp_encoders.12.feed_forward.w_2.weight     |       0x17f643c0 |         0x200000 |
| 1066 | audio_encoder.tp_encoders.12.feed_forward.w_2.bias       |       0x181643c0 |            0x400 |
| 1067 | audio_encoder.tp_encoders.12.norm1.weight                |       0x181647c0 |            0x400 |
| 1068 | audio_encoder.tp_encoders.12.norm1.bias                  |       0x18164bc0 |            0x400 |
| 1069 | audio_encoder.tp_encoders.12.norm2.weight                |       0x18164fc0 |            0x400 |
| 1070 | audio_encoder.tp_encoders.12.norm2.bias                  |       0x181653c0 |            0x400 |
| 1071 | audio_encoder.tp_encoders.13.self_attn.linear_out.weight |       0x181657c0 |          0x80000 |
| 1072 | audio_encoder.tp_encoders.13.self_attn.linear_out.bias   |       0x181e57c0 |            0x400 |
| 1073 | audio_encoder.tp_encoders.13.self_attn.linear_q.weight   |       0x181e5bc0 |          0x80000 |
| 1074 | audio_encoder.tp_encoders.13.self_attn.linear_k.weight   |       0x18265bc0 |          0x80000 |
| 1075 | audio_encoder.tp_encoders.13.self_attn.linear_v.weight   |       0x182e5bc0 |          0x80000 |
| 1076 | audio_encoder.tp_encoders.13.self_attn.linear_q.bias     |       0x18365bc0 |            0x400 |
| 1077 | audio_encoder.tp_encoders.13.self_attn.linear_k.bias     |       0x18365fc0 |            0x400 |
| 1078 | audio_encoder.tp_encoders.13.self_attn.linear_v.bias     |       0x183663c0 |            0x400 |
| 1079 | audio_encoder.tp_encoders.13.self_attn.fsmn_block.weight |       0x183667c0 |           0x2c00 |
| 1080 | audio_encoder.tp_encoders.13.feed_forward.w_1.weight     |       0x183693c0 |         0x200000 |
| 1081 | audio_encoder.tp_encoders.13.feed_forward.w_1.bias       |       0x185693c0 |           0x1000 |
| 1082 | audio_encoder.tp_encoders.13.feed_forward.w_2.weight     |       0x1856a3c0 |         0x200000 |
| 1083 | audio_encoder.tp_encoders.13.feed_forward.w_2.bias       |       0x1876a3c0 |            0x400 |
| 1084 | audio_encoder.tp_encoders.13.norm1.weight                |       0x1876a7c0 |            0x400 |
| 1085 | audio_encoder.tp_encoders.13.norm1.bias                  |       0x1876abc0 |            0x400 |
| 1086 | audio_encoder.tp_encoders.13.norm2.weight                |       0x1876afc0 |            0x400 |
| 1087 | audio_encoder.tp_encoders.13.norm2.bias                  |       0x1876b3c0 |            0x400 |
| 1088 | audio_encoder.tp_encoders.14.self_attn.linear_out.weight |       0x1876b7c0 |          0x80000 |
| 1089 | audio_encoder.tp_encoders.14.self_attn.linear_out.bias   |       0x187eb7c0 |            0x400 |
| 1090 | audio_encoder.tp_encoders.14.self_attn.linear_q.weight   |       0x187ebbc0 |          0x80000 |
| 1091 | audio_encoder.tp_encoders.14.self_attn.linear_k.weight   |       0x1886bbc0 |          0x80000 |
| 1092 | audio_encoder.tp_encoders.14.self_attn.linear_v.weight   |       0x188ebbc0 |          0x80000 |
| 1093 | audio_encoder.tp_encoders.14.self_attn.linear_q.bias     |       0x1896bbc0 |            0x400 |
| 1094 | audio_encoder.tp_encoders.14.self_attn.linear_k.bias     |       0x1896bfc0 |            0x400 |
| 1095 | audio_encoder.tp_encoders.14.self_attn.linear_v.bias     |       0x1896c3c0 |            0x400 |
| 1096 | audio_encoder.tp_encoders.14.self_attn.fsmn_block.weight |       0x1896c7c0 |           0x2c00 |
| 1097 | audio_encoder.tp_encoders.14.feed_forward.w_1.weight     |       0x1896f3c0 |         0x200000 |
| 1098 | audio_encoder.tp_encoders.14.feed_forward.w_1.bias       |       0x18b6f3c0 |           0x1000 |
| 1099 | audio_encoder.tp_encoders.14.feed_forward.w_2.weight     |       0x18b703c0 |         0x200000 |
| 1100 | audio_encoder.tp_encoders.14.feed_forward.w_2.bias       |       0x18d703c0 |            0x400 |
| 1101 | audio_encoder.tp_encoders.14.norm1.weight                |       0x18d707c0 |            0x400 |
| 1102 | audio_encoder.tp_encoders.14.norm1.bias                  |       0x18d70bc0 |            0x400 |
| 1103 | audio_encoder.tp_encoders.14.norm2.weight                |       0x18d70fc0 |            0x400 |
| 1104 | audio_encoder.tp_encoders.14.norm2.bias                  |       0x18d713c0 |            0x400 |
| 1105 | audio_encoder.tp_encoders.15.self_attn.linear_out.weight |       0x18d717c0 |          0x80000 |
| 1106 | audio_encoder.tp_encoders.15.self_attn.linear_out.bias   |       0x18df17c0 |            0x400 |
| 1107 | audio_encoder.tp_encoders.15.self_attn.linear_q.weight   |       0x18df1bc0 |          0x80000 |
| 1108 | audio_encoder.tp_encoders.15.self_attn.linear_k.weight   |       0x18e71bc0 |          0x80000 |
| 1109 | audio_encoder.tp_encoders.15.self_attn.linear_v.weight   |       0x18ef1bc0 |          0x80000 |
| 1110 | audio_encoder.tp_encoders.15.self_attn.linear_q.bias     |       0x18f71bc0 |            0x400 |
| 1111 | audio_encoder.tp_encoders.15.self_attn.linear_k.bias     |       0x18f71fc0 |            0x400 |
| 1112 | audio_encoder.tp_encoders.15.self_attn.linear_v.bias     |       0x18f723c0 |            0x400 |
| 1113 | audio_encoder.tp_encoders.15.self_attn.fsmn_block.weight |       0x18f727c0 |           0x2c00 |
| 1114 | audio_encoder.tp_encoders.15.feed_forward.w_1.weight     |       0x18f753c0 |         0x200000 |
| 1115 | audio_encoder.tp_encoders.15.feed_forward.w_1.bias       |       0x191753c0 |           0x1000 |
| 1116 | audio_encoder.tp_encoders.15.feed_forward.w_2.weight     |       0x191763c0 |         0x200000 |
| 1117 | audio_encoder.tp_encoders.15.feed_forward.w_2.bias       |       0x193763c0 |            0x400 |
| 1118 | audio_encoder.tp_encoders.15.norm1.weight                |       0x193767c0 |            0x400 |
| 1119 | audio_encoder.tp_encoders.15.norm1.bias                  |       0x19376bc0 |            0x400 |
| 1120 | audio_encoder.tp_encoders.15.norm2.weight                |       0x19376fc0 |            0x400 |
| 1121 | audio_encoder.tp_encoders.15.norm2.bias                  |       0x193773c0 |            0x400 |
| 1122 | audio_encoder.tp_encoders.16.self_attn.linear_out.weight |       0x193777c0 |          0x80000 |
| 1123 | audio_encoder.tp_encoders.16.self_attn.linear_out.bias   |       0x193f77c0 |            0x400 |
| 1124 | audio_encoder.tp_encoders.16.self_attn.linear_q.weight   |       0x193f7bc0 |          0x80000 |
| 1125 | audio_encoder.tp_encoders.16.self_attn.linear_k.weight   |       0x19477bc0 |          0x80000 |
| 1126 | audio_encoder.tp_encoders.16.self_attn.linear_v.weight   |       0x194f7bc0 |          0x80000 |
| 1127 | audio_encoder.tp_encoders.16.self_attn.linear_q.bias     |       0x19577bc0 |            0x400 |
| 1128 | audio_encoder.tp_encoders.16.self_attn.linear_k.bias     |       0x19577fc0 |            0x400 |
| 1129 | audio_encoder.tp_encoders.16.self_attn.linear_v.bias     |       0x195783c0 |            0x400 |
| 1130 | audio_encoder.tp_encoders.16.self_attn.fsmn_block.weight |       0x195787c0 |           0x2c00 |
| 1131 | audio_encoder.tp_encoders.16.feed_forward.w_1.weight     |       0x1957b3c0 |         0x200000 |
| 1132 | audio_encoder.tp_encoders.16.feed_forward.w_1.bias       |       0x1977b3c0 |           0x1000 |
| 1133 | audio_encoder.tp_encoders.16.feed_forward.w_2.weight     |       0x1977c3c0 |         0x200000 |
| 1134 | audio_encoder.tp_encoders.16.feed_forward.w_2.bias       |       0x1997c3c0 |            0x400 |
| 1135 | audio_encoder.tp_encoders.16.norm1.weight                |       0x1997c7c0 |            0x400 |
| 1136 | audio_encoder.tp_encoders.16.norm1.bias                  |       0x1997cbc0 |            0x400 |
| 1137 | audio_encoder.tp_encoders.16.norm2.weight                |       0x1997cfc0 |            0x400 |
| 1138 | audio_encoder.tp_encoders.16.norm2.bias                  |       0x1997d3c0 |            0x400 |
| 1139 | audio_encoder.tp_encoders.17.self_attn.linear_out.weight |       0x1997d7c0 |          0x80000 |
| 1140 | audio_encoder.tp_encoders.17.self_attn.linear_out.bias   |       0x199fd7c0 |            0x400 |
| 1141 | audio_encoder.tp_encoders.17.self_attn.linear_q.weight   |       0x199fdbc0 |          0x80000 |
| 1142 | audio_encoder.tp_encoders.17.self_attn.linear_k.weight   |       0x19a7dbc0 |          0x80000 |
| 1143 | audio_encoder.tp_encoders.17.self_attn.linear_v.weight   |       0x19afdbc0 |          0x80000 |
| 1144 | audio_encoder.tp_encoders.17.self_attn.linear_q.bias     |       0x19b7dbc0 |            0x400 |
| 1145 | audio_encoder.tp_encoders.17.self_attn.linear_k.bias     |       0x19b7dfc0 |            0x400 |
| 1146 | audio_encoder.tp_encoders.17.self_attn.linear_v.bias     |       0x19b7e3c0 |            0x400 |
| 1147 | audio_encoder.tp_encoders.17.self_attn.fsmn_block.weight |       0x19b7e7c0 |           0x2c00 |
| 1148 | audio_encoder.tp_encoders.17.feed_forward.w_1.weight     |       0x19b813c0 |         0x200000 |
| 1149 | audio_encoder.tp_encoders.17.feed_forward.w_1.bias       |       0x19d813c0 |           0x1000 |
| 1150 | audio_encoder.tp_encoders.17.feed_forward.w_2.weight     |       0x19d823c0 |         0x200000 |
| 1151 | audio_encoder.tp_encoders.17.feed_forward.w_2.bias       |       0x19f823c0 |            0x400 |
| 1152 | audio_encoder.tp_encoders.17.norm1.weight                |       0x19f827c0 |            0x400 |
| 1153 | audio_encoder.tp_encoders.17.norm1.bias                  |       0x19f82bc0 |            0x400 |
| 1154 | audio_encoder.tp_encoders.17.norm2.weight                |       0x19f82fc0 |            0x400 |
| 1155 | audio_encoder.tp_encoders.17.norm2.bias                  |       0x19f833c0 |            0x400 |
| 1156 | audio_encoder.tp_encoders.18.self_attn.linear_out.weight |       0x19f837c0 |          0x80000 |
| 1157 | audio_encoder.tp_encoders.18.self_attn.linear_out.bias   |       0x1a0037c0 |            0x400 |
| 1158 | audio_encoder.tp_encoders.18.self_attn.linear_q.weight   |       0x1a003bc0 |          0x80000 |
| 1159 | audio_encoder.tp_encoders.18.self_attn.linear_k.weight   |       0x1a083bc0 |          0x80000 |
| 1160 | audio_encoder.tp_encoders.18.self_attn.linear_v.weight   |       0x1a103bc0 |          0x80000 |
| 1161 | audio_encoder.tp_encoders.18.self_attn.linear_q.bias     |       0x1a183bc0 |            0x400 |
| 1162 | audio_encoder.tp_encoders.18.self_attn.linear_k.bias     |       0x1a183fc0 |            0x400 |
| 1163 | audio_encoder.tp_encoders.18.self_attn.linear_v.bias     |       0x1a1843c0 |            0x400 |
| 1164 | audio_encoder.tp_encoders.18.self_attn.fsmn_block.weight |       0x1a1847c0 |           0x2c00 |
| 1165 | audio_encoder.tp_encoders.18.feed_forward.w_1.weight     |       0x1a1873c0 |         0x200000 |
| 1166 | audio_encoder.tp_encoders.18.feed_forward.w_1.bias       |       0x1a3873c0 |           0x1000 |
| 1167 | audio_encoder.tp_encoders.18.feed_forward.w_2.weight     |       0x1a3883c0 |         0x200000 |
| 1168 | audio_encoder.tp_encoders.18.feed_forward.w_2.bias       |       0x1a5883c0 |            0x400 |
| 1169 | audio_encoder.tp_encoders.18.norm1.weight                |       0x1a5887c0 |            0x400 |
| 1170 | audio_encoder.tp_encoders.18.norm1.bias                  |       0x1a588bc0 |            0x400 |
| 1171 | audio_encoder.tp_encoders.18.norm2.weight                |       0x1a588fc0 |            0x400 |
| 1172 | audio_encoder.tp_encoders.18.norm2.bias                  |       0x1a5893c0 |            0x400 |
| 1173 | audio_encoder.tp_encoders.19.self_attn.linear_out.weight |       0x1a5897c0 |          0x80000 |
| 1174 | audio_encoder.tp_encoders.19.self_attn.linear_out.bias   |       0x1a6097c0 |            0x400 |
| 1175 | audio_encoder.tp_encoders.19.self_attn.linear_q.weight   |       0x1a609bc0 |          0x80000 |
| 1176 | audio_encoder.tp_encoders.19.self_attn.linear_k.weight   |       0x1a689bc0 |          0x80000 |
| 1177 | audio_encoder.tp_encoders.19.self_attn.linear_v.weight   |       0x1a709bc0 |          0x80000 |
| 1178 | audio_encoder.tp_encoders.19.self_attn.linear_q.bias     |       0x1a789bc0 |            0x400 |
| 1179 | audio_encoder.tp_encoders.19.self_attn.linear_k.bias     |       0x1a789fc0 |            0x400 |
| 1180 | audio_encoder.tp_encoders.19.self_attn.linear_v.bias     |       0x1a78a3c0 |            0x400 |
| 1181 | audio_encoder.tp_encoders.19.self_attn.fsmn_block.weight |       0x1a78a7c0 |           0x2c00 |
| 1182 | audio_encoder.tp_encoders.19.feed_forward.w_1.weight     |       0x1a78d3c0 |         0x200000 |
| 1183 | audio_encoder.tp_encoders.19.feed_forward.w_1.bias       |       0x1a98d3c0 |           0x1000 |
| 1184 | audio_encoder.tp_encoders.19.feed_forward.w_2.weight     |       0x1a98e3c0 |         0x200000 |
| 1185 | audio_encoder.tp_encoders.19.feed_forward.w_2.bias       |       0x1ab8e3c0 |            0x400 |
| 1186 | audio_encoder.tp_encoders.19.norm1.weight                |       0x1ab8e7c0 |            0x400 |
| 1187 | audio_encoder.tp_encoders.19.norm1.bias                  |       0x1ab8ebc0 |            0x400 |
| 1188 | audio_encoder.tp_encoders.19.norm2.weight                |       0x1ab8efc0 |            0x400 |
| 1189 | audio_encoder.tp_encoders.19.norm2.bias                  |       0x1ab8f3c0 |            0x400 |
| 1190 | audio_encoder.after_norm.weight                          |       0x1ab8f7c0 |            0x400 |
| 1191 | audio_encoder.after_norm.bias                            |       0x1ab8fbc0 |            0x400 |
| 1192 | audio_encoder.tp_norm.weight                             |       0x1ab8ffc0 |            0x400 |
| 1193 | audio_encoder.tp_norm.bias                               |       0x1ab903c0 |            0x400 |
| 1194 | audio_adaptor.linear1.weight                             |       0x1ab907c0 |         0x200000 |
| 1195 | audio_adaptor.linear1.bias                               |       0x1ad907c0 |           0x1000 |
| 1196 | audio_adaptor.linear2.weight                             |       0x1ad917c0 |         0x400000 |
| 1197 | audio_adaptor.linear2.bias                               |       0x1b1917c0 |            0x800 |
| 1198 | audio_adaptor.blocks.0.self_attn.linear_q.weight         |       0x1b191fc0 |         0x200000 |
| 1199 | audio_adaptor.blocks.0.self_attn.linear_q.bias           |       0x1b391fc0 |            0x800 |
| 1200 | audio_adaptor.blocks.0.self_attn.linear_k.weight         |       0x1b3927c0 |         0x200000 |
| 1201 | audio_adaptor.blocks.0.self_attn.linear_k.bias           |       0x1b5927c0 |            0x800 |
| 1202 | audio_adaptor.blocks.0.self_attn.linear_v.weight         |       0x1b592fc0 |         0x200000 |
| 1203 | audio_adaptor.blocks.0.self_attn.linear_v.bias           |       0x1b792fc0 |            0x800 |
| 1204 | audio_adaptor.blocks.0.self_attn.linear_out.weight       |       0x1b7937c0 |         0x200000 |
| 1205 | audio_adaptor.blocks.0.self_attn.linear_out.bias         |       0x1b9937c0 |            0x800 |
| 1206 | audio_adaptor.blocks.0.feed_forward.w_1.weight           |       0x1b993fc0 |          0x80000 |
| 1207 | audio_adaptor.blocks.0.feed_forward.w_1.bias             |       0x1ba13fc0 |            0x200 |
| 1208 | audio_adaptor.blocks.0.feed_forward.w_2.weight           |       0x1ba141c0 |          0x80000 |
| 1209 | audio_adaptor.blocks.0.feed_forward.w_2.bias             |       0x1ba941c0 |            0x800 |
| 1210 | audio_adaptor.blocks.0.norm1.weight                      |       0x1ba949c0 |            0x800 |
| 1211 | audio_adaptor.blocks.0.norm1.bias                        |       0x1ba951c0 |            0x800 |
| 1212 | audio_adaptor.blocks.0.norm2.weight                      |       0x1ba959c0 |            0x800 |
| 1213 | audio_adaptor.blocks.0.norm2.bias                        |       0x1ba961c0 |            0x800 |
| 1214 | audio_adaptor.blocks.1.self_attn.linear_q.weight         |       0x1ba969c0 |         0x200000 |
| 1215 | audio_adaptor.blocks.1.self_attn.linear_q.bias           |       0x1bc969c0 |            0x800 |
| 1216 | audio_adaptor.blocks.1.self_attn.linear_k.weight         |       0x1bc971c0 |         0x200000 |
| 1217 | audio_adaptor.blocks.1.self_attn.linear_k.bias           |       0x1be971c0 |            0x800 |
| 1218 | audio_adaptor.blocks.1.self_attn.linear_v.weight         |       0x1be979c0 |         0x200000 |
| 1219 | audio_adaptor.blocks.1.self_attn.linear_v.bias           |       0x1c0979c0 |            0x800 |
| 1220 | audio_adaptor.blocks.1.self_attn.linear_out.weight       |       0x1c0981c0 |         0x200000 |
| 1221 | audio_adaptor.blocks.1.self_attn.linear_out.bias         |       0x1c2981c0 |            0x800 |
| 1222 | audio_adaptor.blocks.1.feed_forward.w_1.weight           |       0x1c2989c0 |          0x80000 |
| 1223 | audio_adaptor.blocks.1.feed_forward.w_1.bias             |       0x1c3189c0 |            0x200 |
| 1224 | audio_adaptor.blocks.1.feed_forward.w_2.weight           |       0x1c318bc0 |          0x80000 |
| 1225 | audio_adaptor.blocks.1.feed_forward.w_2.bias             |       0x1c398bc0 |            0x800 |
| 1226 | audio_adaptor.blocks.1.norm1.weight                      |       0x1c3993c0 |            0x800 |
| 1227 | audio_adaptor.blocks.1.norm1.bias                        |       0x1c399bc0 |            0x800 |
| 1228 | audio_adaptor.blocks.1.norm2.weight                      |       0x1c39a3c0 |            0x800 |
| 1229 | audio_adaptor.blocks.1.norm2.bias                        |       0x1c39abc0 |            0x800 |
| 1230 | llm.lm_head.weight                                       |       0x1c39b3c0 |       0x128c0000 |
| 1231 | llm.model.embed_tokens.weight                            |       0x2ec5b3c0 |       0x128c0000 |
| 1232 | llm.model.layers.0.input_layernorm.weight                |       0x4151b3c0 |            0x800 |
| 1233 | llm.model.layers.0.mlp.down_proj.weight                  |       0x4151bbc0 |         0x600000 |
| 1234 | llm.model.layers.0.mlp.gate_proj.weight                  |       0x41b1bbc0 |         0x600000 |
| 1235 | llm.model.layers.0.mlp.up_proj.weight                    |       0x4211bbc0 |         0x600000 |
| 1236 | llm.model.layers.0.post_attention_layernorm.weight       |       0x4271bbc0 |            0x800 |
| 1237 | llm.model.layers.0.self_attn.k_norm.weight               |       0x4271c3c0 |            0x100 |
| 1238 | llm.model.layers.0.self_attn.k_proj.weight               |       0x4271c4c0 |         0x200000 |
| 1239 | llm.model.layers.0.self_attn.o_proj.weight               |       0x4291c4c0 |         0x400000 |
| 1240 | llm.model.layers.0.self_attn.q_norm.weight               |       0x42d1c4c0 |            0x100 |
| 1241 | llm.model.layers.0.self_attn.q_proj.weight               |       0x42d1c5c0 |         0x400000 |
| 1242 | llm.model.layers.0.self_attn.v_proj.weight               |       0x4311c5c0 |         0x200000 |
| 1243 | llm.model.layers.1.input_layernorm.weight                |       0x4331c5c0 |            0x800 |
| 1244 | llm.model.layers.1.mlp.down_proj.weight                  |       0x4331cdc0 |         0x600000 |
| 1245 | llm.model.layers.1.mlp.gate_proj.weight                  |       0x4391cdc0 |         0x600000 |
| 1246 | llm.model.layers.1.mlp.up_proj.weight                    |       0x43f1cdc0 |         0x600000 |
| 1247 | llm.model.layers.1.post_attention_layernorm.weight       |       0x4451cdc0 |            0x800 |
| 1248 | llm.model.layers.1.self_attn.k_norm.weight               |       0x4451d5c0 |            0x100 |
| 1249 | llm.model.layers.1.self_attn.k_proj.weight               |       0x4451d6c0 |         0x200000 |
| 1250 | llm.model.layers.1.self_attn.o_proj.weight               |       0x4471d6c0 |         0x400000 |
| 1251 | llm.model.layers.1.self_attn.q_norm.weight               |       0x44b1d6c0 |            0x100 |
| 1252 | llm.model.layers.1.self_attn.q_proj.weight               |       0x44b1d7c0 |         0x400000 |
| 1253 | llm.model.layers.1.self_attn.v_proj.weight               |       0x44f1d7c0 |         0x200000 |
| 1254 | llm.model.layers.10.input_layernorm.weight               |       0x4511d7c0 |            0x800 |
| 1255 | llm.model.layers.10.mlp.down_proj.weight                 |       0x4511dfc0 |         0x600000 |
| 1256 | llm.model.layers.10.mlp.gate_proj.weight                 |       0x4571dfc0 |         0x600000 |
| 1257 | llm.model.layers.10.mlp.up_proj.weight                   |       0x45d1dfc0 |         0x600000 |
| 1258 | llm.model.layers.10.post_attention_layernorm.weight      |       0x4631dfc0 |            0x800 |
| 1259 | llm.model.layers.10.self_attn.k_norm.weight              |       0x4631e7c0 |            0x100 |
| 1260 | llm.model.layers.10.self_attn.k_proj.weight              |       0x4631e8c0 |         0x200000 |
| 1261 | llm.model.layers.10.self_attn.o_proj.weight              |       0x4651e8c0 |         0x400000 |
| 1262 | llm.model.layers.10.self_attn.q_norm.weight              |       0x4691e8c0 |            0x100 |
| 1263 | llm.model.layers.10.self_attn.q_proj.weight              |       0x4691e9c0 |         0x400000 |
| 1264 | llm.model.layers.10.self_attn.v_proj.weight              |       0x46d1e9c0 |         0x200000 |
| 1265 | llm.model.layers.11.input_layernorm.weight               |       0x46f1e9c0 |            0x800 |
| 1266 | llm.model.layers.11.mlp.down_proj.weight                 |       0x46f1f1c0 |         0x600000 |
| 1267 | llm.model.layers.11.mlp.gate_proj.weight                 |       0x4751f1c0 |         0x600000 |
| 1268 | llm.model.layers.11.mlp.up_proj.weight                   |       0x47b1f1c0 |         0x600000 |
| 1269 | llm.model.layers.11.post_attention_layernorm.weight      |       0x4811f1c0 |            0x800 |
| 1270 | llm.model.layers.11.self_attn.k_norm.weight              |       0x4811f9c0 |            0x100 |
| 1271 | llm.model.layers.11.self_attn.k_proj.weight              |       0x4811fac0 |         0x200000 |
| 1272 | llm.model.layers.11.self_attn.o_proj.weight              |       0x4831fac0 |         0x400000 |
| 1273 | llm.model.layers.11.self_attn.q_norm.weight              |       0x4871fac0 |            0x100 |
| 1274 | llm.model.layers.11.self_attn.q_proj.weight              |       0x4871fbc0 |         0x400000 |
| 1275 | llm.model.layers.11.self_attn.v_proj.weight              |       0x48b1fbc0 |         0x200000 |
| 1276 | llm.model.layers.12.input_layernorm.weight               |       0x48d1fbc0 |            0x800 |
| 1277 | llm.model.layers.12.mlp.down_proj.weight                 |       0x48d203c0 |         0x600000 |
| 1278 | llm.model.layers.12.mlp.gate_proj.weight                 |       0x493203c0 |         0x600000 |
| 1279 | llm.model.layers.12.mlp.up_proj.weight                   |       0x499203c0 |         0x600000 |
| 1280 | llm.model.layers.12.post_attention_layernorm.weight      |       0x49f203c0 |            0x800 |
| 1281 | llm.model.layers.12.self_attn.k_norm.weight              |       0x49f20bc0 |            0x100 |
| 1282 | llm.model.layers.12.self_attn.k_proj.weight              |       0x49f20cc0 |         0x200000 |
| 1283 | llm.model.layers.12.self_attn.o_proj.weight              |       0x4a120cc0 |         0x400000 |
| 1284 | llm.model.layers.12.self_attn.q_norm.weight              |       0x4a520cc0 |            0x100 |
| 1285 | llm.model.layers.12.self_attn.q_proj.weight              |       0x4a520dc0 |         0x400000 |
| 1286 | llm.model.layers.12.self_attn.v_proj.weight              |       0x4a920dc0 |         0x200000 |
| 1287 | llm.model.layers.13.input_layernorm.weight               |       0x4ab20dc0 |            0x800 |
| 1288 | llm.model.layers.13.mlp.down_proj.weight                 |       0x4ab215c0 |         0x600000 |
| 1289 | llm.model.layers.13.mlp.gate_proj.weight                 |       0x4b1215c0 |         0x600000 |
| 1290 | llm.model.layers.13.mlp.up_proj.weight                   |       0x4b7215c0 |         0x600000 |
| 1291 | llm.model.layers.13.post_attention_layernorm.weight      |       0x4bd215c0 |            0x800 |
| 1292 | llm.model.layers.13.self_attn.k_norm.weight              |       0x4bd21dc0 |            0x100 |
| 1293 | llm.model.layers.13.self_attn.k_proj.weight              |       0x4bd21ec0 |         0x200000 |
| 1294 | llm.model.layers.13.self_attn.o_proj.weight              |       0x4bf21ec0 |         0x400000 |
| 1295 | llm.model.layers.13.self_attn.q_norm.weight              |       0x4c321ec0 |            0x100 |
| 1296 | llm.model.layers.13.self_attn.q_proj.weight              |       0x4c321fc0 |         0x400000 |
| 1297 | llm.model.layers.13.self_attn.v_proj.weight              |       0x4c721fc0 |         0x200000 |
| 1298 | llm.model.layers.14.input_layernorm.weight               |       0x4c921fc0 |            0x800 |
| 1299 | llm.model.layers.14.mlp.down_proj.weight                 |       0x4c9227c0 |         0x600000 |
| 1300 | llm.model.layers.14.mlp.gate_proj.weight                 |       0x4cf227c0 |         0x600000 |
| 1301 | llm.model.layers.14.mlp.up_proj.weight                   |       0x4d5227c0 |         0x600000 |
| 1302 | llm.model.layers.14.post_attention_layernorm.weight      |       0x4db227c0 |            0x800 |
| 1303 | llm.model.layers.14.self_attn.k_norm.weight              |       0x4db22fc0 |            0x100 |
| 1304 | llm.model.layers.14.self_attn.k_proj.weight              |       0x4db230c0 |         0x200000 |
| 1305 | llm.model.layers.14.self_attn.o_proj.weight              |       0x4dd230c0 |         0x400000 |
| 1306 | llm.model.layers.14.self_attn.q_norm.weight              |       0x4e1230c0 |            0x100 |
| 1307 | llm.model.layers.14.self_attn.q_proj.weight              |       0x4e1231c0 |         0x400000 |
| 1308 | llm.model.layers.14.self_attn.v_proj.weight              |       0x4e5231c0 |         0x200000 |
| 1309 | llm.model.layers.15.input_layernorm.weight               |       0x4e7231c0 |            0x800 |
| 1310 | llm.model.layers.15.mlp.down_proj.weight                 |       0x4e7239c0 |         0x600000 |
| 1311 | llm.model.layers.15.mlp.gate_proj.weight                 |       0x4ed239c0 |         0x600000 |
| 1312 | llm.model.layers.15.mlp.up_proj.weight                   |       0x4f3239c0 |         0x600000 |
| 1313 | llm.model.layers.15.post_attention_layernorm.weight      |       0x4f9239c0 |            0x800 |
| 1314 | llm.model.layers.15.self_attn.k_norm.weight              |       0x4f9241c0 |            0x100 |
| 1315 | llm.model.layers.15.self_attn.k_proj.weight              |       0x4f9242c0 |         0x200000 |
| 1316 | llm.model.layers.15.self_attn.o_proj.weight              |       0x4fb242c0 |         0x400000 |
| 1317 | llm.model.layers.15.self_attn.q_norm.weight              |       0x4ff242c0 |            0x100 |
| 1318 | llm.model.layers.15.self_attn.q_proj.weight              |       0x4ff243c0 |         0x400000 |
| 1319 | llm.model.layers.15.self_attn.v_proj.weight              |       0x503243c0 |         0x200000 |
| 1320 | llm.model.layers.16.input_layernorm.weight               |       0x505243c0 |            0x800 |
| 1321 | llm.model.layers.16.mlp.down_proj.weight                 |       0x50524bc0 |         0x600000 |
| 1322 | llm.model.layers.16.mlp.gate_proj.weight                 |       0x50b24bc0 |         0x600000 |
| 1323 | llm.model.layers.16.mlp.up_proj.weight                   |       0x51124bc0 |         0x600000 |
| 1324 | llm.model.layers.16.post_attention_layernorm.weight      |       0x51724bc0 |            0x800 |
| 1325 | llm.model.layers.16.self_attn.k_norm.weight              |       0x517253c0 |            0x100 |
| 1326 | llm.model.layers.16.self_attn.k_proj.weight              |       0x517254c0 |         0x200000 |
| 1327 | llm.model.layers.16.self_attn.o_proj.weight              |       0x519254c0 |         0x400000 |
| 1328 | llm.model.layers.16.self_attn.q_norm.weight              |       0x51d254c0 |            0x100 |
| 1329 | llm.model.layers.16.self_attn.q_proj.weight              |       0x51d255c0 |         0x400000 |
| 1330 | llm.model.layers.16.self_attn.v_proj.weight              |       0x521255c0 |         0x200000 |
| 1331 | llm.model.layers.17.input_layernorm.weight               |       0x523255c0 |            0x800 |
| 1332 | llm.model.layers.17.mlp.down_proj.weight                 |       0x52325dc0 |         0x600000 |
| 1333 | llm.model.layers.17.mlp.gate_proj.weight                 |       0x52925dc0 |         0x600000 |
| 1334 | llm.model.layers.17.mlp.up_proj.weight                   |       0x52f25dc0 |         0x600000 |
| 1335 | llm.model.layers.17.post_attention_layernorm.weight      |       0x53525dc0 |            0x800 |
| 1336 | llm.model.layers.17.self_attn.k_norm.weight              |       0x535265c0 |            0x100 |
| 1337 | llm.model.layers.17.self_attn.k_proj.weight              |       0x535266c0 |         0x200000 |
| 1338 | llm.model.layers.17.self_attn.o_proj.weight              |       0x537266c0 |         0x400000 |
| 1339 | llm.model.layers.17.self_attn.q_norm.weight              |       0x53b266c0 |            0x100 |
| 1340 | llm.model.layers.17.self_attn.q_proj.weight              |       0x53b267c0 |         0x400000 |
| 1341 | llm.model.layers.17.self_attn.v_proj.weight              |       0x53f267c0 |         0x200000 |
| 1342 | llm.model.layers.18.input_layernorm.weight               |       0x541267c0 |            0x800 |
| 1343 | llm.model.layers.18.mlp.down_proj.weight                 |       0x54126fc0 |         0x600000 |
| 1344 | llm.model.layers.18.mlp.gate_proj.weight                 |       0x54726fc0 |         0x600000 |
| 1345 | llm.model.layers.18.mlp.up_proj.weight                   |       0x54d26fc0 |         0x600000 |
| 1346 | llm.model.layers.18.post_attention_layernorm.weight      |       0x55326fc0 |            0x800 |
| 1347 | llm.model.layers.18.self_attn.k_norm.weight              |       0x553277c0 |            0x100 |
| 1348 | llm.model.layers.18.self_attn.k_proj.weight              |       0x553278c0 |         0x200000 |
| 1349 | llm.model.layers.18.self_attn.o_proj.weight              |       0x555278c0 |         0x400000 |
| 1350 | llm.model.layers.18.self_attn.q_norm.weight              |       0x559278c0 |            0x100 |
| 1351 | llm.model.layers.18.self_attn.q_proj.weight              |       0x559279c0 |         0x400000 |
| 1352 | llm.model.layers.18.self_attn.v_proj.weight              |       0x55d279c0 |         0x200000 |
| 1353 | llm.model.layers.19.input_layernorm.weight               |       0x55f279c0 |            0x800 |
| 1354 | llm.model.layers.19.mlp.down_proj.weight                 |       0x55f281c0 |         0x600000 |
| 1355 | llm.model.layers.19.mlp.gate_proj.weight                 |       0x565281c0 |         0x600000 |
| 1356 | llm.model.layers.19.mlp.up_proj.weight                   |       0x56b281c0 |         0x600000 |
| 1357 | llm.model.layers.19.post_attention_layernorm.weight      |       0x571281c0 |            0x800 |
| 1358 | llm.model.layers.19.self_attn.k_norm.weight              |       0x571289c0 |            0x100 |
| 1359 | llm.model.layers.19.self_attn.k_proj.weight              |       0x57128ac0 |         0x200000 |
| 1360 | llm.model.layers.19.self_attn.o_proj.weight              |       0x57328ac0 |         0x400000 |
| 1361 | llm.model.layers.19.self_attn.q_norm.weight              |       0x57728ac0 |            0x100 |
| 1362 | llm.model.layers.19.self_attn.q_proj.weight              |       0x57728bc0 |         0x400000 |
| 1363 | llm.model.layers.19.self_attn.v_proj.weight              |       0x57b28bc0 |         0x200000 |
| 1364 | llm.model.layers.2.input_layernorm.weight                |       0x57d28bc0 |            0x800 |
| 1365 | llm.model.layers.2.mlp.down_proj.weight                  |       0x57d293c0 |         0x600000 |
| 1366 | llm.model.layers.2.mlp.gate_proj.weight                  |       0x583293c0 |         0x600000 |
| 1367 | llm.model.layers.2.mlp.up_proj.weight                    |       0x589293c0 |         0x600000 |
| 1368 | llm.model.layers.2.post_attention_layernorm.weight       |       0x58f293c0 |            0x800 |
| 1369 | llm.model.layers.2.self_attn.k_norm.weight               |       0x58f29bc0 |            0x100 |
| 1370 | llm.model.layers.2.self_attn.k_proj.weight               |       0x58f29cc0 |         0x200000 |
| 1371 | llm.model.layers.2.self_attn.o_proj.weight               |       0x59129cc0 |         0x400000 |
| 1372 | llm.model.layers.2.self_attn.q_norm.weight               |       0x59529cc0 |            0x100 |
| 1373 | llm.model.layers.2.self_attn.q_proj.weight               |       0x59529dc0 |         0x400000 |
| 1374 | llm.model.layers.2.self_attn.v_proj.weight               |       0x59929dc0 |         0x200000 |
| 1375 | llm.model.layers.20.input_layernorm.weight               |       0x59b29dc0 |            0x800 |
| 1376 | llm.model.layers.20.mlp.down_proj.weight                 |       0x59b2a5c0 |         0x600000 |
| 1377 | llm.model.layers.20.mlp.gate_proj.weight                 |       0x5a12a5c0 |         0x600000 |
| 1378 | llm.model.layers.20.mlp.up_proj.weight                   |       0x5a72a5c0 |         0x600000 |
| 1379 | llm.model.layers.20.post_attention_layernorm.weight      |       0x5ad2a5c0 |            0x800 |
| 1380 | llm.model.layers.20.self_attn.k_norm.weight              |       0x5ad2adc0 |            0x100 |
| 1381 | llm.model.layers.20.self_attn.k_proj.weight              |       0x5ad2aec0 |         0x200000 |
| 1382 | llm.model.layers.20.self_attn.o_proj.weight              |       0x5af2aec0 |         0x400000 |
| 1383 | llm.model.layers.20.self_attn.q_norm.weight              |       0x5b32aec0 |            0x100 |
| 1384 | llm.model.layers.20.self_attn.q_proj.weight              |       0x5b32afc0 |         0x400000 |
| 1385 | llm.model.layers.20.self_attn.v_proj.weight              |       0x5b72afc0 |         0x200000 |
| 1386 | llm.model.layers.21.input_layernorm.weight               |       0x5b92afc0 |            0x800 |
| 1387 | llm.model.layers.21.mlp.down_proj.weight                 |       0x5b92b7c0 |         0x600000 |
| 1388 | llm.model.layers.21.mlp.gate_proj.weight                 |       0x5bf2b7c0 |         0x600000 |
| 1389 | llm.model.layers.21.mlp.up_proj.weight                   |       0x5c52b7c0 |         0x600000 |
| 1390 | llm.model.layers.21.post_attention_layernorm.weight      |       0x5cb2b7c0 |            0x800 |
| 1391 | llm.model.layers.21.self_attn.k_norm.weight              |       0x5cb2bfc0 |            0x100 |
| 1392 | llm.model.layers.21.self_attn.k_proj.weight              |       0x5cb2c0c0 |         0x200000 |
| 1393 | llm.model.layers.21.self_attn.o_proj.weight              |       0x5cd2c0c0 |         0x400000 |
| 1394 | llm.model.layers.21.self_attn.q_norm.weight              |       0x5d12c0c0 |            0x100 |
| 1395 | llm.model.layers.21.self_attn.q_proj.weight              |       0x5d12c1c0 |         0x400000 |
| 1396 | llm.model.layers.21.self_attn.v_proj.weight              |       0x5d52c1c0 |         0x200000 |
| 1397 | llm.model.layers.22.input_layernorm.weight               |       0x5d72c1c0 |            0x800 |
| 1398 | llm.model.layers.22.mlp.down_proj.weight                 |       0x5d72c9c0 |         0x600000 |
| 1399 | llm.model.layers.22.mlp.gate_proj.weight                 |       0x5dd2c9c0 |         0x600000 |
| 1400 | llm.model.layers.22.mlp.up_proj.weight                   |       0x5e32c9c0 |         0x600000 |
| 1401 | llm.model.layers.22.post_attention_layernorm.weight      |       0x5e92c9c0 |            0x800 |
| 1402 | llm.model.layers.22.self_attn.k_norm.weight              |       0x5e92d1c0 |            0x100 |
| 1403 | llm.model.layers.22.self_attn.k_proj.weight              |       0x5e92d2c0 |         0x200000 |
| 1404 | llm.model.layers.22.self_attn.o_proj.weight              |       0x5eb2d2c0 |         0x400000 |
| 1405 | llm.model.layers.22.self_attn.q_norm.weight              |       0x5ef2d2c0 |            0x100 |
| 1406 | llm.model.layers.22.self_attn.q_proj.weight              |       0x5ef2d3c0 |         0x400000 |
| 1407 | llm.model.layers.22.self_attn.v_proj.weight              |       0x5f32d3c0 |         0x200000 |
| 1408 | llm.model.layers.23.input_layernorm.weight               |       0x5f52d3c0 |            0x800 |
| 1409 | llm.model.layers.23.mlp.down_proj.weight                 |       0x5f52dbc0 |         0x600000 |
| 1410 | llm.model.layers.23.mlp.gate_proj.weight                 |       0x5fb2dbc0 |         0x600000 |
| 1411 | llm.model.layers.23.mlp.up_proj.weight                   |       0x6012dbc0 |         0x600000 |
| 1412 | llm.model.layers.23.post_attention_layernorm.weight      |       0x6072dbc0 |            0x800 |
| 1413 | llm.model.layers.23.self_attn.k_norm.weight              |       0x6072e3c0 |            0x100 |
| 1414 | llm.model.layers.23.self_attn.k_proj.weight              |       0x6072e4c0 |         0x200000 |
| 1415 | llm.model.layers.23.self_attn.o_proj.weight              |       0x6092e4c0 |         0x400000 |
| 1416 | llm.model.layers.23.self_attn.q_norm.weight              |       0x60d2e4c0 |            0x100 |
| 1417 | llm.model.layers.23.self_attn.q_proj.weight              |       0x60d2e5c0 |         0x400000 |
| 1418 | llm.model.layers.23.self_attn.v_proj.weight              |       0x6112e5c0 |         0x200000 |
| 1419 | llm.model.layers.24.input_layernorm.weight               |       0x6132e5c0 |            0x800 |
| 1420 | llm.model.layers.24.mlp.down_proj.weight                 |       0x6132edc0 |         0x600000 |
| 1421 | llm.model.layers.24.mlp.gate_proj.weight                 |       0x6192edc0 |         0x600000 |
| 1422 | llm.model.layers.24.mlp.up_proj.weight                   |       0x61f2edc0 |         0x600000 |
| 1423 | llm.model.layers.24.post_attention_layernorm.weight      |       0x6252edc0 |            0x800 |
| 1424 | llm.model.layers.24.self_attn.k_norm.weight              |       0x6252f5c0 |            0x100 |
| 1425 | llm.model.layers.24.self_attn.k_proj.weight              |       0x6252f6c0 |         0x200000 |
| 1426 | llm.model.layers.24.self_attn.o_proj.weight              |       0x6272f6c0 |         0x400000 |
| 1427 | llm.model.layers.24.self_attn.q_norm.weight              |       0x62b2f6c0 |            0x100 |
| 1428 | llm.model.layers.24.self_attn.q_proj.weight              |       0x62b2f7c0 |         0x400000 |
| 1429 | llm.model.layers.24.self_attn.v_proj.weight              |       0x62f2f7c0 |         0x200000 |
| 1430 | llm.model.layers.25.input_layernorm.weight               |       0x6312f7c0 |            0x800 |
| 1431 | llm.model.layers.25.mlp.down_proj.weight                 |       0x6312ffc0 |         0x600000 |
| 1432 | llm.model.layers.25.mlp.gate_proj.weight                 |       0x6372ffc0 |         0x600000 |
| 1433 | llm.model.layers.25.mlp.up_proj.weight                   |       0x63d2ffc0 |         0x600000 |
| 1434 | llm.model.layers.25.post_attention_layernorm.weight      |       0x6432ffc0 |            0x800 |
| 1435 | llm.model.layers.25.self_attn.k_norm.weight              |       0x643307c0 |            0x100 |
| 1436 | llm.model.layers.25.self_attn.k_proj.weight              |       0x643308c0 |         0x200000 |
| 1437 | llm.model.layers.25.self_attn.o_proj.weight              |       0x645308c0 |         0x400000 |
| 1438 | llm.model.layers.25.self_attn.q_norm.weight              |       0x649308c0 |            0x100 |
| 1439 | llm.model.layers.25.self_attn.q_proj.weight              |       0x649309c0 |         0x400000 |
| 1440 | llm.model.layers.25.self_attn.v_proj.weight              |       0x64d309c0 |         0x200000 |
| 1441 | llm.model.layers.26.input_layernorm.weight               |       0x64f309c0 |            0x800 |
| 1442 | llm.model.layers.26.mlp.down_proj.weight                 |       0x64f311c0 |         0x600000 |
| 1443 | llm.model.layers.26.mlp.gate_proj.weight                 |       0x655311c0 |         0x600000 |
| 1444 | llm.model.layers.26.mlp.up_proj.weight                   |       0x65b311c0 |         0x600000 |
| 1445 | llm.model.layers.26.post_attention_layernorm.weight      |       0x661311c0 |            0x800 |
| 1446 | llm.model.layers.26.self_attn.k_norm.weight              |       0x661319c0 |            0x100 |
| 1447 | llm.model.layers.26.self_attn.k_proj.weight              |       0x66131ac0 |         0x200000 |
| 1448 | llm.model.layers.26.self_attn.o_proj.weight              |       0x66331ac0 |         0x400000 |
| 1449 | llm.model.layers.26.self_attn.q_norm.weight              |       0x66731ac0 |            0x100 |
| 1450 | llm.model.layers.26.self_attn.q_proj.weight              |       0x66731bc0 |         0x400000 |
| 1451 | llm.model.layers.26.self_attn.v_proj.weight              |       0x66b31bc0 |         0x200000 |
| 1452 | llm.model.layers.27.input_layernorm.weight               |       0x66d31bc0 |            0x800 |
| 1453 | llm.model.layers.27.mlp.down_proj.weight                 |       0x66d323c0 |         0x600000 |
| 1454 | llm.model.layers.27.mlp.gate_proj.weight                 |       0x673323c0 |         0x600000 |
| 1455 | llm.model.layers.27.mlp.up_proj.weight                   |       0x679323c0 |         0x600000 |
| 1456 | llm.model.layers.27.post_attention_layernorm.weight      |       0x67f323c0 |            0x800 |
| 1457 | llm.model.layers.27.self_attn.k_norm.weight              |       0x67f32bc0 |            0x100 |
| 1458 | llm.model.layers.27.self_attn.k_proj.weight              |       0x67f32cc0 |         0x200000 |
| 1459 | llm.model.layers.27.self_attn.o_proj.weight              |       0x68132cc0 |         0x400000 |
| 1460 | llm.model.layers.27.self_attn.q_norm.weight              |       0x68532cc0 |            0x100 |
| 1461 | llm.model.layers.27.self_attn.q_proj.weight              |       0x68532dc0 |         0x400000 |
| 1462 | llm.model.layers.27.self_attn.v_proj.weight              |       0x68932dc0 |         0x200000 |
| 1463 | llm.model.layers.3.input_layernorm.weight                |       0x68b32dc0 |            0x800 |
| 1464 | llm.model.layers.3.mlp.down_proj.weight                  |       0x68b335c0 |         0x600000 |
| 1465 | llm.model.layers.3.mlp.gate_proj.weight                  |       0x691335c0 |         0x600000 |
| 1466 | llm.model.layers.3.mlp.up_proj.weight                    |       0x697335c0 |         0x600000 |
| 1467 | llm.model.layers.3.post_attention_layernorm.weight       |       0x69d335c0 |            0x800 |
| 1468 | llm.model.layers.3.self_attn.k_norm.weight               |       0x69d33dc0 |            0x100 |
| 1469 | llm.model.layers.3.self_attn.k_proj.weight               |       0x69d33ec0 |         0x200000 |
| 1470 | llm.model.layers.3.self_attn.o_proj.weight               |       0x69f33ec0 |         0x400000 |
| 1471 | llm.model.layers.3.self_attn.q_norm.weight               |       0x6a333ec0 |            0x100 |
| 1472 | llm.model.layers.3.self_attn.q_proj.weight               |       0x6a333fc0 |         0x400000 |
| 1473 | llm.model.layers.3.self_attn.v_proj.weight               |       0x6a733fc0 |         0x200000 |
| 1474 | llm.model.layers.4.input_layernorm.weight                |       0x6a933fc0 |            0x800 |
| 1475 | llm.model.layers.4.mlp.down_proj.weight                  |       0x6a9347c0 |         0x600000 |
| 1476 | llm.model.layers.4.mlp.gate_proj.weight                  |       0x6af347c0 |         0x600000 |
| 1477 | llm.model.layers.4.mlp.up_proj.weight                    |       0x6b5347c0 |         0x600000 |
| 1478 | llm.model.layers.4.post_attention_layernorm.weight       |       0x6bb347c0 |            0x800 |
| 1479 | llm.model.layers.4.self_attn.k_norm.weight               |       0x6bb34fc0 |            0x100 |
| 1480 | llm.model.layers.4.self_attn.k_proj.weight               |       0x6bb350c0 |         0x200000 |
| 1481 | llm.model.layers.4.self_attn.o_proj.weight               |       0x6bd350c0 |         0x400000 |
| 1482 | llm.model.layers.4.self_attn.q_norm.weight               |       0x6c1350c0 |            0x100 |
| 1483 | llm.model.layers.4.self_attn.q_proj.weight               |       0x6c1351c0 |         0x400000 |
| 1484 | llm.model.layers.4.self_attn.v_proj.weight               |       0x6c5351c0 |         0x200000 |
| 1485 | llm.model.layers.5.input_layernorm.weight                |       0x6c7351c0 |            0x800 |
| 1486 | llm.model.layers.5.mlp.down_proj.weight                  |       0x6c7359c0 |         0x600000 |
| 1487 | llm.model.layers.5.mlp.gate_proj.weight                  |       0x6cd359c0 |         0x600000 |
| 1488 | llm.model.layers.5.mlp.up_proj.weight                    |       0x6d3359c0 |         0x600000 |
| 1489 | llm.model.layers.5.post_attention_layernorm.weight       |       0x6d9359c0 |            0x800 |
| 1490 | llm.model.layers.5.self_attn.k_norm.weight               |       0x6d9361c0 |            0x100 |
| 1491 | llm.model.layers.5.self_attn.k_proj.weight               |       0x6d9362c0 |         0x200000 |
| 1492 | llm.model.layers.5.self_attn.o_proj.weight               |       0x6db362c0 |         0x400000 |
| 1493 | llm.model.layers.5.self_attn.q_norm.weight               |       0x6df362c0 |            0x100 |
| 1494 | llm.model.layers.5.self_attn.q_proj.weight               |       0x6df363c0 |         0x400000 |
| 1495 | llm.model.layers.5.self_attn.v_proj.weight               |       0x6e3363c0 |         0x200000 |
| 1496 | llm.model.layers.6.input_layernorm.weight                |       0x6e5363c0 |            0x800 |
| 1497 | llm.model.layers.6.mlp.down_proj.weight                  |       0x6e536bc0 |         0x600000 |
| 1498 | llm.model.layers.6.mlp.gate_proj.weight                  |       0x6eb36bc0 |         0x600000 |
| 1499 | llm.model.layers.6.mlp.up_proj.weight                    |       0x6f136bc0 |         0x600000 |
| 1500 | llm.model.layers.6.post_attention_layernorm.weight       |       0x6f736bc0 |            0x800 |
| 1501 | llm.model.layers.6.self_attn.k_norm.weight               |       0x6f7373c0 |            0x100 |
| 1502 | llm.model.layers.6.self_attn.k_proj.weight               |       0x6f7374c0 |         0x200000 |
| 1503 | llm.model.layers.6.self_attn.o_proj.weight               |       0x6f9374c0 |         0x400000 |
| 1504 | llm.model.layers.6.self_attn.q_norm.weight               |       0x6fd374c0 |            0x100 |
| 1505 | llm.model.layers.6.self_attn.q_proj.weight               |       0x6fd375c0 |         0x400000 |
| 1506 | llm.model.layers.6.self_attn.v_proj.weight               |       0x701375c0 |         0x200000 |
| 1507 | llm.model.layers.7.input_layernorm.weight                |       0x703375c0 |            0x800 |
| 1508 | llm.model.layers.7.mlp.down_proj.weight                  |       0x70337dc0 |         0x600000 |
| 1509 | llm.model.layers.7.mlp.gate_proj.weight                  |       0x70937dc0 |         0x600000 |
| 1510 | llm.model.layers.7.mlp.up_proj.weight                    |       0x70f37dc0 |         0x600000 |
| 1511 | llm.model.layers.7.post_attention_layernorm.weight       |       0x71537dc0 |            0x800 |
| 1512 | llm.model.layers.7.self_attn.k_norm.weight               |       0x715385c0 |            0x100 |
| 1513 | llm.model.layers.7.self_attn.k_proj.weight               |       0x715386c0 |         0x200000 |
| 1514 | llm.model.layers.7.self_attn.o_proj.weight               |       0x717386c0 |         0x400000 |
| 1515 | llm.model.layers.7.self_attn.q_norm.weight               |       0x71b386c0 |            0x100 |
| 1516 | llm.model.layers.7.self_attn.q_proj.weight               |       0x71b387c0 |         0x400000 |
| 1517 | llm.model.layers.7.self_attn.v_proj.weight               |       0x71f387c0 |         0x200000 |
| 1518 | llm.model.layers.8.input_layernorm.weight                |       0x721387c0 |            0x800 |
| 1519 | llm.model.layers.8.mlp.down_proj.weight                  |       0x72138fc0 |         0x600000 |
| 1520 | llm.model.layers.8.mlp.gate_proj.weight                  |       0x72738fc0 |         0x600000 |
| 1521 | llm.model.layers.8.mlp.up_proj.weight                    |       0x72d38fc0 |         0x600000 |
| 1522 | llm.model.layers.8.post_attention_layernorm.weight       |       0x73338fc0 |            0x800 |
| 1523 | llm.model.layers.8.self_attn.k_norm.weight               |       0x733397c0 |            0x100 |
| 1524 | llm.model.layers.8.self_attn.k_proj.weight               |       0x733398c0 |         0x200000 |
| 1525 | llm.model.layers.8.self_attn.o_proj.weight               |       0x735398c0 |         0x400000 |
| 1526 | llm.model.layers.8.self_attn.q_norm.weight               |       0x739398c0 |            0x100 |
| 1527 | llm.model.layers.8.self_attn.q_proj.weight               |       0x739399c0 |         0x400000 |
| 1528 | llm.model.layers.8.self_attn.v_proj.weight               |       0x73d399c0 |         0x200000 |
| 1529 | llm.model.layers.9.input_layernorm.weight                |       0x73f399c0 |            0x800 |
| 1530 | llm.model.layers.9.mlp.down_proj.weight                  |       0x73f3a1c0 |         0x600000 |
| 1531 | llm.model.layers.9.mlp.gate_proj.weight                  |       0x7453a1c0 |         0x600000 |
| 1532 | llm.model.layers.9.mlp.up_proj.weight                    |       0x74b3a1c0 |         0x600000 |
| 1533 | llm.model.layers.9.post_attention_layernorm.weight       |       0x7513a1c0 |            0x800 |
| 1534 | llm.model.layers.9.self_attn.k_norm.weight               |       0x7513a9c0 |            0x100 |
| 1535 | llm.model.layers.9.self_attn.k_proj.weight               |       0x7513aac0 |         0x200000 |
| 1536 | llm.model.layers.9.self_attn.o_proj.weight               |       0x7533aac0 |         0x400000 |
| 1537 | llm.model.layers.9.self_attn.q_norm.weight               |       0x7573aac0 |            0x100 |
| 1538 | llm.model.layers.9.self_attn.q_proj.weight               |       0x7573abc0 |         0x400000 |
| 1539 | llm.model.layers.9.self_attn.v_proj.weight               |       0x75b3abc0 |         0x200000 |
| 1540 | llm.model.norm.weight                                    |       0x75d3abc0 |            0x800 |

### <a name="base">Base Tensor Group : ~985M Elements</a>

| T_ID | Tensor Layer Name                                        | Human Friendly Tensor Layer Name                      | Elements          | Shape                   | Type |
|-----:|:---------------------------------------------------------|:------------------------------------------------------|:------------------|:------------------------|:-----|
|    0 | audio_encoder.encoders0.0.self_attn.linear_out.weight    | Audio_Encoder Encoders0 0 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|    1 | audio_encoder.encoders0.0.self_attn.linear_out.bias      | Audio_Encoder Encoders0 0 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|    2 | audio_encoder.encoders0.0.self_attn.linear_q.weight      | Audio_Encoder Encoders0 0 Self_Attn Linear_Q (W)      | (~287K)    286720 |  560 x    512 x   1 x 1 | BF16 |
|    3 | audio_encoder.encoders0.0.self_attn.linear_k.weight      | Audio_Encoder Encoders0 0 Self_Attn Linear_K (W)      | (~287K)    286720 |  560 x    512 x   1 x 1 | BF16 |
|    4 | audio_encoder.encoders0.0.self_attn.linear_v.weight      | Audio_Encoder Encoders0 0 Self_Attn Linear_V (W)      | (~287K)    286720 |  560 x    512 x   1 x 1 | BF16 |
|    5 | audio_encoder.encoders0.0.self_attn.linear_q.bias        | Audio_Encoder Encoders0 0 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|    6 | audio_encoder.encoders0.0.self_attn.linear_k.bias        | Audio_Encoder Encoders0 0 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|    7 | audio_encoder.encoders0.0.self_attn.linear_v.bias        | Audio_Encoder Encoders0 0 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|    8 | audio_encoder.encoders0.0.self_attn.fsmn_block.weight    | Audio_Encoder Encoders0 0 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|    9 | audio_encoder.encoders0.0.feed_forward.w_1.weight        | Audio_Encoder Encoders0 0 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   10 | audio_encoder.encoders0.0.feed_forward.w_1.bias          | Audio_Encoder Encoders0 0 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   11 | audio_encoder.encoders0.0.feed_forward.w_2.weight        | Audio_Encoder Encoders0 0 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   12 | audio_encoder.encoders0.0.feed_forward.w_2.bias          | Audio_Encoder Encoders0 0 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   13 | audio_encoder.encoders0.0.norm1.weight                   | Audio_Encoder Encoders0 0 Norm1 (W)                   | (  560)       560 |  560 x      1 x   1 x 1 | BF16 |
|   14 | audio_encoder.encoders0.0.norm1.bias                     | Audio_Encoder Encoders0 0 Norm1 (B)                   | (  560)       560 |  560 x      1 x   1 x 1 | BF16 |
|   15 | audio_encoder.encoders0.0.norm2.weight                   | Audio_Encoder Encoders0 0 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   16 | audio_encoder.encoders0.0.norm2.bias                     | Audio_Encoder Encoders0 0 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   17 | audio_encoder.encoders.0.self_attn.linear_out.weight     | Audio_Encoder Encoders 0 Self_Attn Linear_Out (W)     | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   18 | audio_encoder.encoders.0.self_attn.linear_out.bias       | Audio_Encoder Encoders 0 Self_Attn Linear_Out (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   19 | audio_encoder.encoders.0.self_attn.linear_q.weight       | Audio_Encoder Encoders 0 Self_Attn Linear_Q (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   20 | audio_encoder.encoders.0.self_attn.linear_k.weight       | Audio_Encoder Encoders 0 Self_Attn Linear_K (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   21 | audio_encoder.encoders.0.self_attn.linear_v.weight       | Audio_Encoder Encoders 0 Self_Attn Linear_V (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   22 | audio_encoder.encoders.0.self_attn.linear_q.bias         | Audio_Encoder Encoders 0 Self_Attn Linear_Q (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   23 | audio_encoder.encoders.0.self_attn.linear_k.bias         | Audio_Encoder Encoders 0 Self_Attn Linear_K (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   24 | audio_encoder.encoders.0.self_attn.linear_v.bias         | Audio_Encoder Encoders 0 Self_Attn Linear_V (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   25 | audio_encoder.encoders.0.self_attn.fsmn_block.weight     | Audio_Encoder Encoders 0 Self_Attn Fsmn_Block (W)     | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|   26 | audio_encoder.encoders.0.feed_forward.w_1.weight         | Audio_Encoder Encoders 0 Feed_Forward W_1 (W)         | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   27 | audio_encoder.encoders.0.feed_forward.w_1.bias           | Audio_Encoder Encoders 0 Feed_Forward W_1 (B)         | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   28 | audio_encoder.encoders.0.feed_forward.w_2.weight         | Audio_Encoder Encoders 0 Feed_Forward W_2 (W)         | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   29 | audio_encoder.encoders.0.feed_forward.w_2.bias           | Audio_Encoder Encoders 0 Feed_Forward W_2 (B)         | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   30 | audio_encoder.encoders.0.norm1.weight                    | Audio_Encoder Encoders 0 Norm1 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   31 | audio_encoder.encoders.0.norm1.bias                      | Audio_Encoder Encoders 0 Norm1 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   32 | audio_encoder.encoders.0.norm2.weight                    | Audio_Encoder Encoders 0 Norm2 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   33 | audio_encoder.encoders.0.norm2.bias                      | Audio_Encoder Encoders 0 Norm2 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   34 | audio_encoder.encoders.1.self_attn.linear_out.weight     | Audio_Encoder Encoders 1 Self_Attn Linear_Out (W)     | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   35 | audio_encoder.encoders.1.self_attn.linear_out.bias       | Audio_Encoder Encoders 1 Self_Attn Linear_Out (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   36 | audio_encoder.encoders.1.self_attn.linear_q.weight       | Audio_Encoder Encoders 1 Self_Attn Linear_Q (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   37 | audio_encoder.encoders.1.self_attn.linear_k.weight       | Audio_Encoder Encoders 1 Self_Attn Linear_K (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   38 | audio_encoder.encoders.1.self_attn.linear_v.weight       | Audio_Encoder Encoders 1 Self_Attn Linear_V (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   39 | audio_encoder.encoders.1.self_attn.linear_q.bias         | Audio_Encoder Encoders 1 Self_Attn Linear_Q (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   40 | audio_encoder.encoders.1.self_attn.linear_k.bias         | Audio_Encoder Encoders 1 Self_Attn Linear_K (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   41 | audio_encoder.encoders.1.self_attn.linear_v.bias         | Audio_Encoder Encoders 1 Self_Attn Linear_V (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   42 | audio_encoder.encoders.1.self_attn.fsmn_block.weight     | Audio_Encoder Encoders 1 Self_Attn Fsmn_Block (W)     | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|   43 | audio_encoder.encoders.1.feed_forward.w_1.weight         | Audio_Encoder Encoders 1 Feed_Forward W_1 (W)         | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   44 | audio_encoder.encoders.1.feed_forward.w_1.bias           | Audio_Encoder Encoders 1 Feed_Forward W_1 (B)         | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   45 | audio_encoder.encoders.1.feed_forward.w_2.weight         | Audio_Encoder Encoders 1 Feed_Forward W_2 (W)         | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   46 | audio_encoder.encoders.1.feed_forward.w_2.bias           | Audio_Encoder Encoders 1 Feed_Forward W_2 (B)         | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   47 | audio_encoder.encoders.1.norm1.weight                    | Audio_Encoder Encoders 1 Norm1 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   48 | audio_encoder.encoders.1.norm1.bias                      | Audio_Encoder Encoders 1 Norm1 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   49 | audio_encoder.encoders.1.norm2.weight                    | Audio_Encoder Encoders 1 Norm2 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   50 | audio_encoder.encoders.1.norm2.bias                      | Audio_Encoder Encoders 1 Norm2 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   51 | audio_encoder.encoders.2.self_attn.linear_out.weight     | Audio_Encoder Encoders 2 Self_Attn Linear_Out (W)     | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   52 | audio_encoder.encoders.2.self_attn.linear_out.bias       | Audio_Encoder Encoders 2 Self_Attn Linear_Out (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   53 | audio_encoder.encoders.2.self_attn.linear_q.weight       | Audio_Encoder Encoders 2 Self_Attn Linear_Q (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   54 | audio_encoder.encoders.2.self_attn.linear_k.weight       | Audio_Encoder Encoders 2 Self_Attn Linear_K (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   55 | audio_encoder.encoders.2.self_attn.linear_v.weight       | Audio_Encoder Encoders 2 Self_Attn Linear_V (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   56 | audio_encoder.encoders.2.self_attn.linear_q.bias         | Audio_Encoder Encoders 2 Self_Attn Linear_Q (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   57 | audio_encoder.encoders.2.self_attn.linear_k.bias         | Audio_Encoder Encoders 2 Self_Attn Linear_K (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   58 | audio_encoder.encoders.2.self_attn.linear_v.bias         | Audio_Encoder Encoders 2 Self_Attn Linear_V (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   59 | audio_encoder.encoders.2.self_attn.fsmn_block.weight     | Audio_Encoder Encoders 2 Self_Attn Fsmn_Block (W)     | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|   60 | audio_encoder.encoders.2.feed_forward.w_1.weight         | Audio_Encoder Encoders 2 Feed_Forward W_1 (W)         | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   61 | audio_encoder.encoders.2.feed_forward.w_1.bias           | Audio_Encoder Encoders 2 Feed_Forward W_1 (B)         | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   62 | audio_encoder.encoders.2.feed_forward.w_2.weight         | Audio_Encoder Encoders 2 Feed_Forward W_2 (W)         | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   63 | audio_encoder.encoders.2.feed_forward.w_2.bias           | Audio_Encoder Encoders 2 Feed_Forward W_2 (B)         | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   64 | audio_encoder.encoders.2.norm1.weight                    | Audio_Encoder Encoders 2 Norm1 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   65 | audio_encoder.encoders.2.norm1.bias                      | Audio_Encoder Encoders 2 Norm1 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   66 | audio_encoder.encoders.2.norm2.weight                    | Audio_Encoder Encoders 2 Norm2 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   67 | audio_encoder.encoders.2.norm2.bias                      | Audio_Encoder Encoders 2 Norm2 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   68 | audio_encoder.encoders.3.self_attn.linear_out.weight     | Audio_Encoder Encoders 3 Self_Attn Linear_Out (W)     | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   69 | audio_encoder.encoders.3.self_attn.linear_out.bias       | Audio_Encoder Encoders 3 Self_Attn Linear_Out (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   70 | audio_encoder.encoders.3.self_attn.linear_q.weight       | Audio_Encoder Encoders 3 Self_Attn Linear_Q (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   71 | audio_encoder.encoders.3.self_attn.linear_k.weight       | Audio_Encoder Encoders 3 Self_Attn Linear_K (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   72 | audio_encoder.encoders.3.self_attn.linear_v.weight       | Audio_Encoder Encoders 3 Self_Attn Linear_V (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   73 | audio_encoder.encoders.3.self_attn.linear_q.bias         | Audio_Encoder Encoders 3 Self_Attn Linear_Q (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   74 | audio_encoder.encoders.3.self_attn.linear_k.bias         | Audio_Encoder Encoders 3 Self_Attn Linear_K (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   75 | audio_encoder.encoders.3.self_attn.linear_v.bias         | Audio_Encoder Encoders 3 Self_Attn Linear_V (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   76 | audio_encoder.encoders.3.self_attn.fsmn_block.weight     | Audio_Encoder Encoders 3 Self_Attn Fsmn_Block (W)     | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|   77 | audio_encoder.encoders.3.feed_forward.w_1.weight         | Audio_Encoder Encoders 3 Feed_Forward W_1 (W)         | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   78 | audio_encoder.encoders.3.feed_forward.w_1.bias           | Audio_Encoder Encoders 3 Feed_Forward W_1 (B)         | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   79 | audio_encoder.encoders.3.feed_forward.w_2.weight         | Audio_Encoder Encoders 3 Feed_Forward W_2 (W)         | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   80 | audio_encoder.encoders.3.feed_forward.w_2.bias           | Audio_Encoder Encoders 3 Feed_Forward W_2 (B)         | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   81 | audio_encoder.encoders.3.norm1.weight                    | Audio_Encoder Encoders 3 Norm1 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   82 | audio_encoder.encoders.3.norm1.bias                      | Audio_Encoder Encoders 3 Norm1 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   83 | audio_encoder.encoders.3.norm2.weight                    | Audio_Encoder Encoders 3 Norm2 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   84 | audio_encoder.encoders.3.norm2.bias                      | Audio_Encoder Encoders 3 Norm2 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   85 | audio_encoder.encoders.4.self_attn.linear_out.weight     | Audio_Encoder Encoders 4 Self_Attn Linear_Out (W)     | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   86 | audio_encoder.encoders.4.self_attn.linear_out.bias       | Audio_Encoder Encoders 4 Self_Attn Linear_Out (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   87 | audio_encoder.encoders.4.self_attn.linear_q.weight       | Audio_Encoder Encoders 4 Self_Attn Linear_Q (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   88 | audio_encoder.encoders.4.self_attn.linear_k.weight       | Audio_Encoder Encoders 4 Self_Attn Linear_K (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   89 | audio_encoder.encoders.4.self_attn.linear_v.weight       | Audio_Encoder Encoders 4 Self_Attn Linear_V (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   90 | audio_encoder.encoders.4.self_attn.linear_q.bias         | Audio_Encoder Encoders 4 Self_Attn Linear_Q (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   91 | audio_encoder.encoders.4.self_attn.linear_k.bias         | Audio_Encoder Encoders 4 Self_Attn Linear_K (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   92 | audio_encoder.encoders.4.self_attn.linear_v.bias         | Audio_Encoder Encoders 4 Self_Attn Linear_V (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   93 | audio_encoder.encoders.4.self_attn.fsmn_block.weight     | Audio_Encoder Encoders 4 Self_Attn Fsmn_Block (W)     | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|   94 | audio_encoder.encoders.4.feed_forward.w_1.weight         | Audio_Encoder Encoders 4 Feed_Forward W_1 (W)         | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   95 | audio_encoder.encoders.4.feed_forward.w_1.bias           | Audio_Encoder Encoders 4 Feed_Forward W_1 (B)         | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   96 | audio_encoder.encoders.4.feed_forward.w_2.weight         | Audio_Encoder Encoders 4 Feed_Forward W_2 (W)         | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   97 | audio_encoder.encoders.4.feed_forward.w_2.bias           | Audio_Encoder Encoders 4 Feed_Forward W_2 (B)         | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   98 | audio_encoder.encoders.4.norm1.weight                    | Audio_Encoder Encoders 4 Norm1 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   99 | audio_encoder.encoders.4.norm1.bias                      | Audio_Encoder Encoders 4 Norm1 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  100 | audio_encoder.encoders.4.norm2.weight                    | Audio_Encoder Encoders 4 Norm2 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  101 | audio_encoder.encoders.4.norm2.bias                      | Audio_Encoder Encoders 4 Norm2 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  102 | audio_encoder.encoders.5.self_attn.linear_out.weight     | Audio_Encoder Encoders 5 Self_Attn Linear_Out (W)     | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  103 | audio_encoder.encoders.5.self_attn.linear_out.bias       | Audio_Encoder Encoders 5 Self_Attn Linear_Out (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  104 | audio_encoder.encoders.5.self_attn.linear_q.weight       | Audio_Encoder Encoders 5 Self_Attn Linear_Q (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  105 | audio_encoder.encoders.5.self_attn.linear_k.weight       | Audio_Encoder Encoders 5 Self_Attn Linear_K (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  106 | audio_encoder.encoders.5.self_attn.linear_v.weight       | Audio_Encoder Encoders 5 Self_Attn Linear_V (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  107 | audio_encoder.encoders.5.self_attn.linear_q.bias         | Audio_Encoder Encoders 5 Self_Attn Linear_Q (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  108 | audio_encoder.encoders.5.self_attn.linear_k.bias         | Audio_Encoder Encoders 5 Self_Attn Linear_K (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  109 | audio_encoder.encoders.5.self_attn.linear_v.bias         | Audio_Encoder Encoders 5 Self_Attn Linear_V (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  110 | audio_encoder.encoders.5.self_attn.fsmn_block.weight     | Audio_Encoder Encoders 5 Self_Attn Fsmn_Block (W)     | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  111 | audio_encoder.encoders.5.feed_forward.w_1.weight         | Audio_Encoder Encoders 5 Feed_Forward W_1 (W)         | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  112 | audio_encoder.encoders.5.feed_forward.w_1.bias           | Audio_Encoder Encoders 5 Feed_Forward W_1 (B)         | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  113 | audio_encoder.encoders.5.feed_forward.w_2.weight         | Audio_Encoder Encoders 5 Feed_Forward W_2 (W)         | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  114 | audio_encoder.encoders.5.feed_forward.w_2.bias           | Audio_Encoder Encoders 5 Feed_Forward W_2 (B)         | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  115 | audio_encoder.encoders.5.norm1.weight                    | Audio_Encoder Encoders 5 Norm1 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  116 | audio_encoder.encoders.5.norm1.bias                      | Audio_Encoder Encoders 5 Norm1 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  117 | audio_encoder.encoders.5.norm2.weight                    | Audio_Encoder Encoders 5 Norm2 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  118 | audio_encoder.encoders.5.norm2.bias                      | Audio_Encoder Encoders 5 Norm2 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  119 | audio_encoder.encoders.6.self_attn.linear_out.weight     | Audio_Encoder Encoders 6 Self_Attn Linear_Out (W)     | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  120 | audio_encoder.encoders.6.self_attn.linear_out.bias       | Audio_Encoder Encoders 6 Self_Attn Linear_Out (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  121 | audio_encoder.encoders.6.self_attn.linear_q.weight       | Audio_Encoder Encoders 6 Self_Attn Linear_Q (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  122 | audio_encoder.encoders.6.self_attn.linear_k.weight       | Audio_Encoder Encoders 6 Self_Attn Linear_K (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  123 | audio_encoder.encoders.6.self_attn.linear_v.weight       | Audio_Encoder Encoders 6 Self_Attn Linear_V (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  124 | audio_encoder.encoders.6.self_attn.linear_q.bias         | Audio_Encoder Encoders 6 Self_Attn Linear_Q (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  125 | audio_encoder.encoders.6.self_attn.linear_k.bias         | Audio_Encoder Encoders 6 Self_Attn Linear_K (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  126 | audio_encoder.encoders.6.self_attn.linear_v.bias         | Audio_Encoder Encoders 6 Self_Attn Linear_V (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  127 | audio_encoder.encoders.6.self_attn.fsmn_block.weight     | Audio_Encoder Encoders 6 Self_Attn Fsmn_Block (W)     | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  128 | audio_encoder.encoders.6.feed_forward.w_1.weight         | Audio_Encoder Encoders 6 Feed_Forward W_1 (W)         | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  129 | audio_encoder.encoders.6.feed_forward.w_1.bias           | Audio_Encoder Encoders 6 Feed_Forward W_1 (B)         | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  130 | audio_encoder.encoders.6.feed_forward.w_2.weight         | Audio_Encoder Encoders 6 Feed_Forward W_2 (W)         | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  131 | audio_encoder.encoders.6.feed_forward.w_2.bias           | Audio_Encoder Encoders 6 Feed_Forward W_2 (B)         | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  132 | audio_encoder.encoders.6.norm1.weight                    | Audio_Encoder Encoders 6 Norm1 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  133 | audio_encoder.encoders.6.norm1.bias                      | Audio_Encoder Encoders 6 Norm1 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  134 | audio_encoder.encoders.6.norm2.weight                    | Audio_Encoder Encoders 6 Norm2 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  135 | audio_encoder.encoders.6.norm2.bias                      | Audio_Encoder Encoders 6 Norm2 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  136 | audio_encoder.encoders.7.self_attn.linear_out.weight     | Audio_Encoder Encoders 7 Self_Attn Linear_Out (W)     | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  137 | audio_encoder.encoders.7.self_attn.linear_out.bias       | Audio_Encoder Encoders 7 Self_Attn Linear_Out (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  138 | audio_encoder.encoders.7.self_attn.linear_q.weight       | Audio_Encoder Encoders 7 Self_Attn Linear_Q (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  139 | audio_encoder.encoders.7.self_attn.linear_k.weight       | Audio_Encoder Encoders 7 Self_Attn Linear_K (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  140 | audio_encoder.encoders.7.self_attn.linear_v.weight       | Audio_Encoder Encoders 7 Self_Attn Linear_V (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  141 | audio_encoder.encoders.7.self_attn.linear_q.bias         | Audio_Encoder Encoders 7 Self_Attn Linear_Q (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  142 | audio_encoder.encoders.7.self_attn.linear_k.bias         | Audio_Encoder Encoders 7 Self_Attn Linear_K (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  143 | audio_encoder.encoders.7.self_attn.linear_v.bias         | Audio_Encoder Encoders 7 Self_Attn Linear_V (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  144 | audio_encoder.encoders.7.self_attn.fsmn_block.weight     | Audio_Encoder Encoders 7 Self_Attn Fsmn_Block (W)     | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  145 | audio_encoder.encoders.7.feed_forward.w_1.weight         | Audio_Encoder Encoders 7 Feed_Forward W_1 (W)         | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  146 | audio_encoder.encoders.7.feed_forward.w_1.bias           | Audio_Encoder Encoders 7 Feed_Forward W_1 (B)         | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  147 | audio_encoder.encoders.7.feed_forward.w_2.weight         | Audio_Encoder Encoders 7 Feed_Forward W_2 (W)         | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  148 | audio_encoder.encoders.7.feed_forward.w_2.bias           | Audio_Encoder Encoders 7 Feed_Forward W_2 (B)         | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  149 | audio_encoder.encoders.7.norm1.weight                    | Audio_Encoder Encoders 7 Norm1 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  150 | audio_encoder.encoders.7.norm1.bias                      | Audio_Encoder Encoders 7 Norm1 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  151 | audio_encoder.encoders.7.norm2.weight                    | Audio_Encoder Encoders 7 Norm2 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  152 | audio_encoder.encoders.7.norm2.bias                      | Audio_Encoder Encoders 7 Norm2 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  153 | audio_encoder.encoders.8.self_attn.linear_out.weight     | Audio_Encoder Encoders 8 Self_Attn Linear_Out (W)     | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  154 | audio_encoder.encoders.8.self_attn.linear_out.bias       | Audio_Encoder Encoders 8 Self_Attn Linear_Out (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  155 | audio_encoder.encoders.8.self_attn.linear_q.weight       | Audio_Encoder Encoders 8 Self_Attn Linear_Q (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  156 | audio_encoder.encoders.8.self_attn.linear_k.weight       | Audio_Encoder Encoders 8 Self_Attn Linear_K (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  157 | audio_encoder.encoders.8.self_attn.linear_v.weight       | Audio_Encoder Encoders 8 Self_Attn Linear_V (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  158 | audio_encoder.encoders.8.self_attn.linear_q.bias         | Audio_Encoder Encoders 8 Self_Attn Linear_Q (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  159 | audio_encoder.encoders.8.self_attn.linear_k.bias         | Audio_Encoder Encoders 8 Self_Attn Linear_K (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  160 | audio_encoder.encoders.8.self_attn.linear_v.bias         | Audio_Encoder Encoders 8 Self_Attn Linear_V (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  161 | audio_encoder.encoders.8.self_attn.fsmn_block.weight     | Audio_Encoder Encoders 8 Self_Attn Fsmn_Block (W)     | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  162 | audio_encoder.encoders.8.feed_forward.w_1.weight         | Audio_Encoder Encoders 8 Feed_Forward W_1 (W)         | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  163 | audio_encoder.encoders.8.feed_forward.w_1.bias           | Audio_Encoder Encoders 8 Feed_Forward W_1 (B)         | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  164 | audio_encoder.encoders.8.feed_forward.w_2.weight         | Audio_Encoder Encoders 8 Feed_Forward W_2 (W)         | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  165 | audio_encoder.encoders.8.feed_forward.w_2.bias           | Audio_Encoder Encoders 8 Feed_Forward W_2 (B)         | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  166 | audio_encoder.encoders.8.norm1.weight                    | Audio_Encoder Encoders 8 Norm1 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  167 | audio_encoder.encoders.8.norm1.bias                      | Audio_Encoder Encoders 8 Norm1 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  168 | audio_encoder.encoders.8.norm2.weight                    | Audio_Encoder Encoders 8 Norm2 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  169 | audio_encoder.encoders.8.norm2.bias                      | Audio_Encoder Encoders 8 Norm2 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  170 | audio_encoder.encoders.9.self_attn.linear_out.weight     | Audio_Encoder Encoders 9 Self_Attn Linear_Out (W)     | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  171 | audio_encoder.encoders.9.self_attn.linear_out.bias       | Audio_Encoder Encoders 9 Self_Attn Linear_Out (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  172 | audio_encoder.encoders.9.self_attn.linear_q.weight       | Audio_Encoder Encoders 9 Self_Attn Linear_Q (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  173 | audio_encoder.encoders.9.self_attn.linear_k.weight       | Audio_Encoder Encoders 9 Self_Attn Linear_K (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  174 | audio_encoder.encoders.9.self_attn.linear_v.weight       | Audio_Encoder Encoders 9 Self_Attn Linear_V (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  175 | audio_encoder.encoders.9.self_attn.linear_q.bias         | Audio_Encoder Encoders 9 Self_Attn Linear_Q (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  176 | audio_encoder.encoders.9.self_attn.linear_k.bias         | Audio_Encoder Encoders 9 Self_Attn Linear_K (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  177 | audio_encoder.encoders.9.self_attn.linear_v.bias         | Audio_Encoder Encoders 9 Self_Attn Linear_V (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  178 | audio_encoder.encoders.9.self_attn.fsmn_block.weight     | Audio_Encoder Encoders 9 Self_Attn Fsmn_Block (W)     | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  179 | audio_encoder.encoders.9.feed_forward.w_1.weight         | Audio_Encoder Encoders 9 Feed_Forward W_1 (W)         | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  180 | audio_encoder.encoders.9.feed_forward.w_1.bias           | Audio_Encoder Encoders 9 Feed_Forward W_1 (B)         | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  181 | audio_encoder.encoders.9.feed_forward.w_2.weight         | Audio_Encoder Encoders 9 Feed_Forward W_2 (W)         | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  182 | audio_encoder.encoders.9.feed_forward.w_2.bias           | Audio_Encoder Encoders 9 Feed_Forward W_2 (B)         | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  183 | audio_encoder.encoders.9.norm1.weight                    | Audio_Encoder Encoders 9 Norm1 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  184 | audio_encoder.encoders.9.norm1.bias                      | Audio_Encoder Encoders 9 Norm1 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  185 | audio_encoder.encoders.9.norm2.weight                    | Audio_Encoder Encoders 9 Norm2 (W)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  186 | audio_encoder.encoders.9.norm2.bias                      | Audio_Encoder Encoders 9 Norm2 (B)                    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  187 | audio_encoder.encoders.10.self_attn.linear_out.weight    | Audio_Encoder Encoders 10 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  188 | audio_encoder.encoders.10.self_attn.linear_out.bias      | Audio_Encoder Encoders 10 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  189 | audio_encoder.encoders.10.self_attn.linear_q.weight      | Audio_Encoder Encoders 10 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  190 | audio_encoder.encoders.10.self_attn.linear_k.weight      | Audio_Encoder Encoders 10 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  191 | audio_encoder.encoders.10.self_attn.linear_v.weight      | Audio_Encoder Encoders 10 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  192 | audio_encoder.encoders.10.self_attn.linear_q.bias        | Audio_Encoder Encoders 10 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  193 | audio_encoder.encoders.10.self_attn.linear_k.bias        | Audio_Encoder Encoders 10 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  194 | audio_encoder.encoders.10.self_attn.linear_v.bias        | Audio_Encoder Encoders 10 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  195 | audio_encoder.encoders.10.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 10 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  196 | audio_encoder.encoders.10.feed_forward.w_1.weight        | Audio_Encoder Encoders 10 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  197 | audio_encoder.encoders.10.feed_forward.w_1.bias          | Audio_Encoder Encoders 10 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  198 | audio_encoder.encoders.10.feed_forward.w_2.weight        | Audio_Encoder Encoders 10 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  199 | audio_encoder.encoders.10.feed_forward.w_2.bias          | Audio_Encoder Encoders 10 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  200 | audio_encoder.encoders.10.norm1.weight                   | Audio_Encoder Encoders 10 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  201 | audio_encoder.encoders.10.norm1.bias                     | Audio_Encoder Encoders 10 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  202 | audio_encoder.encoders.10.norm2.weight                   | Audio_Encoder Encoders 10 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  203 | audio_encoder.encoders.10.norm2.bias                     | Audio_Encoder Encoders 10 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  204 | audio_encoder.encoders.11.self_attn.linear_out.weight    | Audio_Encoder Encoders 11 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  205 | audio_encoder.encoders.11.self_attn.linear_out.bias      | Audio_Encoder Encoders 11 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  206 | audio_encoder.encoders.11.self_attn.linear_q.weight      | Audio_Encoder Encoders 11 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  207 | audio_encoder.encoders.11.self_attn.linear_k.weight      | Audio_Encoder Encoders 11 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  208 | audio_encoder.encoders.11.self_attn.linear_v.weight      | Audio_Encoder Encoders 11 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  209 | audio_encoder.encoders.11.self_attn.linear_q.bias        | Audio_Encoder Encoders 11 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  210 | audio_encoder.encoders.11.self_attn.linear_k.bias        | Audio_Encoder Encoders 11 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  211 | audio_encoder.encoders.11.self_attn.linear_v.bias        | Audio_Encoder Encoders 11 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  212 | audio_encoder.encoders.11.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 11 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  213 | audio_encoder.encoders.11.feed_forward.w_1.weight        | Audio_Encoder Encoders 11 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  214 | audio_encoder.encoders.11.feed_forward.w_1.bias          | Audio_Encoder Encoders 11 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  215 | audio_encoder.encoders.11.feed_forward.w_2.weight        | Audio_Encoder Encoders 11 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  216 | audio_encoder.encoders.11.feed_forward.w_2.bias          | Audio_Encoder Encoders 11 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  217 | audio_encoder.encoders.11.norm1.weight                   | Audio_Encoder Encoders 11 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  218 | audio_encoder.encoders.11.norm1.bias                     | Audio_Encoder Encoders 11 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  219 | audio_encoder.encoders.11.norm2.weight                   | Audio_Encoder Encoders 11 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  220 | audio_encoder.encoders.11.norm2.bias                     | Audio_Encoder Encoders 11 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  221 | audio_encoder.encoders.12.self_attn.linear_out.weight    | Audio_Encoder Encoders 12 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  222 | audio_encoder.encoders.12.self_attn.linear_out.bias      | Audio_Encoder Encoders 12 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  223 | audio_encoder.encoders.12.self_attn.linear_q.weight      | Audio_Encoder Encoders 12 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  224 | audio_encoder.encoders.12.self_attn.linear_k.weight      | Audio_Encoder Encoders 12 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  225 | audio_encoder.encoders.12.self_attn.linear_v.weight      | Audio_Encoder Encoders 12 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  226 | audio_encoder.encoders.12.self_attn.linear_q.bias        | Audio_Encoder Encoders 12 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  227 | audio_encoder.encoders.12.self_attn.linear_k.bias        | Audio_Encoder Encoders 12 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  228 | audio_encoder.encoders.12.self_attn.linear_v.bias        | Audio_Encoder Encoders 12 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  229 | audio_encoder.encoders.12.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 12 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  230 | audio_encoder.encoders.12.feed_forward.w_1.weight        | Audio_Encoder Encoders 12 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  231 | audio_encoder.encoders.12.feed_forward.w_1.bias          | Audio_Encoder Encoders 12 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  232 | audio_encoder.encoders.12.feed_forward.w_2.weight        | Audio_Encoder Encoders 12 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  233 | audio_encoder.encoders.12.feed_forward.w_2.bias          | Audio_Encoder Encoders 12 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  234 | audio_encoder.encoders.12.norm1.weight                   | Audio_Encoder Encoders 12 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  235 | audio_encoder.encoders.12.norm1.bias                     | Audio_Encoder Encoders 12 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  236 | audio_encoder.encoders.12.norm2.weight                   | Audio_Encoder Encoders 12 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  237 | audio_encoder.encoders.12.norm2.bias                     | Audio_Encoder Encoders 12 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  238 | audio_encoder.encoders.13.self_attn.linear_out.weight    | Audio_Encoder Encoders 13 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  239 | audio_encoder.encoders.13.self_attn.linear_out.bias      | Audio_Encoder Encoders 13 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  240 | audio_encoder.encoders.13.self_attn.linear_q.weight      | Audio_Encoder Encoders 13 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  241 | audio_encoder.encoders.13.self_attn.linear_k.weight      | Audio_Encoder Encoders 13 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  242 | audio_encoder.encoders.13.self_attn.linear_v.weight      | Audio_Encoder Encoders 13 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  243 | audio_encoder.encoders.13.self_attn.linear_q.bias        | Audio_Encoder Encoders 13 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  244 | audio_encoder.encoders.13.self_attn.linear_k.bias        | Audio_Encoder Encoders 13 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  245 | audio_encoder.encoders.13.self_attn.linear_v.bias        | Audio_Encoder Encoders 13 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  246 | audio_encoder.encoders.13.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 13 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  247 | audio_encoder.encoders.13.feed_forward.w_1.weight        | Audio_Encoder Encoders 13 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  248 | audio_encoder.encoders.13.feed_forward.w_1.bias          | Audio_Encoder Encoders 13 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  249 | audio_encoder.encoders.13.feed_forward.w_2.weight        | Audio_Encoder Encoders 13 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  250 | audio_encoder.encoders.13.feed_forward.w_2.bias          | Audio_Encoder Encoders 13 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  251 | audio_encoder.encoders.13.norm1.weight                   | Audio_Encoder Encoders 13 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  252 | audio_encoder.encoders.13.norm1.bias                     | Audio_Encoder Encoders 13 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  253 | audio_encoder.encoders.13.norm2.weight                   | Audio_Encoder Encoders 13 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  254 | audio_encoder.encoders.13.norm2.bias                     | Audio_Encoder Encoders 13 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  255 | audio_encoder.encoders.14.self_attn.linear_out.weight    | Audio_Encoder Encoders 14 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  256 | audio_encoder.encoders.14.self_attn.linear_out.bias      | Audio_Encoder Encoders 14 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  257 | audio_encoder.encoders.14.self_attn.linear_q.weight      | Audio_Encoder Encoders 14 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  258 | audio_encoder.encoders.14.self_attn.linear_k.weight      | Audio_Encoder Encoders 14 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  259 | audio_encoder.encoders.14.self_attn.linear_v.weight      | Audio_Encoder Encoders 14 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  260 | audio_encoder.encoders.14.self_attn.linear_q.bias        | Audio_Encoder Encoders 14 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  261 | audio_encoder.encoders.14.self_attn.linear_k.bias        | Audio_Encoder Encoders 14 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  262 | audio_encoder.encoders.14.self_attn.linear_v.bias        | Audio_Encoder Encoders 14 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  263 | audio_encoder.encoders.14.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 14 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  264 | audio_encoder.encoders.14.feed_forward.w_1.weight        | Audio_Encoder Encoders 14 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  265 | audio_encoder.encoders.14.feed_forward.w_1.bias          | Audio_Encoder Encoders 14 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  266 | audio_encoder.encoders.14.feed_forward.w_2.weight        | Audio_Encoder Encoders 14 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  267 | audio_encoder.encoders.14.feed_forward.w_2.bias          | Audio_Encoder Encoders 14 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  268 | audio_encoder.encoders.14.norm1.weight                   | Audio_Encoder Encoders 14 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  269 | audio_encoder.encoders.14.norm1.bias                     | Audio_Encoder Encoders 14 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  270 | audio_encoder.encoders.14.norm2.weight                   | Audio_Encoder Encoders 14 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  271 | audio_encoder.encoders.14.norm2.bias                     | Audio_Encoder Encoders 14 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  272 | audio_encoder.encoders.15.self_attn.linear_out.weight    | Audio_Encoder Encoders 15 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  273 | audio_encoder.encoders.15.self_attn.linear_out.bias      | Audio_Encoder Encoders 15 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  274 | audio_encoder.encoders.15.self_attn.linear_q.weight      | Audio_Encoder Encoders 15 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  275 | audio_encoder.encoders.15.self_attn.linear_k.weight      | Audio_Encoder Encoders 15 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  276 | audio_encoder.encoders.15.self_attn.linear_v.weight      | Audio_Encoder Encoders 15 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  277 | audio_encoder.encoders.15.self_attn.linear_q.bias        | Audio_Encoder Encoders 15 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  278 | audio_encoder.encoders.15.self_attn.linear_k.bias        | Audio_Encoder Encoders 15 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  279 | audio_encoder.encoders.15.self_attn.linear_v.bias        | Audio_Encoder Encoders 15 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  280 | audio_encoder.encoders.15.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 15 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  281 | audio_encoder.encoders.15.feed_forward.w_1.weight        | Audio_Encoder Encoders 15 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  282 | audio_encoder.encoders.15.feed_forward.w_1.bias          | Audio_Encoder Encoders 15 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  283 | audio_encoder.encoders.15.feed_forward.w_2.weight        | Audio_Encoder Encoders 15 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  284 | audio_encoder.encoders.15.feed_forward.w_2.bias          | Audio_Encoder Encoders 15 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  285 | audio_encoder.encoders.15.norm1.weight                   | Audio_Encoder Encoders 15 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  286 | audio_encoder.encoders.15.norm1.bias                     | Audio_Encoder Encoders 15 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  287 | audio_encoder.encoders.15.norm2.weight                   | Audio_Encoder Encoders 15 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  288 | audio_encoder.encoders.15.norm2.bias                     | Audio_Encoder Encoders 15 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  289 | audio_encoder.encoders.16.self_attn.linear_out.weight    | Audio_Encoder Encoders 16 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  290 | audio_encoder.encoders.16.self_attn.linear_out.bias      | Audio_Encoder Encoders 16 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  291 | audio_encoder.encoders.16.self_attn.linear_q.weight      | Audio_Encoder Encoders 16 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  292 | audio_encoder.encoders.16.self_attn.linear_k.weight      | Audio_Encoder Encoders 16 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  293 | audio_encoder.encoders.16.self_attn.linear_v.weight      | Audio_Encoder Encoders 16 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  294 | audio_encoder.encoders.16.self_attn.linear_q.bias        | Audio_Encoder Encoders 16 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  295 | audio_encoder.encoders.16.self_attn.linear_k.bias        | Audio_Encoder Encoders 16 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  296 | audio_encoder.encoders.16.self_attn.linear_v.bias        | Audio_Encoder Encoders 16 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  297 | audio_encoder.encoders.16.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 16 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  298 | audio_encoder.encoders.16.feed_forward.w_1.weight        | Audio_Encoder Encoders 16 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  299 | audio_encoder.encoders.16.feed_forward.w_1.bias          | Audio_Encoder Encoders 16 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  300 | audio_encoder.encoders.16.feed_forward.w_2.weight        | Audio_Encoder Encoders 16 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  301 | audio_encoder.encoders.16.feed_forward.w_2.bias          | Audio_Encoder Encoders 16 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  302 | audio_encoder.encoders.16.norm1.weight                   | Audio_Encoder Encoders 16 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  303 | audio_encoder.encoders.16.norm1.bias                     | Audio_Encoder Encoders 16 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  304 | audio_encoder.encoders.16.norm2.weight                   | Audio_Encoder Encoders 16 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  305 | audio_encoder.encoders.16.norm2.bias                     | Audio_Encoder Encoders 16 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  306 | audio_encoder.encoders.17.self_attn.linear_out.weight    | Audio_Encoder Encoders 17 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  307 | audio_encoder.encoders.17.self_attn.linear_out.bias      | Audio_Encoder Encoders 17 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  308 | audio_encoder.encoders.17.self_attn.linear_q.weight      | Audio_Encoder Encoders 17 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  309 | audio_encoder.encoders.17.self_attn.linear_k.weight      | Audio_Encoder Encoders 17 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  310 | audio_encoder.encoders.17.self_attn.linear_v.weight      | Audio_Encoder Encoders 17 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  311 | audio_encoder.encoders.17.self_attn.linear_q.bias        | Audio_Encoder Encoders 17 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  312 | audio_encoder.encoders.17.self_attn.linear_k.bias        | Audio_Encoder Encoders 17 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  313 | audio_encoder.encoders.17.self_attn.linear_v.bias        | Audio_Encoder Encoders 17 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  314 | audio_encoder.encoders.17.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 17 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  315 | audio_encoder.encoders.17.feed_forward.w_1.weight        | Audio_Encoder Encoders 17 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  316 | audio_encoder.encoders.17.feed_forward.w_1.bias          | Audio_Encoder Encoders 17 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  317 | audio_encoder.encoders.17.feed_forward.w_2.weight        | Audio_Encoder Encoders 17 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  318 | audio_encoder.encoders.17.feed_forward.w_2.bias          | Audio_Encoder Encoders 17 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  319 | audio_encoder.encoders.17.norm1.weight                   | Audio_Encoder Encoders 17 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  320 | audio_encoder.encoders.17.norm1.bias                     | Audio_Encoder Encoders 17 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  321 | audio_encoder.encoders.17.norm2.weight                   | Audio_Encoder Encoders 17 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  322 | audio_encoder.encoders.17.norm2.bias                     | Audio_Encoder Encoders 17 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  323 | audio_encoder.encoders.18.self_attn.linear_out.weight    | Audio_Encoder Encoders 18 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  324 | audio_encoder.encoders.18.self_attn.linear_out.bias      | Audio_Encoder Encoders 18 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  325 | audio_encoder.encoders.18.self_attn.linear_q.weight      | Audio_Encoder Encoders 18 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  326 | audio_encoder.encoders.18.self_attn.linear_k.weight      | Audio_Encoder Encoders 18 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  327 | audio_encoder.encoders.18.self_attn.linear_v.weight      | Audio_Encoder Encoders 18 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  328 | audio_encoder.encoders.18.self_attn.linear_q.bias        | Audio_Encoder Encoders 18 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  329 | audio_encoder.encoders.18.self_attn.linear_k.bias        | Audio_Encoder Encoders 18 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  330 | audio_encoder.encoders.18.self_attn.linear_v.bias        | Audio_Encoder Encoders 18 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  331 | audio_encoder.encoders.18.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 18 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  332 | audio_encoder.encoders.18.feed_forward.w_1.weight        | Audio_Encoder Encoders 18 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  333 | audio_encoder.encoders.18.feed_forward.w_1.bias          | Audio_Encoder Encoders 18 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  334 | audio_encoder.encoders.18.feed_forward.w_2.weight        | Audio_Encoder Encoders 18 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  335 | audio_encoder.encoders.18.feed_forward.w_2.bias          | Audio_Encoder Encoders 18 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  336 | audio_encoder.encoders.18.norm1.weight                   | Audio_Encoder Encoders 18 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  337 | audio_encoder.encoders.18.norm1.bias                     | Audio_Encoder Encoders 18 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  338 | audio_encoder.encoders.18.norm2.weight                   | Audio_Encoder Encoders 18 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  339 | audio_encoder.encoders.18.norm2.bias                     | Audio_Encoder Encoders 18 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  340 | audio_encoder.encoders.19.self_attn.linear_out.weight    | Audio_Encoder Encoders 19 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  341 | audio_encoder.encoders.19.self_attn.linear_out.bias      | Audio_Encoder Encoders 19 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  342 | audio_encoder.encoders.19.self_attn.linear_q.weight      | Audio_Encoder Encoders 19 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  343 | audio_encoder.encoders.19.self_attn.linear_k.weight      | Audio_Encoder Encoders 19 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  344 | audio_encoder.encoders.19.self_attn.linear_v.weight      | Audio_Encoder Encoders 19 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  345 | audio_encoder.encoders.19.self_attn.linear_q.bias        | Audio_Encoder Encoders 19 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  346 | audio_encoder.encoders.19.self_attn.linear_k.bias        | Audio_Encoder Encoders 19 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  347 | audio_encoder.encoders.19.self_attn.linear_v.bias        | Audio_Encoder Encoders 19 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  348 | audio_encoder.encoders.19.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 19 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  349 | audio_encoder.encoders.19.feed_forward.w_1.weight        | Audio_Encoder Encoders 19 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  350 | audio_encoder.encoders.19.feed_forward.w_1.bias          | Audio_Encoder Encoders 19 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  351 | audio_encoder.encoders.19.feed_forward.w_2.weight        | Audio_Encoder Encoders 19 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  352 | audio_encoder.encoders.19.feed_forward.w_2.bias          | Audio_Encoder Encoders 19 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  353 | audio_encoder.encoders.19.norm1.weight                   | Audio_Encoder Encoders 19 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  354 | audio_encoder.encoders.19.norm1.bias                     | Audio_Encoder Encoders 19 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  355 | audio_encoder.encoders.19.norm2.weight                   | Audio_Encoder Encoders 19 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  356 | audio_encoder.encoders.19.norm2.bias                     | Audio_Encoder Encoders 19 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  357 | audio_encoder.encoders.20.self_attn.linear_out.weight    | Audio_Encoder Encoders 20 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  358 | audio_encoder.encoders.20.self_attn.linear_out.bias      | Audio_Encoder Encoders 20 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  359 | audio_encoder.encoders.20.self_attn.linear_q.weight      | Audio_Encoder Encoders 20 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  360 | audio_encoder.encoders.20.self_attn.linear_k.weight      | Audio_Encoder Encoders 20 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  361 | audio_encoder.encoders.20.self_attn.linear_v.weight      | Audio_Encoder Encoders 20 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  362 | audio_encoder.encoders.20.self_attn.linear_q.bias        | Audio_Encoder Encoders 20 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  363 | audio_encoder.encoders.20.self_attn.linear_k.bias        | Audio_Encoder Encoders 20 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  364 | audio_encoder.encoders.20.self_attn.linear_v.bias        | Audio_Encoder Encoders 20 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  365 | audio_encoder.encoders.20.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 20 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  366 | audio_encoder.encoders.20.feed_forward.w_1.weight        | Audio_Encoder Encoders 20 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  367 | audio_encoder.encoders.20.feed_forward.w_1.bias          | Audio_Encoder Encoders 20 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  368 | audio_encoder.encoders.20.feed_forward.w_2.weight        | Audio_Encoder Encoders 20 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  369 | audio_encoder.encoders.20.feed_forward.w_2.bias          | Audio_Encoder Encoders 20 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  370 | audio_encoder.encoders.20.norm1.weight                   | Audio_Encoder Encoders 20 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  371 | audio_encoder.encoders.20.norm1.bias                     | Audio_Encoder Encoders 20 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  372 | audio_encoder.encoders.20.norm2.weight                   | Audio_Encoder Encoders 20 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  373 | audio_encoder.encoders.20.norm2.bias                     | Audio_Encoder Encoders 20 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  374 | audio_encoder.encoders.21.self_attn.linear_out.weight    | Audio_Encoder Encoders 21 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  375 | audio_encoder.encoders.21.self_attn.linear_out.bias      | Audio_Encoder Encoders 21 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  376 | audio_encoder.encoders.21.self_attn.linear_q.weight      | Audio_Encoder Encoders 21 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  377 | audio_encoder.encoders.21.self_attn.linear_k.weight      | Audio_Encoder Encoders 21 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  378 | audio_encoder.encoders.21.self_attn.linear_v.weight      | Audio_Encoder Encoders 21 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  379 | audio_encoder.encoders.21.self_attn.linear_q.bias        | Audio_Encoder Encoders 21 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  380 | audio_encoder.encoders.21.self_attn.linear_k.bias        | Audio_Encoder Encoders 21 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  381 | audio_encoder.encoders.21.self_attn.linear_v.bias        | Audio_Encoder Encoders 21 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  382 | audio_encoder.encoders.21.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 21 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  383 | audio_encoder.encoders.21.feed_forward.w_1.weight        | Audio_Encoder Encoders 21 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  384 | audio_encoder.encoders.21.feed_forward.w_1.bias          | Audio_Encoder Encoders 21 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  385 | audio_encoder.encoders.21.feed_forward.w_2.weight        | Audio_Encoder Encoders 21 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  386 | audio_encoder.encoders.21.feed_forward.w_2.bias          | Audio_Encoder Encoders 21 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  387 | audio_encoder.encoders.21.norm1.weight                   | Audio_Encoder Encoders 21 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  388 | audio_encoder.encoders.21.norm1.bias                     | Audio_Encoder Encoders 21 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  389 | audio_encoder.encoders.21.norm2.weight                   | Audio_Encoder Encoders 21 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  390 | audio_encoder.encoders.21.norm2.bias                     | Audio_Encoder Encoders 21 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  391 | audio_encoder.encoders.22.self_attn.linear_out.weight    | Audio_Encoder Encoders 22 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  392 | audio_encoder.encoders.22.self_attn.linear_out.bias      | Audio_Encoder Encoders 22 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  393 | audio_encoder.encoders.22.self_attn.linear_q.weight      | Audio_Encoder Encoders 22 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  394 | audio_encoder.encoders.22.self_attn.linear_k.weight      | Audio_Encoder Encoders 22 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  395 | audio_encoder.encoders.22.self_attn.linear_v.weight      | Audio_Encoder Encoders 22 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  396 | audio_encoder.encoders.22.self_attn.linear_q.bias        | Audio_Encoder Encoders 22 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  397 | audio_encoder.encoders.22.self_attn.linear_k.bias        | Audio_Encoder Encoders 22 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  398 | audio_encoder.encoders.22.self_attn.linear_v.bias        | Audio_Encoder Encoders 22 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  399 | audio_encoder.encoders.22.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 22 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  400 | audio_encoder.encoders.22.feed_forward.w_1.weight        | Audio_Encoder Encoders 22 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  401 | audio_encoder.encoders.22.feed_forward.w_1.bias          | Audio_Encoder Encoders 22 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  402 | audio_encoder.encoders.22.feed_forward.w_2.weight        | Audio_Encoder Encoders 22 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  403 | audio_encoder.encoders.22.feed_forward.w_2.bias          | Audio_Encoder Encoders 22 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  404 | audio_encoder.encoders.22.norm1.weight                   | Audio_Encoder Encoders 22 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  405 | audio_encoder.encoders.22.norm1.bias                     | Audio_Encoder Encoders 22 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  406 | audio_encoder.encoders.22.norm2.weight                   | Audio_Encoder Encoders 22 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  407 | audio_encoder.encoders.22.norm2.bias                     | Audio_Encoder Encoders 22 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  408 | audio_encoder.encoders.23.self_attn.linear_out.weight    | Audio_Encoder Encoders 23 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  409 | audio_encoder.encoders.23.self_attn.linear_out.bias      | Audio_Encoder Encoders 23 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  410 | audio_encoder.encoders.23.self_attn.linear_q.weight      | Audio_Encoder Encoders 23 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  411 | audio_encoder.encoders.23.self_attn.linear_k.weight      | Audio_Encoder Encoders 23 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  412 | audio_encoder.encoders.23.self_attn.linear_v.weight      | Audio_Encoder Encoders 23 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  413 | audio_encoder.encoders.23.self_attn.linear_q.bias        | Audio_Encoder Encoders 23 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  414 | audio_encoder.encoders.23.self_attn.linear_k.bias        | Audio_Encoder Encoders 23 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  415 | audio_encoder.encoders.23.self_attn.linear_v.bias        | Audio_Encoder Encoders 23 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  416 | audio_encoder.encoders.23.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 23 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  417 | audio_encoder.encoders.23.feed_forward.w_1.weight        | Audio_Encoder Encoders 23 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  418 | audio_encoder.encoders.23.feed_forward.w_1.bias          | Audio_Encoder Encoders 23 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  419 | audio_encoder.encoders.23.feed_forward.w_2.weight        | Audio_Encoder Encoders 23 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  420 | audio_encoder.encoders.23.feed_forward.w_2.bias          | Audio_Encoder Encoders 23 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  421 | audio_encoder.encoders.23.norm1.weight                   | Audio_Encoder Encoders 23 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  422 | audio_encoder.encoders.23.norm1.bias                     | Audio_Encoder Encoders 23 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  423 | audio_encoder.encoders.23.norm2.weight                   | Audio_Encoder Encoders 23 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  424 | audio_encoder.encoders.23.norm2.bias                     | Audio_Encoder Encoders 23 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  425 | audio_encoder.encoders.24.self_attn.linear_out.weight    | Audio_Encoder Encoders 24 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  426 | audio_encoder.encoders.24.self_attn.linear_out.bias      | Audio_Encoder Encoders 24 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  427 | audio_encoder.encoders.24.self_attn.linear_q.weight      | Audio_Encoder Encoders 24 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  428 | audio_encoder.encoders.24.self_attn.linear_k.weight      | Audio_Encoder Encoders 24 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  429 | audio_encoder.encoders.24.self_attn.linear_v.weight      | Audio_Encoder Encoders 24 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  430 | audio_encoder.encoders.24.self_attn.linear_q.bias        | Audio_Encoder Encoders 24 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  431 | audio_encoder.encoders.24.self_attn.linear_k.bias        | Audio_Encoder Encoders 24 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  432 | audio_encoder.encoders.24.self_attn.linear_v.bias        | Audio_Encoder Encoders 24 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  433 | audio_encoder.encoders.24.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 24 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  434 | audio_encoder.encoders.24.feed_forward.w_1.weight        | Audio_Encoder Encoders 24 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  435 | audio_encoder.encoders.24.feed_forward.w_1.bias          | Audio_Encoder Encoders 24 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  436 | audio_encoder.encoders.24.feed_forward.w_2.weight        | Audio_Encoder Encoders 24 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  437 | audio_encoder.encoders.24.feed_forward.w_2.bias          | Audio_Encoder Encoders 24 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  438 | audio_encoder.encoders.24.norm1.weight                   | Audio_Encoder Encoders 24 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  439 | audio_encoder.encoders.24.norm1.bias                     | Audio_Encoder Encoders 24 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  440 | audio_encoder.encoders.24.norm2.weight                   | Audio_Encoder Encoders 24 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  441 | audio_encoder.encoders.24.norm2.bias                     | Audio_Encoder Encoders 24 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  442 | audio_encoder.encoders.25.self_attn.linear_out.weight    | Audio_Encoder Encoders 25 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  443 | audio_encoder.encoders.25.self_attn.linear_out.bias      | Audio_Encoder Encoders 25 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  444 | audio_encoder.encoders.25.self_attn.linear_q.weight      | Audio_Encoder Encoders 25 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  445 | audio_encoder.encoders.25.self_attn.linear_k.weight      | Audio_Encoder Encoders 25 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  446 | audio_encoder.encoders.25.self_attn.linear_v.weight      | Audio_Encoder Encoders 25 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  447 | audio_encoder.encoders.25.self_attn.linear_q.bias        | Audio_Encoder Encoders 25 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  448 | audio_encoder.encoders.25.self_attn.linear_k.bias        | Audio_Encoder Encoders 25 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  449 | audio_encoder.encoders.25.self_attn.linear_v.bias        | Audio_Encoder Encoders 25 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  450 | audio_encoder.encoders.25.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 25 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  451 | audio_encoder.encoders.25.feed_forward.w_1.weight        | Audio_Encoder Encoders 25 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  452 | audio_encoder.encoders.25.feed_forward.w_1.bias          | Audio_Encoder Encoders 25 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  453 | audio_encoder.encoders.25.feed_forward.w_2.weight        | Audio_Encoder Encoders 25 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  454 | audio_encoder.encoders.25.feed_forward.w_2.bias          | Audio_Encoder Encoders 25 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  455 | audio_encoder.encoders.25.norm1.weight                   | Audio_Encoder Encoders 25 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  456 | audio_encoder.encoders.25.norm1.bias                     | Audio_Encoder Encoders 25 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  457 | audio_encoder.encoders.25.norm2.weight                   | Audio_Encoder Encoders 25 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  458 | audio_encoder.encoders.25.norm2.bias                     | Audio_Encoder Encoders 25 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  459 | audio_encoder.encoders.26.self_attn.linear_out.weight    | Audio_Encoder Encoders 26 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  460 | audio_encoder.encoders.26.self_attn.linear_out.bias      | Audio_Encoder Encoders 26 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  461 | audio_encoder.encoders.26.self_attn.linear_q.weight      | Audio_Encoder Encoders 26 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  462 | audio_encoder.encoders.26.self_attn.linear_k.weight      | Audio_Encoder Encoders 26 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  463 | audio_encoder.encoders.26.self_attn.linear_v.weight      | Audio_Encoder Encoders 26 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  464 | audio_encoder.encoders.26.self_attn.linear_q.bias        | Audio_Encoder Encoders 26 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  465 | audio_encoder.encoders.26.self_attn.linear_k.bias        | Audio_Encoder Encoders 26 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  466 | audio_encoder.encoders.26.self_attn.linear_v.bias        | Audio_Encoder Encoders 26 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  467 | audio_encoder.encoders.26.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 26 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  468 | audio_encoder.encoders.26.feed_forward.w_1.weight        | Audio_Encoder Encoders 26 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  469 | audio_encoder.encoders.26.feed_forward.w_1.bias          | Audio_Encoder Encoders 26 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  470 | audio_encoder.encoders.26.feed_forward.w_2.weight        | Audio_Encoder Encoders 26 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  471 | audio_encoder.encoders.26.feed_forward.w_2.bias          | Audio_Encoder Encoders 26 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  472 | audio_encoder.encoders.26.norm1.weight                   | Audio_Encoder Encoders 26 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  473 | audio_encoder.encoders.26.norm1.bias                     | Audio_Encoder Encoders 26 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  474 | audio_encoder.encoders.26.norm2.weight                   | Audio_Encoder Encoders 26 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  475 | audio_encoder.encoders.26.norm2.bias                     | Audio_Encoder Encoders 26 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  476 | audio_encoder.encoders.27.self_attn.linear_out.weight    | Audio_Encoder Encoders 27 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  477 | audio_encoder.encoders.27.self_attn.linear_out.bias      | Audio_Encoder Encoders 27 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  478 | audio_encoder.encoders.27.self_attn.linear_q.weight      | Audio_Encoder Encoders 27 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  479 | audio_encoder.encoders.27.self_attn.linear_k.weight      | Audio_Encoder Encoders 27 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  480 | audio_encoder.encoders.27.self_attn.linear_v.weight      | Audio_Encoder Encoders 27 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  481 | audio_encoder.encoders.27.self_attn.linear_q.bias        | Audio_Encoder Encoders 27 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  482 | audio_encoder.encoders.27.self_attn.linear_k.bias        | Audio_Encoder Encoders 27 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  483 | audio_encoder.encoders.27.self_attn.linear_v.bias        | Audio_Encoder Encoders 27 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  484 | audio_encoder.encoders.27.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 27 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  485 | audio_encoder.encoders.27.feed_forward.w_1.weight        | Audio_Encoder Encoders 27 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  486 | audio_encoder.encoders.27.feed_forward.w_1.bias          | Audio_Encoder Encoders 27 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  487 | audio_encoder.encoders.27.feed_forward.w_2.weight        | Audio_Encoder Encoders 27 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  488 | audio_encoder.encoders.27.feed_forward.w_2.bias          | Audio_Encoder Encoders 27 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  489 | audio_encoder.encoders.27.norm1.weight                   | Audio_Encoder Encoders 27 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  490 | audio_encoder.encoders.27.norm1.bias                     | Audio_Encoder Encoders 27 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  491 | audio_encoder.encoders.27.norm2.weight                   | Audio_Encoder Encoders 27 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  492 | audio_encoder.encoders.27.norm2.bias                     | Audio_Encoder Encoders 27 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  493 | audio_encoder.encoders.28.self_attn.linear_out.weight    | Audio_Encoder Encoders 28 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  494 | audio_encoder.encoders.28.self_attn.linear_out.bias      | Audio_Encoder Encoders 28 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  495 | audio_encoder.encoders.28.self_attn.linear_q.weight      | Audio_Encoder Encoders 28 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  496 | audio_encoder.encoders.28.self_attn.linear_k.weight      | Audio_Encoder Encoders 28 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  497 | audio_encoder.encoders.28.self_attn.linear_v.weight      | Audio_Encoder Encoders 28 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  498 | audio_encoder.encoders.28.self_attn.linear_q.bias        | Audio_Encoder Encoders 28 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  499 | audio_encoder.encoders.28.self_attn.linear_k.bias        | Audio_Encoder Encoders 28 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  500 | audio_encoder.encoders.28.self_attn.linear_v.bias        | Audio_Encoder Encoders 28 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  501 | audio_encoder.encoders.28.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 28 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  502 | audio_encoder.encoders.28.feed_forward.w_1.weight        | Audio_Encoder Encoders 28 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  503 | audio_encoder.encoders.28.feed_forward.w_1.bias          | Audio_Encoder Encoders 28 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  504 | audio_encoder.encoders.28.feed_forward.w_2.weight        | Audio_Encoder Encoders 28 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  505 | audio_encoder.encoders.28.feed_forward.w_2.bias          | Audio_Encoder Encoders 28 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  506 | audio_encoder.encoders.28.norm1.weight                   | Audio_Encoder Encoders 28 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  507 | audio_encoder.encoders.28.norm1.bias                     | Audio_Encoder Encoders 28 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  508 | audio_encoder.encoders.28.norm2.weight                   | Audio_Encoder Encoders 28 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  509 | audio_encoder.encoders.28.norm2.bias                     | Audio_Encoder Encoders 28 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  510 | audio_encoder.encoders.29.self_attn.linear_out.weight    | Audio_Encoder Encoders 29 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  511 | audio_encoder.encoders.29.self_attn.linear_out.bias      | Audio_Encoder Encoders 29 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  512 | audio_encoder.encoders.29.self_attn.linear_q.weight      | Audio_Encoder Encoders 29 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  513 | audio_encoder.encoders.29.self_attn.linear_k.weight      | Audio_Encoder Encoders 29 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  514 | audio_encoder.encoders.29.self_attn.linear_v.weight      | Audio_Encoder Encoders 29 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  515 | audio_encoder.encoders.29.self_attn.linear_q.bias        | Audio_Encoder Encoders 29 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  516 | audio_encoder.encoders.29.self_attn.linear_k.bias        | Audio_Encoder Encoders 29 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  517 | audio_encoder.encoders.29.self_attn.linear_v.bias        | Audio_Encoder Encoders 29 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  518 | audio_encoder.encoders.29.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 29 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  519 | audio_encoder.encoders.29.feed_forward.w_1.weight        | Audio_Encoder Encoders 29 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  520 | audio_encoder.encoders.29.feed_forward.w_1.bias          | Audio_Encoder Encoders 29 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  521 | audio_encoder.encoders.29.feed_forward.w_2.weight        | Audio_Encoder Encoders 29 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  522 | audio_encoder.encoders.29.feed_forward.w_2.bias          | Audio_Encoder Encoders 29 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  523 | audio_encoder.encoders.29.norm1.weight                   | Audio_Encoder Encoders 29 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  524 | audio_encoder.encoders.29.norm1.bias                     | Audio_Encoder Encoders 29 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  525 | audio_encoder.encoders.29.norm2.weight                   | Audio_Encoder Encoders 29 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  526 | audio_encoder.encoders.29.norm2.bias                     | Audio_Encoder Encoders 29 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  527 | audio_encoder.encoders.30.self_attn.linear_out.weight    | Audio_Encoder Encoders 30 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  528 | audio_encoder.encoders.30.self_attn.linear_out.bias      | Audio_Encoder Encoders 30 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  529 | audio_encoder.encoders.30.self_attn.linear_q.weight      | Audio_Encoder Encoders 30 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  530 | audio_encoder.encoders.30.self_attn.linear_k.weight      | Audio_Encoder Encoders 30 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  531 | audio_encoder.encoders.30.self_attn.linear_v.weight      | Audio_Encoder Encoders 30 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  532 | audio_encoder.encoders.30.self_attn.linear_q.bias        | Audio_Encoder Encoders 30 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  533 | audio_encoder.encoders.30.self_attn.linear_k.bias        | Audio_Encoder Encoders 30 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  534 | audio_encoder.encoders.30.self_attn.linear_v.bias        | Audio_Encoder Encoders 30 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  535 | audio_encoder.encoders.30.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 30 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  536 | audio_encoder.encoders.30.feed_forward.w_1.weight        | Audio_Encoder Encoders 30 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  537 | audio_encoder.encoders.30.feed_forward.w_1.bias          | Audio_Encoder Encoders 30 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  538 | audio_encoder.encoders.30.feed_forward.w_2.weight        | Audio_Encoder Encoders 30 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  539 | audio_encoder.encoders.30.feed_forward.w_2.bias          | Audio_Encoder Encoders 30 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  540 | audio_encoder.encoders.30.norm1.weight                   | Audio_Encoder Encoders 30 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  541 | audio_encoder.encoders.30.norm1.bias                     | Audio_Encoder Encoders 30 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  542 | audio_encoder.encoders.30.norm2.weight                   | Audio_Encoder Encoders 30 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  543 | audio_encoder.encoders.30.norm2.bias                     | Audio_Encoder Encoders 30 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  544 | audio_encoder.encoders.31.self_attn.linear_out.weight    | Audio_Encoder Encoders 31 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  545 | audio_encoder.encoders.31.self_attn.linear_out.bias      | Audio_Encoder Encoders 31 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  546 | audio_encoder.encoders.31.self_attn.linear_q.weight      | Audio_Encoder Encoders 31 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  547 | audio_encoder.encoders.31.self_attn.linear_k.weight      | Audio_Encoder Encoders 31 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  548 | audio_encoder.encoders.31.self_attn.linear_v.weight      | Audio_Encoder Encoders 31 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  549 | audio_encoder.encoders.31.self_attn.linear_q.bias        | Audio_Encoder Encoders 31 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  550 | audio_encoder.encoders.31.self_attn.linear_k.bias        | Audio_Encoder Encoders 31 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  551 | audio_encoder.encoders.31.self_attn.linear_v.bias        | Audio_Encoder Encoders 31 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  552 | audio_encoder.encoders.31.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 31 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  553 | audio_encoder.encoders.31.feed_forward.w_1.weight        | Audio_Encoder Encoders 31 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  554 | audio_encoder.encoders.31.feed_forward.w_1.bias          | Audio_Encoder Encoders 31 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  555 | audio_encoder.encoders.31.feed_forward.w_2.weight        | Audio_Encoder Encoders 31 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  556 | audio_encoder.encoders.31.feed_forward.w_2.bias          | Audio_Encoder Encoders 31 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  557 | audio_encoder.encoders.31.norm1.weight                   | Audio_Encoder Encoders 31 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  558 | audio_encoder.encoders.31.norm1.bias                     | Audio_Encoder Encoders 31 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  559 | audio_encoder.encoders.31.norm2.weight                   | Audio_Encoder Encoders 31 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  560 | audio_encoder.encoders.31.norm2.bias                     | Audio_Encoder Encoders 31 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  561 | audio_encoder.encoders.32.self_attn.linear_out.weight    | Audio_Encoder Encoders 32 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  562 | audio_encoder.encoders.32.self_attn.linear_out.bias      | Audio_Encoder Encoders 32 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  563 | audio_encoder.encoders.32.self_attn.linear_q.weight      | Audio_Encoder Encoders 32 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  564 | audio_encoder.encoders.32.self_attn.linear_k.weight      | Audio_Encoder Encoders 32 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  565 | audio_encoder.encoders.32.self_attn.linear_v.weight      | Audio_Encoder Encoders 32 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  566 | audio_encoder.encoders.32.self_attn.linear_q.bias        | Audio_Encoder Encoders 32 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  567 | audio_encoder.encoders.32.self_attn.linear_k.bias        | Audio_Encoder Encoders 32 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  568 | audio_encoder.encoders.32.self_attn.linear_v.bias        | Audio_Encoder Encoders 32 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  569 | audio_encoder.encoders.32.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 32 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  570 | audio_encoder.encoders.32.feed_forward.w_1.weight        | Audio_Encoder Encoders 32 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  571 | audio_encoder.encoders.32.feed_forward.w_1.bias          | Audio_Encoder Encoders 32 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  572 | audio_encoder.encoders.32.feed_forward.w_2.weight        | Audio_Encoder Encoders 32 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  573 | audio_encoder.encoders.32.feed_forward.w_2.bias          | Audio_Encoder Encoders 32 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  574 | audio_encoder.encoders.32.norm1.weight                   | Audio_Encoder Encoders 32 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  575 | audio_encoder.encoders.32.norm1.bias                     | Audio_Encoder Encoders 32 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  576 | audio_encoder.encoders.32.norm2.weight                   | Audio_Encoder Encoders 32 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  577 | audio_encoder.encoders.32.norm2.bias                     | Audio_Encoder Encoders 32 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  578 | audio_encoder.encoders.33.self_attn.linear_out.weight    | Audio_Encoder Encoders 33 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  579 | audio_encoder.encoders.33.self_attn.linear_out.bias      | Audio_Encoder Encoders 33 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  580 | audio_encoder.encoders.33.self_attn.linear_q.weight      | Audio_Encoder Encoders 33 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  581 | audio_encoder.encoders.33.self_attn.linear_k.weight      | Audio_Encoder Encoders 33 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  582 | audio_encoder.encoders.33.self_attn.linear_v.weight      | Audio_Encoder Encoders 33 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  583 | audio_encoder.encoders.33.self_attn.linear_q.bias        | Audio_Encoder Encoders 33 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  584 | audio_encoder.encoders.33.self_attn.linear_k.bias        | Audio_Encoder Encoders 33 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  585 | audio_encoder.encoders.33.self_attn.linear_v.bias        | Audio_Encoder Encoders 33 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  586 | audio_encoder.encoders.33.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 33 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  587 | audio_encoder.encoders.33.feed_forward.w_1.weight        | Audio_Encoder Encoders 33 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  588 | audio_encoder.encoders.33.feed_forward.w_1.bias          | Audio_Encoder Encoders 33 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  589 | audio_encoder.encoders.33.feed_forward.w_2.weight        | Audio_Encoder Encoders 33 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  590 | audio_encoder.encoders.33.feed_forward.w_2.bias          | Audio_Encoder Encoders 33 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  591 | audio_encoder.encoders.33.norm1.weight                   | Audio_Encoder Encoders 33 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  592 | audio_encoder.encoders.33.norm1.bias                     | Audio_Encoder Encoders 33 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  593 | audio_encoder.encoders.33.norm2.weight                   | Audio_Encoder Encoders 33 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  594 | audio_encoder.encoders.33.norm2.bias                     | Audio_Encoder Encoders 33 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  595 | audio_encoder.encoders.34.self_attn.linear_out.weight    | Audio_Encoder Encoders 34 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  596 | audio_encoder.encoders.34.self_attn.linear_out.bias      | Audio_Encoder Encoders 34 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  597 | audio_encoder.encoders.34.self_attn.linear_q.weight      | Audio_Encoder Encoders 34 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  598 | audio_encoder.encoders.34.self_attn.linear_k.weight      | Audio_Encoder Encoders 34 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  599 | audio_encoder.encoders.34.self_attn.linear_v.weight      | Audio_Encoder Encoders 34 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  600 | audio_encoder.encoders.34.self_attn.linear_q.bias        | Audio_Encoder Encoders 34 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  601 | audio_encoder.encoders.34.self_attn.linear_k.bias        | Audio_Encoder Encoders 34 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  602 | audio_encoder.encoders.34.self_attn.linear_v.bias        | Audio_Encoder Encoders 34 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  603 | audio_encoder.encoders.34.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 34 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  604 | audio_encoder.encoders.34.feed_forward.w_1.weight        | Audio_Encoder Encoders 34 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  605 | audio_encoder.encoders.34.feed_forward.w_1.bias          | Audio_Encoder Encoders 34 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  606 | audio_encoder.encoders.34.feed_forward.w_2.weight        | Audio_Encoder Encoders 34 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  607 | audio_encoder.encoders.34.feed_forward.w_2.bias          | Audio_Encoder Encoders 34 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  608 | audio_encoder.encoders.34.norm1.weight                   | Audio_Encoder Encoders 34 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  609 | audio_encoder.encoders.34.norm1.bias                     | Audio_Encoder Encoders 34 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  610 | audio_encoder.encoders.34.norm2.weight                   | Audio_Encoder Encoders 34 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  611 | audio_encoder.encoders.34.norm2.bias                     | Audio_Encoder Encoders 34 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  612 | audio_encoder.encoders.35.self_attn.linear_out.weight    | Audio_Encoder Encoders 35 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  613 | audio_encoder.encoders.35.self_attn.linear_out.bias      | Audio_Encoder Encoders 35 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  614 | audio_encoder.encoders.35.self_attn.linear_q.weight      | Audio_Encoder Encoders 35 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  615 | audio_encoder.encoders.35.self_attn.linear_k.weight      | Audio_Encoder Encoders 35 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  616 | audio_encoder.encoders.35.self_attn.linear_v.weight      | Audio_Encoder Encoders 35 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  617 | audio_encoder.encoders.35.self_attn.linear_q.bias        | Audio_Encoder Encoders 35 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  618 | audio_encoder.encoders.35.self_attn.linear_k.bias        | Audio_Encoder Encoders 35 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  619 | audio_encoder.encoders.35.self_attn.linear_v.bias        | Audio_Encoder Encoders 35 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  620 | audio_encoder.encoders.35.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 35 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  621 | audio_encoder.encoders.35.feed_forward.w_1.weight        | Audio_Encoder Encoders 35 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  622 | audio_encoder.encoders.35.feed_forward.w_1.bias          | Audio_Encoder Encoders 35 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  623 | audio_encoder.encoders.35.feed_forward.w_2.weight        | Audio_Encoder Encoders 35 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  624 | audio_encoder.encoders.35.feed_forward.w_2.bias          | Audio_Encoder Encoders 35 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  625 | audio_encoder.encoders.35.norm1.weight                   | Audio_Encoder Encoders 35 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  626 | audio_encoder.encoders.35.norm1.bias                     | Audio_Encoder Encoders 35 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  627 | audio_encoder.encoders.35.norm2.weight                   | Audio_Encoder Encoders 35 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  628 | audio_encoder.encoders.35.norm2.bias                     | Audio_Encoder Encoders 35 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  629 | audio_encoder.encoders.36.self_attn.linear_out.weight    | Audio_Encoder Encoders 36 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  630 | audio_encoder.encoders.36.self_attn.linear_out.bias      | Audio_Encoder Encoders 36 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  631 | audio_encoder.encoders.36.self_attn.linear_q.weight      | Audio_Encoder Encoders 36 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  632 | audio_encoder.encoders.36.self_attn.linear_k.weight      | Audio_Encoder Encoders 36 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  633 | audio_encoder.encoders.36.self_attn.linear_v.weight      | Audio_Encoder Encoders 36 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  634 | audio_encoder.encoders.36.self_attn.linear_q.bias        | Audio_Encoder Encoders 36 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  635 | audio_encoder.encoders.36.self_attn.linear_k.bias        | Audio_Encoder Encoders 36 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  636 | audio_encoder.encoders.36.self_attn.linear_v.bias        | Audio_Encoder Encoders 36 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  637 | audio_encoder.encoders.36.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 36 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  638 | audio_encoder.encoders.36.feed_forward.w_1.weight        | Audio_Encoder Encoders 36 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  639 | audio_encoder.encoders.36.feed_forward.w_1.bias          | Audio_Encoder Encoders 36 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  640 | audio_encoder.encoders.36.feed_forward.w_2.weight        | Audio_Encoder Encoders 36 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  641 | audio_encoder.encoders.36.feed_forward.w_2.bias          | Audio_Encoder Encoders 36 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  642 | audio_encoder.encoders.36.norm1.weight                   | Audio_Encoder Encoders 36 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  643 | audio_encoder.encoders.36.norm1.bias                     | Audio_Encoder Encoders 36 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  644 | audio_encoder.encoders.36.norm2.weight                   | Audio_Encoder Encoders 36 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  645 | audio_encoder.encoders.36.norm2.bias                     | Audio_Encoder Encoders 36 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  646 | audio_encoder.encoders.37.self_attn.linear_out.weight    | Audio_Encoder Encoders 37 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  647 | audio_encoder.encoders.37.self_attn.linear_out.bias      | Audio_Encoder Encoders 37 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  648 | audio_encoder.encoders.37.self_attn.linear_q.weight      | Audio_Encoder Encoders 37 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  649 | audio_encoder.encoders.37.self_attn.linear_k.weight      | Audio_Encoder Encoders 37 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  650 | audio_encoder.encoders.37.self_attn.linear_v.weight      | Audio_Encoder Encoders 37 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  651 | audio_encoder.encoders.37.self_attn.linear_q.bias        | Audio_Encoder Encoders 37 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  652 | audio_encoder.encoders.37.self_attn.linear_k.bias        | Audio_Encoder Encoders 37 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  653 | audio_encoder.encoders.37.self_attn.linear_v.bias        | Audio_Encoder Encoders 37 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  654 | audio_encoder.encoders.37.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 37 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  655 | audio_encoder.encoders.37.feed_forward.w_1.weight        | Audio_Encoder Encoders 37 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  656 | audio_encoder.encoders.37.feed_forward.w_1.bias          | Audio_Encoder Encoders 37 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  657 | audio_encoder.encoders.37.feed_forward.w_2.weight        | Audio_Encoder Encoders 37 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  658 | audio_encoder.encoders.37.feed_forward.w_2.bias          | Audio_Encoder Encoders 37 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  659 | audio_encoder.encoders.37.norm1.weight                   | Audio_Encoder Encoders 37 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  660 | audio_encoder.encoders.37.norm1.bias                     | Audio_Encoder Encoders 37 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  661 | audio_encoder.encoders.37.norm2.weight                   | Audio_Encoder Encoders 37 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  662 | audio_encoder.encoders.37.norm2.bias                     | Audio_Encoder Encoders 37 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  663 | audio_encoder.encoders.38.self_attn.linear_out.weight    | Audio_Encoder Encoders 38 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  664 | audio_encoder.encoders.38.self_attn.linear_out.bias      | Audio_Encoder Encoders 38 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  665 | audio_encoder.encoders.38.self_attn.linear_q.weight      | Audio_Encoder Encoders 38 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  666 | audio_encoder.encoders.38.self_attn.linear_k.weight      | Audio_Encoder Encoders 38 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  667 | audio_encoder.encoders.38.self_attn.linear_v.weight      | Audio_Encoder Encoders 38 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  668 | audio_encoder.encoders.38.self_attn.linear_q.bias        | Audio_Encoder Encoders 38 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  669 | audio_encoder.encoders.38.self_attn.linear_k.bias        | Audio_Encoder Encoders 38 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  670 | audio_encoder.encoders.38.self_attn.linear_v.bias        | Audio_Encoder Encoders 38 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  671 | audio_encoder.encoders.38.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 38 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  672 | audio_encoder.encoders.38.feed_forward.w_1.weight        | Audio_Encoder Encoders 38 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  673 | audio_encoder.encoders.38.feed_forward.w_1.bias          | Audio_Encoder Encoders 38 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  674 | audio_encoder.encoders.38.feed_forward.w_2.weight        | Audio_Encoder Encoders 38 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  675 | audio_encoder.encoders.38.feed_forward.w_2.bias          | Audio_Encoder Encoders 38 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  676 | audio_encoder.encoders.38.norm1.weight                   | Audio_Encoder Encoders 38 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  677 | audio_encoder.encoders.38.norm1.bias                     | Audio_Encoder Encoders 38 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  678 | audio_encoder.encoders.38.norm2.weight                   | Audio_Encoder Encoders 38 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  679 | audio_encoder.encoders.38.norm2.bias                     | Audio_Encoder Encoders 38 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  680 | audio_encoder.encoders.39.self_attn.linear_out.weight    | Audio_Encoder Encoders 39 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  681 | audio_encoder.encoders.39.self_attn.linear_out.bias      | Audio_Encoder Encoders 39 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  682 | audio_encoder.encoders.39.self_attn.linear_q.weight      | Audio_Encoder Encoders 39 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  683 | audio_encoder.encoders.39.self_attn.linear_k.weight      | Audio_Encoder Encoders 39 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  684 | audio_encoder.encoders.39.self_attn.linear_v.weight      | Audio_Encoder Encoders 39 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  685 | audio_encoder.encoders.39.self_attn.linear_q.bias        | Audio_Encoder Encoders 39 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  686 | audio_encoder.encoders.39.self_attn.linear_k.bias        | Audio_Encoder Encoders 39 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  687 | audio_encoder.encoders.39.self_attn.linear_v.bias        | Audio_Encoder Encoders 39 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  688 | audio_encoder.encoders.39.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 39 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  689 | audio_encoder.encoders.39.feed_forward.w_1.weight        | Audio_Encoder Encoders 39 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  690 | audio_encoder.encoders.39.feed_forward.w_1.bias          | Audio_Encoder Encoders 39 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  691 | audio_encoder.encoders.39.feed_forward.w_2.weight        | Audio_Encoder Encoders 39 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  692 | audio_encoder.encoders.39.feed_forward.w_2.bias          | Audio_Encoder Encoders 39 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  693 | audio_encoder.encoders.39.norm1.weight                   | Audio_Encoder Encoders 39 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  694 | audio_encoder.encoders.39.norm1.bias                     | Audio_Encoder Encoders 39 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  695 | audio_encoder.encoders.39.norm2.weight                   | Audio_Encoder Encoders 39 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  696 | audio_encoder.encoders.39.norm2.bias                     | Audio_Encoder Encoders 39 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  697 | audio_encoder.encoders.40.self_attn.linear_out.weight    | Audio_Encoder Encoders 40 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  698 | audio_encoder.encoders.40.self_attn.linear_out.bias      | Audio_Encoder Encoders 40 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  699 | audio_encoder.encoders.40.self_attn.linear_q.weight      | Audio_Encoder Encoders 40 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  700 | audio_encoder.encoders.40.self_attn.linear_k.weight      | Audio_Encoder Encoders 40 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  701 | audio_encoder.encoders.40.self_attn.linear_v.weight      | Audio_Encoder Encoders 40 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  702 | audio_encoder.encoders.40.self_attn.linear_q.bias        | Audio_Encoder Encoders 40 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  703 | audio_encoder.encoders.40.self_attn.linear_k.bias        | Audio_Encoder Encoders 40 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  704 | audio_encoder.encoders.40.self_attn.linear_v.bias        | Audio_Encoder Encoders 40 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  705 | audio_encoder.encoders.40.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 40 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  706 | audio_encoder.encoders.40.feed_forward.w_1.weight        | Audio_Encoder Encoders 40 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  707 | audio_encoder.encoders.40.feed_forward.w_1.bias          | Audio_Encoder Encoders 40 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  708 | audio_encoder.encoders.40.feed_forward.w_2.weight        | Audio_Encoder Encoders 40 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  709 | audio_encoder.encoders.40.feed_forward.w_2.bias          | Audio_Encoder Encoders 40 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  710 | audio_encoder.encoders.40.norm1.weight                   | Audio_Encoder Encoders 40 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  711 | audio_encoder.encoders.40.norm1.bias                     | Audio_Encoder Encoders 40 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  712 | audio_encoder.encoders.40.norm2.weight                   | Audio_Encoder Encoders 40 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  713 | audio_encoder.encoders.40.norm2.bias                     | Audio_Encoder Encoders 40 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  714 | audio_encoder.encoders.41.self_attn.linear_out.weight    | Audio_Encoder Encoders 41 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  715 | audio_encoder.encoders.41.self_attn.linear_out.bias      | Audio_Encoder Encoders 41 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  716 | audio_encoder.encoders.41.self_attn.linear_q.weight      | Audio_Encoder Encoders 41 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  717 | audio_encoder.encoders.41.self_attn.linear_k.weight      | Audio_Encoder Encoders 41 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  718 | audio_encoder.encoders.41.self_attn.linear_v.weight      | Audio_Encoder Encoders 41 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  719 | audio_encoder.encoders.41.self_attn.linear_q.bias        | Audio_Encoder Encoders 41 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  720 | audio_encoder.encoders.41.self_attn.linear_k.bias        | Audio_Encoder Encoders 41 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  721 | audio_encoder.encoders.41.self_attn.linear_v.bias        | Audio_Encoder Encoders 41 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  722 | audio_encoder.encoders.41.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 41 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  723 | audio_encoder.encoders.41.feed_forward.w_1.weight        | Audio_Encoder Encoders 41 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  724 | audio_encoder.encoders.41.feed_forward.w_1.bias          | Audio_Encoder Encoders 41 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  725 | audio_encoder.encoders.41.feed_forward.w_2.weight        | Audio_Encoder Encoders 41 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  726 | audio_encoder.encoders.41.feed_forward.w_2.bias          | Audio_Encoder Encoders 41 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  727 | audio_encoder.encoders.41.norm1.weight                   | Audio_Encoder Encoders 41 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  728 | audio_encoder.encoders.41.norm1.bias                     | Audio_Encoder Encoders 41 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  729 | audio_encoder.encoders.41.norm2.weight                   | Audio_Encoder Encoders 41 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  730 | audio_encoder.encoders.41.norm2.bias                     | Audio_Encoder Encoders 41 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  731 | audio_encoder.encoders.42.self_attn.linear_out.weight    | Audio_Encoder Encoders 42 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  732 | audio_encoder.encoders.42.self_attn.linear_out.bias      | Audio_Encoder Encoders 42 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  733 | audio_encoder.encoders.42.self_attn.linear_q.weight      | Audio_Encoder Encoders 42 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  734 | audio_encoder.encoders.42.self_attn.linear_k.weight      | Audio_Encoder Encoders 42 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  735 | audio_encoder.encoders.42.self_attn.linear_v.weight      | Audio_Encoder Encoders 42 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  736 | audio_encoder.encoders.42.self_attn.linear_q.bias        | Audio_Encoder Encoders 42 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  737 | audio_encoder.encoders.42.self_attn.linear_k.bias        | Audio_Encoder Encoders 42 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  738 | audio_encoder.encoders.42.self_attn.linear_v.bias        | Audio_Encoder Encoders 42 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  739 | audio_encoder.encoders.42.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 42 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  740 | audio_encoder.encoders.42.feed_forward.w_1.weight        | Audio_Encoder Encoders 42 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  741 | audio_encoder.encoders.42.feed_forward.w_1.bias          | Audio_Encoder Encoders 42 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  742 | audio_encoder.encoders.42.feed_forward.w_2.weight        | Audio_Encoder Encoders 42 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  743 | audio_encoder.encoders.42.feed_forward.w_2.bias          | Audio_Encoder Encoders 42 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  744 | audio_encoder.encoders.42.norm1.weight                   | Audio_Encoder Encoders 42 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  745 | audio_encoder.encoders.42.norm1.bias                     | Audio_Encoder Encoders 42 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  746 | audio_encoder.encoders.42.norm2.weight                   | Audio_Encoder Encoders 42 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  747 | audio_encoder.encoders.42.norm2.bias                     | Audio_Encoder Encoders 42 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  748 | audio_encoder.encoders.43.self_attn.linear_out.weight    | Audio_Encoder Encoders 43 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  749 | audio_encoder.encoders.43.self_attn.linear_out.bias      | Audio_Encoder Encoders 43 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  750 | audio_encoder.encoders.43.self_attn.linear_q.weight      | Audio_Encoder Encoders 43 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  751 | audio_encoder.encoders.43.self_attn.linear_k.weight      | Audio_Encoder Encoders 43 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  752 | audio_encoder.encoders.43.self_attn.linear_v.weight      | Audio_Encoder Encoders 43 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  753 | audio_encoder.encoders.43.self_attn.linear_q.bias        | Audio_Encoder Encoders 43 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  754 | audio_encoder.encoders.43.self_attn.linear_k.bias        | Audio_Encoder Encoders 43 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  755 | audio_encoder.encoders.43.self_attn.linear_v.bias        | Audio_Encoder Encoders 43 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  756 | audio_encoder.encoders.43.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 43 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  757 | audio_encoder.encoders.43.feed_forward.w_1.weight        | Audio_Encoder Encoders 43 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  758 | audio_encoder.encoders.43.feed_forward.w_1.bias          | Audio_Encoder Encoders 43 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  759 | audio_encoder.encoders.43.feed_forward.w_2.weight        | Audio_Encoder Encoders 43 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  760 | audio_encoder.encoders.43.feed_forward.w_2.bias          | Audio_Encoder Encoders 43 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  761 | audio_encoder.encoders.43.norm1.weight                   | Audio_Encoder Encoders 43 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  762 | audio_encoder.encoders.43.norm1.bias                     | Audio_Encoder Encoders 43 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  763 | audio_encoder.encoders.43.norm2.weight                   | Audio_Encoder Encoders 43 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  764 | audio_encoder.encoders.43.norm2.bias                     | Audio_Encoder Encoders 43 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  765 | audio_encoder.encoders.44.self_attn.linear_out.weight    | Audio_Encoder Encoders 44 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  766 | audio_encoder.encoders.44.self_attn.linear_out.bias      | Audio_Encoder Encoders 44 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  767 | audio_encoder.encoders.44.self_attn.linear_q.weight      | Audio_Encoder Encoders 44 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  768 | audio_encoder.encoders.44.self_attn.linear_k.weight      | Audio_Encoder Encoders 44 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  769 | audio_encoder.encoders.44.self_attn.linear_v.weight      | Audio_Encoder Encoders 44 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  770 | audio_encoder.encoders.44.self_attn.linear_q.bias        | Audio_Encoder Encoders 44 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  771 | audio_encoder.encoders.44.self_attn.linear_k.bias        | Audio_Encoder Encoders 44 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  772 | audio_encoder.encoders.44.self_attn.linear_v.bias        | Audio_Encoder Encoders 44 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  773 | audio_encoder.encoders.44.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 44 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  774 | audio_encoder.encoders.44.feed_forward.w_1.weight        | Audio_Encoder Encoders 44 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  775 | audio_encoder.encoders.44.feed_forward.w_1.bias          | Audio_Encoder Encoders 44 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  776 | audio_encoder.encoders.44.feed_forward.w_2.weight        | Audio_Encoder Encoders 44 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  777 | audio_encoder.encoders.44.feed_forward.w_2.bias          | Audio_Encoder Encoders 44 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  778 | audio_encoder.encoders.44.norm1.weight                   | Audio_Encoder Encoders 44 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  779 | audio_encoder.encoders.44.norm1.bias                     | Audio_Encoder Encoders 44 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  780 | audio_encoder.encoders.44.norm2.weight                   | Audio_Encoder Encoders 44 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  781 | audio_encoder.encoders.44.norm2.bias                     | Audio_Encoder Encoders 44 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  782 | audio_encoder.encoders.45.self_attn.linear_out.weight    | Audio_Encoder Encoders 45 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  783 | audio_encoder.encoders.45.self_attn.linear_out.bias      | Audio_Encoder Encoders 45 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  784 | audio_encoder.encoders.45.self_attn.linear_q.weight      | Audio_Encoder Encoders 45 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  785 | audio_encoder.encoders.45.self_attn.linear_k.weight      | Audio_Encoder Encoders 45 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  786 | audio_encoder.encoders.45.self_attn.linear_v.weight      | Audio_Encoder Encoders 45 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  787 | audio_encoder.encoders.45.self_attn.linear_q.bias        | Audio_Encoder Encoders 45 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  788 | audio_encoder.encoders.45.self_attn.linear_k.bias        | Audio_Encoder Encoders 45 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  789 | audio_encoder.encoders.45.self_attn.linear_v.bias        | Audio_Encoder Encoders 45 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  790 | audio_encoder.encoders.45.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 45 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  791 | audio_encoder.encoders.45.feed_forward.w_1.weight        | Audio_Encoder Encoders 45 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  792 | audio_encoder.encoders.45.feed_forward.w_1.bias          | Audio_Encoder Encoders 45 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  793 | audio_encoder.encoders.45.feed_forward.w_2.weight        | Audio_Encoder Encoders 45 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  794 | audio_encoder.encoders.45.feed_forward.w_2.bias          | Audio_Encoder Encoders 45 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  795 | audio_encoder.encoders.45.norm1.weight                   | Audio_Encoder Encoders 45 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  796 | audio_encoder.encoders.45.norm1.bias                     | Audio_Encoder Encoders 45 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  797 | audio_encoder.encoders.45.norm2.weight                   | Audio_Encoder Encoders 45 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  798 | audio_encoder.encoders.45.norm2.bias                     | Audio_Encoder Encoders 45 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  799 | audio_encoder.encoders.46.self_attn.linear_out.weight    | Audio_Encoder Encoders 46 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  800 | audio_encoder.encoders.46.self_attn.linear_out.bias      | Audio_Encoder Encoders 46 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  801 | audio_encoder.encoders.46.self_attn.linear_q.weight      | Audio_Encoder Encoders 46 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  802 | audio_encoder.encoders.46.self_attn.linear_k.weight      | Audio_Encoder Encoders 46 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  803 | audio_encoder.encoders.46.self_attn.linear_v.weight      | Audio_Encoder Encoders 46 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  804 | audio_encoder.encoders.46.self_attn.linear_q.bias        | Audio_Encoder Encoders 46 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  805 | audio_encoder.encoders.46.self_attn.linear_k.bias        | Audio_Encoder Encoders 46 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  806 | audio_encoder.encoders.46.self_attn.linear_v.bias        | Audio_Encoder Encoders 46 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  807 | audio_encoder.encoders.46.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 46 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  808 | audio_encoder.encoders.46.feed_forward.w_1.weight        | Audio_Encoder Encoders 46 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  809 | audio_encoder.encoders.46.feed_forward.w_1.bias          | Audio_Encoder Encoders 46 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  810 | audio_encoder.encoders.46.feed_forward.w_2.weight        | Audio_Encoder Encoders 46 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  811 | audio_encoder.encoders.46.feed_forward.w_2.bias          | Audio_Encoder Encoders 46 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  812 | audio_encoder.encoders.46.norm1.weight                   | Audio_Encoder Encoders 46 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  813 | audio_encoder.encoders.46.norm1.bias                     | Audio_Encoder Encoders 46 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  814 | audio_encoder.encoders.46.norm2.weight                   | Audio_Encoder Encoders 46 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  815 | audio_encoder.encoders.46.norm2.bias                     | Audio_Encoder Encoders 46 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  816 | audio_encoder.encoders.47.self_attn.linear_out.weight    | Audio_Encoder Encoders 47 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  817 | audio_encoder.encoders.47.self_attn.linear_out.bias      | Audio_Encoder Encoders 47 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  818 | audio_encoder.encoders.47.self_attn.linear_q.weight      | Audio_Encoder Encoders 47 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  819 | audio_encoder.encoders.47.self_attn.linear_k.weight      | Audio_Encoder Encoders 47 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  820 | audio_encoder.encoders.47.self_attn.linear_v.weight      | Audio_Encoder Encoders 47 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  821 | audio_encoder.encoders.47.self_attn.linear_q.bias        | Audio_Encoder Encoders 47 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  822 | audio_encoder.encoders.47.self_attn.linear_k.bias        | Audio_Encoder Encoders 47 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  823 | audio_encoder.encoders.47.self_attn.linear_v.bias        | Audio_Encoder Encoders 47 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  824 | audio_encoder.encoders.47.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 47 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  825 | audio_encoder.encoders.47.feed_forward.w_1.weight        | Audio_Encoder Encoders 47 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  826 | audio_encoder.encoders.47.feed_forward.w_1.bias          | Audio_Encoder Encoders 47 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  827 | audio_encoder.encoders.47.feed_forward.w_2.weight        | Audio_Encoder Encoders 47 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  828 | audio_encoder.encoders.47.feed_forward.w_2.bias          | Audio_Encoder Encoders 47 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  829 | audio_encoder.encoders.47.norm1.weight                   | Audio_Encoder Encoders 47 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  830 | audio_encoder.encoders.47.norm1.bias                     | Audio_Encoder Encoders 47 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  831 | audio_encoder.encoders.47.norm2.weight                   | Audio_Encoder Encoders 47 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  832 | audio_encoder.encoders.47.norm2.bias                     | Audio_Encoder Encoders 47 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  833 | audio_encoder.encoders.48.self_attn.linear_out.weight    | Audio_Encoder Encoders 48 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  834 | audio_encoder.encoders.48.self_attn.linear_out.bias      | Audio_Encoder Encoders 48 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  835 | audio_encoder.encoders.48.self_attn.linear_q.weight      | Audio_Encoder Encoders 48 Self_Attn Linear_Q (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  836 | audio_encoder.encoders.48.self_attn.linear_k.weight      | Audio_Encoder Encoders 48 Self_Attn Linear_K (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  837 | audio_encoder.encoders.48.self_attn.linear_v.weight      | Audio_Encoder Encoders 48 Self_Attn Linear_V (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  838 | audio_encoder.encoders.48.self_attn.linear_q.bias        | Audio_Encoder Encoders 48 Self_Attn Linear_Q (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  839 | audio_encoder.encoders.48.self_attn.linear_k.bias        | Audio_Encoder Encoders 48 Self_Attn Linear_K (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  840 | audio_encoder.encoders.48.self_attn.linear_v.bias        | Audio_Encoder Encoders 48 Self_Attn Linear_V (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  841 | audio_encoder.encoders.48.self_attn.fsmn_block.weight    | Audio_Encoder Encoders 48 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  842 | audio_encoder.encoders.48.feed_forward.w_1.weight        | Audio_Encoder Encoders 48 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  843 | audio_encoder.encoders.48.feed_forward.w_1.bias          | Audio_Encoder Encoders 48 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  844 | audio_encoder.encoders.48.feed_forward.w_2.weight        | Audio_Encoder Encoders 48 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  845 | audio_encoder.encoders.48.feed_forward.w_2.bias          | Audio_Encoder Encoders 48 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  846 | audio_encoder.encoders.48.norm1.weight                   | Audio_Encoder Encoders 48 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  847 | audio_encoder.encoders.48.norm1.bias                     | Audio_Encoder Encoders 48 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  848 | audio_encoder.encoders.48.norm2.weight                   | Audio_Encoder Encoders 48 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  849 | audio_encoder.encoders.48.norm2.bias                     | Audio_Encoder Encoders 48 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  850 | audio_encoder.tp_encoders.0.self_attn.linear_out.weight  | Audio_Encoder Tp_Encoders 0 Self_Attn Linear_Out (W)  | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  851 | audio_encoder.tp_encoders.0.self_attn.linear_out.bias    | Audio_Encoder Tp_Encoders 0 Self_Attn Linear_Out (B)  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  852 | audio_encoder.tp_encoders.0.self_attn.linear_q.weight    | Audio_Encoder Tp_Encoders 0 Self_Attn Linear_Q (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  853 | audio_encoder.tp_encoders.0.self_attn.linear_k.weight    | Audio_Encoder Tp_Encoders 0 Self_Attn Linear_K (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  854 | audio_encoder.tp_encoders.0.self_attn.linear_v.weight    | Audio_Encoder Tp_Encoders 0 Self_Attn Linear_V (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  855 | audio_encoder.tp_encoders.0.self_attn.linear_q.bias      | Audio_Encoder Tp_Encoders 0 Self_Attn Linear_Q (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  856 | audio_encoder.tp_encoders.0.self_attn.linear_k.bias      | Audio_Encoder Tp_Encoders 0 Self_Attn Linear_K (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  857 | audio_encoder.tp_encoders.0.self_attn.linear_v.bias      | Audio_Encoder Tp_Encoders 0 Self_Attn Linear_V (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  858 | audio_encoder.tp_encoders.0.self_attn.fsmn_block.weight  | Audio_Encoder Tp_Encoders 0 Self_Attn Fsmn_Block (W)  | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  859 | audio_encoder.tp_encoders.0.feed_forward.w_1.weight      | Audio_Encoder Tp_Encoders 0 Feed_Forward W_1 (W)      | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  860 | audio_encoder.tp_encoders.0.feed_forward.w_1.bias        | Audio_Encoder Tp_Encoders 0 Feed_Forward W_1 (B)      | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  861 | audio_encoder.tp_encoders.0.feed_forward.w_2.weight      | Audio_Encoder Tp_Encoders 0 Feed_Forward W_2 (W)      | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  862 | audio_encoder.tp_encoders.0.feed_forward.w_2.bias        | Audio_Encoder Tp_Encoders 0 Feed_Forward W_2 (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  863 | audio_encoder.tp_encoders.0.norm1.weight                 | Audio_Encoder Tp_Encoders 0 Norm1 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  864 | audio_encoder.tp_encoders.0.norm1.bias                   | Audio_Encoder Tp_Encoders 0 Norm1 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  865 | audio_encoder.tp_encoders.0.norm2.weight                 | Audio_Encoder Tp_Encoders 0 Norm2 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  866 | audio_encoder.tp_encoders.0.norm2.bias                   | Audio_Encoder Tp_Encoders 0 Norm2 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  867 | audio_encoder.tp_encoders.1.self_attn.linear_out.weight  | Audio_Encoder Tp_Encoders 1 Self_Attn Linear_Out (W)  | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  868 | audio_encoder.tp_encoders.1.self_attn.linear_out.bias    | Audio_Encoder Tp_Encoders 1 Self_Attn Linear_Out (B)  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  869 | audio_encoder.tp_encoders.1.self_attn.linear_q.weight    | Audio_Encoder Tp_Encoders 1 Self_Attn Linear_Q (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  870 | audio_encoder.tp_encoders.1.self_attn.linear_k.weight    | Audio_Encoder Tp_Encoders 1 Self_Attn Linear_K (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  871 | audio_encoder.tp_encoders.1.self_attn.linear_v.weight    | Audio_Encoder Tp_Encoders 1 Self_Attn Linear_V (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  872 | audio_encoder.tp_encoders.1.self_attn.linear_q.bias      | Audio_Encoder Tp_Encoders 1 Self_Attn Linear_Q (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  873 | audio_encoder.tp_encoders.1.self_attn.linear_k.bias      | Audio_Encoder Tp_Encoders 1 Self_Attn Linear_K (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  874 | audio_encoder.tp_encoders.1.self_attn.linear_v.bias      | Audio_Encoder Tp_Encoders 1 Self_Attn Linear_V (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  875 | audio_encoder.tp_encoders.1.self_attn.fsmn_block.weight  | Audio_Encoder Tp_Encoders 1 Self_Attn Fsmn_Block (W)  | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  876 | audio_encoder.tp_encoders.1.feed_forward.w_1.weight      | Audio_Encoder Tp_Encoders 1 Feed_Forward W_1 (W)      | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  877 | audio_encoder.tp_encoders.1.feed_forward.w_1.bias        | Audio_Encoder Tp_Encoders 1 Feed_Forward W_1 (B)      | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  878 | audio_encoder.tp_encoders.1.feed_forward.w_2.weight      | Audio_Encoder Tp_Encoders 1 Feed_Forward W_2 (W)      | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  879 | audio_encoder.tp_encoders.1.feed_forward.w_2.bias        | Audio_Encoder Tp_Encoders 1 Feed_Forward W_2 (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  880 | audio_encoder.tp_encoders.1.norm1.weight                 | Audio_Encoder Tp_Encoders 1 Norm1 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  881 | audio_encoder.tp_encoders.1.norm1.bias                   | Audio_Encoder Tp_Encoders 1 Norm1 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  882 | audio_encoder.tp_encoders.1.norm2.weight                 | Audio_Encoder Tp_Encoders 1 Norm2 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  883 | audio_encoder.tp_encoders.1.norm2.bias                   | Audio_Encoder Tp_Encoders 1 Norm2 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  884 | audio_encoder.tp_encoders.2.self_attn.linear_out.weight  | Audio_Encoder Tp_Encoders 2 Self_Attn Linear_Out (W)  | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  885 | audio_encoder.tp_encoders.2.self_attn.linear_out.bias    | Audio_Encoder Tp_Encoders 2 Self_Attn Linear_Out (B)  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  886 | audio_encoder.tp_encoders.2.self_attn.linear_q.weight    | Audio_Encoder Tp_Encoders 2 Self_Attn Linear_Q (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  887 | audio_encoder.tp_encoders.2.self_attn.linear_k.weight    | Audio_Encoder Tp_Encoders 2 Self_Attn Linear_K (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  888 | audio_encoder.tp_encoders.2.self_attn.linear_v.weight    | Audio_Encoder Tp_Encoders 2 Self_Attn Linear_V (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  889 | audio_encoder.tp_encoders.2.self_attn.linear_q.bias      | Audio_Encoder Tp_Encoders 2 Self_Attn Linear_Q (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  890 | audio_encoder.tp_encoders.2.self_attn.linear_k.bias      | Audio_Encoder Tp_Encoders 2 Self_Attn Linear_K (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  891 | audio_encoder.tp_encoders.2.self_attn.linear_v.bias      | Audio_Encoder Tp_Encoders 2 Self_Attn Linear_V (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  892 | audio_encoder.tp_encoders.2.self_attn.fsmn_block.weight  | Audio_Encoder Tp_Encoders 2 Self_Attn Fsmn_Block (W)  | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  893 | audio_encoder.tp_encoders.2.feed_forward.w_1.weight      | Audio_Encoder Tp_Encoders 2 Feed_Forward W_1 (W)      | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  894 | audio_encoder.tp_encoders.2.feed_forward.w_1.bias        | Audio_Encoder Tp_Encoders 2 Feed_Forward W_1 (B)      | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  895 | audio_encoder.tp_encoders.2.feed_forward.w_2.weight      | Audio_Encoder Tp_Encoders 2 Feed_Forward W_2 (W)      | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  896 | audio_encoder.tp_encoders.2.feed_forward.w_2.bias        | Audio_Encoder Tp_Encoders 2 Feed_Forward W_2 (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  897 | audio_encoder.tp_encoders.2.norm1.weight                 | Audio_Encoder Tp_Encoders 2 Norm1 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  898 | audio_encoder.tp_encoders.2.norm1.bias                   | Audio_Encoder Tp_Encoders 2 Norm1 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  899 | audio_encoder.tp_encoders.2.norm2.weight                 | Audio_Encoder Tp_Encoders 2 Norm2 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  900 | audio_encoder.tp_encoders.2.norm2.bias                   | Audio_Encoder Tp_Encoders 2 Norm2 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  901 | audio_encoder.tp_encoders.3.self_attn.linear_out.weight  | Audio_Encoder Tp_Encoders 3 Self_Attn Linear_Out (W)  | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  902 | audio_encoder.tp_encoders.3.self_attn.linear_out.bias    | Audio_Encoder Tp_Encoders 3 Self_Attn Linear_Out (B)  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  903 | audio_encoder.tp_encoders.3.self_attn.linear_q.weight    | Audio_Encoder Tp_Encoders 3 Self_Attn Linear_Q (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  904 | audio_encoder.tp_encoders.3.self_attn.linear_k.weight    | Audio_Encoder Tp_Encoders 3 Self_Attn Linear_K (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  905 | audio_encoder.tp_encoders.3.self_attn.linear_v.weight    | Audio_Encoder Tp_Encoders 3 Self_Attn Linear_V (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  906 | audio_encoder.tp_encoders.3.self_attn.linear_q.bias      | Audio_Encoder Tp_Encoders 3 Self_Attn Linear_Q (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  907 | audio_encoder.tp_encoders.3.self_attn.linear_k.bias      | Audio_Encoder Tp_Encoders 3 Self_Attn Linear_K (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  908 | audio_encoder.tp_encoders.3.self_attn.linear_v.bias      | Audio_Encoder Tp_Encoders 3 Self_Attn Linear_V (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  909 | audio_encoder.tp_encoders.3.self_attn.fsmn_block.weight  | Audio_Encoder Tp_Encoders 3 Self_Attn Fsmn_Block (W)  | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  910 | audio_encoder.tp_encoders.3.feed_forward.w_1.weight      | Audio_Encoder Tp_Encoders 3 Feed_Forward W_1 (W)      | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  911 | audio_encoder.tp_encoders.3.feed_forward.w_1.bias        | Audio_Encoder Tp_Encoders 3 Feed_Forward W_1 (B)      | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  912 | audio_encoder.tp_encoders.3.feed_forward.w_2.weight      | Audio_Encoder Tp_Encoders 3 Feed_Forward W_2 (W)      | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  913 | audio_encoder.tp_encoders.3.feed_forward.w_2.bias        | Audio_Encoder Tp_Encoders 3 Feed_Forward W_2 (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  914 | audio_encoder.tp_encoders.3.norm1.weight                 | Audio_Encoder Tp_Encoders 3 Norm1 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  915 | audio_encoder.tp_encoders.3.norm1.bias                   | Audio_Encoder Tp_Encoders 3 Norm1 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  916 | audio_encoder.tp_encoders.3.norm2.weight                 | Audio_Encoder Tp_Encoders 3 Norm2 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  917 | audio_encoder.tp_encoders.3.norm2.bias                   | Audio_Encoder Tp_Encoders 3 Norm2 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  918 | audio_encoder.tp_encoders.4.self_attn.linear_out.weight  | Audio_Encoder Tp_Encoders 4 Self_Attn Linear_Out (W)  | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  919 | audio_encoder.tp_encoders.4.self_attn.linear_out.bias    | Audio_Encoder Tp_Encoders 4 Self_Attn Linear_Out (B)  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  920 | audio_encoder.tp_encoders.4.self_attn.linear_q.weight    | Audio_Encoder Tp_Encoders 4 Self_Attn Linear_Q (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  921 | audio_encoder.tp_encoders.4.self_attn.linear_k.weight    | Audio_Encoder Tp_Encoders 4 Self_Attn Linear_K (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  922 | audio_encoder.tp_encoders.4.self_attn.linear_v.weight    | Audio_Encoder Tp_Encoders 4 Self_Attn Linear_V (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  923 | audio_encoder.tp_encoders.4.self_attn.linear_q.bias      | Audio_Encoder Tp_Encoders 4 Self_Attn Linear_Q (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  924 | audio_encoder.tp_encoders.4.self_attn.linear_k.bias      | Audio_Encoder Tp_Encoders 4 Self_Attn Linear_K (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  925 | audio_encoder.tp_encoders.4.self_attn.linear_v.bias      | Audio_Encoder Tp_Encoders 4 Self_Attn Linear_V (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  926 | audio_encoder.tp_encoders.4.self_attn.fsmn_block.weight  | Audio_Encoder Tp_Encoders 4 Self_Attn Fsmn_Block (W)  | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  927 | audio_encoder.tp_encoders.4.feed_forward.w_1.weight      | Audio_Encoder Tp_Encoders 4 Feed_Forward W_1 (W)      | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  928 | audio_encoder.tp_encoders.4.feed_forward.w_1.bias        | Audio_Encoder Tp_Encoders 4 Feed_Forward W_1 (B)      | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  929 | audio_encoder.tp_encoders.4.feed_forward.w_2.weight      | Audio_Encoder Tp_Encoders 4 Feed_Forward W_2 (W)      | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  930 | audio_encoder.tp_encoders.4.feed_forward.w_2.bias        | Audio_Encoder Tp_Encoders 4 Feed_Forward W_2 (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  931 | audio_encoder.tp_encoders.4.norm1.weight                 | Audio_Encoder Tp_Encoders 4 Norm1 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  932 | audio_encoder.tp_encoders.4.norm1.bias                   | Audio_Encoder Tp_Encoders 4 Norm1 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  933 | audio_encoder.tp_encoders.4.norm2.weight                 | Audio_Encoder Tp_Encoders 4 Norm2 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  934 | audio_encoder.tp_encoders.4.norm2.bias                   | Audio_Encoder Tp_Encoders 4 Norm2 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  935 | audio_encoder.tp_encoders.5.self_attn.linear_out.weight  | Audio_Encoder Tp_Encoders 5 Self_Attn Linear_Out (W)  | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  936 | audio_encoder.tp_encoders.5.self_attn.linear_out.bias    | Audio_Encoder Tp_Encoders 5 Self_Attn Linear_Out (B)  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  937 | audio_encoder.tp_encoders.5.self_attn.linear_q.weight    | Audio_Encoder Tp_Encoders 5 Self_Attn Linear_Q (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  938 | audio_encoder.tp_encoders.5.self_attn.linear_k.weight    | Audio_Encoder Tp_Encoders 5 Self_Attn Linear_K (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  939 | audio_encoder.tp_encoders.5.self_attn.linear_v.weight    | Audio_Encoder Tp_Encoders 5 Self_Attn Linear_V (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  940 | audio_encoder.tp_encoders.5.self_attn.linear_q.bias      | Audio_Encoder Tp_Encoders 5 Self_Attn Linear_Q (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  941 | audio_encoder.tp_encoders.5.self_attn.linear_k.bias      | Audio_Encoder Tp_Encoders 5 Self_Attn Linear_K (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  942 | audio_encoder.tp_encoders.5.self_attn.linear_v.bias      | Audio_Encoder Tp_Encoders 5 Self_Attn Linear_V (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  943 | audio_encoder.tp_encoders.5.self_attn.fsmn_block.weight  | Audio_Encoder Tp_Encoders 5 Self_Attn Fsmn_Block (W)  | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  944 | audio_encoder.tp_encoders.5.feed_forward.w_1.weight      | Audio_Encoder Tp_Encoders 5 Feed_Forward W_1 (W)      | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  945 | audio_encoder.tp_encoders.5.feed_forward.w_1.bias        | Audio_Encoder Tp_Encoders 5 Feed_Forward W_1 (B)      | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  946 | audio_encoder.tp_encoders.5.feed_forward.w_2.weight      | Audio_Encoder Tp_Encoders 5 Feed_Forward W_2 (W)      | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  947 | audio_encoder.tp_encoders.5.feed_forward.w_2.bias        | Audio_Encoder Tp_Encoders 5 Feed_Forward W_2 (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  948 | audio_encoder.tp_encoders.5.norm1.weight                 | Audio_Encoder Tp_Encoders 5 Norm1 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  949 | audio_encoder.tp_encoders.5.norm1.bias                   | Audio_Encoder Tp_Encoders 5 Norm1 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  950 | audio_encoder.tp_encoders.5.norm2.weight                 | Audio_Encoder Tp_Encoders 5 Norm2 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  951 | audio_encoder.tp_encoders.5.norm2.bias                   | Audio_Encoder Tp_Encoders 5 Norm2 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  952 | audio_encoder.tp_encoders.6.self_attn.linear_out.weight  | Audio_Encoder Tp_Encoders 6 Self_Attn Linear_Out (W)  | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  953 | audio_encoder.tp_encoders.6.self_attn.linear_out.bias    | Audio_Encoder Tp_Encoders 6 Self_Attn Linear_Out (B)  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  954 | audio_encoder.tp_encoders.6.self_attn.linear_q.weight    | Audio_Encoder Tp_Encoders 6 Self_Attn Linear_Q (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  955 | audio_encoder.tp_encoders.6.self_attn.linear_k.weight    | Audio_Encoder Tp_Encoders 6 Self_Attn Linear_K (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  956 | audio_encoder.tp_encoders.6.self_attn.linear_v.weight    | Audio_Encoder Tp_Encoders 6 Self_Attn Linear_V (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  957 | audio_encoder.tp_encoders.6.self_attn.linear_q.bias      | Audio_Encoder Tp_Encoders 6 Self_Attn Linear_Q (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  958 | audio_encoder.tp_encoders.6.self_attn.linear_k.bias      | Audio_Encoder Tp_Encoders 6 Self_Attn Linear_K (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  959 | audio_encoder.tp_encoders.6.self_attn.linear_v.bias      | Audio_Encoder Tp_Encoders 6 Self_Attn Linear_V (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  960 | audio_encoder.tp_encoders.6.self_attn.fsmn_block.weight  | Audio_Encoder Tp_Encoders 6 Self_Attn Fsmn_Block (W)  | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  961 | audio_encoder.tp_encoders.6.feed_forward.w_1.weight      | Audio_Encoder Tp_Encoders 6 Feed_Forward W_1 (W)      | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  962 | audio_encoder.tp_encoders.6.feed_forward.w_1.bias        | Audio_Encoder Tp_Encoders 6 Feed_Forward W_1 (B)      | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  963 | audio_encoder.tp_encoders.6.feed_forward.w_2.weight      | Audio_Encoder Tp_Encoders 6 Feed_Forward W_2 (W)      | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  964 | audio_encoder.tp_encoders.6.feed_forward.w_2.bias        | Audio_Encoder Tp_Encoders 6 Feed_Forward W_2 (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  965 | audio_encoder.tp_encoders.6.norm1.weight                 | Audio_Encoder Tp_Encoders 6 Norm1 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  966 | audio_encoder.tp_encoders.6.norm1.bias                   | Audio_Encoder Tp_Encoders 6 Norm1 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  967 | audio_encoder.tp_encoders.6.norm2.weight                 | Audio_Encoder Tp_Encoders 6 Norm2 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  968 | audio_encoder.tp_encoders.6.norm2.bias                   | Audio_Encoder Tp_Encoders 6 Norm2 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  969 | audio_encoder.tp_encoders.7.self_attn.linear_out.weight  | Audio_Encoder Tp_Encoders 7 Self_Attn Linear_Out (W)  | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  970 | audio_encoder.tp_encoders.7.self_attn.linear_out.bias    | Audio_Encoder Tp_Encoders 7 Self_Attn Linear_Out (B)  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  971 | audio_encoder.tp_encoders.7.self_attn.linear_q.weight    | Audio_Encoder Tp_Encoders 7 Self_Attn Linear_Q (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  972 | audio_encoder.tp_encoders.7.self_attn.linear_k.weight    | Audio_Encoder Tp_Encoders 7 Self_Attn Linear_K (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  973 | audio_encoder.tp_encoders.7.self_attn.linear_v.weight    | Audio_Encoder Tp_Encoders 7 Self_Attn Linear_V (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  974 | audio_encoder.tp_encoders.7.self_attn.linear_q.bias      | Audio_Encoder Tp_Encoders 7 Self_Attn Linear_Q (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  975 | audio_encoder.tp_encoders.7.self_attn.linear_k.bias      | Audio_Encoder Tp_Encoders 7 Self_Attn Linear_K (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  976 | audio_encoder.tp_encoders.7.self_attn.linear_v.bias      | Audio_Encoder Tp_Encoders 7 Self_Attn Linear_V (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  977 | audio_encoder.tp_encoders.7.self_attn.fsmn_block.weight  | Audio_Encoder Tp_Encoders 7 Self_Attn Fsmn_Block (W)  | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  978 | audio_encoder.tp_encoders.7.feed_forward.w_1.weight      | Audio_Encoder Tp_Encoders 7 Feed_Forward W_1 (W)      | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  979 | audio_encoder.tp_encoders.7.feed_forward.w_1.bias        | Audio_Encoder Tp_Encoders 7 Feed_Forward W_1 (B)      | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  980 | audio_encoder.tp_encoders.7.feed_forward.w_2.weight      | Audio_Encoder Tp_Encoders 7 Feed_Forward W_2 (W)      | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  981 | audio_encoder.tp_encoders.7.feed_forward.w_2.bias        | Audio_Encoder Tp_Encoders 7 Feed_Forward W_2 (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  982 | audio_encoder.tp_encoders.7.norm1.weight                 | Audio_Encoder Tp_Encoders 7 Norm1 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  983 | audio_encoder.tp_encoders.7.norm1.bias                   | Audio_Encoder Tp_Encoders 7 Norm1 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  984 | audio_encoder.tp_encoders.7.norm2.weight                 | Audio_Encoder Tp_Encoders 7 Norm2 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  985 | audio_encoder.tp_encoders.7.norm2.bias                   | Audio_Encoder Tp_Encoders 7 Norm2 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  986 | audio_encoder.tp_encoders.8.self_attn.linear_out.weight  | Audio_Encoder Tp_Encoders 8 Self_Attn Linear_Out (W)  | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  987 | audio_encoder.tp_encoders.8.self_attn.linear_out.bias    | Audio_Encoder Tp_Encoders 8 Self_Attn Linear_Out (B)  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  988 | audio_encoder.tp_encoders.8.self_attn.linear_q.weight    | Audio_Encoder Tp_Encoders 8 Self_Attn Linear_Q (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  989 | audio_encoder.tp_encoders.8.self_attn.linear_k.weight    | Audio_Encoder Tp_Encoders 8 Self_Attn Linear_K (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  990 | audio_encoder.tp_encoders.8.self_attn.linear_v.weight    | Audio_Encoder Tp_Encoders 8 Self_Attn Linear_V (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  991 | audio_encoder.tp_encoders.8.self_attn.linear_q.bias      | Audio_Encoder Tp_Encoders 8 Self_Attn Linear_Q (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  992 | audio_encoder.tp_encoders.8.self_attn.linear_k.bias      | Audio_Encoder Tp_Encoders 8 Self_Attn Linear_K (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  993 | audio_encoder.tp_encoders.8.self_attn.linear_v.bias      | Audio_Encoder Tp_Encoders 8 Self_Attn Linear_V (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  994 | audio_encoder.tp_encoders.8.self_attn.fsmn_block.weight  | Audio_Encoder Tp_Encoders 8 Self_Attn Fsmn_Block (W)  | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  995 | audio_encoder.tp_encoders.8.feed_forward.w_1.weight      | Audio_Encoder Tp_Encoders 8 Feed_Forward W_1 (W)      | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  996 | audio_encoder.tp_encoders.8.feed_forward.w_1.bias        | Audio_Encoder Tp_Encoders 8 Feed_Forward W_1 (B)      | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  997 | audio_encoder.tp_encoders.8.feed_forward.w_2.weight      | Audio_Encoder Tp_Encoders 8 Feed_Forward W_2 (W)      | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  998 | audio_encoder.tp_encoders.8.feed_forward.w_2.bias        | Audio_Encoder Tp_Encoders 8 Feed_Forward W_2 (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  999 | audio_encoder.tp_encoders.8.norm1.weight                 | Audio_Encoder Tp_Encoders 8 Norm1 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1000 | audio_encoder.tp_encoders.8.norm1.bias                   | Audio_Encoder Tp_Encoders 8 Norm1 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1001 | audio_encoder.tp_encoders.8.norm2.weight                 | Audio_Encoder Tp_Encoders 8 Norm2 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1002 | audio_encoder.tp_encoders.8.norm2.bias                   | Audio_Encoder Tp_Encoders 8 Norm2 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1003 | audio_encoder.tp_encoders.9.self_attn.linear_out.weight  | Audio_Encoder Tp_Encoders 9 Self_Attn Linear_Out (W)  | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1004 | audio_encoder.tp_encoders.9.self_attn.linear_out.bias    | Audio_Encoder Tp_Encoders 9 Self_Attn Linear_Out (B)  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1005 | audio_encoder.tp_encoders.9.self_attn.linear_q.weight    | Audio_Encoder Tp_Encoders 9 Self_Attn Linear_Q (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1006 | audio_encoder.tp_encoders.9.self_attn.linear_k.weight    | Audio_Encoder Tp_Encoders 9 Self_Attn Linear_K (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1007 | audio_encoder.tp_encoders.9.self_attn.linear_v.weight    | Audio_Encoder Tp_Encoders 9 Self_Attn Linear_V (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1008 | audio_encoder.tp_encoders.9.self_attn.linear_q.bias      | Audio_Encoder Tp_Encoders 9 Self_Attn Linear_Q (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1009 | audio_encoder.tp_encoders.9.self_attn.linear_k.bias      | Audio_Encoder Tp_Encoders 9 Self_Attn Linear_K (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1010 | audio_encoder.tp_encoders.9.self_attn.linear_v.bias      | Audio_Encoder Tp_Encoders 9 Self_Attn Linear_V (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1011 | audio_encoder.tp_encoders.9.self_attn.fsmn_block.weight  | Audio_Encoder Tp_Encoders 9 Self_Attn Fsmn_Block (W)  | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
| 1012 | audio_encoder.tp_encoders.9.feed_forward.w_1.weight      | Audio_Encoder Tp_Encoders 9 Feed_Forward W_1 (W)      | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
| 1013 | audio_encoder.tp_encoders.9.feed_forward.w_1.bias        | Audio_Encoder Tp_Encoders 9 Feed_Forward W_1 (B)      | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
| 1014 | audio_encoder.tp_encoders.9.feed_forward.w_2.weight      | Audio_Encoder Tp_Encoders 9 Feed_Forward W_2 (W)      | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
| 1015 | audio_encoder.tp_encoders.9.feed_forward.w_2.bias        | Audio_Encoder Tp_Encoders 9 Feed_Forward W_2 (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1016 | audio_encoder.tp_encoders.9.norm1.weight                 | Audio_Encoder Tp_Encoders 9 Norm1 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1017 | audio_encoder.tp_encoders.9.norm1.bias                   | Audio_Encoder Tp_Encoders 9 Norm1 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1018 | audio_encoder.tp_encoders.9.norm2.weight                 | Audio_Encoder Tp_Encoders 9 Norm2 (W)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1019 | audio_encoder.tp_encoders.9.norm2.bias                   | Audio_Encoder Tp_Encoders 9 Norm2 (B)                 | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1020 | audio_encoder.tp_encoders.10.self_attn.linear_out.weight | Audio_Encoder Tp_Encoders 10 Self_Attn Linear_Out (W) | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1021 | audio_encoder.tp_encoders.10.self_attn.linear_out.bias   | Audio_Encoder Tp_Encoders 10 Self_Attn Linear_Out (B) | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1022 | audio_encoder.tp_encoders.10.self_attn.linear_q.weight   | Audio_Encoder Tp_Encoders 10 Self_Attn Linear_Q (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1023 | audio_encoder.tp_encoders.10.self_attn.linear_k.weight   | Audio_Encoder Tp_Encoders 10 Self_Attn Linear_K (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1024 | audio_encoder.tp_encoders.10.self_attn.linear_v.weight   | Audio_Encoder Tp_Encoders 10 Self_Attn Linear_V (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1025 | audio_encoder.tp_encoders.10.self_attn.linear_q.bias     | Audio_Encoder Tp_Encoders 10 Self_Attn Linear_Q (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1026 | audio_encoder.tp_encoders.10.self_attn.linear_k.bias     | Audio_Encoder Tp_Encoders 10 Self_Attn Linear_K (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1027 | audio_encoder.tp_encoders.10.self_attn.linear_v.bias     | Audio_Encoder Tp_Encoders 10 Self_Attn Linear_V (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1028 | audio_encoder.tp_encoders.10.self_attn.fsmn_block.weight | Audio_Encoder Tp_Encoders 10 Self_Attn Fsmn_Block (W) | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
| 1029 | audio_encoder.tp_encoders.10.feed_forward.w_1.weight     | Audio_Encoder Tp_Encoders 10 Feed_Forward W_1 (W)     | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
| 1030 | audio_encoder.tp_encoders.10.feed_forward.w_1.bias       | Audio_Encoder Tp_Encoders 10 Feed_Forward W_1 (B)     | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
| 1031 | audio_encoder.tp_encoders.10.feed_forward.w_2.weight     | Audio_Encoder Tp_Encoders 10 Feed_Forward W_2 (W)     | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
| 1032 | audio_encoder.tp_encoders.10.feed_forward.w_2.bias       | Audio_Encoder Tp_Encoders 10 Feed_Forward W_2 (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1033 | audio_encoder.tp_encoders.10.norm1.weight                | Audio_Encoder Tp_Encoders 10 Norm1 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1034 | audio_encoder.tp_encoders.10.norm1.bias                  | Audio_Encoder Tp_Encoders 10 Norm1 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1035 | audio_encoder.tp_encoders.10.norm2.weight                | Audio_Encoder Tp_Encoders 10 Norm2 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1036 | audio_encoder.tp_encoders.10.norm2.bias                  | Audio_Encoder Tp_Encoders 10 Norm2 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1037 | audio_encoder.tp_encoders.11.self_attn.linear_out.weight | Audio_Encoder Tp_Encoders 11 Self_Attn Linear_Out (W) | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1038 | audio_encoder.tp_encoders.11.self_attn.linear_out.bias   | Audio_Encoder Tp_Encoders 11 Self_Attn Linear_Out (B) | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1039 | audio_encoder.tp_encoders.11.self_attn.linear_q.weight   | Audio_Encoder Tp_Encoders 11 Self_Attn Linear_Q (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1040 | audio_encoder.tp_encoders.11.self_attn.linear_k.weight   | Audio_Encoder Tp_Encoders 11 Self_Attn Linear_K (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1041 | audio_encoder.tp_encoders.11.self_attn.linear_v.weight   | Audio_Encoder Tp_Encoders 11 Self_Attn Linear_V (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1042 | audio_encoder.tp_encoders.11.self_attn.linear_q.bias     | Audio_Encoder Tp_Encoders 11 Self_Attn Linear_Q (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1043 | audio_encoder.tp_encoders.11.self_attn.linear_k.bias     | Audio_Encoder Tp_Encoders 11 Self_Attn Linear_K (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1044 | audio_encoder.tp_encoders.11.self_attn.linear_v.bias     | Audio_Encoder Tp_Encoders 11 Self_Attn Linear_V (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1045 | audio_encoder.tp_encoders.11.self_attn.fsmn_block.weight | Audio_Encoder Tp_Encoders 11 Self_Attn Fsmn_Block (W) | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
| 1046 | audio_encoder.tp_encoders.11.feed_forward.w_1.weight     | Audio_Encoder Tp_Encoders 11 Feed_Forward W_1 (W)     | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
| 1047 | audio_encoder.tp_encoders.11.feed_forward.w_1.bias       | Audio_Encoder Tp_Encoders 11 Feed_Forward W_1 (B)     | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
| 1048 | audio_encoder.tp_encoders.11.feed_forward.w_2.weight     | Audio_Encoder Tp_Encoders 11 Feed_Forward W_2 (W)     | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
| 1049 | audio_encoder.tp_encoders.11.feed_forward.w_2.bias       | Audio_Encoder Tp_Encoders 11 Feed_Forward W_2 (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1050 | audio_encoder.tp_encoders.11.norm1.weight                | Audio_Encoder Tp_Encoders 11 Norm1 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1051 | audio_encoder.tp_encoders.11.norm1.bias                  | Audio_Encoder Tp_Encoders 11 Norm1 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1052 | audio_encoder.tp_encoders.11.norm2.weight                | Audio_Encoder Tp_Encoders 11 Norm2 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1053 | audio_encoder.tp_encoders.11.norm2.bias                  | Audio_Encoder Tp_Encoders 11 Norm2 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1054 | audio_encoder.tp_encoders.12.self_attn.linear_out.weight | Audio_Encoder Tp_Encoders 12 Self_Attn Linear_Out (W) | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1055 | audio_encoder.tp_encoders.12.self_attn.linear_out.bias   | Audio_Encoder Tp_Encoders 12 Self_Attn Linear_Out (B) | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1056 | audio_encoder.tp_encoders.12.self_attn.linear_q.weight   | Audio_Encoder Tp_Encoders 12 Self_Attn Linear_Q (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1057 | audio_encoder.tp_encoders.12.self_attn.linear_k.weight   | Audio_Encoder Tp_Encoders 12 Self_Attn Linear_K (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1058 | audio_encoder.tp_encoders.12.self_attn.linear_v.weight   | Audio_Encoder Tp_Encoders 12 Self_Attn Linear_V (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1059 | audio_encoder.tp_encoders.12.self_attn.linear_q.bias     | Audio_Encoder Tp_Encoders 12 Self_Attn Linear_Q (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1060 | audio_encoder.tp_encoders.12.self_attn.linear_k.bias     | Audio_Encoder Tp_Encoders 12 Self_Attn Linear_K (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1061 | audio_encoder.tp_encoders.12.self_attn.linear_v.bias     | Audio_Encoder Tp_Encoders 12 Self_Attn Linear_V (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1062 | audio_encoder.tp_encoders.12.self_attn.fsmn_block.weight | Audio_Encoder Tp_Encoders 12 Self_Attn Fsmn_Block (W) | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
| 1063 | audio_encoder.tp_encoders.12.feed_forward.w_1.weight     | Audio_Encoder Tp_Encoders 12 Feed_Forward W_1 (W)     | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
| 1064 | audio_encoder.tp_encoders.12.feed_forward.w_1.bias       | Audio_Encoder Tp_Encoders 12 Feed_Forward W_1 (B)     | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
| 1065 | audio_encoder.tp_encoders.12.feed_forward.w_2.weight     | Audio_Encoder Tp_Encoders 12 Feed_Forward W_2 (W)     | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
| 1066 | audio_encoder.tp_encoders.12.feed_forward.w_2.bias       | Audio_Encoder Tp_Encoders 12 Feed_Forward W_2 (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1067 | audio_encoder.tp_encoders.12.norm1.weight                | Audio_Encoder Tp_Encoders 12 Norm1 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1068 | audio_encoder.tp_encoders.12.norm1.bias                  | Audio_Encoder Tp_Encoders 12 Norm1 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1069 | audio_encoder.tp_encoders.12.norm2.weight                | Audio_Encoder Tp_Encoders 12 Norm2 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1070 | audio_encoder.tp_encoders.12.norm2.bias                  | Audio_Encoder Tp_Encoders 12 Norm2 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1071 | audio_encoder.tp_encoders.13.self_attn.linear_out.weight | Audio_Encoder Tp_Encoders 13 Self_Attn Linear_Out (W) | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1072 | audio_encoder.tp_encoders.13.self_attn.linear_out.bias   | Audio_Encoder Tp_Encoders 13 Self_Attn Linear_Out (B) | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1073 | audio_encoder.tp_encoders.13.self_attn.linear_q.weight   | Audio_Encoder Tp_Encoders 13 Self_Attn Linear_Q (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1074 | audio_encoder.tp_encoders.13.self_attn.linear_k.weight   | Audio_Encoder Tp_Encoders 13 Self_Attn Linear_K (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1075 | audio_encoder.tp_encoders.13.self_attn.linear_v.weight   | Audio_Encoder Tp_Encoders 13 Self_Attn Linear_V (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1076 | audio_encoder.tp_encoders.13.self_attn.linear_q.bias     | Audio_Encoder Tp_Encoders 13 Self_Attn Linear_Q (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1077 | audio_encoder.tp_encoders.13.self_attn.linear_k.bias     | Audio_Encoder Tp_Encoders 13 Self_Attn Linear_K (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1078 | audio_encoder.tp_encoders.13.self_attn.linear_v.bias     | Audio_Encoder Tp_Encoders 13 Self_Attn Linear_V (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1079 | audio_encoder.tp_encoders.13.self_attn.fsmn_block.weight | Audio_Encoder Tp_Encoders 13 Self_Attn Fsmn_Block (W) | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
| 1080 | audio_encoder.tp_encoders.13.feed_forward.w_1.weight     | Audio_Encoder Tp_Encoders 13 Feed_Forward W_1 (W)     | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
| 1081 | audio_encoder.tp_encoders.13.feed_forward.w_1.bias       | Audio_Encoder Tp_Encoders 13 Feed_Forward W_1 (B)     | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
| 1082 | audio_encoder.tp_encoders.13.feed_forward.w_2.weight     | Audio_Encoder Tp_Encoders 13 Feed_Forward W_2 (W)     | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
| 1083 | audio_encoder.tp_encoders.13.feed_forward.w_2.bias       | Audio_Encoder Tp_Encoders 13 Feed_Forward W_2 (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1084 | audio_encoder.tp_encoders.13.norm1.weight                | Audio_Encoder Tp_Encoders 13 Norm1 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1085 | audio_encoder.tp_encoders.13.norm1.bias                  | Audio_Encoder Tp_Encoders 13 Norm1 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1086 | audio_encoder.tp_encoders.13.norm2.weight                | Audio_Encoder Tp_Encoders 13 Norm2 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1087 | audio_encoder.tp_encoders.13.norm2.bias                  | Audio_Encoder Tp_Encoders 13 Norm2 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1088 | audio_encoder.tp_encoders.14.self_attn.linear_out.weight | Audio_Encoder Tp_Encoders 14 Self_Attn Linear_Out (W) | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1089 | audio_encoder.tp_encoders.14.self_attn.linear_out.bias   | Audio_Encoder Tp_Encoders 14 Self_Attn Linear_Out (B) | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1090 | audio_encoder.tp_encoders.14.self_attn.linear_q.weight   | Audio_Encoder Tp_Encoders 14 Self_Attn Linear_Q (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1091 | audio_encoder.tp_encoders.14.self_attn.linear_k.weight   | Audio_Encoder Tp_Encoders 14 Self_Attn Linear_K (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1092 | audio_encoder.tp_encoders.14.self_attn.linear_v.weight   | Audio_Encoder Tp_Encoders 14 Self_Attn Linear_V (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1093 | audio_encoder.tp_encoders.14.self_attn.linear_q.bias     | Audio_Encoder Tp_Encoders 14 Self_Attn Linear_Q (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1094 | audio_encoder.tp_encoders.14.self_attn.linear_k.bias     | Audio_Encoder Tp_Encoders 14 Self_Attn Linear_K (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1095 | audio_encoder.tp_encoders.14.self_attn.linear_v.bias     | Audio_Encoder Tp_Encoders 14 Self_Attn Linear_V (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1096 | audio_encoder.tp_encoders.14.self_attn.fsmn_block.weight | Audio_Encoder Tp_Encoders 14 Self_Attn Fsmn_Block (W) | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
| 1097 | audio_encoder.tp_encoders.14.feed_forward.w_1.weight     | Audio_Encoder Tp_Encoders 14 Feed_Forward W_1 (W)     | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
| 1098 | audio_encoder.tp_encoders.14.feed_forward.w_1.bias       | Audio_Encoder Tp_Encoders 14 Feed_Forward W_1 (B)     | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
| 1099 | audio_encoder.tp_encoders.14.feed_forward.w_2.weight     | Audio_Encoder Tp_Encoders 14 Feed_Forward W_2 (W)     | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
| 1100 | audio_encoder.tp_encoders.14.feed_forward.w_2.bias       | Audio_Encoder Tp_Encoders 14 Feed_Forward W_2 (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1101 | audio_encoder.tp_encoders.14.norm1.weight                | Audio_Encoder Tp_Encoders 14 Norm1 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1102 | audio_encoder.tp_encoders.14.norm1.bias                  | Audio_Encoder Tp_Encoders 14 Norm1 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1103 | audio_encoder.tp_encoders.14.norm2.weight                | Audio_Encoder Tp_Encoders 14 Norm2 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1104 | audio_encoder.tp_encoders.14.norm2.bias                  | Audio_Encoder Tp_Encoders 14 Norm2 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1105 | audio_encoder.tp_encoders.15.self_attn.linear_out.weight | Audio_Encoder Tp_Encoders 15 Self_Attn Linear_Out (W) | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1106 | audio_encoder.tp_encoders.15.self_attn.linear_out.bias   | Audio_Encoder Tp_Encoders 15 Self_Attn Linear_Out (B) | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1107 | audio_encoder.tp_encoders.15.self_attn.linear_q.weight   | Audio_Encoder Tp_Encoders 15 Self_Attn Linear_Q (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1108 | audio_encoder.tp_encoders.15.self_attn.linear_k.weight   | Audio_Encoder Tp_Encoders 15 Self_Attn Linear_K (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1109 | audio_encoder.tp_encoders.15.self_attn.linear_v.weight   | Audio_Encoder Tp_Encoders 15 Self_Attn Linear_V (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1110 | audio_encoder.tp_encoders.15.self_attn.linear_q.bias     | Audio_Encoder Tp_Encoders 15 Self_Attn Linear_Q (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1111 | audio_encoder.tp_encoders.15.self_attn.linear_k.bias     | Audio_Encoder Tp_Encoders 15 Self_Attn Linear_K (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1112 | audio_encoder.tp_encoders.15.self_attn.linear_v.bias     | Audio_Encoder Tp_Encoders 15 Self_Attn Linear_V (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1113 | audio_encoder.tp_encoders.15.self_attn.fsmn_block.weight | Audio_Encoder Tp_Encoders 15 Self_Attn Fsmn_Block (W) | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
| 1114 | audio_encoder.tp_encoders.15.feed_forward.w_1.weight     | Audio_Encoder Tp_Encoders 15 Feed_Forward W_1 (W)     | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
| 1115 | audio_encoder.tp_encoders.15.feed_forward.w_1.bias       | Audio_Encoder Tp_Encoders 15 Feed_Forward W_1 (B)     | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
| 1116 | audio_encoder.tp_encoders.15.feed_forward.w_2.weight     | Audio_Encoder Tp_Encoders 15 Feed_Forward W_2 (W)     | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
| 1117 | audio_encoder.tp_encoders.15.feed_forward.w_2.bias       | Audio_Encoder Tp_Encoders 15 Feed_Forward W_2 (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1118 | audio_encoder.tp_encoders.15.norm1.weight                | Audio_Encoder Tp_Encoders 15 Norm1 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1119 | audio_encoder.tp_encoders.15.norm1.bias                  | Audio_Encoder Tp_Encoders 15 Norm1 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1120 | audio_encoder.tp_encoders.15.norm2.weight                | Audio_Encoder Tp_Encoders 15 Norm2 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1121 | audio_encoder.tp_encoders.15.norm2.bias                  | Audio_Encoder Tp_Encoders 15 Norm2 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1122 | audio_encoder.tp_encoders.16.self_attn.linear_out.weight | Audio_Encoder Tp_Encoders 16 Self_Attn Linear_Out (W) | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1123 | audio_encoder.tp_encoders.16.self_attn.linear_out.bias   | Audio_Encoder Tp_Encoders 16 Self_Attn Linear_Out (B) | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1124 | audio_encoder.tp_encoders.16.self_attn.linear_q.weight   | Audio_Encoder Tp_Encoders 16 Self_Attn Linear_Q (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1125 | audio_encoder.tp_encoders.16.self_attn.linear_k.weight   | Audio_Encoder Tp_Encoders 16 Self_Attn Linear_K (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1126 | audio_encoder.tp_encoders.16.self_attn.linear_v.weight   | Audio_Encoder Tp_Encoders 16 Self_Attn Linear_V (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1127 | audio_encoder.tp_encoders.16.self_attn.linear_q.bias     | Audio_Encoder Tp_Encoders 16 Self_Attn Linear_Q (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1128 | audio_encoder.tp_encoders.16.self_attn.linear_k.bias     | Audio_Encoder Tp_Encoders 16 Self_Attn Linear_K (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1129 | audio_encoder.tp_encoders.16.self_attn.linear_v.bias     | Audio_Encoder Tp_Encoders 16 Self_Attn Linear_V (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1130 | audio_encoder.tp_encoders.16.self_attn.fsmn_block.weight | Audio_Encoder Tp_Encoders 16 Self_Attn Fsmn_Block (W) | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
| 1131 | audio_encoder.tp_encoders.16.feed_forward.w_1.weight     | Audio_Encoder Tp_Encoders 16 Feed_Forward W_1 (W)     | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
| 1132 | audio_encoder.tp_encoders.16.feed_forward.w_1.bias       | Audio_Encoder Tp_Encoders 16 Feed_Forward W_1 (B)     | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
| 1133 | audio_encoder.tp_encoders.16.feed_forward.w_2.weight     | Audio_Encoder Tp_Encoders 16 Feed_Forward W_2 (W)     | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
| 1134 | audio_encoder.tp_encoders.16.feed_forward.w_2.bias       | Audio_Encoder Tp_Encoders 16 Feed_Forward W_2 (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1135 | audio_encoder.tp_encoders.16.norm1.weight                | Audio_Encoder Tp_Encoders 16 Norm1 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1136 | audio_encoder.tp_encoders.16.norm1.bias                  | Audio_Encoder Tp_Encoders 16 Norm1 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1137 | audio_encoder.tp_encoders.16.norm2.weight                | Audio_Encoder Tp_Encoders 16 Norm2 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1138 | audio_encoder.tp_encoders.16.norm2.bias                  | Audio_Encoder Tp_Encoders 16 Norm2 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1139 | audio_encoder.tp_encoders.17.self_attn.linear_out.weight | Audio_Encoder Tp_Encoders 17 Self_Attn Linear_Out (W) | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1140 | audio_encoder.tp_encoders.17.self_attn.linear_out.bias   | Audio_Encoder Tp_Encoders 17 Self_Attn Linear_Out (B) | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1141 | audio_encoder.tp_encoders.17.self_attn.linear_q.weight   | Audio_Encoder Tp_Encoders 17 Self_Attn Linear_Q (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1142 | audio_encoder.tp_encoders.17.self_attn.linear_k.weight   | Audio_Encoder Tp_Encoders 17 Self_Attn Linear_K (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1143 | audio_encoder.tp_encoders.17.self_attn.linear_v.weight   | Audio_Encoder Tp_Encoders 17 Self_Attn Linear_V (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1144 | audio_encoder.tp_encoders.17.self_attn.linear_q.bias     | Audio_Encoder Tp_Encoders 17 Self_Attn Linear_Q (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1145 | audio_encoder.tp_encoders.17.self_attn.linear_k.bias     | Audio_Encoder Tp_Encoders 17 Self_Attn Linear_K (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1146 | audio_encoder.tp_encoders.17.self_attn.linear_v.bias     | Audio_Encoder Tp_Encoders 17 Self_Attn Linear_V (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1147 | audio_encoder.tp_encoders.17.self_attn.fsmn_block.weight | Audio_Encoder Tp_Encoders 17 Self_Attn Fsmn_Block (W) | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
| 1148 | audio_encoder.tp_encoders.17.feed_forward.w_1.weight     | Audio_Encoder Tp_Encoders 17 Feed_Forward W_1 (W)     | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
| 1149 | audio_encoder.tp_encoders.17.feed_forward.w_1.bias       | Audio_Encoder Tp_Encoders 17 Feed_Forward W_1 (B)     | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
| 1150 | audio_encoder.tp_encoders.17.feed_forward.w_2.weight     | Audio_Encoder Tp_Encoders 17 Feed_Forward W_2 (W)     | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
| 1151 | audio_encoder.tp_encoders.17.feed_forward.w_2.bias       | Audio_Encoder Tp_Encoders 17 Feed_Forward W_2 (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1152 | audio_encoder.tp_encoders.17.norm1.weight                | Audio_Encoder Tp_Encoders 17 Norm1 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1153 | audio_encoder.tp_encoders.17.norm1.bias                  | Audio_Encoder Tp_Encoders 17 Norm1 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1154 | audio_encoder.tp_encoders.17.norm2.weight                | Audio_Encoder Tp_Encoders 17 Norm2 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1155 | audio_encoder.tp_encoders.17.norm2.bias                  | Audio_Encoder Tp_Encoders 17 Norm2 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1156 | audio_encoder.tp_encoders.18.self_attn.linear_out.weight | Audio_Encoder Tp_Encoders 18 Self_Attn Linear_Out (W) | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1157 | audio_encoder.tp_encoders.18.self_attn.linear_out.bias   | Audio_Encoder Tp_Encoders 18 Self_Attn Linear_Out (B) | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1158 | audio_encoder.tp_encoders.18.self_attn.linear_q.weight   | Audio_Encoder Tp_Encoders 18 Self_Attn Linear_Q (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1159 | audio_encoder.tp_encoders.18.self_attn.linear_k.weight   | Audio_Encoder Tp_Encoders 18 Self_Attn Linear_K (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1160 | audio_encoder.tp_encoders.18.self_attn.linear_v.weight   | Audio_Encoder Tp_Encoders 18 Self_Attn Linear_V (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1161 | audio_encoder.tp_encoders.18.self_attn.linear_q.bias     | Audio_Encoder Tp_Encoders 18 Self_Attn Linear_Q (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1162 | audio_encoder.tp_encoders.18.self_attn.linear_k.bias     | Audio_Encoder Tp_Encoders 18 Self_Attn Linear_K (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1163 | audio_encoder.tp_encoders.18.self_attn.linear_v.bias     | Audio_Encoder Tp_Encoders 18 Self_Attn Linear_V (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1164 | audio_encoder.tp_encoders.18.self_attn.fsmn_block.weight | Audio_Encoder Tp_Encoders 18 Self_Attn Fsmn_Block (W) | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
| 1165 | audio_encoder.tp_encoders.18.feed_forward.w_1.weight     | Audio_Encoder Tp_Encoders 18 Feed_Forward W_1 (W)     | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
| 1166 | audio_encoder.tp_encoders.18.feed_forward.w_1.bias       | Audio_Encoder Tp_Encoders 18 Feed_Forward W_1 (B)     | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
| 1167 | audio_encoder.tp_encoders.18.feed_forward.w_2.weight     | Audio_Encoder Tp_Encoders 18 Feed_Forward W_2 (W)     | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
| 1168 | audio_encoder.tp_encoders.18.feed_forward.w_2.bias       | Audio_Encoder Tp_Encoders 18 Feed_Forward W_2 (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1169 | audio_encoder.tp_encoders.18.norm1.weight                | Audio_Encoder Tp_Encoders 18 Norm1 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1170 | audio_encoder.tp_encoders.18.norm1.bias                  | Audio_Encoder Tp_Encoders 18 Norm1 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1171 | audio_encoder.tp_encoders.18.norm2.weight                | Audio_Encoder Tp_Encoders 18 Norm2 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1172 | audio_encoder.tp_encoders.18.norm2.bias                  | Audio_Encoder Tp_Encoders 18 Norm2 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1173 | audio_encoder.tp_encoders.19.self_attn.linear_out.weight | Audio_Encoder Tp_Encoders 19 Self_Attn Linear_Out (W) | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1174 | audio_encoder.tp_encoders.19.self_attn.linear_out.bias   | Audio_Encoder Tp_Encoders 19 Self_Attn Linear_Out (B) | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1175 | audio_encoder.tp_encoders.19.self_attn.linear_q.weight   | Audio_Encoder Tp_Encoders 19 Self_Attn Linear_Q (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1176 | audio_encoder.tp_encoders.19.self_attn.linear_k.weight   | Audio_Encoder Tp_Encoders 19 Self_Attn Linear_K (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1177 | audio_encoder.tp_encoders.19.self_attn.linear_v.weight   | Audio_Encoder Tp_Encoders 19 Self_Attn Linear_V (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
| 1178 | audio_encoder.tp_encoders.19.self_attn.linear_q.bias     | Audio_Encoder Tp_Encoders 19 Self_Attn Linear_Q (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1179 | audio_encoder.tp_encoders.19.self_attn.linear_k.bias     | Audio_Encoder Tp_Encoders 19 Self_Attn Linear_K (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1180 | audio_encoder.tp_encoders.19.self_attn.linear_v.bias     | Audio_Encoder Tp_Encoders 19 Self_Attn Linear_V (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1181 | audio_encoder.tp_encoders.19.self_attn.fsmn_block.weight | Audio_Encoder Tp_Encoders 19 Self_Attn Fsmn_Block (W) | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
| 1182 | audio_encoder.tp_encoders.19.feed_forward.w_1.weight     | Audio_Encoder Tp_Encoders 19 Feed_Forward W_1 (W)     | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
| 1183 | audio_encoder.tp_encoders.19.feed_forward.w_1.bias       | Audio_Encoder Tp_Encoders 19 Feed_Forward W_1 (B)     | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
| 1184 | audio_encoder.tp_encoders.19.feed_forward.w_2.weight     | Audio_Encoder Tp_Encoders 19 Feed_Forward W_2 (W)     | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
| 1185 | audio_encoder.tp_encoders.19.feed_forward.w_2.bias       | Audio_Encoder Tp_Encoders 19 Feed_Forward W_2 (B)     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1186 | audio_encoder.tp_encoders.19.norm1.weight                | Audio_Encoder Tp_Encoders 19 Norm1 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1187 | audio_encoder.tp_encoders.19.norm1.bias                  | Audio_Encoder Tp_Encoders 19 Norm1 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1188 | audio_encoder.tp_encoders.19.norm2.weight                | Audio_Encoder Tp_Encoders 19 Norm2 (W)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1189 | audio_encoder.tp_encoders.19.norm2.bias                  | Audio_Encoder Tp_Encoders 19 Norm2 (B)                | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1190 | audio_encoder.after_norm.weight                          | Audio_Encoder After_Norm (W)                          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1191 | audio_encoder.after_norm.bias                            | Audio_Encoder After_Norm (B)                          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1192 | audio_encoder.tp_norm.weight                             | Audio_Encoder Tp_Norm (W)                             | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1193 | audio_encoder.tp_norm.bias                               | Audio_Encoder Tp_Norm (B)                             | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
| 1194 | audio_adaptor.linear1.weight                             | Audio_Adaptor Linear1 (W)                             | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
| 1195 | audio_adaptor.linear1.bias                               | Audio_Adaptor Linear1 (B)                             | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
| 1196 | audio_adaptor.linear2.weight                             | Audio_Adaptor Linear2 (W)                             | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1197 | audio_adaptor.linear2.bias                               | Audio_Adaptor Linear2 (B)                             | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1198 | audio_adaptor.blocks.0.self_attn.linear_q.weight         | Audio_Adaptor Blocks 0 Self_Attn Linear_Q (W)         | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1199 | audio_adaptor.blocks.0.self_attn.linear_q.bias           | Audio_Adaptor Blocks 0 Self_Attn Linear_Q (B)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1200 | audio_adaptor.blocks.0.self_attn.linear_k.weight         | Audio_Adaptor Blocks 0 Self_Attn Linear_K (W)         | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1201 | audio_adaptor.blocks.0.self_attn.linear_k.bias           | Audio_Adaptor Blocks 0 Self_Attn Linear_K (B)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1202 | audio_adaptor.blocks.0.self_attn.linear_v.weight         | Audio_Adaptor Blocks 0 Self_Attn Linear_V (W)         | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1203 | audio_adaptor.blocks.0.self_attn.linear_v.bias           | Audio_Adaptor Blocks 0 Self_Attn Linear_V (B)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1204 | audio_adaptor.blocks.0.self_attn.linear_out.weight       | Audio_Adaptor Blocks 0 Self_Attn Linear_Out (W)       | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1205 | audio_adaptor.blocks.0.self_attn.linear_out.bias         | Audio_Adaptor Blocks 0 Self_Attn Linear_Out (B)       | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1206 | audio_adaptor.blocks.0.feed_forward.w_1.weight           | Audio_Adaptor Blocks 0 Feed_Forward W_1 (W)           | (~262K)    262144 | 1024 x    256 x   1 x 1 | BF16 |
| 1207 | audio_adaptor.blocks.0.feed_forward.w_1.bias             | Audio_Adaptor Blocks 0 Feed_Forward W_1 (B)           | (  256)       256 |  256 x      1 x   1 x 1 | BF16 |
| 1208 | audio_adaptor.blocks.0.feed_forward.w_2.weight           | Audio_Adaptor Blocks 0 Feed_Forward W_2 (W)           | (~262K)    262144 |  256 x   1024 x   1 x 1 | BF16 |
| 1209 | audio_adaptor.blocks.0.feed_forward.w_2.bias             | Audio_Adaptor Blocks 0 Feed_Forward W_2 (B)           | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1210 | audio_adaptor.blocks.0.norm1.weight                      | Audio_Adaptor Blocks 0 Norm1 (W)                      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1211 | audio_adaptor.blocks.0.norm1.bias                        | Audio_Adaptor Blocks 0 Norm1 (B)                      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1212 | audio_adaptor.blocks.0.norm2.weight                      | Audio_Adaptor Blocks 0 Norm2 (W)                      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1213 | audio_adaptor.blocks.0.norm2.bias                        | Audio_Adaptor Blocks 0 Norm2 (B)                      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1214 | audio_adaptor.blocks.1.self_attn.linear_q.weight         | Audio_Adaptor Blocks 1 Self_Attn Linear_Q (W)         | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1215 | audio_adaptor.blocks.1.self_attn.linear_q.bias           | Audio_Adaptor Blocks 1 Self_Attn Linear_Q (B)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1216 | audio_adaptor.blocks.1.self_attn.linear_k.weight         | Audio_Adaptor Blocks 1 Self_Attn Linear_K (W)         | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1217 | audio_adaptor.blocks.1.self_attn.linear_k.bias           | Audio_Adaptor Blocks 1 Self_Attn Linear_K (B)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1218 | audio_adaptor.blocks.1.self_attn.linear_v.weight         | Audio_Adaptor Blocks 1 Self_Attn Linear_V (W)         | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1219 | audio_adaptor.blocks.1.self_attn.linear_v.bias           | Audio_Adaptor Blocks 1 Self_Attn Linear_V (B)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1220 | audio_adaptor.blocks.1.self_attn.linear_out.weight       | Audio_Adaptor Blocks 1 Self_Attn Linear_Out (W)       | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1221 | audio_adaptor.blocks.1.self_attn.linear_out.bias         | Audio_Adaptor Blocks 1 Self_Attn Linear_Out (B)       | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1222 | audio_adaptor.blocks.1.feed_forward.w_1.weight           | Audio_Adaptor Blocks 1 Feed_Forward W_1 (W)           | (~262K)    262144 | 1024 x    256 x   1 x 1 | BF16 |
| 1223 | audio_adaptor.blocks.1.feed_forward.w_1.bias             | Audio_Adaptor Blocks 1 Feed_Forward W_1 (B)           | (  256)       256 |  256 x      1 x   1 x 1 | BF16 |
| 1224 | audio_adaptor.blocks.1.feed_forward.w_2.weight           | Audio_Adaptor Blocks 1 Feed_Forward W_2 (W)           | (~262K)    262144 |  256 x   1024 x   1 x 1 | BF16 |
| 1225 | audio_adaptor.blocks.1.feed_forward.w_2.bias             | Audio_Adaptor Blocks 1 Feed_Forward W_2 (B)           | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1226 | audio_adaptor.blocks.1.norm1.weight                      | Audio_Adaptor Blocks 1 Norm1 (W)                      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1227 | audio_adaptor.blocks.1.norm1.bias                        | Audio_Adaptor Blocks 1 Norm1 (B)                      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1228 | audio_adaptor.blocks.1.norm2.weight                      | Audio_Adaptor Blocks 1 Norm2 (W)                      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1229 | audio_adaptor.blocks.1.norm2.bias                        | Audio_Adaptor Blocks 1 Norm2 (B)                      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1230 | llm.lm_head.weight                                       | Llm Lm_Head (W)                                       | (~156M) 155582464 | 1024 x 151936 x   1 x 1 | BF16 |
| 1231 | llm.model.embed_tokens.weight                            | Llm Model Embed_Tokens (W)                            | (~156M) 155582464 | 1024 x 151936 x   1 x 1 | BF16 |
| 1232 | llm.model.layers.0.input_layernorm.weight                | Llm Model Layers 0 Input_Layernorm (W)                | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1233 | llm.model.layers.0.mlp.down_proj.weight                  | Llm Model Layers 0 Mlp Down_Proj (W)                  | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1234 | llm.model.layers.0.mlp.gate_proj.weight                  | Llm Model Layers 0 Mlp Gate_Proj (W)                  | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1235 | llm.model.layers.0.mlp.up_proj.weight                    | Llm Model Layers 0 Mlp Up_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1236 | llm.model.layers.0.post_attention_layernorm.weight       | Llm Model Layers 0 Post_Attention_Layernorm (W)       | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1237 | llm.model.layers.0.self_attn.k_norm.weight               | Llm Model Layers 0 Self_Attn K_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1238 | llm.model.layers.0.self_attn.k_proj.weight               | Llm Model Layers 0 Self_Attn K_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1239 | llm.model.layers.0.self_attn.o_proj.weight               | Llm Model Layers 0 Self_Attn O_Proj (W)               | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1240 | llm.model.layers.0.self_attn.q_norm.weight               | Llm Model Layers 0 Self_Attn Q_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1241 | llm.model.layers.0.self_attn.q_proj.weight               | Llm Model Layers 0 Self_Attn Q_Proj (W)               | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1242 | llm.model.layers.0.self_attn.v_proj.weight               | Llm Model Layers 0 Self_Attn V_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1243 | llm.model.layers.1.input_layernorm.weight                | Llm Model Layers 1 Input_Layernorm (W)                | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1244 | llm.model.layers.1.mlp.down_proj.weight                  | Llm Model Layers 1 Mlp Down_Proj (W)                  | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1245 | llm.model.layers.1.mlp.gate_proj.weight                  | Llm Model Layers 1 Mlp Gate_Proj (W)                  | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1246 | llm.model.layers.1.mlp.up_proj.weight                    | Llm Model Layers 1 Mlp Up_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1247 | llm.model.layers.1.post_attention_layernorm.weight       | Llm Model Layers 1 Post_Attention_Layernorm (W)       | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1248 | llm.model.layers.1.self_attn.k_norm.weight               | Llm Model Layers 1 Self_Attn K_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1249 | llm.model.layers.1.self_attn.k_proj.weight               | Llm Model Layers 1 Self_Attn K_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1250 | llm.model.layers.1.self_attn.o_proj.weight               | Llm Model Layers 1 Self_Attn O_Proj (W)               | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1251 | llm.model.layers.1.self_attn.q_norm.weight               | Llm Model Layers 1 Self_Attn Q_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1252 | llm.model.layers.1.self_attn.q_proj.weight               | Llm Model Layers 1 Self_Attn Q_Proj (W)               | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1253 | llm.model.layers.1.self_attn.v_proj.weight               | Llm Model Layers 1 Self_Attn V_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1254 | llm.model.layers.10.input_layernorm.weight               | Llm Model Layers 10 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1255 | llm.model.layers.10.mlp.down_proj.weight                 | Llm Model Layers 10 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1256 | llm.model.layers.10.mlp.gate_proj.weight                 | Llm Model Layers 10 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1257 | llm.model.layers.10.mlp.up_proj.weight                   | Llm Model Layers 10 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1258 | llm.model.layers.10.post_attention_layernorm.weight      | Llm Model Layers 10 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1259 | llm.model.layers.10.self_attn.k_norm.weight              | Llm Model Layers 10 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1260 | llm.model.layers.10.self_attn.k_proj.weight              | Llm Model Layers 10 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1261 | llm.model.layers.10.self_attn.o_proj.weight              | Llm Model Layers 10 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1262 | llm.model.layers.10.self_attn.q_norm.weight              | Llm Model Layers 10 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1263 | llm.model.layers.10.self_attn.q_proj.weight              | Llm Model Layers 10 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1264 | llm.model.layers.10.self_attn.v_proj.weight              | Llm Model Layers 10 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1265 | llm.model.layers.11.input_layernorm.weight               | Llm Model Layers 11 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1266 | llm.model.layers.11.mlp.down_proj.weight                 | Llm Model Layers 11 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1267 | llm.model.layers.11.mlp.gate_proj.weight                 | Llm Model Layers 11 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1268 | llm.model.layers.11.mlp.up_proj.weight                   | Llm Model Layers 11 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1269 | llm.model.layers.11.post_attention_layernorm.weight      | Llm Model Layers 11 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1270 | llm.model.layers.11.self_attn.k_norm.weight              | Llm Model Layers 11 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1271 | llm.model.layers.11.self_attn.k_proj.weight              | Llm Model Layers 11 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1272 | llm.model.layers.11.self_attn.o_proj.weight              | Llm Model Layers 11 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1273 | llm.model.layers.11.self_attn.q_norm.weight              | Llm Model Layers 11 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1274 | llm.model.layers.11.self_attn.q_proj.weight              | Llm Model Layers 11 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1275 | llm.model.layers.11.self_attn.v_proj.weight              | Llm Model Layers 11 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1276 | llm.model.layers.12.input_layernorm.weight               | Llm Model Layers 12 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1277 | llm.model.layers.12.mlp.down_proj.weight                 | Llm Model Layers 12 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1278 | llm.model.layers.12.mlp.gate_proj.weight                 | Llm Model Layers 12 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1279 | llm.model.layers.12.mlp.up_proj.weight                   | Llm Model Layers 12 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1280 | llm.model.layers.12.post_attention_layernorm.weight      | Llm Model Layers 12 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1281 | llm.model.layers.12.self_attn.k_norm.weight              | Llm Model Layers 12 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1282 | llm.model.layers.12.self_attn.k_proj.weight              | Llm Model Layers 12 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1283 | llm.model.layers.12.self_attn.o_proj.weight              | Llm Model Layers 12 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1284 | llm.model.layers.12.self_attn.q_norm.weight              | Llm Model Layers 12 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1285 | llm.model.layers.12.self_attn.q_proj.weight              | Llm Model Layers 12 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1286 | llm.model.layers.12.self_attn.v_proj.weight              | Llm Model Layers 12 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1287 | llm.model.layers.13.input_layernorm.weight               | Llm Model Layers 13 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1288 | llm.model.layers.13.mlp.down_proj.weight                 | Llm Model Layers 13 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1289 | llm.model.layers.13.mlp.gate_proj.weight                 | Llm Model Layers 13 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1290 | llm.model.layers.13.mlp.up_proj.weight                   | Llm Model Layers 13 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1291 | llm.model.layers.13.post_attention_layernorm.weight      | Llm Model Layers 13 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1292 | llm.model.layers.13.self_attn.k_norm.weight              | Llm Model Layers 13 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1293 | llm.model.layers.13.self_attn.k_proj.weight              | Llm Model Layers 13 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1294 | llm.model.layers.13.self_attn.o_proj.weight              | Llm Model Layers 13 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1295 | llm.model.layers.13.self_attn.q_norm.weight              | Llm Model Layers 13 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1296 | llm.model.layers.13.self_attn.q_proj.weight              | Llm Model Layers 13 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1297 | llm.model.layers.13.self_attn.v_proj.weight              | Llm Model Layers 13 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1298 | llm.model.layers.14.input_layernorm.weight               | Llm Model Layers 14 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1299 | llm.model.layers.14.mlp.down_proj.weight                 | Llm Model Layers 14 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1300 | llm.model.layers.14.mlp.gate_proj.weight                 | Llm Model Layers 14 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1301 | llm.model.layers.14.mlp.up_proj.weight                   | Llm Model Layers 14 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1302 | llm.model.layers.14.post_attention_layernorm.weight      | Llm Model Layers 14 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1303 | llm.model.layers.14.self_attn.k_norm.weight              | Llm Model Layers 14 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1304 | llm.model.layers.14.self_attn.k_proj.weight              | Llm Model Layers 14 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1305 | llm.model.layers.14.self_attn.o_proj.weight              | Llm Model Layers 14 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1306 | llm.model.layers.14.self_attn.q_norm.weight              | Llm Model Layers 14 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1307 | llm.model.layers.14.self_attn.q_proj.weight              | Llm Model Layers 14 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1308 | llm.model.layers.14.self_attn.v_proj.weight              | Llm Model Layers 14 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1309 | llm.model.layers.15.input_layernorm.weight               | Llm Model Layers 15 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1310 | llm.model.layers.15.mlp.down_proj.weight                 | Llm Model Layers 15 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1311 | llm.model.layers.15.mlp.gate_proj.weight                 | Llm Model Layers 15 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1312 | llm.model.layers.15.mlp.up_proj.weight                   | Llm Model Layers 15 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1313 | llm.model.layers.15.post_attention_layernorm.weight      | Llm Model Layers 15 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1314 | llm.model.layers.15.self_attn.k_norm.weight              | Llm Model Layers 15 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1315 | llm.model.layers.15.self_attn.k_proj.weight              | Llm Model Layers 15 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1316 | llm.model.layers.15.self_attn.o_proj.weight              | Llm Model Layers 15 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1317 | llm.model.layers.15.self_attn.q_norm.weight              | Llm Model Layers 15 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1318 | llm.model.layers.15.self_attn.q_proj.weight              | Llm Model Layers 15 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1319 | llm.model.layers.15.self_attn.v_proj.weight              | Llm Model Layers 15 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1320 | llm.model.layers.16.input_layernorm.weight               | Llm Model Layers 16 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1321 | llm.model.layers.16.mlp.down_proj.weight                 | Llm Model Layers 16 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1322 | llm.model.layers.16.mlp.gate_proj.weight                 | Llm Model Layers 16 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1323 | llm.model.layers.16.mlp.up_proj.weight                   | Llm Model Layers 16 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1324 | llm.model.layers.16.post_attention_layernorm.weight      | Llm Model Layers 16 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1325 | llm.model.layers.16.self_attn.k_norm.weight              | Llm Model Layers 16 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1326 | llm.model.layers.16.self_attn.k_proj.weight              | Llm Model Layers 16 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1327 | llm.model.layers.16.self_attn.o_proj.weight              | Llm Model Layers 16 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1328 | llm.model.layers.16.self_attn.q_norm.weight              | Llm Model Layers 16 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1329 | llm.model.layers.16.self_attn.q_proj.weight              | Llm Model Layers 16 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1330 | llm.model.layers.16.self_attn.v_proj.weight              | Llm Model Layers 16 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1331 | llm.model.layers.17.input_layernorm.weight               | Llm Model Layers 17 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1332 | llm.model.layers.17.mlp.down_proj.weight                 | Llm Model Layers 17 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1333 | llm.model.layers.17.mlp.gate_proj.weight                 | Llm Model Layers 17 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1334 | llm.model.layers.17.mlp.up_proj.weight                   | Llm Model Layers 17 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1335 | llm.model.layers.17.post_attention_layernorm.weight      | Llm Model Layers 17 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1336 | llm.model.layers.17.self_attn.k_norm.weight              | Llm Model Layers 17 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1337 | llm.model.layers.17.self_attn.k_proj.weight              | Llm Model Layers 17 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1338 | llm.model.layers.17.self_attn.o_proj.weight              | Llm Model Layers 17 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1339 | llm.model.layers.17.self_attn.q_norm.weight              | Llm Model Layers 17 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1340 | llm.model.layers.17.self_attn.q_proj.weight              | Llm Model Layers 17 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1341 | llm.model.layers.17.self_attn.v_proj.weight              | Llm Model Layers 17 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1342 | llm.model.layers.18.input_layernorm.weight               | Llm Model Layers 18 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1343 | llm.model.layers.18.mlp.down_proj.weight                 | Llm Model Layers 18 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1344 | llm.model.layers.18.mlp.gate_proj.weight                 | Llm Model Layers 18 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1345 | llm.model.layers.18.mlp.up_proj.weight                   | Llm Model Layers 18 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1346 | llm.model.layers.18.post_attention_layernorm.weight      | Llm Model Layers 18 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1347 | llm.model.layers.18.self_attn.k_norm.weight              | Llm Model Layers 18 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1348 | llm.model.layers.18.self_attn.k_proj.weight              | Llm Model Layers 18 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1349 | llm.model.layers.18.self_attn.o_proj.weight              | Llm Model Layers 18 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1350 | llm.model.layers.18.self_attn.q_norm.weight              | Llm Model Layers 18 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1351 | llm.model.layers.18.self_attn.q_proj.weight              | Llm Model Layers 18 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1352 | llm.model.layers.18.self_attn.v_proj.weight              | Llm Model Layers 18 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1353 | llm.model.layers.19.input_layernorm.weight               | Llm Model Layers 19 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1354 | llm.model.layers.19.mlp.down_proj.weight                 | Llm Model Layers 19 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1355 | llm.model.layers.19.mlp.gate_proj.weight                 | Llm Model Layers 19 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1356 | llm.model.layers.19.mlp.up_proj.weight                   | Llm Model Layers 19 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1357 | llm.model.layers.19.post_attention_layernorm.weight      | Llm Model Layers 19 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1358 | llm.model.layers.19.self_attn.k_norm.weight              | Llm Model Layers 19 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1359 | llm.model.layers.19.self_attn.k_proj.weight              | Llm Model Layers 19 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1360 | llm.model.layers.19.self_attn.o_proj.weight              | Llm Model Layers 19 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1361 | llm.model.layers.19.self_attn.q_norm.weight              | Llm Model Layers 19 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1362 | llm.model.layers.19.self_attn.q_proj.weight              | Llm Model Layers 19 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1363 | llm.model.layers.19.self_attn.v_proj.weight              | Llm Model Layers 19 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1364 | llm.model.layers.2.input_layernorm.weight                | Llm Model Layers 2 Input_Layernorm (W)                | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1365 | llm.model.layers.2.mlp.down_proj.weight                  | Llm Model Layers 2 Mlp Down_Proj (W)                  | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1366 | llm.model.layers.2.mlp.gate_proj.weight                  | Llm Model Layers 2 Mlp Gate_Proj (W)                  | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1367 | llm.model.layers.2.mlp.up_proj.weight                    | Llm Model Layers 2 Mlp Up_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1368 | llm.model.layers.2.post_attention_layernorm.weight       | Llm Model Layers 2 Post_Attention_Layernorm (W)       | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1369 | llm.model.layers.2.self_attn.k_norm.weight               | Llm Model Layers 2 Self_Attn K_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1370 | llm.model.layers.2.self_attn.k_proj.weight               | Llm Model Layers 2 Self_Attn K_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1371 | llm.model.layers.2.self_attn.o_proj.weight               | Llm Model Layers 2 Self_Attn O_Proj (W)               | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1372 | llm.model.layers.2.self_attn.q_norm.weight               | Llm Model Layers 2 Self_Attn Q_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1373 | llm.model.layers.2.self_attn.q_proj.weight               | Llm Model Layers 2 Self_Attn Q_Proj (W)               | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1374 | llm.model.layers.2.self_attn.v_proj.weight               | Llm Model Layers 2 Self_Attn V_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1375 | llm.model.layers.20.input_layernorm.weight               | Llm Model Layers 20 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1376 | llm.model.layers.20.mlp.down_proj.weight                 | Llm Model Layers 20 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1377 | llm.model.layers.20.mlp.gate_proj.weight                 | Llm Model Layers 20 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1378 | llm.model.layers.20.mlp.up_proj.weight                   | Llm Model Layers 20 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1379 | llm.model.layers.20.post_attention_layernorm.weight      | Llm Model Layers 20 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1380 | llm.model.layers.20.self_attn.k_norm.weight              | Llm Model Layers 20 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1381 | llm.model.layers.20.self_attn.k_proj.weight              | Llm Model Layers 20 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1382 | llm.model.layers.20.self_attn.o_proj.weight              | Llm Model Layers 20 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1383 | llm.model.layers.20.self_attn.q_norm.weight              | Llm Model Layers 20 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1384 | llm.model.layers.20.self_attn.q_proj.weight              | Llm Model Layers 20 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1385 | llm.model.layers.20.self_attn.v_proj.weight              | Llm Model Layers 20 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1386 | llm.model.layers.21.input_layernorm.weight               | Llm Model Layers 21 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1387 | llm.model.layers.21.mlp.down_proj.weight                 | Llm Model Layers 21 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1388 | llm.model.layers.21.mlp.gate_proj.weight                 | Llm Model Layers 21 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1389 | llm.model.layers.21.mlp.up_proj.weight                   | Llm Model Layers 21 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1390 | llm.model.layers.21.post_attention_layernorm.weight      | Llm Model Layers 21 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1391 | llm.model.layers.21.self_attn.k_norm.weight              | Llm Model Layers 21 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1392 | llm.model.layers.21.self_attn.k_proj.weight              | Llm Model Layers 21 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1393 | llm.model.layers.21.self_attn.o_proj.weight              | Llm Model Layers 21 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1394 | llm.model.layers.21.self_attn.q_norm.weight              | Llm Model Layers 21 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1395 | llm.model.layers.21.self_attn.q_proj.weight              | Llm Model Layers 21 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1396 | llm.model.layers.21.self_attn.v_proj.weight              | Llm Model Layers 21 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1397 | llm.model.layers.22.input_layernorm.weight               | Llm Model Layers 22 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1398 | llm.model.layers.22.mlp.down_proj.weight                 | Llm Model Layers 22 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1399 | llm.model.layers.22.mlp.gate_proj.weight                 | Llm Model Layers 22 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1400 | llm.model.layers.22.mlp.up_proj.weight                   | Llm Model Layers 22 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1401 | llm.model.layers.22.post_attention_layernorm.weight      | Llm Model Layers 22 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1402 | llm.model.layers.22.self_attn.k_norm.weight              | Llm Model Layers 22 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1403 | llm.model.layers.22.self_attn.k_proj.weight              | Llm Model Layers 22 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1404 | llm.model.layers.22.self_attn.o_proj.weight              | Llm Model Layers 22 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1405 | llm.model.layers.22.self_attn.q_norm.weight              | Llm Model Layers 22 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1406 | llm.model.layers.22.self_attn.q_proj.weight              | Llm Model Layers 22 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1407 | llm.model.layers.22.self_attn.v_proj.weight              | Llm Model Layers 22 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1408 | llm.model.layers.23.input_layernorm.weight               | Llm Model Layers 23 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1409 | llm.model.layers.23.mlp.down_proj.weight                 | Llm Model Layers 23 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1410 | llm.model.layers.23.mlp.gate_proj.weight                 | Llm Model Layers 23 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1411 | llm.model.layers.23.mlp.up_proj.weight                   | Llm Model Layers 23 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1412 | llm.model.layers.23.post_attention_layernorm.weight      | Llm Model Layers 23 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1413 | llm.model.layers.23.self_attn.k_norm.weight              | Llm Model Layers 23 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1414 | llm.model.layers.23.self_attn.k_proj.weight              | Llm Model Layers 23 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1415 | llm.model.layers.23.self_attn.o_proj.weight              | Llm Model Layers 23 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1416 | llm.model.layers.23.self_attn.q_norm.weight              | Llm Model Layers 23 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1417 | llm.model.layers.23.self_attn.q_proj.weight              | Llm Model Layers 23 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1418 | llm.model.layers.23.self_attn.v_proj.weight              | Llm Model Layers 23 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1419 | llm.model.layers.24.input_layernorm.weight               | Llm Model Layers 24 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1420 | llm.model.layers.24.mlp.down_proj.weight                 | Llm Model Layers 24 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1421 | llm.model.layers.24.mlp.gate_proj.weight                 | Llm Model Layers 24 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1422 | llm.model.layers.24.mlp.up_proj.weight                   | Llm Model Layers 24 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1423 | llm.model.layers.24.post_attention_layernorm.weight      | Llm Model Layers 24 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1424 | llm.model.layers.24.self_attn.k_norm.weight              | Llm Model Layers 24 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1425 | llm.model.layers.24.self_attn.k_proj.weight              | Llm Model Layers 24 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1426 | llm.model.layers.24.self_attn.o_proj.weight              | Llm Model Layers 24 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1427 | llm.model.layers.24.self_attn.q_norm.weight              | Llm Model Layers 24 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1428 | llm.model.layers.24.self_attn.q_proj.weight              | Llm Model Layers 24 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1429 | llm.model.layers.24.self_attn.v_proj.weight              | Llm Model Layers 24 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1430 | llm.model.layers.25.input_layernorm.weight               | Llm Model Layers 25 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1431 | llm.model.layers.25.mlp.down_proj.weight                 | Llm Model Layers 25 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1432 | llm.model.layers.25.mlp.gate_proj.weight                 | Llm Model Layers 25 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1433 | llm.model.layers.25.mlp.up_proj.weight                   | Llm Model Layers 25 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1434 | llm.model.layers.25.post_attention_layernorm.weight      | Llm Model Layers 25 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1435 | llm.model.layers.25.self_attn.k_norm.weight              | Llm Model Layers 25 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1436 | llm.model.layers.25.self_attn.k_proj.weight              | Llm Model Layers 25 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1437 | llm.model.layers.25.self_attn.o_proj.weight              | Llm Model Layers 25 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1438 | llm.model.layers.25.self_attn.q_norm.weight              | Llm Model Layers 25 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1439 | llm.model.layers.25.self_attn.q_proj.weight              | Llm Model Layers 25 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1440 | llm.model.layers.25.self_attn.v_proj.weight              | Llm Model Layers 25 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1441 | llm.model.layers.26.input_layernorm.weight               | Llm Model Layers 26 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1442 | llm.model.layers.26.mlp.down_proj.weight                 | Llm Model Layers 26 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1443 | llm.model.layers.26.mlp.gate_proj.weight                 | Llm Model Layers 26 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1444 | llm.model.layers.26.mlp.up_proj.weight                   | Llm Model Layers 26 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1445 | llm.model.layers.26.post_attention_layernorm.weight      | Llm Model Layers 26 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1446 | llm.model.layers.26.self_attn.k_norm.weight              | Llm Model Layers 26 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1447 | llm.model.layers.26.self_attn.k_proj.weight              | Llm Model Layers 26 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1448 | llm.model.layers.26.self_attn.o_proj.weight              | Llm Model Layers 26 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1449 | llm.model.layers.26.self_attn.q_norm.weight              | Llm Model Layers 26 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1450 | llm.model.layers.26.self_attn.q_proj.weight              | Llm Model Layers 26 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1451 | llm.model.layers.26.self_attn.v_proj.weight              | Llm Model Layers 26 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1452 | llm.model.layers.27.input_layernorm.weight               | Llm Model Layers 27 Input_Layernorm (W)               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1453 | llm.model.layers.27.mlp.down_proj.weight                 | Llm Model Layers 27 Mlp Down_Proj (W)                 | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1454 | llm.model.layers.27.mlp.gate_proj.weight                 | Llm Model Layers 27 Mlp Gate_Proj (W)                 | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1455 | llm.model.layers.27.mlp.up_proj.weight                   | Llm Model Layers 27 Mlp Up_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1456 | llm.model.layers.27.post_attention_layernorm.weight      | Llm Model Layers 27 Post_Attention_Layernorm (W)      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1457 | llm.model.layers.27.self_attn.k_norm.weight              | Llm Model Layers 27 Self_Attn K_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1458 | llm.model.layers.27.self_attn.k_proj.weight              | Llm Model Layers 27 Self_Attn K_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1459 | llm.model.layers.27.self_attn.o_proj.weight              | Llm Model Layers 27 Self_Attn O_Proj (W)              | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1460 | llm.model.layers.27.self_attn.q_norm.weight              | Llm Model Layers 27 Self_Attn Q_Norm (W)              | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1461 | llm.model.layers.27.self_attn.q_proj.weight              | Llm Model Layers 27 Self_Attn Q_Proj (W)              | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1462 | llm.model.layers.27.self_attn.v_proj.weight              | Llm Model Layers 27 Self_Attn V_Proj (W)              | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1463 | llm.model.layers.3.input_layernorm.weight                | Llm Model Layers 3 Input_Layernorm (W)                | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1464 | llm.model.layers.3.mlp.down_proj.weight                  | Llm Model Layers 3 Mlp Down_Proj (W)                  | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1465 | llm.model.layers.3.mlp.gate_proj.weight                  | Llm Model Layers 3 Mlp Gate_Proj (W)                  | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1466 | llm.model.layers.3.mlp.up_proj.weight                    | Llm Model Layers 3 Mlp Up_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1467 | llm.model.layers.3.post_attention_layernorm.weight       | Llm Model Layers 3 Post_Attention_Layernorm (W)       | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1468 | llm.model.layers.3.self_attn.k_norm.weight               | Llm Model Layers 3 Self_Attn K_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1469 | llm.model.layers.3.self_attn.k_proj.weight               | Llm Model Layers 3 Self_Attn K_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1470 | llm.model.layers.3.self_attn.o_proj.weight               | Llm Model Layers 3 Self_Attn O_Proj (W)               | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1471 | llm.model.layers.3.self_attn.q_norm.weight               | Llm Model Layers 3 Self_Attn Q_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1472 | llm.model.layers.3.self_attn.q_proj.weight               | Llm Model Layers 3 Self_Attn Q_Proj (W)               | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1473 | llm.model.layers.3.self_attn.v_proj.weight               | Llm Model Layers 3 Self_Attn V_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1474 | llm.model.layers.4.input_layernorm.weight                | Llm Model Layers 4 Input_Layernorm (W)                | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1475 | llm.model.layers.4.mlp.down_proj.weight                  | Llm Model Layers 4 Mlp Down_Proj (W)                  | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1476 | llm.model.layers.4.mlp.gate_proj.weight                  | Llm Model Layers 4 Mlp Gate_Proj (W)                  | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1477 | llm.model.layers.4.mlp.up_proj.weight                    | Llm Model Layers 4 Mlp Up_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1478 | llm.model.layers.4.post_attention_layernorm.weight       | Llm Model Layers 4 Post_Attention_Layernorm (W)       | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1479 | llm.model.layers.4.self_attn.k_norm.weight               | Llm Model Layers 4 Self_Attn K_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1480 | llm.model.layers.4.self_attn.k_proj.weight               | Llm Model Layers 4 Self_Attn K_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1481 | llm.model.layers.4.self_attn.o_proj.weight               | Llm Model Layers 4 Self_Attn O_Proj (W)               | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1482 | llm.model.layers.4.self_attn.q_norm.weight               | Llm Model Layers 4 Self_Attn Q_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1483 | llm.model.layers.4.self_attn.q_proj.weight               | Llm Model Layers 4 Self_Attn Q_Proj (W)               | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1484 | llm.model.layers.4.self_attn.v_proj.weight               | Llm Model Layers 4 Self_Attn V_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1485 | llm.model.layers.5.input_layernorm.weight                | Llm Model Layers 5 Input_Layernorm (W)                | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1486 | llm.model.layers.5.mlp.down_proj.weight                  | Llm Model Layers 5 Mlp Down_Proj (W)                  | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1487 | llm.model.layers.5.mlp.gate_proj.weight                  | Llm Model Layers 5 Mlp Gate_Proj (W)                  | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1488 | llm.model.layers.5.mlp.up_proj.weight                    | Llm Model Layers 5 Mlp Up_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1489 | llm.model.layers.5.post_attention_layernorm.weight       | Llm Model Layers 5 Post_Attention_Layernorm (W)       | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1490 | llm.model.layers.5.self_attn.k_norm.weight               | Llm Model Layers 5 Self_Attn K_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1491 | llm.model.layers.5.self_attn.k_proj.weight               | Llm Model Layers 5 Self_Attn K_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1492 | llm.model.layers.5.self_attn.o_proj.weight               | Llm Model Layers 5 Self_Attn O_Proj (W)               | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1493 | llm.model.layers.5.self_attn.q_norm.weight               | Llm Model Layers 5 Self_Attn Q_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1494 | llm.model.layers.5.self_attn.q_proj.weight               | Llm Model Layers 5 Self_Attn Q_Proj (W)               | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1495 | llm.model.layers.5.self_attn.v_proj.weight               | Llm Model Layers 5 Self_Attn V_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1496 | llm.model.layers.6.input_layernorm.weight                | Llm Model Layers 6 Input_Layernorm (W)                | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1497 | llm.model.layers.6.mlp.down_proj.weight                  | Llm Model Layers 6 Mlp Down_Proj (W)                  | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1498 | llm.model.layers.6.mlp.gate_proj.weight                  | Llm Model Layers 6 Mlp Gate_Proj (W)                  | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1499 | llm.model.layers.6.mlp.up_proj.weight                    | Llm Model Layers 6 Mlp Up_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1500 | llm.model.layers.6.post_attention_layernorm.weight       | Llm Model Layers 6 Post_Attention_Layernorm (W)       | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1501 | llm.model.layers.6.self_attn.k_norm.weight               | Llm Model Layers 6 Self_Attn K_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1502 | llm.model.layers.6.self_attn.k_proj.weight               | Llm Model Layers 6 Self_Attn K_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1503 | llm.model.layers.6.self_attn.o_proj.weight               | Llm Model Layers 6 Self_Attn O_Proj (W)               | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1504 | llm.model.layers.6.self_attn.q_norm.weight               | Llm Model Layers 6 Self_Attn Q_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1505 | llm.model.layers.6.self_attn.q_proj.weight               | Llm Model Layers 6 Self_Attn Q_Proj (W)               | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1506 | llm.model.layers.6.self_attn.v_proj.weight               | Llm Model Layers 6 Self_Attn V_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1507 | llm.model.layers.7.input_layernorm.weight                | Llm Model Layers 7 Input_Layernorm (W)                | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1508 | llm.model.layers.7.mlp.down_proj.weight                  | Llm Model Layers 7 Mlp Down_Proj (W)                  | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1509 | llm.model.layers.7.mlp.gate_proj.weight                  | Llm Model Layers 7 Mlp Gate_Proj (W)                  | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1510 | llm.model.layers.7.mlp.up_proj.weight                    | Llm Model Layers 7 Mlp Up_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1511 | llm.model.layers.7.post_attention_layernorm.weight       | Llm Model Layers 7 Post_Attention_Layernorm (W)       | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1512 | llm.model.layers.7.self_attn.k_norm.weight               | Llm Model Layers 7 Self_Attn K_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1513 | llm.model.layers.7.self_attn.k_proj.weight               | Llm Model Layers 7 Self_Attn K_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1514 | llm.model.layers.7.self_attn.o_proj.weight               | Llm Model Layers 7 Self_Attn O_Proj (W)               | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1515 | llm.model.layers.7.self_attn.q_norm.weight               | Llm Model Layers 7 Self_Attn Q_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1516 | llm.model.layers.7.self_attn.q_proj.weight               | Llm Model Layers 7 Self_Attn Q_Proj (W)               | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1517 | llm.model.layers.7.self_attn.v_proj.weight               | Llm Model Layers 7 Self_Attn V_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1518 | llm.model.layers.8.input_layernorm.weight                | Llm Model Layers 8 Input_Layernorm (W)                | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1519 | llm.model.layers.8.mlp.down_proj.weight                  | Llm Model Layers 8 Mlp Down_Proj (W)                  | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1520 | llm.model.layers.8.mlp.gate_proj.weight                  | Llm Model Layers 8 Mlp Gate_Proj (W)                  | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1521 | llm.model.layers.8.mlp.up_proj.weight                    | Llm Model Layers 8 Mlp Up_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1522 | llm.model.layers.8.post_attention_layernorm.weight       | Llm Model Layers 8 Post_Attention_Layernorm (W)       | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1523 | llm.model.layers.8.self_attn.k_norm.weight               | Llm Model Layers 8 Self_Attn K_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1524 | llm.model.layers.8.self_attn.k_proj.weight               | Llm Model Layers 8 Self_Attn K_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1525 | llm.model.layers.8.self_attn.o_proj.weight               | Llm Model Layers 8 Self_Attn O_Proj (W)               | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1526 | llm.model.layers.8.self_attn.q_norm.weight               | Llm Model Layers 8 Self_Attn Q_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1527 | llm.model.layers.8.self_attn.q_proj.weight               | Llm Model Layers 8 Self_Attn Q_Proj (W)               | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1528 | llm.model.layers.8.self_attn.v_proj.weight               | Llm Model Layers 8 Self_Attn V_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1529 | llm.model.layers.9.input_layernorm.weight                | Llm Model Layers 9 Input_Layernorm (W)                | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1530 | llm.model.layers.9.mlp.down_proj.weight                  | Llm Model Layers 9 Mlp Down_Proj (W)                  | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1531 | llm.model.layers.9.mlp.gate_proj.weight                  | Llm Model Layers 9 Mlp Gate_Proj (W)                  | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1532 | llm.model.layers.9.mlp.up_proj.weight                    | Llm Model Layers 9 Mlp Up_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1533 | llm.model.layers.9.post_attention_layernorm.weight       | Llm Model Layers 9 Post_Attention_Layernorm (W)       | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1534 | llm.model.layers.9.self_attn.k_norm.weight               | Llm Model Layers 9 Self_Attn K_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1535 | llm.model.layers.9.self_attn.k_proj.weight               | Llm Model Layers 9 Self_Attn K_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1536 | llm.model.layers.9.self_attn.o_proj.weight               | Llm Model Layers 9 Self_Attn O_Proj (W)               | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1537 | llm.model.layers.9.self_attn.q_norm.weight               | Llm Model Layers 9 Self_Attn Q_Norm (W)               | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1538 | llm.model.layers.9.self_attn.q_proj.weight               | Llm Model Layers 9 Self_Attn Q_Proj (W)               | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1539 | llm.model.layers.9.self_attn.v_proj.weight               | Llm Model Layers 9 Self_Attn V_Proj (W)               | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1540 | llm.model.norm.weight                                    | Llm Model Norm (W)                                    | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |

- Total elements in base: (~985M) 985374304
- Percentage of total elements: 100.00%



