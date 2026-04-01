# FunAsr.bin - GGUF Internal File Dump

- Endian: BIG endian

## Key Value Metadata Store

There are 38 key-value pairs in this file

| POS | TYPE     |  Count | Key                             | Value                                                               |
|----:|:---------|-------:|:--------------------------------|:--------------------------------------------------------------------|
|   1 | UINT32   |      1 | GGUF.version                    | 3                                                                   |
|   2 | UINT64   |      1 | GGUF.tensor_count               | 1261                                                                |
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

| T_ID | Tensor Layer Name                                          |  Data Offset (B) |    Data Size (B) |
|-----:|:-----------------------------------------------------------|-----------------:|-----------------:|
|    0 | audio_encoder.encoders0.0.self_attn.linear_out.weight      |         0x5c1a80 |          0x80000 |
|    1 | audio_encoder.encoders0.0.self_attn.linear_out.bias        |         0x641a80 |            0x400 |
|    2 | audio_encoder.encoders0.0.self_attn.linear_q_k_v.weight    |         0x641e80 |         0x1a4000 |
|    3 | audio_encoder.encoders0.0.self_attn.linear_q_k_v.bias      |         0x7e5e80 |            0xc00 |
|    4 | audio_encoder.encoders0.0.self_attn.fsmn_block.weight      |         0x7e6a80 |           0x2c00 |
|    5 | audio_encoder.encoders0.0.feed_forward.w_1.weight          |         0x7e9680 |         0x200000 |
|    6 | audio_encoder.encoders0.0.feed_forward.w_1.bias            |         0x9e9680 |           0x1000 |
|    7 | audio_encoder.encoders0.0.feed_forward.w_2.weight          |         0x9ea680 |         0x200000 |
|    8 | audio_encoder.encoders0.0.feed_forward.w_2.bias            |         0xbea680 |            0x400 |
|    9 | audio_encoder.encoders0.0.norm1.weight                     |         0xbeaa80 |            0x460 |
|   10 | audio_encoder.encoders0.0.norm1.bias                       |         0xbeaee0 |            0x460 |
|   11 | audio_encoder.encoders0.0.norm2.weight                     |         0xbeb340 |            0x400 |
|   12 | audio_encoder.encoders0.0.norm2.bias                       |         0xbeb740 |            0x400 |
|   13 | audio_encoder.encoders.0.self_attn.linear_out.weight       |         0xbebb40 |          0x80000 |
|   14 | audio_encoder.encoders.0.self_attn.linear_out.bias         |         0xc6bb40 |            0x400 |
|   15 | audio_encoder.encoders.0.self_attn.linear_q_k_v.weight     |         0xc6bf40 |         0x180000 |
|   16 | audio_encoder.encoders.0.self_attn.linear_q_k_v.bias       |         0xdebf40 |            0xc00 |
|   17 | audio_encoder.encoders.0.self_attn.fsmn_block.weight       |         0xdecb40 |           0x2c00 |
|   18 | audio_encoder.encoders.0.feed_forward.w_1.weight           |         0xdef740 |         0x200000 |
|   19 | audio_encoder.encoders.0.feed_forward.w_1.bias             |         0xfef740 |           0x1000 |
|   20 | audio_encoder.encoders.0.feed_forward.w_2.weight           |         0xff0740 |         0x200000 |
|   21 | audio_encoder.encoders.0.feed_forward.w_2.bias             |        0x11f0740 |            0x400 |
|   22 | audio_encoder.encoders.0.norm1.weight                      |        0x11f0b40 |            0x400 |
|   23 | audio_encoder.encoders.0.norm1.bias                        |        0x11f0f40 |            0x400 |
|   24 | audio_encoder.encoders.0.norm2.weight                      |        0x11f1340 |            0x400 |
|   25 | audio_encoder.encoders.0.norm2.bias                        |        0x11f1740 |            0x400 |
|   26 | audio_encoder.encoders.1.self_attn.linear_out.weight       |        0x11f1b40 |          0x80000 |
|   27 | audio_encoder.encoders.1.self_attn.linear_out.bias         |        0x1271b40 |            0x400 |
|   28 | audio_encoder.encoders.1.self_attn.linear_q_k_v.weight     |        0x1271f40 |         0x180000 |
|   29 | audio_encoder.encoders.1.self_attn.linear_q_k_v.bias       |        0x13f1f40 |            0xc00 |
|   30 | audio_encoder.encoders.1.self_attn.fsmn_block.weight       |        0x13f2b40 |           0x2c00 |
|   31 | audio_encoder.encoders.1.feed_forward.w_1.weight           |        0x13f5740 |         0x200000 |
|   32 | audio_encoder.encoders.1.feed_forward.w_1.bias             |        0x15f5740 |           0x1000 |
|   33 | audio_encoder.encoders.1.feed_forward.w_2.weight           |        0x15f6740 |         0x200000 |
|   34 | audio_encoder.encoders.1.feed_forward.w_2.bias             |        0x17f6740 |            0x400 |
|   35 | audio_encoder.encoders.1.norm1.weight                      |        0x17f6b40 |            0x400 |
|   36 | audio_encoder.encoders.1.norm1.bias                        |        0x17f6f40 |            0x400 |
|   37 | audio_encoder.encoders.1.norm2.weight                      |        0x17f7340 |            0x400 |
|   38 | audio_encoder.encoders.1.norm2.bias                        |        0x17f7740 |            0x400 |
|   39 | audio_encoder.encoders.2.self_attn.linear_out.weight       |        0x17f7b40 |          0x80000 |
|   40 | audio_encoder.encoders.2.self_attn.linear_out.bias         |        0x1877b40 |            0x400 |
|   41 | audio_encoder.encoders.2.self_attn.linear_q_k_v.weight     |        0x1877f40 |         0x180000 |
|   42 | audio_encoder.encoders.2.self_attn.linear_q_k_v.bias       |        0x19f7f40 |            0xc00 |
|   43 | audio_encoder.encoders.2.self_attn.fsmn_block.weight       |        0x19f8b40 |           0x2c00 |
|   44 | audio_encoder.encoders.2.feed_forward.w_1.weight           |        0x19fb740 |         0x200000 |
|   45 | audio_encoder.encoders.2.feed_forward.w_1.bias             |        0x1bfb740 |           0x1000 |
|   46 | audio_encoder.encoders.2.feed_forward.w_2.weight           |        0x1bfc740 |         0x200000 |
|   47 | audio_encoder.encoders.2.feed_forward.w_2.bias             |        0x1dfc740 |            0x400 |
|   48 | audio_encoder.encoders.2.norm1.weight                      |        0x1dfcb40 |            0x400 |
|   49 | audio_encoder.encoders.2.norm1.bias                        |        0x1dfcf40 |            0x400 |
|   50 | audio_encoder.encoders.2.norm2.weight                      |        0x1dfd340 |            0x400 |
|   51 | audio_encoder.encoders.2.norm2.bias                        |        0x1dfd740 |            0x400 |
|   52 | audio_encoder.encoders.3.self_attn.linear_out.weight       |        0x1dfdb40 |          0x80000 |
|   53 | audio_encoder.encoders.3.self_attn.linear_out.bias         |        0x1e7db40 |            0x400 |
|   54 | audio_encoder.encoders.3.self_attn.linear_q_k_v.weight     |        0x1e7df40 |         0x180000 |
|   55 | audio_encoder.encoders.3.self_attn.linear_q_k_v.bias       |        0x1ffdf40 |            0xc00 |
|   56 | audio_encoder.encoders.3.self_attn.fsmn_block.weight       |        0x1ffeb40 |           0x2c00 |
|   57 | audio_encoder.encoders.3.feed_forward.w_1.weight           |        0x2001740 |         0x200000 |
|   58 | audio_encoder.encoders.3.feed_forward.w_1.bias             |        0x2201740 |           0x1000 |
|   59 | audio_encoder.encoders.3.feed_forward.w_2.weight           |        0x2202740 |         0x200000 |
|   60 | audio_encoder.encoders.3.feed_forward.w_2.bias             |        0x2402740 |            0x400 |
|   61 | audio_encoder.encoders.3.norm1.weight                      |        0x2402b40 |            0x400 |
|   62 | audio_encoder.encoders.3.norm1.bias                        |        0x2402f40 |            0x400 |
|   63 | audio_encoder.encoders.3.norm2.weight                      |        0x2403340 |            0x400 |
|   64 | audio_encoder.encoders.3.norm2.bias                        |        0x2403740 |            0x400 |
|   65 | audio_encoder.encoders.4.self_attn.linear_out.weight       |        0x2403b40 |          0x80000 |
|   66 | audio_encoder.encoders.4.self_attn.linear_out.bias         |        0x2483b40 |            0x400 |
|   67 | audio_encoder.encoders.4.self_attn.linear_q_k_v.weight     |        0x2483f40 |         0x180000 |
|   68 | audio_encoder.encoders.4.self_attn.linear_q_k_v.bias       |        0x2603f40 |            0xc00 |
|   69 | audio_encoder.encoders.4.self_attn.fsmn_block.weight       |        0x2604b40 |           0x2c00 |
|   70 | audio_encoder.encoders.4.feed_forward.w_1.weight           |        0x2607740 |         0x200000 |
|   71 | audio_encoder.encoders.4.feed_forward.w_1.bias             |        0x2807740 |           0x1000 |
|   72 | audio_encoder.encoders.4.feed_forward.w_2.weight           |        0x2808740 |         0x200000 |
|   73 | audio_encoder.encoders.4.feed_forward.w_2.bias             |        0x2a08740 |            0x400 |
|   74 | audio_encoder.encoders.4.norm1.weight                      |        0x2a08b40 |            0x400 |
|   75 | audio_encoder.encoders.4.norm1.bias                        |        0x2a08f40 |            0x400 |
|   76 | audio_encoder.encoders.4.norm2.weight                      |        0x2a09340 |            0x400 |
|   77 | audio_encoder.encoders.4.norm2.bias                        |        0x2a09740 |            0x400 |
|   78 | audio_encoder.encoders.5.self_attn.linear_out.weight       |        0x2a09b40 |          0x80000 |
|   79 | audio_encoder.encoders.5.self_attn.linear_out.bias         |        0x2a89b40 |            0x400 |
|   80 | audio_encoder.encoders.5.self_attn.linear_q_k_v.weight     |        0x2a89f40 |         0x180000 |
|   81 | audio_encoder.encoders.5.self_attn.linear_q_k_v.bias       |        0x2c09f40 |            0xc00 |
|   82 | audio_encoder.encoders.5.self_attn.fsmn_block.weight       |        0x2c0ab40 |           0x2c00 |
|   83 | audio_encoder.encoders.5.feed_forward.w_1.weight           |        0x2c0d740 |         0x200000 |
|   84 | audio_encoder.encoders.5.feed_forward.w_1.bias             |        0x2e0d740 |           0x1000 |
|   85 | audio_encoder.encoders.5.feed_forward.w_2.weight           |        0x2e0e740 |         0x200000 |
|   86 | audio_encoder.encoders.5.feed_forward.w_2.bias             |        0x300e740 |            0x400 |
|   87 | audio_encoder.encoders.5.norm1.weight                      |        0x300eb40 |            0x400 |
|   88 | audio_encoder.encoders.5.norm1.bias                        |        0x300ef40 |            0x400 |
|   89 | audio_encoder.encoders.5.norm2.weight                      |        0x300f340 |            0x400 |
|   90 | audio_encoder.encoders.5.norm2.bias                        |        0x300f740 |            0x400 |
|   91 | audio_encoder.encoders.6.self_attn.linear_out.weight       |        0x300fb40 |          0x80000 |
|   92 | audio_encoder.encoders.6.self_attn.linear_out.bias         |        0x308fb40 |            0x400 |
|   93 | audio_encoder.encoders.6.self_attn.linear_q_k_v.weight     |        0x308ff40 |         0x180000 |
|   94 | audio_encoder.encoders.6.self_attn.linear_q_k_v.bias       |        0x320ff40 |            0xc00 |
|   95 | audio_encoder.encoders.6.self_attn.fsmn_block.weight       |        0x3210b40 |           0x2c00 |
|   96 | audio_encoder.encoders.6.feed_forward.w_1.weight           |        0x3213740 |         0x200000 |
|   97 | audio_encoder.encoders.6.feed_forward.w_1.bias             |        0x3413740 |           0x1000 |
|   98 | audio_encoder.encoders.6.feed_forward.w_2.weight           |        0x3414740 |         0x200000 |
|   99 | audio_encoder.encoders.6.feed_forward.w_2.bias             |        0x3614740 |            0x400 |
|  100 | audio_encoder.encoders.6.norm1.weight                      |        0x3614b40 |            0x400 |
|  101 | audio_encoder.encoders.6.norm1.bias                        |        0x3614f40 |            0x400 |
|  102 | audio_encoder.encoders.6.norm2.weight                      |        0x3615340 |            0x400 |
|  103 | audio_encoder.encoders.6.norm2.bias                        |        0x3615740 |            0x400 |
|  104 | audio_encoder.encoders.7.self_attn.linear_out.weight       |        0x3615b40 |          0x80000 |
|  105 | audio_encoder.encoders.7.self_attn.linear_out.bias         |        0x3695b40 |            0x400 |
|  106 | audio_encoder.encoders.7.self_attn.linear_q_k_v.weight     |        0x3695f40 |         0x180000 |
|  107 | audio_encoder.encoders.7.self_attn.linear_q_k_v.bias       |        0x3815f40 |            0xc00 |
|  108 | audio_encoder.encoders.7.self_attn.fsmn_block.weight       |        0x3816b40 |           0x2c00 |
|  109 | audio_encoder.encoders.7.feed_forward.w_1.weight           |        0x3819740 |         0x200000 |
|  110 | audio_encoder.encoders.7.feed_forward.w_1.bias             |        0x3a19740 |           0x1000 |
|  111 | audio_encoder.encoders.7.feed_forward.w_2.weight           |        0x3a1a740 |         0x200000 |
|  112 | audio_encoder.encoders.7.feed_forward.w_2.bias             |        0x3c1a740 |            0x400 |
|  113 | audio_encoder.encoders.7.norm1.weight                      |        0x3c1ab40 |            0x400 |
|  114 | audio_encoder.encoders.7.norm1.bias                        |        0x3c1af40 |            0x400 |
|  115 | audio_encoder.encoders.7.norm2.weight                      |        0x3c1b340 |            0x400 |
|  116 | audio_encoder.encoders.7.norm2.bias                        |        0x3c1b740 |            0x400 |
|  117 | audio_encoder.encoders.8.self_attn.linear_out.weight       |        0x3c1bb40 |          0x80000 |
|  118 | audio_encoder.encoders.8.self_attn.linear_out.bias         |        0x3c9bb40 |            0x400 |
|  119 | audio_encoder.encoders.8.self_attn.linear_q_k_v.weight     |        0x3c9bf40 |         0x180000 |
|  120 | audio_encoder.encoders.8.self_attn.linear_q_k_v.bias       |        0x3e1bf40 |            0xc00 |
|  121 | audio_encoder.encoders.8.self_attn.fsmn_block.weight       |        0x3e1cb40 |           0x2c00 |
|  122 | audio_encoder.encoders.8.feed_forward.w_1.weight           |        0x3e1f740 |         0x200000 |
|  123 | audio_encoder.encoders.8.feed_forward.w_1.bias             |        0x401f740 |           0x1000 |
|  124 | audio_encoder.encoders.8.feed_forward.w_2.weight           |        0x4020740 |         0x200000 |
|  125 | audio_encoder.encoders.8.feed_forward.w_2.bias             |        0x4220740 |            0x400 |
|  126 | audio_encoder.encoders.8.norm1.weight                      |        0x4220b40 |            0x400 |
|  127 | audio_encoder.encoders.8.norm1.bias                        |        0x4220f40 |            0x400 |
|  128 | audio_encoder.encoders.8.norm2.weight                      |        0x4221340 |            0x400 |
|  129 | audio_encoder.encoders.8.norm2.bias                        |        0x4221740 |            0x400 |
|  130 | audio_encoder.encoders.9.self_attn.linear_out.weight       |        0x4221b40 |          0x80000 |
|  131 | audio_encoder.encoders.9.self_attn.linear_out.bias         |        0x42a1b40 |            0x400 |
|  132 | audio_encoder.encoders.9.self_attn.linear_q_k_v.weight     |        0x42a1f40 |         0x180000 |
|  133 | audio_encoder.encoders.9.self_attn.linear_q_k_v.bias       |        0x4421f40 |            0xc00 |
|  134 | audio_encoder.encoders.9.self_attn.fsmn_block.weight       |        0x4422b40 |           0x2c00 |
|  135 | audio_encoder.encoders.9.feed_forward.w_1.weight           |        0x4425740 |         0x200000 |
|  136 | audio_encoder.encoders.9.feed_forward.w_1.bias             |        0x4625740 |           0x1000 |
|  137 | audio_encoder.encoders.9.feed_forward.w_2.weight           |        0x4626740 |         0x200000 |
|  138 | audio_encoder.encoders.9.feed_forward.w_2.bias             |        0x4826740 |            0x400 |
|  139 | audio_encoder.encoders.9.norm1.weight                      |        0x4826b40 |            0x400 |
|  140 | audio_encoder.encoders.9.norm1.bias                        |        0x4826f40 |            0x400 |
|  141 | audio_encoder.encoders.9.norm2.weight                      |        0x4827340 |            0x400 |
|  142 | audio_encoder.encoders.9.norm2.bias                        |        0x4827740 |            0x400 |
|  143 | audio_encoder.encoders.10.self_attn.linear_out.weight      |        0x4827b40 |          0x80000 |
|  144 | audio_encoder.encoders.10.self_attn.linear_out.bias        |        0x48a7b40 |            0x400 |
|  145 | audio_encoder.encoders.10.self_attn.linear_q_k_v.weight    |        0x48a7f40 |         0x180000 |
|  146 | audio_encoder.encoders.10.self_attn.linear_q_k_v.bias      |        0x4a27f40 |            0xc00 |
|  147 | audio_encoder.encoders.10.self_attn.fsmn_block.weight      |        0x4a28b40 |           0x2c00 |
|  148 | audio_encoder.encoders.10.feed_forward.w_1.weight          |        0x4a2b740 |         0x200000 |
|  149 | audio_encoder.encoders.10.feed_forward.w_1.bias            |        0x4c2b740 |           0x1000 |
|  150 | audio_encoder.encoders.10.feed_forward.w_2.weight          |        0x4c2c740 |         0x200000 |
|  151 | audio_encoder.encoders.10.feed_forward.w_2.bias            |        0x4e2c740 |            0x400 |
|  152 | audio_encoder.encoders.10.norm1.weight                     |        0x4e2cb40 |            0x400 |
|  153 | audio_encoder.encoders.10.norm1.bias                       |        0x4e2cf40 |            0x400 |
|  154 | audio_encoder.encoders.10.norm2.weight                     |        0x4e2d340 |            0x400 |
|  155 | audio_encoder.encoders.10.norm2.bias                       |        0x4e2d740 |            0x400 |
|  156 | audio_encoder.encoders.11.self_attn.linear_out.weight      |        0x4e2db40 |          0x80000 |
|  157 | audio_encoder.encoders.11.self_attn.linear_out.bias        |        0x4eadb40 |            0x400 |
|  158 | audio_encoder.encoders.11.self_attn.linear_q_k_v.weight    |        0x4eadf40 |         0x180000 |
|  159 | audio_encoder.encoders.11.self_attn.linear_q_k_v.bias      |        0x502df40 |            0xc00 |
|  160 | audio_encoder.encoders.11.self_attn.fsmn_block.weight      |        0x502eb40 |           0x2c00 |
|  161 | audio_encoder.encoders.11.feed_forward.w_1.weight          |        0x5031740 |         0x200000 |
|  162 | audio_encoder.encoders.11.feed_forward.w_1.bias            |        0x5231740 |           0x1000 |
|  163 | audio_encoder.encoders.11.feed_forward.w_2.weight          |        0x5232740 |         0x200000 |
|  164 | audio_encoder.encoders.11.feed_forward.w_2.bias            |        0x5432740 |            0x400 |
|  165 | audio_encoder.encoders.11.norm1.weight                     |        0x5432b40 |            0x400 |
|  166 | audio_encoder.encoders.11.norm1.bias                       |        0x5432f40 |            0x400 |
|  167 | audio_encoder.encoders.11.norm2.weight                     |        0x5433340 |            0x400 |
|  168 | audio_encoder.encoders.11.norm2.bias                       |        0x5433740 |            0x400 |
|  169 | audio_encoder.encoders.12.self_attn.linear_out.weight      |        0x5433b40 |          0x80000 |
|  170 | audio_encoder.encoders.12.self_attn.linear_out.bias        |        0x54b3b40 |            0x400 |
|  171 | audio_encoder.encoders.12.self_attn.linear_q_k_v.weight    |        0x54b3f40 |         0x180000 |
|  172 | audio_encoder.encoders.12.self_attn.linear_q_k_v.bias      |        0x5633f40 |            0xc00 |
|  173 | audio_encoder.encoders.12.self_attn.fsmn_block.weight      |        0x5634b40 |           0x2c00 |
|  174 | audio_encoder.encoders.12.feed_forward.w_1.weight          |        0x5637740 |         0x200000 |
|  175 | audio_encoder.encoders.12.feed_forward.w_1.bias            |        0x5837740 |           0x1000 |
|  176 | audio_encoder.encoders.12.feed_forward.w_2.weight          |        0x5838740 |         0x200000 |
|  177 | audio_encoder.encoders.12.feed_forward.w_2.bias            |        0x5a38740 |            0x400 |
|  178 | audio_encoder.encoders.12.norm1.weight                     |        0x5a38b40 |            0x400 |
|  179 | audio_encoder.encoders.12.norm1.bias                       |        0x5a38f40 |            0x400 |
|  180 | audio_encoder.encoders.12.norm2.weight                     |        0x5a39340 |            0x400 |
|  181 | audio_encoder.encoders.12.norm2.bias                       |        0x5a39740 |            0x400 |
|  182 | audio_encoder.encoders.13.self_attn.linear_out.weight      |        0x5a39b40 |          0x80000 |
|  183 | audio_encoder.encoders.13.self_attn.linear_out.bias        |        0x5ab9b40 |            0x400 |
|  184 | audio_encoder.encoders.13.self_attn.linear_q_k_v.weight    |        0x5ab9f40 |         0x180000 |
|  185 | audio_encoder.encoders.13.self_attn.linear_q_k_v.bias      |        0x5c39f40 |            0xc00 |
|  186 | audio_encoder.encoders.13.self_attn.fsmn_block.weight      |        0x5c3ab40 |           0x2c00 |
|  187 | audio_encoder.encoders.13.feed_forward.w_1.weight          |        0x5c3d740 |         0x200000 |
|  188 | audio_encoder.encoders.13.feed_forward.w_1.bias            |        0x5e3d740 |           0x1000 |
|  189 | audio_encoder.encoders.13.feed_forward.w_2.weight          |        0x5e3e740 |         0x200000 |
|  190 | audio_encoder.encoders.13.feed_forward.w_2.bias            |        0x603e740 |            0x400 |
|  191 | audio_encoder.encoders.13.norm1.weight                     |        0x603eb40 |            0x400 |
|  192 | audio_encoder.encoders.13.norm1.bias                       |        0x603ef40 |            0x400 |
|  193 | audio_encoder.encoders.13.norm2.weight                     |        0x603f340 |            0x400 |
|  194 | audio_encoder.encoders.13.norm2.bias                       |        0x603f740 |            0x400 |
|  195 | audio_encoder.encoders.14.self_attn.linear_out.weight      |        0x603fb40 |          0x80000 |
|  196 | audio_encoder.encoders.14.self_attn.linear_out.bias        |        0x60bfb40 |            0x400 |
|  197 | audio_encoder.encoders.14.self_attn.linear_q_k_v.weight    |        0x60bff40 |         0x180000 |
|  198 | audio_encoder.encoders.14.self_attn.linear_q_k_v.bias      |        0x623ff40 |            0xc00 |
|  199 | audio_encoder.encoders.14.self_attn.fsmn_block.weight      |        0x6240b40 |           0x2c00 |
|  200 | audio_encoder.encoders.14.feed_forward.w_1.weight          |        0x6243740 |         0x200000 |
|  201 | audio_encoder.encoders.14.feed_forward.w_1.bias            |        0x6443740 |           0x1000 |
|  202 | audio_encoder.encoders.14.feed_forward.w_2.weight          |        0x6444740 |         0x200000 |
|  203 | audio_encoder.encoders.14.feed_forward.w_2.bias            |        0x6644740 |            0x400 |
|  204 | audio_encoder.encoders.14.norm1.weight                     |        0x6644b40 |            0x400 |
|  205 | audio_encoder.encoders.14.norm1.bias                       |        0x6644f40 |            0x400 |
|  206 | audio_encoder.encoders.14.norm2.weight                     |        0x6645340 |            0x400 |
|  207 | audio_encoder.encoders.14.norm2.bias                       |        0x6645740 |            0x400 |
|  208 | audio_encoder.encoders.15.self_attn.linear_out.weight      |        0x6645b40 |          0x80000 |
|  209 | audio_encoder.encoders.15.self_attn.linear_out.bias        |        0x66c5b40 |            0x400 |
|  210 | audio_encoder.encoders.15.self_attn.linear_q_k_v.weight    |        0x66c5f40 |         0x180000 |
|  211 | audio_encoder.encoders.15.self_attn.linear_q_k_v.bias      |        0x6845f40 |            0xc00 |
|  212 | audio_encoder.encoders.15.self_attn.fsmn_block.weight      |        0x6846b40 |           0x2c00 |
|  213 | audio_encoder.encoders.15.feed_forward.w_1.weight          |        0x6849740 |         0x200000 |
|  214 | audio_encoder.encoders.15.feed_forward.w_1.bias            |        0x6a49740 |           0x1000 |
|  215 | audio_encoder.encoders.15.feed_forward.w_2.weight          |        0x6a4a740 |         0x200000 |
|  216 | audio_encoder.encoders.15.feed_forward.w_2.bias            |        0x6c4a740 |            0x400 |
|  217 | audio_encoder.encoders.15.norm1.weight                     |        0x6c4ab40 |            0x400 |
|  218 | audio_encoder.encoders.15.norm1.bias                       |        0x6c4af40 |            0x400 |
|  219 | audio_encoder.encoders.15.norm2.weight                     |        0x6c4b340 |            0x400 |
|  220 | audio_encoder.encoders.15.norm2.bias                       |        0x6c4b740 |            0x400 |
|  221 | audio_encoder.encoders.16.self_attn.linear_out.weight      |        0x6c4bb40 |          0x80000 |
|  222 | audio_encoder.encoders.16.self_attn.linear_out.bias        |        0x6ccbb40 |            0x400 |
|  223 | audio_encoder.encoders.16.self_attn.linear_q_k_v.weight    |        0x6ccbf40 |         0x180000 |
|  224 | audio_encoder.encoders.16.self_attn.linear_q_k_v.bias      |        0x6e4bf40 |            0xc00 |
|  225 | audio_encoder.encoders.16.self_attn.fsmn_block.weight      |        0x6e4cb40 |           0x2c00 |
|  226 | audio_encoder.encoders.16.feed_forward.w_1.weight          |        0x6e4f740 |         0x200000 |
|  227 | audio_encoder.encoders.16.feed_forward.w_1.bias            |        0x704f740 |           0x1000 |
|  228 | audio_encoder.encoders.16.feed_forward.w_2.weight          |        0x7050740 |         0x200000 |
|  229 | audio_encoder.encoders.16.feed_forward.w_2.bias            |        0x7250740 |            0x400 |
|  230 | audio_encoder.encoders.16.norm1.weight                     |        0x7250b40 |            0x400 |
|  231 | audio_encoder.encoders.16.norm1.bias                       |        0x7250f40 |            0x400 |
|  232 | audio_encoder.encoders.16.norm2.weight                     |        0x7251340 |            0x400 |
|  233 | audio_encoder.encoders.16.norm2.bias                       |        0x7251740 |            0x400 |
|  234 | audio_encoder.encoders.17.self_attn.linear_out.weight      |        0x7251b40 |          0x80000 |
|  235 | audio_encoder.encoders.17.self_attn.linear_out.bias        |        0x72d1b40 |            0x400 |
|  236 | audio_encoder.encoders.17.self_attn.linear_q_k_v.weight    |        0x72d1f40 |         0x180000 |
|  237 | audio_encoder.encoders.17.self_attn.linear_q_k_v.bias      |        0x7451f40 |            0xc00 |
|  238 | audio_encoder.encoders.17.self_attn.fsmn_block.weight      |        0x7452b40 |           0x2c00 |
|  239 | audio_encoder.encoders.17.feed_forward.w_1.weight          |        0x7455740 |         0x200000 |
|  240 | audio_encoder.encoders.17.feed_forward.w_1.bias            |        0x7655740 |           0x1000 |
|  241 | audio_encoder.encoders.17.feed_forward.w_2.weight          |        0x7656740 |         0x200000 |
|  242 | audio_encoder.encoders.17.feed_forward.w_2.bias            |        0x7856740 |            0x400 |
|  243 | audio_encoder.encoders.17.norm1.weight                     |        0x7856b40 |            0x400 |
|  244 | audio_encoder.encoders.17.norm1.bias                       |        0x7856f40 |            0x400 |
|  245 | audio_encoder.encoders.17.norm2.weight                     |        0x7857340 |            0x400 |
|  246 | audio_encoder.encoders.17.norm2.bias                       |        0x7857740 |            0x400 |
|  247 | audio_encoder.encoders.18.self_attn.linear_out.weight      |        0x7857b40 |          0x80000 |
|  248 | audio_encoder.encoders.18.self_attn.linear_out.bias        |        0x78d7b40 |            0x400 |
|  249 | audio_encoder.encoders.18.self_attn.linear_q_k_v.weight    |        0x78d7f40 |         0x180000 |
|  250 | audio_encoder.encoders.18.self_attn.linear_q_k_v.bias      |        0x7a57f40 |            0xc00 |
|  251 | audio_encoder.encoders.18.self_attn.fsmn_block.weight      |        0x7a58b40 |           0x2c00 |
|  252 | audio_encoder.encoders.18.feed_forward.w_1.weight          |        0x7a5b740 |         0x200000 |
|  253 | audio_encoder.encoders.18.feed_forward.w_1.bias            |        0x7c5b740 |           0x1000 |
|  254 | audio_encoder.encoders.18.feed_forward.w_2.weight          |        0x7c5c740 |         0x200000 |
|  255 | audio_encoder.encoders.18.feed_forward.w_2.bias            |        0x7e5c740 |            0x400 |
|  256 | audio_encoder.encoders.18.norm1.weight                     |        0x7e5cb40 |            0x400 |
|  257 | audio_encoder.encoders.18.norm1.bias                       |        0x7e5cf40 |            0x400 |
|  258 | audio_encoder.encoders.18.norm2.weight                     |        0x7e5d340 |            0x400 |
|  259 | audio_encoder.encoders.18.norm2.bias                       |        0x7e5d740 |            0x400 |
|  260 | audio_encoder.encoders.19.self_attn.linear_out.weight      |        0x7e5db40 |          0x80000 |
|  261 | audio_encoder.encoders.19.self_attn.linear_out.bias        |        0x7eddb40 |            0x400 |
|  262 | audio_encoder.encoders.19.self_attn.linear_q_k_v.weight    |        0x7eddf40 |         0x180000 |
|  263 | audio_encoder.encoders.19.self_attn.linear_q_k_v.bias      |        0x805df40 |            0xc00 |
|  264 | audio_encoder.encoders.19.self_attn.fsmn_block.weight      |        0x805eb40 |           0x2c00 |
|  265 | audio_encoder.encoders.19.feed_forward.w_1.weight          |        0x8061740 |         0x200000 |
|  266 | audio_encoder.encoders.19.feed_forward.w_1.bias            |        0x8261740 |           0x1000 |
|  267 | audio_encoder.encoders.19.feed_forward.w_2.weight          |        0x8262740 |         0x200000 |
|  268 | audio_encoder.encoders.19.feed_forward.w_2.bias            |        0x8462740 |            0x400 |
|  269 | audio_encoder.encoders.19.norm1.weight                     |        0x8462b40 |            0x400 |
|  270 | audio_encoder.encoders.19.norm1.bias                       |        0x8462f40 |            0x400 |
|  271 | audio_encoder.encoders.19.norm2.weight                     |        0x8463340 |            0x400 |
|  272 | audio_encoder.encoders.19.norm2.bias                       |        0x8463740 |            0x400 |
|  273 | audio_encoder.encoders.20.self_attn.linear_out.weight      |        0x8463b40 |          0x80000 |
|  274 | audio_encoder.encoders.20.self_attn.linear_out.bias        |        0x84e3b40 |            0x400 |
|  275 | audio_encoder.encoders.20.self_attn.linear_q_k_v.weight    |        0x84e3f40 |         0x180000 |
|  276 | audio_encoder.encoders.20.self_attn.linear_q_k_v.bias      |        0x8663f40 |            0xc00 |
|  277 | audio_encoder.encoders.20.self_attn.fsmn_block.weight      |        0x8664b40 |           0x2c00 |
|  278 | audio_encoder.encoders.20.feed_forward.w_1.weight          |        0x8667740 |         0x200000 |
|  279 | audio_encoder.encoders.20.feed_forward.w_1.bias            |        0x8867740 |           0x1000 |
|  280 | audio_encoder.encoders.20.feed_forward.w_2.weight          |        0x8868740 |         0x200000 |
|  281 | audio_encoder.encoders.20.feed_forward.w_2.bias            |        0x8a68740 |            0x400 |
|  282 | audio_encoder.encoders.20.norm1.weight                     |        0x8a68b40 |            0x400 |
|  283 | audio_encoder.encoders.20.norm1.bias                       |        0x8a68f40 |            0x400 |
|  284 | audio_encoder.encoders.20.norm2.weight                     |        0x8a69340 |            0x400 |
|  285 | audio_encoder.encoders.20.norm2.bias                       |        0x8a69740 |            0x400 |
|  286 | audio_encoder.encoders.21.self_attn.linear_out.weight      |        0x8a69b40 |          0x80000 |
|  287 | audio_encoder.encoders.21.self_attn.linear_out.bias        |        0x8ae9b40 |            0x400 |
|  288 | audio_encoder.encoders.21.self_attn.linear_q_k_v.weight    |        0x8ae9f40 |         0x180000 |
|  289 | audio_encoder.encoders.21.self_attn.linear_q_k_v.bias      |        0x8c69f40 |            0xc00 |
|  290 | audio_encoder.encoders.21.self_attn.fsmn_block.weight      |        0x8c6ab40 |           0x2c00 |
|  291 | audio_encoder.encoders.21.feed_forward.w_1.weight          |        0x8c6d740 |         0x200000 |
|  292 | audio_encoder.encoders.21.feed_forward.w_1.bias            |        0x8e6d740 |           0x1000 |
|  293 | audio_encoder.encoders.21.feed_forward.w_2.weight          |        0x8e6e740 |         0x200000 |
|  294 | audio_encoder.encoders.21.feed_forward.w_2.bias            |        0x906e740 |            0x400 |
|  295 | audio_encoder.encoders.21.norm1.weight                     |        0x906eb40 |            0x400 |
|  296 | audio_encoder.encoders.21.norm1.bias                       |        0x906ef40 |            0x400 |
|  297 | audio_encoder.encoders.21.norm2.weight                     |        0x906f340 |            0x400 |
|  298 | audio_encoder.encoders.21.norm2.bias                       |        0x906f740 |            0x400 |
|  299 | audio_encoder.encoders.22.self_attn.linear_out.weight      |        0x906fb40 |          0x80000 |
|  300 | audio_encoder.encoders.22.self_attn.linear_out.bias        |        0x90efb40 |            0x400 |
|  301 | audio_encoder.encoders.22.self_attn.linear_q_k_v.weight    |        0x90eff40 |         0x180000 |
|  302 | audio_encoder.encoders.22.self_attn.linear_q_k_v.bias      |        0x926ff40 |            0xc00 |
|  303 | audio_encoder.encoders.22.self_attn.fsmn_block.weight      |        0x9270b40 |           0x2c00 |
|  304 | audio_encoder.encoders.22.feed_forward.w_1.weight          |        0x9273740 |         0x200000 |
|  305 | audio_encoder.encoders.22.feed_forward.w_1.bias            |        0x9473740 |           0x1000 |
|  306 | audio_encoder.encoders.22.feed_forward.w_2.weight          |        0x9474740 |         0x200000 |
|  307 | audio_encoder.encoders.22.feed_forward.w_2.bias            |        0x9674740 |            0x400 |
|  308 | audio_encoder.encoders.22.norm1.weight                     |        0x9674b40 |            0x400 |
|  309 | audio_encoder.encoders.22.norm1.bias                       |        0x9674f40 |            0x400 |
|  310 | audio_encoder.encoders.22.norm2.weight                     |        0x9675340 |            0x400 |
|  311 | audio_encoder.encoders.22.norm2.bias                       |        0x9675740 |            0x400 |
|  312 | audio_encoder.encoders.23.self_attn.linear_out.weight      |        0x9675b40 |          0x80000 |
|  313 | audio_encoder.encoders.23.self_attn.linear_out.bias        |        0x96f5b40 |            0x400 |
|  314 | audio_encoder.encoders.23.self_attn.linear_q_k_v.weight    |        0x96f5f40 |         0x180000 |
|  315 | audio_encoder.encoders.23.self_attn.linear_q_k_v.bias      |        0x9875f40 |            0xc00 |
|  316 | audio_encoder.encoders.23.self_attn.fsmn_block.weight      |        0x9876b40 |           0x2c00 |
|  317 | audio_encoder.encoders.23.feed_forward.w_1.weight          |        0x9879740 |         0x200000 |
|  318 | audio_encoder.encoders.23.feed_forward.w_1.bias            |        0x9a79740 |           0x1000 |
|  319 | audio_encoder.encoders.23.feed_forward.w_2.weight          |        0x9a7a740 |         0x200000 |
|  320 | audio_encoder.encoders.23.feed_forward.w_2.bias            |        0x9c7a740 |            0x400 |
|  321 | audio_encoder.encoders.23.norm1.weight                     |        0x9c7ab40 |            0x400 |
|  322 | audio_encoder.encoders.23.norm1.bias                       |        0x9c7af40 |            0x400 |
|  323 | audio_encoder.encoders.23.norm2.weight                     |        0x9c7b340 |            0x400 |
|  324 | audio_encoder.encoders.23.norm2.bias                       |        0x9c7b740 |            0x400 |
|  325 | audio_encoder.encoders.24.self_attn.linear_out.weight      |        0x9c7bb40 |          0x80000 |
|  326 | audio_encoder.encoders.24.self_attn.linear_out.bias        |        0x9cfbb40 |            0x400 |
|  327 | audio_encoder.encoders.24.self_attn.linear_q_k_v.weight    |        0x9cfbf40 |         0x180000 |
|  328 | audio_encoder.encoders.24.self_attn.linear_q_k_v.bias      |        0x9e7bf40 |            0xc00 |
|  329 | audio_encoder.encoders.24.self_attn.fsmn_block.weight      |        0x9e7cb40 |           0x2c00 |
|  330 | audio_encoder.encoders.24.feed_forward.w_1.weight          |        0x9e7f740 |         0x200000 |
|  331 | audio_encoder.encoders.24.feed_forward.w_1.bias            |        0xa07f740 |           0x1000 |
|  332 | audio_encoder.encoders.24.feed_forward.w_2.weight          |        0xa080740 |         0x200000 |
|  333 | audio_encoder.encoders.24.feed_forward.w_2.bias            |        0xa280740 |            0x400 |
|  334 | audio_encoder.encoders.24.norm1.weight                     |        0xa280b40 |            0x400 |
|  335 | audio_encoder.encoders.24.norm1.bias                       |        0xa280f40 |            0x400 |
|  336 | audio_encoder.encoders.24.norm2.weight                     |        0xa281340 |            0x400 |
|  337 | audio_encoder.encoders.24.norm2.bias                       |        0xa281740 |            0x400 |
|  338 | audio_encoder.encoders.25.self_attn.linear_out.weight      |        0xa281b40 |          0x80000 |
|  339 | audio_encoder.encoders.25.self_attn.linear_out.bias        |        0xa301b40 |            0x400 |
|  340 | audio_encoder.encoders.25.self_attn.linear_q_k_v.weight    |        0xa301f40 |         0x180000 |
|  341 | audio_encoder.encoders.25.self_attn.linear_q_k_v.bias      |        0xa481f40 |            0xc00 |
|  342 | audio_encoder.encoders.25.self_attn.fsmn_block.weight      |        0xa482b40 |           0x2c00 |
|  343 | audio_encoder.encoders.25.feed_forward.w_1.weight          |        0xa485740 |         0x200000 |
|  344 | audio_encoder.encoders.25.feed_forward.w_1.bias            |        0xa685740 |           0x1000 |
|  345 | audio_encoder.encoders.25.feed_forward.w_2.weight          |        0xa686740 |         0x200000 |
|  346 | audio_encoder.encoders.25.feed_forward.w_2.bias            |        0xa886740 |            0x400 |
|  347 | audio_encoder.encoders.25.norm1.weight                     |        0xa886b40 |            0x400 |
|  348 | audio_encoder.encoders.25.norm1.bias                       |        0xa886f40 |            0x400 |
|  349 | audio_encoder.encoders.25.norm2.weight                     |        0xa887340 |            0x400 |
|  350 | audio_encoder.encoders.25.norm2.bias                       |        0xa887740 |            0x400 |
|  351 | audio_encoder.encoders.26.self_attn.linear_out.weight      |        0xa887b40 |          0x80000 |
|  352 | audio_encoder.encoders.26.self_attn.linear_out.bias        |        0xa907b40 |            0x400 |
|  353 | audio_encoder.encoders.26.self_attn.linear_q_k_v.weight    |        0xa907f40 |         0x180000 |
|  354 | audio_encoder.encoders.26.self_attn.linear_q_k_v.bias      |        0xaa87f40 |            0xc00 |
|  355 | audio_encoder.encoders.26.self_attn.fsmn_block.weight      |        0xaa88b40 |           0x2c00 |
|  356 | audio_encoder.encoders.26.feed_forward.w_1.weight          |        0xaa8b740 |         0x200000 |
|  357 | audio_encoder.encoders.26.feed_forward.w_1.bias            |        0xac8b740 |           0x1000 |
|  358 | audio_encoder.encoders.26.feed_forward.w_2.weight          |        0xac8c740 |         0x200000 |
|  359 | audio_encoder.encoders.26.feed_forward.w_2.bias            |        0xae8c740 |            0x400 |
|  360 | audio_encoder.encoders.26.norm1.weight                     |        0xae8cb40 |            0x400 |
|  361 | audio_encoder.encoders.26.norm1.bias                       |        0xae8cf40 |            0x400 |
|  362 | audio_encoder.encoders.26.norm2.weight                     |        0xae8d340 |            0x400 |
|  363 | audio_encoder.encoders.26.norm2.bias                       |        0xae8d740 |            0x400 |
|  364 | audio_encoder.encoders.27.self_attn.linear_out.weight      |        0xae8db40 |          0x80000 |
|  365 | audio_encoder.encoders.27.self_attn.linear_out.bias        |        0xaf0db40 |            0x400 |
|  366 | audio_encoder.encoders.27.self_attn.linear_q_k_v.weight    |        0xaf0df40 |         0x180000 |
|  367 | audio_encoder.encoders.27.self_attn.linear_q_k_v.bias      |        0xb08df40 |            0xc00 |
|  368 | audio_encoder.encoders.27.self_attn.fsmn_block.weight      |        0xb08eb40 |           0x2c00 |
|  369 | audio_encoder.encoders.27.feed_forward.w_1.weight          |        0xb091740 |         0x200000 |
|  370 | audio_encoder.encoders.27.feed_forward.w_1.bias            |        0xb291740 |           0x1000 |
|  371 | audio_encoder.encoders.27.feed_forward.w_2.weight          |        0xb292740 |         0x200000 |
|  372 | audio_encoder.encoders.27.feed_forward.w_2.bias            |        0xb492740 |            0x400 |
|  373 | audio_encoder.encoders.27.norm1.weight                     |        0xb492b40 |            0x400 |
|  374 | audio_encoder.encoders.27.norm1.bias                       |        0xb492f40 |            0x400 |
|  375 | audio_encoder.encoders.27.norm2.weight                     |        0xb493340 |            0x400 |
|  376 | audio_encoder.encoders.27.norm2.bias                       |        0xb493740 |            0x400 |
|  377 | audio_encoder.encoders.28.self_attn.linear_out.weight      |        0xb493b40 |          0x80000 |
|  378 | audio_encoder.encoders.28.self_attn.linear_out.bias        |        0xb513b40 |            0x400 |
|  379 | audio_encoder.encoders.28.self_attn.linear_q_k_v.weight    |        0xb513f40 |         0x180000 |
|  380 | audio_encoder.encoders.28.self_attn.linear_q_k_v.bias      |        0xb693f40 |            0xc00 |
|  381 | audio_encoder.encoders.28.self_attn.fsmn_block.weight      |        0xb694b40 |           0x2c00 |
|  382 | audio_encoder.encoders.28.feed_forward.w_1.weight          |        0xb697740 |         0x200000 |
|  383 | audio_encoder.encoders.28.feed_forward.w_1.bias            |        0xb897740 |           0x1000 |
|  384 | audio_encoder.encoders.28.feed_forward.w_2.weight          |        0xb898740 |         0x200000 |
|  385 | audio_encoder.encoders.28.feed_forward.w_2.bias            |        0xba98740 |            0x400 |
|  386 | audio_encoder.encoders.28.norm1.weight                     |        0xba98b40 |            0x400 |
|  387 | audio_encoder.encoders.28.norm1.bias                       |        0xba98f40 |            0x400 |
|  388 | audio_encoder.encoders.28.norm2.weight                     |        0xba99340 |            0x400 |
|  389 | audio_encoder.encoders.28.norm2.bias                       |        0xba99740 |            0x400 |
|  390 | audio_encoder.encoders.29.self_attn.linear_out.weight      |        0xba99b40 |          0x80000 |
|  391 | audio_encoder.encoders.29.self_attn.linear_out.bias        |        0xbb19b40 |            0x400 |
|  392 | audio_encoder.encoders.29.self_attn.linear_q_k_v.weight    |        0xbb19f40 |         0x180000 |
|  393 | audio_encoder.encoders.29.self_attn.linear_q_k_v.bias      |        0xbc99f40 |            0xc00 |
|  394 | audio_encoder.encoders.29.self_attn.fsmn_block.weight      |        0xbc9ab40 |           0x2c00 |
|  395 | audio_encoder.encoders.29.feed_forward.w_1.weight          |        0xbc9d740 |         0x200000 |
|  396 | audio_encoder.encoders.29.feed_forward.w_1.bias            |        0xbe9d740 |           0x1000 |
|  397 | audio_encoder.encoders.29.feed_forward.w_2.weight          |        0xbe9e740 |         0x200000 |
|  398 | audio_encoder.encoders.29.feed_forward.w_2.bias            |        0xc09e740 |            0x400 |
|  399 | audio_encoder.encoders.29.norm1.weight                     |        0xc09eb40 |            0x400 |
|  400 | audio_encoder.encoders.29.norm1.bias                       |        0xc09ef40 |            0x400 |
|  401 | audio_encoder.encoders.29.norm2.weight                     |        0xc09f340 |            0x400 |
|  402 | audio_encoder.encoders.29.norm2.bias                       |        0xc09f740 |            0x400 |
|  403 | audio_encoder.encoders.30.self_attn.linear_out.weight      |        0xc09fb40 |          0x80000 |
|  404 | audio_encoder.encoders.30.self_attn.linear_out.bias        |        0xc11fb40 |            0x400 |
|  405 | audio_encoder.encoders.30.self_attn.linear_q_k_v.weight    |        0xc11ff40 |         0x180000 |
|  406 | audio_encoder.encoders.30.self_attn.linear_q_k_v.bias      |        0xc29ff40 |            0xc00 |
|  407 | audio_encoder.encoders.30.self_attn.fsmn_block.weight      |        0xc2a0b40 |           0x2c00 |
|  408 | audio_encoder.encoders.30.feed_forward.w_1.weight          |        0xc2a3740 |         0x200000 |
|  409 | audio_encoder.encoders.30.feed_forward.w_1.bias            |        0xc4a3740 |           0x1000 |
|  410 | audio_encoder.encoders.30.feed_forward.w_2.weight          |        0xc4a4740 |         0x200000 |
|  411 | audio_encoder.encoders.30.feed_forward.w_2.bias            |        0xc6a4740 |            0x400 |
|  412 | audio_encoder.encoders.30.norm1.weight                     |        0xc6a4b40 |            0x400 |
|  413 | audio_encoder.encoders.30.norm1.bias                       |        0xc6a4f40 |            0x400 |
|  414 | audio_encoder.encoders.30.norm2.weight                     |        0xc6a5340 |            0x400 |
|  415 | audio_encoder.encoders.30.norm2.bias                       |        0xc6a5740 |            0x400 |
|  416 | audio_encoder.encoders.31.self_attn.linear_out.weight      |        0xc6a5b40 |          0x80000 |
|  417 | audio_encoder.encoders.31.self_attn.linear_out.bias        |        0xc725b40 |            0x400 |
|  418 | audio_encoder.encoders.31.self_attn.linear_q_k_v.weight    |        0xc725f40 |         0x180000 |
|  419 | audio_encoder.encoders.31.self_attn.linear_q_k_v.bias      |        0xc8a5f40 |            0xc00 |
|  420 | audio_encoder.encoders.31.self_attn.fsmn_block.weight      |        0xc8a6b40 |           0x2c00 |
|  421 | audio_encoder.encoders.31.feed_forward.w_1.weight          |        0xc8a9740 |         0x200000 |
|  422 | audio_encoder.encoders.31.feed_forward.w_1.bias            |        0xcaa9740 |           0x1000 |
|  423 | audio_encoder.encoders.31.feed_forward.w_2.weight          |        0xcaaa740 |         0x200000 |
|  424 | audio_encoder.encoders.31.feed_forward.w_2.bias            |        0xccaa740 |            0x400 |
|  425 | audio_encoder.encoders.31.norm1.weight                     |        0xccaab40 |            0x400 |
|  426 | audio_encoder.encoders.31.norm1.bias                       |        0xccaaf40 |            0x400 |
|  427 | audio_encoder.encoders.31.norm2.weight                     |        0xccab340 |            0x400 |
|  428 | audio_encoder.encoders.31.norm2.bias                       |        0xccab740 |            0x400 |
|  429 | audio_encoder.encoders.32.self_attn.linear_out.weight      |        0xccabb40 |          0x80000 |
|  430 | audio_encoder.encoders.32.self_attn.linear_out.bias        |        0xcd2bb40 |            0x400 |
|  431 | audio_encoder.encoders.32.self_attn.linear_q_k_v.weight    |        0xcd2bf40 |         0x180000 |
|  432 | audio_encoder.encoders.32.self_attn.linear_q_k_v.bias      |        0xceabf40 |            0xc00 |
|  433 | audio_encoder.encoders.32.self_attn.fsmn_block.weight      |        0xceacb40 |           0x2c00 |
|  434 | audio_encoder.encoders.32.feed_forward.w_1.weight          |        0xceaf740 |         0x200000 |
|  435 | audio_encoder.encoders.32.feed_forward.w_1.bias            |        0xd0af740 |           0x1000 |
|  436 | audio_encoder.encoders.32.feed_forward.w_2.weight          |        0xd0b0740 |         0x200000 |
|  437 | audio_encoder.encoders.32.feed_forward.w_2.bias            |        0xd2b0740 |            0x400 |
|  438 | audio_encoder.encoders.32.norm1.weight                     |        0xd2b0b40 |            0x400 |
|  439 | audio_encoder.encoders.32.norm1.bias                       |        0xd2b0f40 |            0x400 |
|  440 | audio_encoder.encoders.32.norm2.weight                     |        0xd2b1340 |            0x400 |
|  441 | audio_encoder.encoders.32.norm2.bias                       |        0xd2b1740 |            0x400 |
|  442 | audio_encoder.encoders.33.self_attn.linear_out.weight      |        0xd2b1b40 |          0x80000 |
|  443 | audio_encoder.encoders.33.self_attn.linear_out.bias        |        0xd331b40 |            0x400 |
|  444 | audio_encoder.encoders.33.self_attn.linear_q_k_v.weight    |        0xd331f40 |         0x180000 |
|  445 | audio_encoder.encoders.33.self_attn.linear_q_k_v.bias      |        0xd4b1f40 |            0xc00 |
|  446 | audio_encoder.encoders.33.self_attn.fsmn_block.weight      |        0xd4b2b40 |           0x2c00 |
|  447 | audio_encoder.encoders.33.feed_forward.w_1.weight          |        0xd4b5740 |         0x200000 |
|  448 | audio_encoder.encoders.33.feed_forward.w_1.bias            |        0xd6b5740 |           0x1000 |
|  449 | audio_encoder.encoders.33.feed_forward.w_2.weight          |        0xd6b6740 |         0x200000 |
|  450 | audio_encoder.encoders.33.feed_forward.w_2.bias            |        0xd8b6740 |            0x400 |
|  451 | audio_encoder.encoders.33.norm1.weight                     |        0xd8b6b40 |            0x400 |
|  452 | audio_encoder.encoders.33.norm1.bias                       |        0xd8b6f40 |            0x400 |
|  453 | audio_encoder.encoders.33.norm2.weight                     |        0xd8b7340 |            0x400 |
|  454 | audio_encoder.encoders.33.norm2.bias                       |        0xd8b7740 |            0x400 |
|  455 | audio_encoder.encoders.34.self_attn.linear_out.weight      |        0xd8b7b40 |          0x80000 |
|  456 | audio_encoder.encoders.34.self_attn.linear_out.bias        |        0xd937b40 |            0x400 |
|  457 | audio_encoder.encoders.34.self_attn.linear_q_k_v.weight    |        0xd937f40 |         0x180000 |
|  458 | audio_encoder.encoders.34.self_attn.linear_q_k_v.bias      |        0xdab7f40 |            0xc00 |
|  459 | audio_encoder.encoders.34.self_attn.fsmn_block.weight      |        0xdab8b40 |           0x2c00 |
|  460 | audio_encoder.encoders.34.feed_forward.w_1.weight          |        0xdabb740 |         0x200000 |
|  461 | audio_encoder.encoders.34.feed_forward.w_1.bias            |        0xdcbb740 |           0x1000 |
|  462 | audio_encoder.encoders.34.feed_forward.w_2.weight          |        0xdcbc740 |         0x200000 |
|  463 | audio_encoder.encoders.34.feed_forward.w_2.bias            |        0xdebc740 |            0x400 |
|  464 | audio_encoder.encoders.34.norm1.weight                     |        0xdebcb40 |            0x400 |
|  465 | audio_encoder.encoders.34.norm1.bias                       |        0xdebcf40 |            0x400 |
|  466 | audio_encoder.encoders.34.norm2.weight                     |        0xdebd340 |            0x400 |
|  467 | audio_encoder.encoders.34.norm2.bias                       |        0xdebd740 |            0x400 |
|  468 | audio_encoder.encoders.35.self_attn.linear_out.weight      |        0xdebdb40 |          0x80000 |
|  469 | audio_encoder.encoders.35.self_attn.linear_out.bias        |        0xdf3db40 |            0x400 |
|  470 | audio_encoder.encoders.35.self_attn.linear_q_k_v.weight    |        0xdf3df40 |         0x180000 |
|  471 | audio_encoder.encoders.35.self_attn.linear_q_k_v.bias      |        0xe0bdf40 |            0xc00 |
|  472 | audio_encoder.encoders.35.self_attn.fsmn_block.weight      |        0xe0beb40 |           0x2c00 |
|  473 | audio_encoder.encoders.35.feed_forward.w_1.weight          |        0xe0c1740 |         0x200000 |
|  474 | audio_encoder.encoders.35.feed_forward.w_1.bias            |        0xe2c1740 |           0x1000 |
|  475 | audio_encoder.encoders.35.feed_forward.w_2.weight          |        0xe2c2740 |         0x200000 |
|  476 | audio_encoder.encoders.35.feed_forward.w_2.bias            |        0xe4c2740 |            0x400 |
|  477 | audio_encoder.encoders.35.norm1.weight                     |        0xe4c2b40 |            0x400 |
|  478 | audio_encoder.encoders.35.norm1.bias                       |        0xe4c2f40 |            0x400 |
|  479 | audio_encoder.encoders.35.norm2.weight                     |        0xe4c3340 |            0x400 |
|  480 | audio_encoder.encoders.35.norm2.bias                       |        0xe4c3740 |            0x400 |
|  481 | audio_encoder.encoders.36.self_attn.linear_out.weight      |        0xe4c3b40 |          0x80000 |
|  482 | audio_encoder.encoders.36.self_attn.linear_out.bias        |        0xe543b40 |            0x400 |
|  483 | audio_encoder.encoders.36.self_attn.linear_q_k_v.weight    |        0xe543f40 |         0x180000 |
|  484 | audio_encoder.encoders.36.self_attn.linear_q_k_v.bias      |        0xe6c3f40 |            0xc00 |
|  485 | audio_encoder.encoders.36.self_attn.fsmn_block.weight      |        0xe6c4b40 |           0x2c00 |
|  486 | audio_encoder.encoders.36.feed_forward.w_1.weight          |        0xe6c7740 |         0x200000 |
|  487 | audio_encoder.encoders.36.feed_forward.w_1.bias            |        0xe8c7740 |           0x1000 |
|  488 | audio_encoder.encoders.36.feed_forward.w_2.weight          |        0xe8c8740 |         0x200000 |
|  489 | audio_encoder.encoders.36.feed_forward.w_2.bias            |        0xeac8740 |            0x400 |
|  490 | audio_encoder.encoders.36.norm1.weight                     |        0xeac8b40 |            0x400 |
|  491 | audio_encoder.encoders.36.norm1.bias                       |        0xeac8f40 |            0x400 |
|  492 | audio_encoder.encoders.36.norm2.weight                     |        0xeac9340 |            0x400 |
|  493 | audio_encoder.encoders.36.norm2.bias                       |        0xeac9740 |            0x400 |
|  494 | audio_encoder.encoders.37.self_attn.linear_out.weight      |        0xeac9b40 |          0x80000 |
|  495 | audio_encoder.encoders.37.self_attn.linear_out.bias        |        0xeb49b40 |            0x400 |
|  496 | audio_encoder.encoders.37.self_attn.linear_q_k_v.weight    |        0xeb49f40 |         0x180000 |
|  497 | audio_encoder.encoders.37.self_attn.linear_q_k_v.bias      |        0xecc9f40 |            0xc00 |
|  498 | audio_encoder.encoders.37.self_attn.fsmn_block.weight      |        0xeccab40 |           0x2c00 |
|  499 | audio_encoder.encoders.37.feed_forward.w_1.weight          |        0xeccd740 |         0x200000 |
|  500 | audio_encoder.encoders.37.feed_forward.w_1.bias            |        0xeecd740 |           0x1000 |
|  501 | audio_encoder.encoders.37.feed_forward.w_2.weight          |        0xeece740 |         0x200000 |
|  502 | audio_encoder.encoders.37.feed_forward.w_2.bias            |        0xf0ce740 |            0x400 |
|  503 | audio_encoder.encoders.37.norm1.weight                     |        0xf0ceb40 |            0x400 |
|  504 | audio_encoder.encoders.37.norm1.bias                       |        0xf0cef40 |            0x400 |
|  505 | audio_encoder.encoders.37.norm2.weight                     |        0xf0cf340 |            0x400 |
|  506 | audio_encoder.encoders.37.norm2.bias                       |        0xf0cf740 |            0x400 |
|  507 | audio_encoder.encoders.38.self_attn.linear_out.weight      |        0xf0cfb40 |          0x80000 |
|  508 | audio_encoder.encoders.38.self_attn.linear_out.bias        |        0xf14fb40 |            0x400 |
|  509 | audio_encoder.encoders.38.self_attn.linear_q_k_v.weight    |        0xf14ff40 |         0x180000 |
|  510 | audio_encoder.encoders.38.self_attn.linear_q_k_v.bias      |        0xf2cff40 |            0xc00 |
|  511 | audio_encoder.encoders.38.self_attn.fsmn_block.weight      |        0xf2d0b40 |           0x2c00 |
|  512 | audio_encoder.encoders.38.feed_forward.w_1.weight          |        0xf2d3740 |         0x200000 |
|  513 | audio_encoder.encoders.38.feed_forward.w_1.bias            |        0xf4d3740 |           0x1000 |
|  514 | audio_encoder.encoders.38.feed_forward.w_2.weight          |        0xf4d4740 |         0x200000 |
|  515 | audio_encoder.encoders.38.feed_forward.w_2.bias            |        0xf6d4740 |            0x400 |
|  516 | audio_encoder.encoders.38.norm1.weight                     |        0xf6d4b40 |            0x400 |
|  517 | audio_encoder.encoders.38.norm1.bias                       |        0xf6d4f40 |            0x400 |
|  518 | audio_encoder.encoders.38.norm2.weight                     |        0xf6d5340 |            0x400 |
|  519 | audio_encoder.encoders.38.norm2.bias                       |        0xf6d5740 |            0x400 |
|  520 | audio_encoder.encoders.39.self_attn.linear_out.weight      |        0xf6d5b40 |          0x80000 |
|  521 | audio_encoder.encoders.39.self_attn.linear_out.bias        |        0xf755b40 |            0x400 |
|  522 | audio_encoder.encoders.39.self_attn.linear_q_k_v.weight    |        0xf755f40 |         0x180000 |
|  523 | audio_encoder.encoders.39.self_attn.linear_q_k_v.bias      |        0xf8d5f40 |            0xc00 |
|  524 | audio_encoder.encoders.39.self_attn.fsmn_block.weight      |        0xf8d6b40 |           0x2c00 |
|  525 | audio_encoder.encoders.39.feed_forward.w_1.weight          |        0xf8d9740 |         0x200000 |
|  526 | audio_encoder.encoders.39.feed_forward.w_1.bias            |        0xfad9740 |           0x1000 |
|  527 | audio_encoder.encoders.39.feed_forward.w_2.weight          |        0xfada740 |         0x200000 |
|  528 | audio_encoder.encoders.39.feed_forward.w_2.bias            |        0xfcda740 |            0x400 |
|  529 | audio_encoder.encoders.39.norm1.weight                     |        0xfcdab40 |            0x400 |
|  530 | audio_encoder.encoders.39.norm1.bias                       |        0xfcdaf40 |            0x400 |
|  531 | audio_encoder.encoders.39.norm2.weight                     |        0xfcdb340 |            0x400 |
|  532 | audio_encoder.encoders.39.norm2.bias                       |        0xfcdb740 |            0x400 |
|  533 | audio_encoder.encoders.40.self_attn.linear_out.weight      |        0xfcdbb40 |          0x80000 |
|  534 | audio_encoder.encoders.40.self_attn.linear_out.bias        |        0xfd5bb40 |            0x400 |
|  535 | audio_encoder.encoders.40.self_attn.linear_q_k_v.weight    |        0xfd5bf40 |         0x180000 |
|  536 | audio_encoder.encoders.40.self_attn.linear_q_k_v.bias      |        0xfedbf40 |            0xc00 |
|  537 | audio_encoder.encoders.40.self_attn.fsmn_block.weight      |        0xfedcb40 |           0x2c00 |
|  538 | audio_encoder.encoders.40.feed_forward.w_1.weight          |        0xfedf740 |         0x200000 |
|  539 | audio_encoder.encoders.40.feed_forward.w_1.bias            |       0x100df740 |           0x1000 |
|  540 | audio_encoder.encoders.40.feed_forward.w_2.weight          |       0x100e0740 |         0x200000 |
|  541 | audio_encoder.encoders.40.feed_forward.w_2.bias            |       0x102e0740 |            0x400 |
|  542 | audio_encoder.encoders.40.norm1.weight                     |       0x102e0b40 |            0x400 |
|  543 | audio_encoder.encoders.40.norm1.bias                       |       0x102e0f40 |            0x400 |
|  544 | audio_encoder.encoders.40.norm2.weight                     |       0x102e1340 |            0x400 |
|  545 | audio_encoder.encoders.40.norm2.bias                       |       0x102e1740 |            0x400 |
|  546 | audio_encoder.encoders.41.self_attn.linear_out.weight      |       0x102e1b40 |          0x80000 |
|  547 | audio_encoder.encoders.41.self_attn.linear_out.bias        |       0x10361b40 |            0x400 |
|  548 | audio_encoder.encoders.41.self_attn.linear_q_k_v.weight    |       0x10361f40 |         0x180000 |
|  549 | audio_encoder.encoders.41.self_attn.linear_q_k_v.bias      |       0x104e1f40 |            0xc00 |
|  550 | audio_encoder.encoders.41.self_attn.fsmn_block.weight      |       0x104e2b40 |           0x2c00 |
|  551 | audio_encoder.encoders.41.feed_forward.w_1.weight          |       0x104e5740 |         0x200000 |
|  552 | audio_encoder.encoders.41.feed_forward.w_1.bias            |       0x106e5740 |           0x1000 |
|  553 | audio_encoder.encoders.41.feed_forward.w_2.weight          |       0x106e6740 |         0x200000 |
|  554 | audio_encoder.encoders.41.feed_forward.w_2.bias            |       0x108e6740 |            0x400 |
|  555 | audio_encoder.encoders.41.norm1.weight                     |       0x108e6b40 |            0x400 |
|  556 | audio_encoder.encoders.41.norm1.bias                       |       0x108e6f40 |            0x400 |
|  557 | audio_encoder.encoders.41.norm2.weight                     |       0x108e7340 |            0x400 |
|  558 | audio_encoder.encoders.41.norm2.bias                       |       0x108e7740 |            0x400 |
|  559 | audio_encoder.encoders.42.self_attn.linear_out.weight      |       0x108e7b40 |          0x80000 |
|  560 | audio_encoder.encoders.42.self_attn.linear_out.bias        |       0x10967b40 |            0x400 |
|  561 | audio_encoder.encoders.42.self_attn.linear_q_k_v.weight    |       0x10967f40 |         0x180000 |
|  562 | audio_encoder.encoders.42.self_attn.linear_q_k_v.bias      |       0x10ae7f40 |            0xc00 |
|  563 | audio_encoder.encoders.42.self_attn.fsmn_block.weight      |       0x10ae8b40 |           0x2c00 |
|  564 | audio_encoder.encoders.42.feed_forward.w_1.weight          |       0x10aeb740 |         0x200000 |
|  565 | audio_encoder.encoders.42.feed_forward.w_1.bias            |       0x10ceb740 |           0x1000 |
|  566 | audio_encoder.encoders.42.feed_forward.w_2.weight          |       0x10cec740 |         0x200000 |
|  567 | audio_encoder.encoders.42.feed_forward.w_2.bias            |       0x10eec740 |            0x400 |
|  568 | audio_encoder.encoders.42.norm1.weight                     |       0x10eecb40 |            0x400 |
|  569 | audio_encoder.encoders.42.norm1.bias                       |       0x10eecf40 |            0x400 |
|  570 | audio_encoder.encoders.42.norm2.weight                     |       0x10eed340 |            0x400 |
|  571 | audio_encoder.encoders.42.norm2.bias                       |       0x10eed740 |            0x400 |
|  572 | audio_encoder.encoders.43.self_attn.linear_out.weight      |       0x10eedb40 |          0x80000 |
|  573 | audio_encoder.encoders.43.self_attn.linear_out.bias        |       0x10f6db40 |            0x400 |
|  574 | audio_encoder.encoders.43.self_attn.linear_q_k_v.weight    |       0x10f6df40 |         0x180000 |
|  575 | audio_encoder.encoders.43.self_attn.linear_q_k_v.bias      |       0x110edf40 |            0xc00 |
|  576 | audio_encoder.encoders.43.self_attn.fsmn_block.weight      |       0x110eeb40 |           0x2c00 |
|  577 | audio_encoder.encoders.43.feed_forward.w_1.weight          |       0x110f1740 |         0x200000 |
|  578 | audio_encoder.encoders.43.feed_forward.w_1.bias            |       0x112f1740 |           0x1000 |
|  579 | audio_encoder.encoders.43.feed_forward.w_2.weight          |       0x112f2740 |         0x200000 |
|  580 | audio_encoder.encoders.43.feed_forward.w_2.bias            |       0x114f2740 |            0x400 |
|  581 | audio_encoder.encoders.43.norm1.weight                     |       0x114f2b40 |            0x400 |
|  582 | audio_encoder.encoders.43.norm1.bias                       |       0x114f2f40 |            0x400 |
|  583 | audio_encoder.encoders.43.norm2.weight                     |       0x114f3340 |            0x400 |
|  584 | audio_encoder.encoders.43.norm2.bias                       |       0x114f3740 |            0x400 |
|  585 | audio_encoder.encoders.44.self_attn.linear_out.weight      |       0x114f3b40 |          0x80000 |
|  586 | audio_encoder.encoders.44.self_attn.linear_out.bias        |       0x11573b40 |            0x400 |
|  587 | audio_encoder.encoders.44.self_attn.linear_q_k_v.weight    |       0x11573f40 |         0x180000 |
|  588 | audio_encoder.encoders.44.self_attn.linear_q_k_v.bias      |       0x116f3f40 |            0xc00 |
|  589 | audio_encoder.encoders.44.self_attn.fsmn_block.weight      |       0x116f4b40 |           0x2c00 |
|  590 | audio_encoder.encoders.44.feed_forward.w_1.weight          |       0x116f7740 |         0x200000 |
|  591 | audio_encoder.encoders.44.feed_forward.w_1.bias            |       0x118f7740 |           0x1000 |
|  592 | audio_encoder.encoders.44.feed_forward.w_2.weight          |       0x118f8740 |         0x200000 |
|  593 | audio_encoder.encoders.44.feed_forward.w_2.bias            |       0x11af8740 |            0x400 |
|  594 | audio_encoder.encoders.44.norm1.weight                     |       0x11af8b40 |            0x400 |
|  595 | audio_encoder.encoders.44.norm1.bias                       |       0x11af8f40 |            0x400 |
|  596 | audio_encoder.encoders.44.norm2.weight                     |       0x11af9340 |            0x400 |
|  597 | audio_encoder.encoders.44.norm2.bias                       |       0x11af9740 |            0x400 |
|  598 | audio_encoder.encoders.45.self_attn.linear_out.weight      |       0x11af9b40 |          0x80000 |
|  599 | audio_encoder.encoders.45.self_attn.linear_out.bias        |       0x11b79b40 |            0x400 |
|  600 | audio_encoder.encoders.45.self_attn.linear_q_k_v.weight    |       0x11b79f40 |         0x180000 |
|  601 | audio_encoder.encoders.45.self_attn.linear_q_k_v.bias      |       0x11cf9f40 |            0xc00 |
|  602 | audio_encoder.encoders.45.self_attn.fsmn_block.weight      |       0x11cfab40 |           0x2c00 |
|  603 | audio_encoder.encoders.45.feed_forward.w_1.weight          |       0x11cfd740 |         0x200000 |
|  604 | audio_encoder.encoders.45.feed_forward.w_1.bias            |       0x11efd740 |           0x1000 |
|  605 | audio_encoder.encoders.45.feed_forward.w_2.weight          |       0x11efe740 |         0x200000 |
|  606 | audio_encoder.encoders.45.feed_forward.w_2.bias            |       0x120fe740 |            0x400 |
|  607 | audio_encoder.encoders.45.norm1.weight                     |       0x120feb40 |            0x400 |
|  608 | audio_encoder.encoders.45.norm1.bias                       |       0x120fef40 |            0x400 |
|  609 | audio_encoder.encoders.45.norm2.weight                     |       0x120ff340 |            0x400 |
|  610 | audio_encoder.encoders.45.norm2.bias                       |       0x120ff740 |            0x400 |
|  611 | audio_encoder.encoders.46.self_attn.linear_out.weight      |       0x120ffb40 |          0x80000 |
|  612 | audio_encoder.encoders.46.self_attn.linear_out.bias        |       0x1217fb40 |            0x400 |
|  613 | audio_encoder.encoders.46.self_attn.linear_q_k_v.weight    |       0x1217ff40 |         0x180000 |
|  614 | audio_encoder.encoders.46.self_attn.linear_q_k_v.bias      |       0x122fff40 |            0xc00 |
|  615 | audio_encoder.encoders.46.self_attn.fsmn_block.weight      |       0x12300b40 |           0x2c00 |
|  616 | audio_encoder.encoders.46.feed_forward.w_1.weight          |       0x12303740 |         0x200000 |
|  617 | audio_encoder.encoders.46.feed_forward.w_1.bias            |       0x12503740 |           0x1000 |
|  618 | audio_encoder.encoders.46.feed_forward.w_2.weight          |       0x12504740 |         0x200000 |
|  619 | audio_encoder.encoders.46.feed_forward.w_2.bias            |       0x12704740 |            0x400 |
|  620 | audio_encoder.encoders.46.norm1.weight                     |       0x12704b40 |            0x400 |
|  621 | audio_encoder.encoders.46.norm1.bias                       |       0x12704f40 |            0x400 |
|  622 | audio_encoder.encoders.46.norm2.weight                     |       0x12705340 |            0x400 |
|  623 | audio_encoder.encoders.46.norm2.bias                       |       0x12705740 |            0x400 |
|  624 | audio_encoder.encoders.47.self_attn.linear_out.weight      |       0x12705b40 |          0x80000 |
|  625 | audio_encoder.encoders.47.self_attn.linear_out.bias        |       0x12785b40 |            0x400 |
|  626 | audio_encoder.encoders.47.self_attn.linear_q_k_v.weight    |       0x12785f40 |         0x180000 |
|  627 | audio_encoder.encoders.47.self_attn.linear_q_k_v.bias      |       0x12905f40 |            0xc00 |
|  628 | audio_encoder.encoders.47.self_attn.fsmn_block.weight      |       0x12906b40 |           0x2c00 |
|  629 | audio_encoder.encoders.47.feed_forward.w_1.weight          |       0x12909740 |         0x200000 |
|  630 | audio_encoder.encoders.47.feed_forward.w_1.bias            |       0x12b09740 |           0x1000 |
|  631 | audio_encoder.encoders.47.feed_forward.w_2.weight          |       0x12b0a740 |         0x200000 |
|  632 | audio_encoder.encoders.47.feed_forward.w_2.bias            |       0x12d0a740 |            0x400 |
|  633 | audio_encoder.encoders.47.norm1.weight                     |       0x12d0ab40 |            0x400 |
|  634 | audio_encoder.encoders.47.norm1.bias                       |       0x12d0af40 |            0x400 |
|  635 | audio_encoder.encoders.47.norm2.weight                     |       0x12d0b340 |            0x400 |
|  636 | audio_encoder.encoders.47.norm2.bias                       |       0x12d0b740 |            0x400 |
|  637 | audio_encoder.encoders.48.self_attn.linear_out.weight      |       0x12d0bb40 |          0x80000 |
|  638 | audio_encoder.encoders.48.self_attn.linear_out.bias        |       0x12d8bb40 |            0x400 |
|  639 | audio_encoder.encoders.48.self_attn.linear_q_k_v.weight    |       0x12d8bf40 |         0x180000 |
|  640 | audio_encoder.encoders.48.self_attn.linear_q_k_v.bias      |       0x12f0bf40 |            0xc00 |
|  641 | audio_encoder.encoders.48.self_attn.fsmn_block.weight      |       0x12f0cb40 |           0x2c00 |
|  642 | audio_encoder.encoders.48.feed_forward.w_1.weight          |       0x12f0f740 |         0x200000 |
|  643 | audio_encoder.encoders.48.feed_forward.w_1.bias            |       0x1310f740 |           0x1000 |
|  644 | audio_encoder.encoders.48.feed_forward.w_2.weight          |       0x13110740 |         0x200000 |
|  645 | audio_encoder.encoders.48.feed_forward.w_2.bias            |       0x13310740 |            0x400 |
|  646 | audio_encoder.encoders.48.norm1.weight                     |       0x13310b40 |            0x400 |
|  647 | audio_encoder.encoders.48.norm1.bias                       |       0x13310f40 |            0x400 |
|  648 | audio_encoder.encoders.48.norm2.weight                     |       0x13311340 |            0x400 |
|  649 | audio_encoder.encoders.48.norm2.bias                       |       0x13311740 |            0x400 |
|  650 | audio_encoder.tp_encoders.0.self_attn.linear_out.weight    |       0x13311b40 |          0x80000 |
|  651 | audio_encoder.tp_encoders.0.self_attn.linear_out.bias      |       0x13391b40 |            0x400 |
|  652 | audio_encoder.tp_encoders.0.self_attn.linear_q_k_v.weight  |       0x13391f40 |         0x180000 |
|  653 | audio_encoder.tp_encoders.0.self_attn.linear_q_k_v.bias    |       0x13511f40 |            0xc00 |
|  654 | audio_encoder.tp_encoders.0.self_attn.fsmn_block.weight    |       0x13512b40 |           0x2c00 |
|  655 | audio_encoder.tp_encoders.0.feed_forward.w_1.weight        |       0x13515740 |         0x200000 |
|  656 | audio_encoder.tp_encoders.0.feed_forward.w_1.bias          |       0x13715740 |           0x1000 |
|  657 | audio_encoder.tp_encoders.0.feed_forward.w_2.weight        |       0x13716740 |         0x200000 |
|  658 | audio_encoder.tp_encoders.0.feed_forward.w_2.bias          |       0x13916740 |            0x400 |
|  659 | audio_encoder.tp_encoders.0.norm1.weight                   |       0x13916b40 |            0x400 |
|  660 | audio_encoder.tp_encoders.0.norm1.bias                     |       0x13916f40 |            0x400 |
|  661 | audio_encoder.tp_encoders.0.norm2.weight                   |       0x13917340 |            0x400 |
|  662 | audio_encoder.tp_encoders.0.norm2.bias                     |       0x13917740 |            0x400 |
|  663 | audio_encoder.tp_encoders.1.self_attn.linear_out.weight    |       0x13917b40 |          0x80000 |
|  664 | audio_encoder.tp_encoders.1.self_attn.linear_out.bias      |       0x13997b40 |            0x400 |
|  665 | audio_encoder.tp_encoders.1.self_attn.linear_q_k_v.weight  |       0x13997f40 |         0x180000 |
|  666 | audio_encoder.tp_encoders.1.self_attn.linear_q_k_v.bias    |       0x13b17f40 |            0xc00 |
|  667 | audio_encoder.tp_encoders.1.self_attn.fsmn_block.weight    |       0x13b18b40 |           0x2c00 |
|  668 | audio_encoder.tp_encoders.1.feed_forward.w_1.weight        |       0x13b1b740 |         0x200000 |
|  669 | audio_encoder.tp_encoders.1.feed_forward.w_1.bias          |       0x13d1b740 |           0x1000 |
|  670 | audio_encoder.tp_encoders.1.feed_forward.w_2.weight        |       0x13d1c740 |         0x200000 |
|  671 | audio_encoder.tp_encoders.1.feed_forward.w_2.bias          |       0x13f1c740 |            0x400 |
|  672 | audio_encoder.tp_encoders.1.norm1.weight                   |       0x13f1cb40 |            0x400 |
|  673 | audio_encoder.tp_encoders.1.norm1.bias                     |       0x13f1cf40 |            0x400 |
|  674 | audio_encoder.tp_encoders.1.norm2.weight                   |       0x13f1d340 |            0x400 |
|  675 | audio_encoder.tp_encoders.1.norm2.bias                     |       0x13f1d740 |            0x400 |
|  676 | audio_encoder.tp_encoders.2.self_attn.linear_out.weight    |       0x13f1db40 |          0x80000 |
|  677 | audio_encoder.tp_encoders.2.self_attn.linear_out.bias      |       0x13f9db40 |            0x400 |
|  678 | audio_encoder.tp_encoders.2.self_attn.linear_q_k_v.weight  |       0x13f9df40 |         0x180000 |
|  679 | audio_encoder.tp_encoders.2.self_attn.linear_q_k_v.bias    |       0x1411df40 |            0xc00 |
|  680 | audio_encoder.tp_encoders.2.self_attn.fsmn_block.weight    |       0x1411eb40 |           0x2c00 |
|  681 | audio_encoder.tp_encoders.2.feed_forward.w_1.weight        |       0x14121740 |         0x200000 |
|  682 | audio_encoder.tp_encoders.2.feed_forward.w_1.bias          |       0x14321740 |           0x1000 |
|  683 | audio_encoder.tp_encoders.2.feed_forward.w_2.weight        |       0x14322740 |         0x200000 |
|  684 | audio_encoder.tp_encoders.2.feed_forward.w_2.bias          |       0x14522740 |            0x400 |
|  685 | audio_encoder.tp_encoders.2.norm1.weight                   |       0x14522b40 |            0x400 |
|  686 | audio_encoder.tp_encoders.2.norm1.bias                     |       0x14522f40 |            0x400 |
|  687 | audio_encoder.tp_encoders.2.norm2.weight                   |       0x14523340 |            0x400 |
|  688 | audio_encoder.tp_encoders.2.norm2.bias                     |       0x14523740 |            0x400 |
|  689 | audio_encoder.tp_encoders.3.self_attn.linear_out.weight    |       0x14523b40 |          0x80000 |
|  690 | audio_encoder.tp_encoders.3.self_attn.linear_out.bias      |       0x145a3b40 |            0x400 |
|  691 | audio_encoder.tp_encoders.3.self_attn.linear_q_k_v.weight  |       0x145a3f40 |         0x180000 |
|  692 | audio_encoder.tp_encoders.3.self_attn.linear_q_k_v.bias    |       0x14723f40 |            0xc00 |
|  693 | audio_encoder.tp_encoders.3.self_attn.fsmn_block.weight    |       0x14724b40 |           0x2c00 |
|  694 | audio_encoder.tp_encoders.3.feed_forward.w_1.weight        |       0x14727740 |         0x200000 |
|  695 | audio_encoder.tp_encoders.3.feed_forward.w_1.bias          |       0x14927740 |           0x1000 |
|  696 | audio_encoder.tp_encoders.3.feed_forward.w_2.weight        |       0x14928740 |         0x200000 |
|  697 | audio_encoder.tp_encoders.3.feed_forward.w_2.bias          |       0x14b28740 |            0x400 |
|  698 | audio_encoder.tp_encoders.3.norm1.weight                   |       0x14b28b40 |            0x400 |
|  699 | audio_encoder.tp_encoders.3.norm1.bias                     |       0x14b28f40 |            0x400 |
|  700 | audio_encoder.tp_encoders.3.norm2.weight                   |       0x14b29340 |            0x400 |
|  701 | audio_encoder.tp_encoders.3.norm2.bias                     |       0x14b29740 |            0x400 |
|  702 | audio_encoder.tp_encoders.4.self_attn.linear_out.weight    |       0x14b29b40 |          0x80000 |
|  703 | audio_encoder.tp_encoders.4.self_attn.linear_out.bias      |       0x14ba9b40 |            0x400 |
|  704 | audio_encoder.tp_encoders.4.self_attn.linear_q_k_v.weight  |       0x14ba9f40 |         0x180000 |
|  705 | audio_encoder.tp_encoders.4.self_attn.linear_q_k_v.bias    |       0x14d29f40 |            0xc00 |
|  706 | audio_encoder.tp_encoders.4.self_attn.fsmn_block.weight    |       0x14d2ab40 |           0x2c00 |
|  707 | audio_encoder.tp_encoders.4.feed_forward.w_1.weight        |       0x14d2d740 |         0x200000 |
|  708 | audio_encoder.tp_encoders.4.feed_forward.w_1.bias          |       0x14f2d740 |           0x1000 |
|  709 | audio_encoder.tp_encoders.4.feed_forward.w_2.weight        |       0x14f2e740 |         0x200000 |
|  710 | audio_encoder.tp_encoders.4.feed_forward.w_2.bias          |       0x1512e740 |            0x400 |
|  711 | audio_encoder.tp_encoders.4.norm1.weight                   |       0x1512eb40 |            0x400 |
|  712 | audio_encoder.tp_encoders.4.norm1.bias                     |       0x1512ef40 |            0x400 |
|  713 | audio_encoder.tp_encoders.4.norm2.weight                   |       0x1512f340 |            0x400 |
|  714 | audio_encoder.tp_encoders.4.norm2.bias                     |       0x1512f740 |            0x400 |
|  715 | audio_encoder.tp_encoders.5.self_attn.linear_out.weight    |       0x1512fb40 |          0x80000 |
|  716 | audio_encoder.tp_encoders.5.self_attn.linear_out.bias      |       0x151afb40 |            0x400 |
|  717 | audio_encoder.tp_encoders.5.self_attn.linear_q_k_v.weight  |       0x151aff40 |         0x180000 |
|  718 | audio_encoder.tp_encoders.5.self_attn.linear_q_k_v.bias    |       0x1532ff40 |            0xc00 |
|  719 | audio_encoder.tp_encoders.5.self_attn.fsmn_block.weight    |       0x15330b40 |           0x2c00 |
|  720 | audio_encoder.tp_encoders.5.feed_forward.w_1.weight        |       0x15333740 |         0x200000 |
|  721 | audio_encoder.tp_encoders.5.feed_forward.w_1.bias          |       0x15533740 |           0x1000 |
|  722 | audio_encoder.tp_encoders.5.feed_forward.w_2.weight        |       0x15534740 |         0x200000 |
|  723 | audio_encoder.tp_encoders.5.feed_forward.w_2.bias          |       0x15734740 |            0x400 |
|  724 | audio_encoder.tp_encoders.5.norm1.weight                   |       0x15734b40 |            0x400 |
|  725 | audio_encoder.tp_encoders.5.norm1.bias                     |       0x15734f40 |            0x400 |
|  726 | audio_encoder.tp_encoders.5.norm2.weight                   |       0x15735340 |            0x400 |
|  727 | audio_encoder.tp_encoders.5.norm2.bias                     |       0x15735740 |            0x400 |
|  728 | audio_encoder.tp_encoders.6.self_attn.linear_out.weight    |       0x15735b40 |          0x80000 |
|  729 | audio_encoder.tp_encoders.6.self_attn.linear_out.bias      |       0x157b5b40 |            0x400 |
|  730 | audio_encoder.tp_encoders.6.self_attn.linear_q_k_v.weight  |       0x157b5f40 |         0x180000 |
|  731 | audio_encoder.tp_encoders.6.self_attn.linear_q_k_v.bias    |       0x15935f40 |            0xc00 |
|  732 | audio_encoder.tp_encoders.6.self_attn.fsmn_block.weight    |       0x15936b40 |           0x2c00 |
|  733 | audio_encoder.tp_encoders.6.feed_forward.w_1.weight        |       0x15939740 |         0x200000 |
|  734 | audio_encoder.tp_encoders.6.feed_forward.w_1.bias          |       0x15b39740 |           0x1000 |
|  735 | audio_encoder.tp_encoders.6.feed_forward.w_2.weight        |       0x15b3a740 |         0x200000 |
|  736 | audio_encoder.tp_encoders.6.feed_forward.w_2.bias          |       0x15d3a740 |            0x400 |
|  737 | audio_encoder.tp_encoders.6.norm1.weight                   |       0x15d3ab40 |            0x400 |
|  738 | audio_encoder.tp_encoders.6.norm1.bias                     |       0x15d3af40 |            0x400 |
|  739 | audio_encoder.tp_encoders.6.norm2.weight                   |       0x15d3b340 |            0x400 |
|  740 | audio_encoder.tp_encoders.6.norm2.bias                     |       0x15d3b740 |            0x400 |
|  741 | audio_encoder.tp_encoders.7.self_attn.linear_out.weight    |       0x15d3bb40 |          0x80000 |
|  742 | audio_encoder.tp_encoders.7.self_attn.linear_out.bias      |       0x15dbbb40 |            0x400 |
|  743 | audio_encoder.tp_encoders.7.self_attn.linear_q_k_v.weight  |       0x15dbbf40 |         0x180000 |
|  744 | audio_encoder.tp_encoders.7.self_attn.linear_q_k_v.bias    |       0x15f3bf40 |            0xc00 |
|  745 | audio_encoder.tp_encoders.7.self_attn.fsmn_block.weight    |       0x15f3cb40 |           0x2c00 |
|  746 | audio_encoder.tp_encoders.7.feed_forward.w_1.weight        |       0x15f3f740 |         0x200000 |
|  747 | audio_encoder.tp_encoders.7.feed_forward.w_1.bias          |       0x1613f740 |           0x1000 |
|  748 | audio_encoder.tp_encoders.7.feed_forward.w_2.weight        |       0x16140740 |         0x200000 |
|  749 | audio_encoder.tp_encoders.7.feed_forward.w_2.bias          |       0x16340740 |            0x400 |
|  750 | audio_encoder.tp_encoders.7.norm1.weight                   |       0x16340b40 |            0x400 |
|  751 | audio_encoder.tp_encoders.7.norm1.bias                     |       0x16340f40 |            0x400 |
|  752 | audio_encoder.tp_encoders.7.norm2.weight                   |       0x16341340 |            0x400 |
|  753 | audio_encoder.tp_encoders.7.norm2.bias                     |       0x16341740 |            0x400 |
|  754 | audio_encoder.tp_encoders.8.self_attn.linear_out.weight    |       0x16341b40 |          0x80000 |
|  755 | audio_encoder.tp_encoders.8.self_attn.linear_out.bias      |       0x163c1b40 |            0x400 |
|  756 | audio_encoder.tp_encoders.8.self_attn.linear_q_k_v.weight  |       0x163c1f40 |         0x180000 |
|  757 | audio_encoder.tp_encoders.8.self_attn.linear_q_k_v.bias    |       0x16541f40 |            0xc00 |
|  758 | audio_encoder.tp_encoders.8.self_attn.fsmn_block.weight    |       0x16542b40 |           0x2c00 |
|  759 | audio_encoder.tp_encoders.8.feed_forward.w_1.weight        |       0x16545740 |         0x200000 |
|  760 | audio_encoder.tp_encoders.8.feed_forward.w_1.bias          |       0x16745740 |           0x1000 |
|  761 | audio_encoder.tp_encoders.8.feed_forward.w_2.weight        |       0x16746740 |         0x200000 |
|  762 | audio_encoder.tp_encoders.8.feed_forward.w_2.bias          |       0x16946740 |            0x400 |
|  763 | audio_encoder.tp_encoders.8.norm1.weight                   |       0x16946b40 |            0x400 |
|  764 | audio_encoder.tp_encoders.8.norm1.bias                     |       0x16946f40 |            0x400 |
|  765 | audio_encoder.tp_encoders.8.norm2.weight                   |       0x16947340 |            0x400 |
|  766 | audio_encoder.tp_encoders.8.norm2.bias                     |       0x16947740 |            0x400 |
|  767 | audio_encoder.tp_encoders.9.self_attn.linear_out.weight    |       0x16947b40 |          0x80000 |
|  768 | audio_encoder.tp_encoders.9.self_attn.linear_out.bias      |       0x169c7b40 |            0x400 |
|  769 | audio_encoder.tp_encoders.9.self_attn.linear_q_k_v.weight  |       0x169c7f40 |         0x180000 |
|  770 | audio_encoder.tp_encoders.9.self_attn.linear_q_k_v.bias    |       0x16b47f40 |            0xc00 |
|  771 | audio_encoder.tp_encoders.9.self_attn.fsmn_block.weight    |       0x16b48b40 |           0x2c00 |
|  772 | audio_encoder.tp_encoders.9.feed_forward.w_1.weight        |       0x16b4b740 |         0x200000 |
|  773 | audio_encoder.tp_encoders.9.feed_forward.w_1.bias          |       0x16d4b740 |           0x1000 |
|  774 | audio_encoder.tp_encoders.9.feed_forward.w_2.weight        |       0x16d4c740 |         0x200000 |
|  775 | audio_encoder.tp_encoders.9.feed_forward.w_2.bias          |       0x16f4c740 |            0x400 |
|  776 | audio_encoder.tp_encoders.9.norm1.weight                   |       0x16f4cb40 |            0x400 |
|  777 | audio_encoder.tp_encoders.9.norm1.bias                     |       0x16f4cf40 |            0x400 |
|  778 | audio_encoder.tp_encoders.9.norm2.weight                   |       0x16f4d340 |            0x400 |
|  779 | audio_encoder.tp_encoders.9.norm2.bias                     |       0x16f4d740 |            0x400 |
|  780 | audio_encoder.tp_encoders.10.self_attn.linear_out.weight   |       0x16f4db40 |          0x80000 |
|  781 | audio_encoder.tp_encoders.10.self_attn.linear_out.bias     |       0x16fcdb40 |            0x400 |
|  782 | audio_encoder.tp_encoders.10.self_attn.linear_q_k_v.weight |       0x16fcdf40 |         0x180000 |
|  783 | audio_encoder.tp_encoders.10.self_attn.linear_q_k_v.bias   |       0x1714df40 |            0xc00 |
|  784 | audio_encoder.tp_encoders.10.self_attn.fsmn_block.weight   |       0x1714eb40 |           0x2c00 |
|  785 | audio_encoder.tp_encoders.10.feed_forward.w_1.weight       |       0x17151740 |         0x200000 |
|  786 | audio_encoder.tp_encoders.10.feed_forward.w_1.bias         |       0x17351740 |           0x1000 |
|  787 | audio_encoder.tp_encoders.10.feed_forward.w_2.weight       |       0x17352740 |         0x200000 |
|  788 | audio_encoder.tp_encoders.10.feed_forward.w_2.bias         |       0x17552740 |            0x400 |
|  789 | audio_encoder.tp_encoders.10.norm1.weight                  |       0x17552b40 |            0x400 |
|  790 | audio_encoder.tp_encoders.10.norm1.bias                    |       0x17552f40 |            0x400 |
|  791 | audio_encoder.tp_encoders.10.norm2.weight                  |       0x17553340 |            0x400 |
|  792 | audio_encoder.tp_encoders.10.norm2.bias                    |       0x17553740 |            0x400 |
|  793 | audio_encoder.tp_encoders.11.self_attn.linear_out.weight   |       0x17553b40 |          0x80000 |
|  794 | audio_encoder.tp_encoders.11.self_attn.linear_out.bias     |       0x175d3b40 |            0x400 |
|  795 | audio_encoder.tp_encoders.11.self_attn.linear_q_k_v.weight |       0x175d3f40 |         0x180000 |
|  796 | audio_encoder.tp_encoders.11.self_attn.linear_q_k_v.bias   |       0x17753f40 |            0xc00 |
|  797 | audio_encoder.tp_encoders.11.self_attn.fsmn_block.weight   |       0x17754b40 |           0x2c00 |
|  798 | audio_encoder.tp_encoders.11.feed_forward.w_1.weight       |       0x17757740 |         0x200000 |
|  799 | audio_encoder.tp_encoders.11.feed_forward.w_1.bias         |       0x17957740 |           0x1000 |
|  800 | audio_encoder.tp_encoders.11.feed_forward.w_2.weight       |       0x17958740 |         0x200000 |
|  801 | audio_encoder.tp_encoders.11.feed_forward.w_2.bias         |       0x17b58740 |            0x400 |
|  802 | audio_encoder.tp_encoders.11.norm1.weight                  |       0x17b58b40 |            0x400 |
|  803 | audio_encoder.tp_encoders.11.norm1.bias                    |       0x17b58f40 |            0x400 |
|  804 | audio_encoder.tp_encoders.11.norm2.weight                  |       0x17b59340 |            0x400 |
|  805 | audio_encoder.tp_encoders.11.norm2.bias                    |       0x17b59740 |            0x400 |
|  806 | audio_encoder.tp_encoders.12.self_attn.linear_out.weight   |       0x17b59b40 |          0x80000 |
|  807 | audio_encoder.tp_encoders.12.self_attn.linear_out.bias     |       0x17bd9b40 |            0x400 |
|  808 | audio_encoder.tp_encoders.12.self_attn.linear_q_k_v.weight |       0x17bd9f40 |         0x180000 |
|  809 | audio_encoder.tp_encoders.12.self_attn.linear_q_k_v.bias   |       0x17d59f40 |            0xc00 |
|  810 | audio_encoder.tp_encoders.12.self_attn.fsmn_block.weight   |       0x17d5ab40 |           0x2c00 |
|  811 | audio_encoder.tp_encoders.12.feed_forward.w_1.weight       |       0x17d5d740 |         0x200000 |
|  812 | audio_encoder.tp_encoders.12.feed_forward.w_1.bias         |       0x17f5d740 |           0x1000 |
|  813 | audio_encoder.tp_encoders.12.feed_forward.w_2.weight       |       0x17f5e740 |         0x200000 |
|  814 | audio_encoder.tp_encoders.12.feed_forward.w_2.bias         |       0x1815e740 |            0x400 |
|  815 | audio_encoder.tp_encoders.12.norm1.weight                  |       0x1815eb40 |            0x400 |
|  816 | audio_encoder.tp_encoders.12.norm1.bias                    |       0x1815ef40 |            0x400 |
|  817 | audio_encoder.tp_encoders.12.norm2.weight                  |       0x1815f340 |            0x400 |
|  818 | audio_encoder.tp_encoders.12.norm2.bias                    |       0x1815f740 |            0x400 |
|  819 | audio_encoder.tp_encoders.13.self_attn.linear_out.weight   |       0x1815fb40 |          0x80000 |
|  820 | audio_encoder.tp_encoders.13.self_attn.linear_out.bias     |       0x181dfb40 |            0x400 |
|  821 | audio_encoder.tp_encoders.13.self_attn.linear_q_k_v.weight |       0x181dff40 |         0x180000 |
|  822 | audio_encoder.tp_encoders.13.self_attn.linear_q_k_v.bias   |       0x1835ff40 |            0xc00 |
|  823 | audio_encoder.tp_encoders.13.self_attn.fsmn_block.weight   |       0x18360b40 |           0x2c00 |
|  824 | audio_encoder.tp_encoders.13.feed_forward.w_1.weight       |       0x18363740 |         0x200000 |
|  825 | audio_encoder.tp_encoders.13.feed_forward.w_1.bias         |       0x18563740 |           0x1000 |
|  826 | audio_encoder.tp_encoders.13.feed_forward.w_2.weight       |       0x18564740 |         0x200000 |
|  827 | audio_encoder.tp_encoders.13.feed_forward.w_2.bias         |       0x18764740 |            0x400 |
|  828 | audio_encoder.tp_encoders.13.norm1.weight                  |       0x18764b40 |            0x400 |
|  829 | audio_encoder.tp_encoders.13.norm1.bias                    |       0x18764f40 |            0x400 |
|  830 | audio_encoder.tp_encoders.13.norm2.weight                  |       0x18765340 |            0x400 |
|  831 | audio_encoder.tp_encoders.13.norm2.bias                    |       0x18765740 |            0x400 |
|  832 | audio_encoder.tp_encoders.14.self_attn.linear_out.weight   |       0x18765b40 |          0x80000 |
|  833 | audio_encoder.tp_encoders.14.self_attn.linear_out.bias     |       0x187e5b40 |            0x400 |
|  834 | audio_encoder.tp_encoders.14.self_attn.linear_q_k_v.weight |       0x187e5f40 |         0x180000 |
|  835 | audio_encoder.tp_encoders.14.self_attn.linear_q_k_v.bias   |       0x18965f40 |            0xc00 |
|  836 | audio_encoder.tp_encoders.14.self_attn.fsmn_block.weight   |       0x18966b40 |           0x2c00 |
|  837 | audio_encoder.tp_encoders.14.feed_forward.w_1.weight       |       0x18969740 |         0x200000 |
|  838 | audio_encoder.tp_encoders.14.feed_forward.w_1.bias         |       0x18b69740 |           0x1000 |
|  839 | audio_encoder.tp_encoders.14.feed_forward.w_2.weight       |       0x18b6a740 |         0x200000 |
|  840 | audio_encoder.tp_encoders.14.feed_forward.w_2.bias         |       0x18d6a740 |            0x400 |
|  841 | audio_encoder.tp_encoders.14.norm1.weight                  |       0x18d6ab40 |            0x400 |
|  842 | audio_encoder.tp_encoders.14.norm1.bias                    |       0x18d6af40 |            0x400 |
|  843 | audio_encoder.tp_encoders.14.norm2.weight                  |       0x18d6b340 |            0x400 |
|  844 | audio_encoder.tp_encoders.14.norm2.bias                    |       0x18d6b740 |            0x400 |
|  845 | audio_encoder.tp_encoders.15.self_attn.linear_out.weight   |       0x18d6bb40 |          0x80000 |
|  846 | audio_encoder.tp_encoders.15.self_attn.linear_out.bias     |       0x18debb40 |            0x400 |
|  847 | audio_encoder.tp_encoders.15.self_attn.linear_q_k_v.weight |       0x18debf40 |         0x180000 |
|  848 | audio_encoder.tp_encoders.15.self_attn.linear_q_k_v.bias   |       0x18f6bf40 |            0xc00 |
|  849 | audio_encoder.tp_encoders.15.self_attn.fsmn_block.weight   |       0x18f6cb40 |           0x2c00 |
|  850 | audio_encoder.tp_encoders.15.feed_forward.w_1.weight       |       0x18f6f740 |         0x200000 |
|  851 | audio_encoder.tp_encoders.15.feed_forward.w_1.bias         |       0x1916f740 |           0x1000 |
|  852 | audio_encoder.tp_encoders.15.feed_forward.w_2.weight       |       0x19170740 |         0x200000 |
|  853 | audio_encoder.tp_encoders.15.feed_forward.w_2.bias         |       0x19370740 |            0x400 |
|  854 | audio_encoder.tp_encoders.15.norm1.weight                  |       0x19370b40 |            0x400 |
|  855 | audio_encoder.tp_encoders.15.norm1.bias                    |       0x19370f40 |            0x400 |
|  856 | audio_encoder.tp_encoders.15.norm2.weight                  |       0x19371340 |            0x400 |
|  857 | audio_encoder.tp_encoders.15.norm2.bias                    |       0x19371740 |            0x400 |
|  858 | audio_encoder.tp_encoders.16.self_attn.linear_out.weight   |       0x19371b40 |          0x80000 |
|  859 | audio_encoder.tp_encoders.16.self_attn.linear_out.bias     |       0x193f1b40 |            0x400 |
|  860 | audio_encoder.tp_encoders.16.self_attn.linear_q_k_v.weight |       0x193f1f40 |         0x180000 |
|  861 | audio_encoder.tp_encoders.16.self_attn.linear_q_k_v.bias   |       0x19571f40 |            0xc00 |
|  862 | audio_encoder.tp_encoders.16.self_attn.fsmn_block.weight   |       0x19572b40 |           0x2c00 |
|  863 | audio_encoder.tp_encoders.16.feed_forward.w_1.weight       |       0x19575740 |         0x200000 |
|  864 | audio_encoder.tp_encoders.16.feed_forward.w_1.bias         |       0x19775740 |           0x1000 |
|  865 | audio_encoder.tp_encoders.16.feed_forward.w_2.weight       |       0x19776740 |         0x200000 |
|  866 | audio_encoder.tp_encoders.16.feed_forward.w_2.bias         |       0x19976740 |            0x400 |
|  867 | audio_encoder.tp_encoders.16.norm1.weight                  |       0x19976b40 |            0x400 |
|  868 | audio_encoder.tp_encoders.16.norm1.bias                    |       0x19976f40 |            0x400 |
|  869 | audio_encoder.tp_encoders.16.norm2.weight                  |       0x19977340 |            0x400 |
|  870 | audio_encoder.tp_encoders.16.norm2.bias                    |       0x19977740 |            0x400 |
|  871 | audio_encoder.tp_encoders.17.self_attn.linear_out.weight   |       0x19977b40 |          0x80000 |
|  872 | audio_encoder.tp_encoders.17.self_attn.linear_out.bias     |       0x199f7b40 |            0x400 |
|  873 | audio_encoder.tp_encoders.17.self_attn.linear_q_k_v.weight |       0x199f7f40 |         0x180000 |
|  874 | audio_encoder.tp_encoders.17.self_attn.linear_q_k_v.bias   |       0x19b77f40 |            0xc00 |
|  875 | audio_encoder.tp_encoders.17.self_attn.fsmn_block.weight   |       0x19b78b40 |           0x2c00 |
|  876 | audio_encoder.tp_encoders.17.feed_forward.w_1.weight       |       0x19b7b740 |         0x200000 |
|  877 | audio_encoder.tp_encoders.17.feed_forward.w_1.bias         |       0x19d7b740 |           0x1000 |
|  878 | audio_encoder.tp_encoders.17.feed_forward.w_2.weight       |       0x19d7c740 |         0x200000 |
|  879 | audio_encoder.tp_encoders.17.feed_forward.w_2.bias         |       0x19f7c740 |            0x400 |
|  880 | audio_encoder.tp_encoders.17.norm1.weight                  |       0x19f7cb40 |            0x400 |
|  881 | audio_encoder.tp_encoders.17.norm1.bias                    |       0x19f7cf40 |            0x400 |
|  882 | audio_encoder.tp_encoders.17.norm2.weight                  |       0x19f7d340 |            0x400 |
|  883 | audio_encoder.tp_encoders.17.norm2.bias                    |       0x19f7d740 |            0x400 |
|  884 | audio_encoder.tp_encoders.18.self_attn.linear_out.weight   |       0x19f7db40 |          0x80000 |
|  885 | audio_encoder.tp_encoders.18.self_attn.linear_out.bias     |       0x19ffdb40 |            0x400 |
|  886 | audio_encoder.tp_encoders.18.self_attn.linear_q_k_v.weight |       0x19ffdf40 |         0x180000 |
|  887 | audio_encoder.tp_encoders.18.self_attn.linear_q_k_v.bias   |       0x1a17df40 |            0xc00 |
|  888 | audio_encoder.tp_encoders.18.self_attn.fsmn_block.weight   |       0x1a17eb40 |           0x2c00 |
|  889 | audio_encoder.tp_encoders.18.feed_forward.w_1.weight       |       0x1a181740 |         0x200000 |
|  890 | audio_encoder.tp_encoders.18.feed_forward.w_1.bias         |       0x1a381740 |           0x1000 |
|  891 | audio_encoder.tp_encoders.18.feed_forward.w_2.weight       |       0x1a382740 |         0x200000 |
|  892 | audio_encoder.tp_encoders.18.feed_forward.w_2.bias         |       0x1a582740 |            0x400 |
|  893 | audio_encoder.tp_encoders.18.norm1.weight                  |       0x1a582b40 |            0x400 |
|  894 | audio_encoder.tp_encoders.18.norm1.bias                    |       0x1a582f40 |            0x400 |
|  895 | audio_encoder.tp_encoders.18.norm2.weight                  |       0x1a583340 |            0x400 |
|  896 | audio_encoder.tp_encoders.18.norm2.bias                    |       0x1a583740 |            0x400 |
|  897 | audio_encoder.tp_encoders.19.self_attn.linear_out.weight   |       0x1a583b40 |          0x80000 |
|  898 | audio_encoder.tp_encoders.19.self_attn.linear_out.bias     |       0x1a603b40 |            0x400 |
|  899 | audio_encoder.tp_encoders.19.self_attn.linear_q_k_v.weight |       0x1a603f40 |         0x180000 |
|  900 | audio_encoder.tp_encoders.19.self_attn.linear_q_k_v.bias   |       0x1a783f40 |            0xc00 |
|  901 | audio_encoder.tp_encoders.19.self_attn.fsmn_block.weight   |       0x1a784b40 |           0x2c00 |
|  902 | audio_encoder.tp_encoders.19.feed_forward.w_1.weight       |       0x1a787740 |         0x200000 |
|  903 | audio_encoder.tp_encoders.19.feed_forward.w_1.bias         |       0x1a987740 |           0x1000 |
|  904 | audio_encoder.tp_encoders.19.feed_forward.w_2.weight       |       0x1a988740 |         0x200000 |
|  905 | audio_encoder.tp_encoders.19.feed_forward.w_2.bias         |       0x1ab88740 |            0x400 |
|  906 | audio_encoder.tp_encoders.19.norm1.weight                  |       0x1ab88b40 |            0x400 |
|  907 | audio_encoder.tp_encoders.19.norm1.bias                    |       0x1ab88f40 |            0x400 |
|  908 | audio_encoder.tp_encoders.19.norm2.weight                  |       0x1ab89340 |            0x400 |
|  909 | audio_encoder.tp_encoders.19.norm2.bias                    |       0x1ab89740 |            0x400 |
|  910 | audio_encoder.after_norm.weight                            |       0x1ab89b40 |            0x400 |
|  911 | audio_encoder.after_norm.bias                              |       0x1ab89f40 |            0x400 |
|  912 | audio_encoder.tp_norm.weight                               |       0x1ab8a340 |            0x400 |
|  913 | audio_encoder.tp_norm.bias                                 |       0x1ab8a740 |            0x400 |
|  914 | audio_adaptor.linear1.weight                               |       0x1ab8ab40 |         0x200000 |
|  915 | audio_adaptor.linear1.bias                                 |       0x1ad8ab40 |           0x1000 |
|  916 | audio_adaptor.linear2.weight                               |       0x1ad8bb40 |         0x400000 |
|  917 | audio_adaptor.linear2.bias                                 |       0x1b18bb40 |            0x800 |
|  918 | audio_adaptor.blocks.0.self_attn.linear_q.weight           |       0x1b18c340 |         0x200000 |
|  919 | audio_adaptor.blocks.0.self_attn.linear_q.bias             |       0x1b38c340 |            0x800 |
|  920 | audio_adaptor.blocks.0.self_attn.linear_k.weight           |       0x1b38cb40 |         0x200000 |
|  921 | audio_adaptor.blocks.0.self_attn.linear_k.bias             |       0x1b58cb40 |            0x800 |
|  922 | audio_adaptor.blocks.0.self_attn.linear_v.weight           |       0x1b58d340 |         0x200000 |
|  923 | audio_adaptor.blocks.0.self_attn.linear_v.bias             |       0x1b78d340 |            0x800 |
|  924 | audio_adaptor.blocks.0.self_attn.linear_out.weight         |       0x1b78db40 |         0x200000 |
|  925 | audio_adaptor.blocks.0.self_attn.linear_out.bias           |       0x1b98db40 |            0x800 |
|  926 | audio_adaptor.blocks.0.feed_forward.w_1.weight             |       0x1b98e340 |          0x80000 |
|  927 | audio_adaptor.blocks.0.feed_forward.w_1.bias               |       0x1ba0e340 |            0x200 |
|  928 | audio_adaptor.blocks.0.feed_forward.w_2.weight             |       0x1ba0e540 |          0x80000 |
|  929 | audio_adaptor.blocks.0.feed_forward.w_2.bias               |       0x1ba8e540 |            0x800 |
|  930 | audio_adaptor.blocks.0.norm1.weight                        |       0x1ba8ed40 |            0x800 |
|  931 | audio_adaptor.blocks.0.norm1.bias                          |       0x1ba8f540 |            0x800 |
|  932 | audio_adaptor.blocks.0.norm2.weight                        |       0x1ba8fd40 |            0x800 |
|  933 | audio_adaptor.blocks.0.norm2.bias                          |       0x1ba90540 |            0x800 |
|  934 | audio_adaptor.blocks.1.self_attn.linear_q.weight           |       0x1ba90d40 |         0x200000 |
|  935 | audio_adaptor.blocks.1.self_attn.linear_q.bias             |       0x1bc90d40 |            0x800 |
|  936 | audio_adaptor.blocks.1.self_attn.linear_k.weight           |       0x1bc91540 |         0x200000 |
|  937 | audio_adaptor.blocks.1.self_attn.linear_k.bias             |       0x1be91540 |            0x800 |
|  938 | audio_adaptor.blocks.1.self_attn.linear_v.weight           |       0x1be91d40 |         0x200000 |
|  939 | audio_adaptor.blocks.1.self_attn.linear_v.bias             |       0x1c091d40 |            0x800 |
|  940 | audio_adaptor.blocks.1.self_attn.linear_out.weight         |       0x1c092540 |         0x200000 |
|  941 | audio_adaptor.blocks.1.self_attn.linear_out.bias           |       0x1c292540 |            0x800 |
|  942 | audio_adaptor.blocks.1.feed_forward.w_1.weight             |       0x1c292d40 |          0x80000 |
|  943 | audio_adaptor.blocks.1.feed_forward.w_1.bias               |       0x1c312d40 |            0x200 |
|  944 | audio_adaptor.blocks.1.feed_forward.w_2.weight             |       0x1c312f40 |          0x80000 |
|  945 | audio_adaptor.blocks.1.feed_forward.w_2.bias               |       0x1c392f40 |            0x800 |
|  946 | audio_adaptor.blocks.1.norm1.weight                        |       0x1c393740 |            0x800 |
|  947 | audio_adaptor.blocks.1.norm1.bias                          |       0x1c393f40 |            0x800 |
|  948 | audio_adaptor.blocks.1.norm2.weight                        |       0x1c394740 |            0x800 |
|  949 | audio_adaptor.blocks.1.norm2.bias                          |       0x1c394f40 |            0x800 |
|  950 | llm.lm_head.weight                                         |       0x1c395740 |       0x128c0000 |
|  951 | llm.model.embed_tokens.weight                              |       0x2ec55740 |       0x128c0000 |
|  952 | llm.model.layers.0.input_layernorm.weight                  |       0x41515740 |            0x800 |
|  953 | llm.model.layers.0.mlp.down_proj.weight                    |       0x41515f40 |         0x600000 |
|  954 | llm.model.layers.0.mlp.gate_proj.weight                    |       0x41b15f40 |         0x600000 |
|  955 | llm.model.layers.0.mlp.up_proj.weight                      |       0x42115f40 |         0x600000 |
|  956 | llm.model.layers.0.post_attention_layernorm.weight         |       0x42715f40 |            0x800 |
|  957 | llm.model.layers.0.self_attn.k_norm.weight                 |       0x42716740 |            0x100 |
|  958 | llm.model.layers.0.self_attn.k_proj.weight                 |       0x42716840 |         0x200000 |
|  959 | llm.model.layers.0.self_attn.o_proj.weight                 |       0x42916840 |         0x400000 |
|  960 | llm.model.layers.0.self_attn.q_norm.weight                 |       0x42d16840 |            0x100 |
|  961 | llm.model.layers.0.self_attn.q_proj.weight                 |       0x42d16940 |         0x400000 |
|  962 | llm.model.layers.0.self_attn.v_proj.weight                 |       0x43116940 |         0x200000 |
|  963 | llm.model.layers.1.input_layernorm.weight                  |       0x43316940 |            0x800 |
|  964 | llm.model.layers.1.mlp.down_proj.weight                    |       0x43317140 |         0x600000 |
|  965 | llm.model.layers.1.mlp.gate_proj.weight                    |       0x43917140 |         0x600000 |
|  966 | llm.model.layers.1.mlp.up_proj.weight                      |       0x43f17140 |         0x600000 |
|  967 | llm.model.layers.1.post_attention_layernorm.weight         |       0x44517140 |            0x800 |
|  968 | llm.model.layers.1.self_attn.k_norm.weight                 |       0x44517940 |            0x100 |
|  969 | llm.model.layers.1.self_attn.k_proj.weight                 |       0x44517a40 |         0x200000 |
|  970 | llm.model.layers.1.self_attn.o_proj.weight                 |       0x44717a40 |         0x400000 |
|  971 | llm.model.layers.1.self_attn.q_norm.weight                 |       0x44b17a40 |            0x100 |
|  972 | llm.model.layers.1.self_attn.q_proj.weight                 |       0x44b17b40 |         0x400000 |
|  973 | llm.model.layers.1.self_attn.v_proj.weight                 |       0x44f17b40 |         0x200000 |
|  974 | llm.model.layers.10.input_layernorm.weight                 |       0x45117b40 |            0x800 |
|  975 | llm.model.layers.10.mlp.down_proj.weight                   |       0x45118340 |         0x600000 |
|  976 | llm.model.layers.10.mlp.gate_proj.weight                   |       0x45718340 |         0x600000 |
|  977 | llm.model.layers.10.mlp.up_proj.weight                     |       0x45d18340 |         0x600000 |
|  978 | llm.model.layers.10.post_attention_layernorm.weight        |       0x46318340 |            0x800 |
|  979 | llm.model.layers.10.self_attn.k_norm.weight                |       0x46318b40 |            0x100 |
|  980 | llm.model.layers.10.self_attn.k_proj.weight                |       0x46318c40 |         0x200000 |
|  981 | llm.model.layers.10.self_attn.o_proj.weight                |       0x46518c40 |         0x400000 |
|  982 | llm.model.layers.10.self_attn.q_norm.weight                |       0x46918c40 |            0x100 |
|  983 | llm.model.layers.10.self_attn.q_proj.weight                |       0x46918d40 |         0x400000 |
|  984 | llm.model.layers.10.self_attn.v_proj.weight                |       0x46d18d40 |         0x200000 |
|  985 | llm.model.layers.11.input_layernorm.weight                 |       0x46f18d40 |            0x800 |
|  986 | llm.model.layers.11.mlp.down_proj.weight                   |       0x46f19540 |         0x600000 |
|  987 | llm.model.layers.11.mlp.gate_proj.weight                   |       0x47519540 |         0x600000 |
|  988 | llm.model.layers.11.mlp.up_proj.weight                     |       0x47b19540 |         0x600000 |
|  989 | llm.model.layers.11.post_attention_layernorm.weight        |       0x48119540 |            0x800 |
|  990 | llm.model.layers.11.self_attn.k_norm.weight                |       0x48119d40 |            0x100 |
|  991 | llm.model.layers.11.self_attn.k_proj.weight                |       0x48119e40 |         0x200000 |
|  992 | llm.model.layers.11.self_attn.o_proj.weight                |       0x48319e40 |         0x400000 |
|  993 | llm.model.layers.11.self_attn.q_norm.weight                |       0x48719e40 |            0x100 |
|  994 | llm.model.layers.11.self_attn.q_proj.weight                |       0x48719f40 |         0x400000 |
|  995 | llm.model.layers.11.self_attn.v_proj.weight                |       0x48b19f40 |         0x200000 |
|  996 | llm.model.layers.12.input_layernorm.weight                 |       0x48d19f40 |            0x800 |
|  997 | llm.model.layers.12.mlp.down_proj.weight                   |       0x48d1a740 |         0x600000 |
|  998 | llm.model.layers.12.mlp.gate_proj.weight                   |       0x4931a740 |         0x600000 |
|  999 | llm.model.layers.12.mlp.up_proj.weight                     |       0x4991a740 |         0x600000 |
| 1000 | llm.model.layers.12.post_attention_layernorm.weight        |       0x49f1a740 |            0x800 |
| 1001 | llm.model.layers.12.self_attn.k_norm.weight                |       0x49f1af40 |            0x100 |
| 1002 | llm.model.layers.12.self_attn.k_proj.weight                |       0x49f1b040 |         0x200000 |
| 1003 | llm.model.layers.12.self_attn.o_proj.weight                |       0x4a11b040 |         0x400000 |
| 1004 | llm.model.layers.12.self_attn.q_norm.weight                |       0x4a51b040 |            0x100 |
| 1005 | llm.model.layers.12.self_attn.q_proj.weight                |       0x4a51b140 |         0x400000 |
| 1006 | llm.model.layers.12.self_attn.v_proj.weight                |       0x4a91b140 |         0x200000 |
| 1007 | llm.model.layers.13.input_layernorm.weight                 |       0x4ab1b140 |            0x800 |
| 1008 | llm.model.layers.13.mlp.down_proj.weight                   |       0x4ab1b940 |         0x600000 |
| 1009 | llm.model.layers.13.mlp.gate_proj.weight                   |       0x4b11b940 |         0x600000 |
| 1010 | llm.model.layers.13.mlp.up_proj.weight                     |       0x4b71b940 |         0x600000 |
| 1011 | llm.model.layers.13.post_attention_layernorm.weight        |       0x4bd1b940 |            0x800 |
| 1012 | llm.model.layers.13.self_attn.k_norm.weight                |       0x4bd1c140 |            0x100 |
| 1013 | llm.model.layers.13.self_attn.k_proj.weight                |       0x4bd1c240 |         0x200000 |
| 1014 | llm.model.layers.13.self_attn.o_proj.weight                |       0x4bf1c240 |         0x400000 |
| 1015 | llm.model.layers.13.self_attn.q_norm.weight                |       0x4c31c240 |            0x100 |
| 1016 | llm.model.layers.13.self_attn.q_proj.weight                |       0x4c31c340 |         0x400000 |
| 1017 | llm.model.layers.13.self_attn.v_proj.weight                |       0x4c71c340 |         0x200000 |
| 1018 | llm.model.layers.14.input_layernorm.weight                 |       0x4c91c340 |            0x800 |
| 1019 | llm.model.layers.14.mlp.down_proj.weight                   |       0x4c91cb40 |         0x600000 |
| 1020 | llm.model.layers.14.mlp.gate_proj.weight                   |       0x4cf1cb40 |         0x600000 |
| 1021 | llm.model.layers.14.mlp.up_proj.weight                     |       0x4d51cb40 |         0x600000 |
| 1022 | llm.model.layers.14.post_attention_layernorm.weight        |       0x4db1cb40 |            0x800 |
| 1023 | llm.model.layers.14.self_attn.k_norm.weight                |       0x4db1d340 |            0x100 |
| 1024 | llm.model.layers.14.self_attn.k_proj.weight                |       0x4db1d440 |         0x200000 |
| 1025 | llm.model.layers.14.self_attn.o_proj.weight                |       0x4dd1d440 |         0x400000 |
| 1026 | llm.model.layers.14.self_attn.q_norm.weight                |       0x4e11d440 |            0x100 |
| 1027 | llm.model.layers.14.self_attn.q_proj.weight                |       0x4e11d540 |         0x400000 |
| 1028 | llm.model.layers.14.self_attn.v_proj.weight                |       0x4e51d540 |         0x200000 |
| 1029 | llm.model.layers.15.input_layernorm.weight                 |       0x4e71d540 |            0x800 |
| 1030 | llm.model.layers.15.mlp.down_proj.weight                   |       0x4e71dd40 |         0x600000 |
| 1031 | llm.model.layers.15.mlp.gate_proj.weight                   |       0x4ed1dd40 |         0x600000 |
| 1032 | llm.model.layers.15.mlp.up_proj.weight                     |       0x4f31dd40 |         0x600000 |
| 1033 | llm.model.layers.15.post_attention_layernorm.weight        |       0x4f91dd40 |            0x800 |
| 1034 | llm.model.layers.15.self_attn.k_norm.weight                |       0x4f91e540 |            0x100 |
| 1035 | llm.model.layers.15.self_attn.k_proj.weight                |       0x4f91e640 |         0x200000 |
| 1036 | llm.model.layers.15.self_attn.o_proj.weight                |       0x4fb1e640 |         0x400000 |
| 1037 | llm.model.layers.15.self_attn.q_norm.weight                |       0x4ff1e640 |            0x100 |
| 1038 | llm.model.layers.15.self_attn.q_proj.weight                |       0x4ff1e740 |         0x400000 |
| 1039 | llm.model.layers.15.self_attn.v_proj.weight                |       0x5031e740 |         0x200000 |
| 1040 | llm.model.layers.16.input_layernorm.weight                 |       0x5051e740 |            0x800 |
| 1041 | llm.model.layers.16.mlp.down_proj.weight                   |       0x5051ef40 |         0x600000 |
| 1042 | llm.model.layers.16.mlp.gate_proj.weight                   |       0x50b1ef40 |         0x600000 |
| 1043 | llm.model.layers.16.mlp.up_proj.weight                     |       0x5111ef40 |         0x600000 |
| 1044 | llm.model.layers.16.post_attention_layernorm.weight        |       0x5171ef40 |            0x800 |
| 1045 | llm.model.layers.16.self_attn.k_norm.weight                |       0x5171f740 |            0x100 |
| 1046 | llm.model.layers.16.self_attn.k_proj.weight                |       0x5171f840 |         0x200000 |
| 1047 | llm.model.layers.16.self_attn.o_proj.weight                |       0x5191f840 |         0x400000 |
| 1048 | llm.model.layers.16.self_attn.q_norm.weight                |       0x51d1f840 |            0x100 |
| 1049 | llm.model.layers.16.self_attn.q_proj.weight                |       0x51d1f940 |         0x400000 |
| 1050 | llm.model.layers.16.self_attn.v_proj.weight                |       0x5211f940 |         0x200000 |
| 1051 | llm.model.layers.17.input_layernorm.weight                 |       0x5231f940 |            0x800 |
| 1052 | llm.model.layers.17.mlp.down_proj.weight                   |       0x52320140 |         0x600000 |
| 1053 | llm.model.layers.17.mlp.gate_proj.weight                   |       0x52920140 |         0x600000 |
| 1054 | llm.model.layers.17.mlp.up_proj.weight                     |       0x52f20140 |         0x600000 |
| 1055 | llm.model.layers.17.post_attention_layernorm.weight        |       0x53520140 |            0x800 |
| 1056 | llm.model.layers.17.self_attn.k_norm.weight                |       0x53520940 |            0x100 |
| 1057 | llm.model.layers.17.self_attn.k_proj.weight                |       0x53520a40 |         0x200000 |
| 1058 | llm.model.layers.17.self_attn.o_proj.weight                |       0x53720a40 |         0x400000 |
| 1059 | llm.model.layers.17.self_attn.q_norm.weight                |       0x53b20a40 |            0x100 |
| 1060 | llm.model.layers.17.self_attn.q_proj.weight                |       0x53b20b40 |         0x400000 |
| 1061 | llm.model.layers.17.self_attn.v_proj.weight                |       0x53f20b40 |         0x200000 |
| 1062 | llm.model.layers.18.input_layernorm.weight                 |       0x54120b40 |            0x800 |
| 1063 | llm.model.layers.18.mlp.down_proj.weight                   |       0x54121340 |         0x600000 |
| 1064 | llm.model.layers.18.mlp.gate_proj.weight                   |       0x54721340 |         0x600000 |
| 1065 | llm.model.layers.18.mlp.up_proj.weight                     |       0x54d21340 |         0x600000 |
| 1066 | llm.model.layers.18.post_attention_layernorm.weight        |       0x55321340 |            0x800 |
| 1067 | llm.model.layers.18.self_attn.k_norm.weight                |       0x55321b40 |            0x100 |
| 1068 | llm.model.layers.18.self_attn.k_proj.weight                |       0x55321c40 |         0x200000 |
| 1069 | llm.model.layers.18.self_attn.o_proj.weight                |       0x55521c40 |         0x400000 |
| 1070 | llm.model.layers.18.self_attn.q_norm.weight                |       0x55921c40 |            0x100 |
| 1071 | llm.model.layers.18.self_attn.q_proj.weight                |       0x55921d40 |         0x400000 |
| 1072 | llm.model.layers.18.self_attn.v_proj.weight                |       0x55d21d40 |         0x200000 |
| 1073 | llm.model.layers.19.input_layernorm.weight                 |       0x55f21d40 |            0x800 |
| 1074 | llm.model.layers.19.mlp.down_proj.weight                   |       0x55f22540 |         0x600000 |
| 1075 | llm.model.layers.19.mlp.gate_proj.weight                   |       0x56522540 |         0x600000 |
| 1076 | llm.model.layers.19.mlp.up_proj.weight                     |       0x56b22540 |         0x600000 |
| 1077 | llm.model.layers.19.post_attention_layernorm.weight        |       0x57122540 |            0x800 |
| 1078 | llm.model.layers.19.self_attn.k_norm.weight                |       0x57122d40 |            0x100 |
| 1079 | llm.model.layers.19.self_attn.k_proj.weight                |       0x57122e40 |         0x200000 |
| 1080 | llm.model.layers.19.self_attn.o_proj.weight                |       0x57322e40 |         0x400000 |
| 1081 | llm.model.layers.19.self_attn.q_norm.weight                |       0x57722e40 |            0x100 |
| 1082 | llm.model.layers.19.self_attn.q_proj.weight                |       0x57722f40 |         0x400000 |
| 1083 | llm.model.layers.19.self_attn.v_proj.weight                |       0x57b22f40 |         0x200000 |
| 1084 | llm.model.layers.2.input_layernorm.weight                  |       0x57d22f40 |            0x800 |
| 1085 | llm.model.layers.2.mlp.down_proj.weight                    |       0x57d23740 |         0x600000 |
| 1086 | llm.model.layers.2.mlp.gate_proj.weight                    |       0x58323740 |         0x600000 |
| 1087 | llm.model.layers.2.mlp.up_proj.weight                      |       0x58923740 |         0x600000 |
| 1088 | llm.model.layers.2.post_attention_layernorm.weight         |       0x58f23740 |            0x800 |
| 1089 | llm.model.layers.2.self_attn.k_norm.weight                 |       0x58f23f40 |            0x100 |
| 1090 | llm.model.layers.2.self_attn.k_proj.weight                 |       0x58f24040 |         0x200000 |
| 1091 | llm.model.layers.2.self_attn.o_proj.weight                 |       0x59124040 |         0x400000 |
| 1092 | llm.model.layers.2.self_attn.q_norm.weight                 |       0x59524040 |            0x100 |
| 1093 | llm.model.layers.2.self_attn.q_proj.weight                 |       0x59524140 |         0x400000 |
| 1094 | llm.model.layers.2.self_attn.v_proj.weight                 |       0x59924140 |         0x200000 |
| 1095 | llm.model.layers.20.input_layernorm.weight                 |       0x59b24140 |            0x800 |
| 1096 | llm.model.layers.20.mlp.down_proj.weight                   |       0x59b24940 |         0x600000 |
| 1097 | llm.model.layers.20.mlp.gate_proj.weight                   |       0x5a124940 |         0x600000 |
| 1098 | llm.model.layers.20.mlp.up_proj.weight                     |       0x5a724940 |         0x600000 |
| 1099 | llm.model.layers.20.post_attention_layernorm.weight        |       0x5ad24940 |            0x800 |
| 1100 | llm.model.layers.20.self_attn.k_norm.weight                |       0x5ad25140 |            0x100 |
| 1101 | llm.model.layers.20.self_attn.k_proj.weight                |       0x5ad25240 |         0x200000 |
| 1102 | llm.model.layers.20.self_attn.o_proj.weight                |       0x5af25240 |         0x400000 |
| 1103 | llm.model.layers.20.self_attn.q_norm.weight                |       0x5b325240 |            0x100 |
| 1104 | llm.model.layers.20.self_attn.q_proj.weight                |       0x5b325340 |         0x400000 |
| 1105 | llm.model.layers.20.self_attn.v_proj.weight                |       0x5b725340 |         0x200000 |
| 1106 | llm.model.layers.21.input_layernorm.weight                 |       0x5b925340 |            0x800 |
| 1107 | llm.model.layers.21.mlp.down_proj.weight                   |       0x5b925b40 |         0x600000 |
| 1108 | llm.model.layers.21.mlp.gate_proj.weight                   |       0x5bf25b40 |         0x600000 |
| 1109 | llm.model.layers.21.mlp.up_proj.weight                     |       0x5c525b40 |         0x600000 |
| 1110 | llm.model.layers.21.post_attention_layernorm.weight        |       0x5cb25b40 |            0x800 |
| 1111 | llm.model.layers.21.self_attn.k_norm.weight                |       0x5cb26340 |            0x100 |
| 1112 | llm.model.layers.21.self_attn.k_proj.weight                |       0x5cb26440 |         0x200000 |
| 1113 | llm.model.layers.21.self_attn.o_proj.weight                |       0x5cd26440 |         0x400000 |
| 1114 | llm.model.layers.21.self_attn.q_norm.weight                |       0x5d126440 |            0x100 |
| 1115 | llm.model.layers.21.self_attn.q_proj.weight                |       0x5d126540 |         0x400000 |
| 1116 | llm.model.layers.21.self_attn.v_proj.weight                |       0x5d526540 |         0x200000 |
| 1117 | llm.model.layers.22.input_layernorm.weight                 |       0x5d726540 |            0x800 |
| 1118 | llm.model.layers.22.mlp.down_proj.weight                   |       0x5d726d40 |         0x600000 |
| 1119 | llm.model.layers.22.mlp.gate_proj.weight                   |       0x5dd26d40 |         0x600000 |
| 1120 | llm.model.layers.22.mlp.up_proj.weight                     |       0x5e326d40 |         0x600000 |
| 1121 | llm.model.layers.22.post_attention_layernorm.weight        |       0x5e926d40 |            0x800 |
| 1122 | llm.model.layers.22.self_attn.k_norm.weight                |       0x5e927540 |            0x100 |
| 1123 | llm.model.layers.22.self_attn.k_proj.weight                |       0x5e927640 |         0x200000 |
| 1124 | llm.model.layers.22.self_attn.o_proj.weight                |       0x5eb27640 |         0x400000 |
| 1125 | llm.model.layers.22.self_attn.q_norm.weight                |       0x5ef27640 |            0x100 |
| 1126 | llm.model.layers.22.self_attn.q_proj.weight                |       0x5ef27740 |         0x400000 |
| 1127 | llm.model.layers.22.self_attn.v_proj.weight                |       0x5f327740 |         0x200000 |
| 1128 | llm.model.layers.23.input_layernorm.weight                 |       0x5f527740 |            0x800 |
| 1129 | llm.model.layers.23.mlp.down_proj.weight                   |       0x5f527f40 |         0x600000 |
| 1130 | llm.model.layers.23.mlp.gate_proj.weight                   |       0x5fb27f40 |         0x600000 |
| 1131 | llm.model.layers.23.mlp.up_proj.weight                     |       0x60127f40 |         0x600000 |
| 1132 | llm.model.layers.23.post_attention_layernorm.weight        |       0x60727f40 |            0x800 |
| 1133 | llm.model.layers.23.self_attn.k_norm.weight                |       0x60728740 |            0x100 |
| 1134 | llm.model.layers.23.self_attn.k_proj.weight                |       0x60728840 |         0x200000 |
| 1135 | llm.model.layers.23.self_attn.o_proj.weight                |       0x60928840 |         0x400000 |
| 1136 | llm.model.layers.23.self_attn.q_norm.weight                |       0x60d28840 |            0x100 |
| 1137 | llm.model.layers.23.self_attn.q_proj.weight                |       0x60d28940 |         0x400000 |
| 1138 | llm.model.layers.23.self_attn.v_proj.weight                |       0x61128940 |         0x200000 |
| 1139 | llm.model.layers.24.input_layernorm.weight                 |       0x61328940 |            0x800 |
| 1140 | llm.model.layers.24.mlp.down_proj.weight                   |       0x61329140 |         0x600000 |
| 1141 | llm.model.layers.24.mlp.gate_proj.weight                   |       0x61929140 |         0x600000 |
| 1142 | llm.model.layers.24.mlp.up_proj.weight                     |       0x61f29140 |         0x600000 |
| 1143 | llm.model.layers.24.post_attention_layernorm.weight        |       0x62529140 |            0x800 |
| 1144 | llm.model.layers.24.self_attn.k_norm.weight                |       0x62529940 |            0x100 |
| 1145 | llm.model.layers.24.self_attn.k_proj.weight                |       0x62529a40 |         0x200000 |
| 1146 | llm.model.layers.24.self_attn.o_proj.weight                |       0x62729a40 |         0x400000 |
| 1147 | llm.model.layers.24.self_attn.q_norm.weight                |       0x62b29a40 |            0x100 |
| 1148 | llm.model.layers.24.self_attn.q_proj.weight                |       0x62b29b40 |         0x400000 |
| 1149 | llm.model.layers.24.self_attn.v_proj.weight                |       0x62f29b40 |         0x200000 |
| 1150 | llm.model.layers.25.input_layernorm.weight                 |       0x63129b40 |            0x800 |
| 1151 | llm.model.layers.25.mlp.down_proj.weight                   |       0x6312a340 |         0x600000 |
| 1152 | llm.model.layers.25.mlp.gate_proj.weight                   |       0x6372a340 |         0x600000 |
| 1153 | llm.model.layers.25.mlp.up_proj.weight                     |       0x63d2a340 |         0x600000 |
| 1154 | llm.model.layers.25.post_attention_layernorm.weight        |       0x6432a340 |            0x800 |
| 1155 | llm.model.layers.25.self_attn.k_norm.weight                |       0x6432ab40 |            0x100 |
| 1156 | llm.model.layers.25.self_attn.k_proj.weight                |       0x6432ac40 |         0x200000 |
| 1157 | llm.model.layers.25.self_attn.o_proj.weight                |       0x6452ac40 |         0x400000 |
| 1158 | llm.model.layers.25.self_attn.q_norm.weight                |       0x6492ac40 |            0x100 |
| 1159 | llm.model.layers.25.self_attn.q_proj.weight                |       0x6492ad40 |         0x400000 |
| 1160 | llm.model.layers.25.self_attn.v_proj.weight                |       0x64d2ad40 |         0x200000 |
| 1161 | llm.model.layers.26.input_layernorm.weight                 |       0x64f2ad40 |            0x800 |
| 1162 | llm.model.layers.26.mlp.down_proj.weight                   |       0x64f2b540 |         0x600000 |
| 1163 | llm.model.layers.26.mlp.gate_proj.weight                   |       0x6552b540 |         0x600000 |
| 1164 | llm.model.layers.26.mlp.up_proj.weight                     |       0x65b2b540 |         0x600000 |
| 1165 | llm.model.layers.26.post_attention_layernorm.weight        |       0x6612b540 |            0x800 |
| 1166 | llm.model.layers.26.self_attn.k_norm.weight                |       0x6612bd40 |            0x100 |
| 1167 | llm.model.layers.26.self_attn.k_proj.weight                |       0x6612be40 |         0x200000 |
| 1168 | llm.model.layers.26.self_attn.o_proj.weight                |       0x6632be40 |         0x400000 |
| 1169 | llm.model.layers.26.self_attn.q_norm.weight                |       0x6672be40 |            0x100 |
| 1170 | llm.model.layers.26.self_attn.q_proj.weight                |       0x6672bf40 |         0x400000 |
| 1171 | llm.model.layers.26.self_attn.v_proj.weight                |       0x66b2bf40 |         0x200000 |
| 1172 | llm.model.layers.27.input_layernorm.weight                 |       0x66d2bf40 |            0x800 |
| 1173 | llm.model.layers.27.mlp.down_proj.weight                   |       0x66d2c740 |         0x600000 |
| 1174 | llm.model.layers.27.mlp.gate_proj.weight                   |       0x6732c740 |         0x600000 |
| 1175 | llm.model.layers.27.mlp.up_proj.weight                     |       0x6792c740 |         0x600000 |
| 1176 | llm.model.layers.27.post_attention_layernorm.weight        |       0x67f2c740 |            0x800 |
| 1177 | llm.model.layers.27.self_attn.k_norm.weight                |       0x67f2cf40 |            0x100 |
| 1178 | llm.model.layers.27.self_attn.k_proj.weight                |       0x67f2d040 |         0x200000 |
| 1179 | llm.model.layers.27.self_attn.o_proj.weight                |       0x6812d040 |         0x400000 |
| 1180 | llm.model.layers.27.self_attn.q_norm.weight                |       0x6852d040 |            0x100 |
| 1181 | llm.model.layers.27.self_attn.q_proj.weight                |       0x6852d140 |         0x400000 |
| 1182 | llm.model.layers.27.self_attn.v_proj.weight                |       0x6892d140 |         0x200000 |
| 1183 | llm.model.layers.3.input_layernorm.weight                  |       0x68b2d140 |            0x800 |
| 1184 | llm.model.layers.3.mlp.down_proj.weight                    |       0x68b2d940 |         0x600000 |
| 1185 | llm.model.layers.3.mlp.gate_proj.weight                    |       0x6912d940 |         0x600000 |
| 1186 | llm.model.layers.3.mlp.up_proj.weight                      |       0x6972d940 |         0x600000 |
| 1187 | llm.model.layers.3.post_attention_layernorm.weight         |       0x69d2d940 |            0x800 |
| 1188 | llm.model.layers.3.self_attn.k_norm.weight                 |       0x69d2e140 |            0x100 |
| 1189 | llm.model.layers.3.self_attn.k_proj.weight                 |       0x69d2e240 |         0x200000 |
| 1190 | llm.model.layers.3.self_attn.o_proj.weight                 |       0x69f2e240 |         0x400000 |
| 1191 | llm.model.layers.3.self_attn.q_norm.weight                 |       0x6a32e240 |            0x100 |
| 1192 | llm.model.layers.3.self_attn.q_proj.weight                 |       0x6a32e340 |         0x400000 |
| 1193 | llm.model.layers.3.self_attn.v_proj.weight                 |       0x6a72e340 |         0x200000 |
| 1194 | llm.model.layers.4.input_layernorm.weight                  |       0x6a92e340 |            0x800 |
| 1195 | llm.model.layers.4.mlp.down_proj.weight                    |       0x6a92eb40 |         0x600000 |
| 1196 | llm.model.layers.4.mlp.gate_proj.weight                    |       0x6af2eb40 |         0x600000 |
| 1197 | llm.model.layers.4.mlp.up_proj.weight                      |       0x6b52eb40 |         0x600000 |
| 1198 | llm.model.layers.4.post_attention_layernorm.weight         |       0x6bb2eb40 |            0x800 |
| 1199 | llm.model.layers.4.self_attn.k_norm.weight                 |       0x6bb2f340 |            0x100 |
| 1200 | llm.model.layers.4.self_attn.k_proj.weight                 |       0x6bb2f440 |         0x200000 |
| 1201 | llm.model.layers.4.self_attn.o_proj.weight                 |       0x6bd2f440 |         0x400000 |
| 1202 | llm.model.layers.4.self_attn.q_norm.weight                 |       0x6c12f440 |            0x100 |
| 1203 | llm.model.layers.4.self_attn.q_proj.weight                 |       0x6c12f540 |         0x400000 |
| 1204 | llm.model.layers.4.self_attn.v_proj.weight                 |       0x6c52f540 |         0x200000 |
| 1205 | llm.model.layers.5.input_layernorm.weight                  |       0x6c72f540 |            0x800 |
| 1206 | llm.model.layers.5.mlp.down_proj.weight                    |       0x6c72fd40 |         0x600000 |
| 1207 | llm.model.layers.5.mlp.gate_proj.weight                    |       0x6cd2fd40 |         0x600000 |
| 1208 | llm.model.layers.5.mlp.up_proj.weight                      |       0x6d32fd40 |         0x600000 |
| 1209 | llm.model.layers.5.post_attention_layernorm.weight         |       0x6d92fd40 |            0x800 |
| 1210 | llm.model.layers.5.self_attn.k_norm.weight                 |       0x6d930540 |            0x100 |
| 1211 | llm.model.layers.5.self_attn.k_proj.weight                 |       0x6d930640 |         0x200000 |
| 1212 | llm.model.layers.5.self_attn.o_proj.weight                 |       0x6db30640 |         0x400000 |
| 1213 | llm.model.layers.5.self_attn.q_norm.weight                 |       0x6df30640 |            0x100 |
| 1214 | llm.model.layers.5.self_attn.q_proj.weight                 |       0x6df30740 |         0x400000 |
| 1215 | llm.model.layers.5.self_attn.v_proj.weight                 |       0x6e330740 |         0x200000 |
| 1216 | llm.model.layers.6.input_layernorm.weight                  |       0x6e530740 |            0x800 |
| 1217 | llm.model.layers.6.mlp.down_proj.weight                    |       0x6e530f40 |         0x600000 |
| 1218 | llm.model.layers.6.mlp.gate_proj.weight                    |       0x6eb30f40 |         0x600000 |
| 1219 | llm.model.layers.6.mlp.up_proj.weight                      |       0x6f130f40 |         0x600000 |
| 1220 | llm.model.layers.6.post_attention_layernorm.weight         |       0x6f730f40 |            0x800 |
| 1221 | llm.model.layers.6.self_attn.k_norm.weight                 |       0x6f731740 |            0x100 |
| 1222 | llm.model.layers.6.self_attn.k_proj.weight                 |       0x6f731840 |         0x200000 |
| 1223 | llm.model.layers.6.self_attn.o_proj.weight                 |       0x6f931840 |         0x400000 |
| 1224 | llm.model.layers.6.self_attn.q_norm.weight                 |       0x6fd31840 |            0x100 |
| 1225 | llm.model.layers.6.self_attn.q_proj.weight                 |       0x6fd31940 |         0x400000 |
| 1226 | llm.model.layers.6.self_attn.v_proj.weight                 |       0x70131940 |         0x200000 |
| 1227 | llm.model.layers.7.input_layernorm.weight                  |       0x70331940 |            0x800 |
| 1228 | llm.model.layers.7.mlp.down_proj.weight                    |       0x70332140 |         0x600000 |
| 1229 | llm.model.layers.7.mlp.gate_proj.weight                    |       0x70932140 |         0x600000 |
| 1230 | llm.model.layers.7.mlp.up_proj.weight                      |       0x70f32140 |         0x600000 |
| 1231 | llm.model.layers.7.post_attention_layernorm.weight         |       0x71532140 |            0x800 |
| 1232 | llm.model.layers.7.self_attn.k_norm.weight                 |       0x71532940 |            0x100 |
| 1233 | llm.model.layers.7.self_attn.k_proj.weight                 |       0x71532a40 |         0x200000 |
| 1234 | llm.model.layers.7.self_attn.o_proj.weight                 |       0x71732a40 |         0x400000 |
| 1235 | llm.model.layers.7.self_attn.q_norm.weight                 |       0x71b32a40 |            0x100 |
| 1236 | llm.model.layers.7.self_attn.q_proj.weight                 |       0x71b32b40 |         0x400000 |
| 1237 | llm.model.layers.7.self_attn.v_proj.weight                 |       0x71f32b40 |         0x200000 |
| 1238 | llm.model.layers.8.input_layernorm.weight                  |       0x72132b40 |            0x800 |
| 1239 | llm.model.layers.8.mlp.down_proj.weight                    |       0x72133340 |         0x600000 |
| 1240 | llm.model.layers.8.mlp.gate_proj.weight                    |       0x72733340 |         0x600000 |
| 1241 | llm.model.layers.8.mlp.up_proj.weight                      |       0x72d33340 |         0x600000 |
| 1242 | llm.model.layers.8.post_attention_layernorm.weight         |       0x73333340 |            0x800 |
| 1243 | llm.model.layers.8.self_attn.k_norm.weight                 |       0x73333b40 |            0x100 |
| 1244 | llm.model.layers.8.self_attn.k_proj.weight                 |       0x73333c40 |         0x200000 |
| 1245 | llm.model.layers.8.self_attn.o_proj.weight                 |       0x73533c40 |         0x400000 |
| 1246 | llm.model.layers.8.self_attn.q_norm.weight                 |       0x73933c40 |            0x100 |
| 1247 | llm.model.layers.8.self_attn.q_proj.weight                 |       0x73933d40 |         0x400000 |
| 1248 | llm.model.layers.8.self_attn.v_proj.weight                 |       0x73d33d40 |         0x200000 |
| 1249 | llm.model.layers.9.input_layernorm.weight                  |       0x73f33d40 |            0x800 |
| 1250 | llm.model.layers.9.mlp.down_proj.weight                    |       0x73f34540 |         0x600000 |
| 1251 | llm.model.layers.9.mlp.gate_proj.weight                    |       0x74534540 |         0x600000 |
| 1252 | llm.model.layers.9.mlp.up_proj.weight                      |       0x74b34540 |         0x600000 |
| 1253 | llm.model.layers.9.post_attention_layernorm.weight         |       0x75134540 |            0x800 |
| 1254 | llm.model.layers.9.self_attn.k_norm.weight                 |       0x75134d40 |            0x100 |
| 1255 | llm.model.layers.9.self_attn.k_proj.weight                 |       0x75134e40 |         0x200000 |
| 1256 | llm.model.layers.9.self_attn.o_proj.weight                 |       0x75334e40 |         0x400000 |
| 1257 | llm.model.layers.9.self_attn.q_norm.weight                 |       0x75734e40 |            0x100 |
| 1258 | llm.model.layers.9.self_attn.q_proj.weight                 |       0x75734f40 |         0x400000 |
| 1259 | llm.model.layers.9.self_attn.v_proj.weight                 |       0x75b34f40 |         0x200000 |
| 1260 | llm.model.norm.weight                                      |       0x75d34f40 |            0x800 |

### <a name="base">Base Tensor Group : ~985M Elements</a>

| T_ID | Tensor Layer Name                                          | Human Friendly Tensor Layer Name                        | Elements          | Shape                   | Type |
|-----:|:-----------------------------------------------------------|:--------------------------------------------------------|:------------------|:------------------------|:-----|
|    0 | audio_encoder.encoders0.0.self_attn.linear_out.weight      | Audio_Encoder Encoders0 0 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|    1 | audio_encoder.encoders0.0.self_attn.linear_out.bias        | Audio_Encoder Encoders0 0 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|    2 | audio_encoder.encoders0.0.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders0 0 Self_Attn Linear_Q_K_V (W)    | (~860K)    860160 |  560 x   1536 x   1 x 1 | BF16 |
|    3 | audio_encoder.encoders0.0.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders0 0 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|    4 | audio_encoder.encoders0.0.self_attn.fsmn_block.weight      | Audio_Encoder Encoders0 0 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|    5 | audio_encoder.encoders0.0.feed_forward.w_1.weight          | Audio_Encoder Encoders0 0 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|    6 | audio_encoder.encoders0.0.feed_forward.w_1.bias            | Audio_Encoder Encoders0 0 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|    7 | audio_encoder.encoders0.0.feed_forward.w_2.weight          | Audio_Encoder Encoders0 0 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|    8 | audio_encoder.encoders0.0.feed_forward.w_2.bias            | Audio_Encoder Encoders0 0 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|    9 | audio_encoder.encoders0.0.norm1.weight                     | Audio_Encoder Encoders0 0 Norm1 (W)                     | (  560)       560 |  560 x      1 x   1 x 1 | BF16 |
|   10 | audio_encoder.encoders0.0.norm1.bias                       | Audio_Encoder Encoders0 0 Norm1 (B)                     | (  560)       560 |  560 x      1 x   1 x 1 | BF16 |
|   11 | audio_encoder.encoders0.0.norm2.weight                     | Audio_Encoder Encoders0 0 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   12 | audio_encoder.encoders0.0.norm2.bias                       | Audio_Encoder Encoders0 0 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   13 | audio_encoder.encoders.0.self_attn.linear_out.weight       | Audio_Encoder Encoders 0 Self_Attn Linear_Out (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   14 | audio_encoder.encoders.0.self_attn.linear_out.bias         | Audio_Encoder Encoders 0 Self_Attn Linear_Out (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   15 | audio_encoder.encoders.0.self_attn.linear_q_k_v.weight     | Audio_Encoder Encoders 0 Self_Attn Linear_Q_K_V (W)     | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|   16 | audio_encoder.encoders.0.self_attn.linear_q_k_v.bias       | Audio_Encoder Encoders 0 Self_Attn Linear_Q_K_V (B)     | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|   17 | audio_encoder.encoders.0.self_attn.fsmn_block.weight       | Audio_Encoder Encoders 0 Self_Attn Fsmn_Block (W)       | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|   18 | audio_encoder.encoders.0.feed_forward.w_1.weight           | Audio_Encoder Encoders 0 Feed_Forward W_1 (W)           | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   19 | audio_encoder.encoders.0.feed_forward.w_1.bias             | Audio_Encoder Encoders 0 Feed_Forward W_1 (B)           | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   20 | audio_encoder.encoders.0.feed_forward.w_2.weight           | Audio_Encoder Encoders 0 Feed_Forward W_2 (W)           | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   21 | audio_encoder.encoders.0.feed_forward.w_2.bias             | Audio_Encoder Encoders 0 Feed_Forward W_2 (B)           | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   22 | audio_encoder.encoders.0.norm1.weight                      | Audio_Encoder Encoders 0 Norm1 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   23 | audio_encoder.encoders.0.norm1.bias                        | Audio_Encoder Encoders 0 Norm1 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   24 | audio_encoder.encoders.0.norm2.weight                      | Audio_Encoder Encoders 0 Norm2 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   25 | audio_encoder.encoders.0.norm2.bias                        | Audio_Encoder Encoders 0 Norm2 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   26 | audio_encoder.encoders.1.self_attn.linear_out.weight       | Audio_Encoder Encoders 1 Self_Attn Linear_Out (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   27 | audio_encoder.encoders.1.self_attn.linear_out.bias         | Audio_Encoder Encoders 1 Self_Attn Linear_Out (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   28 | audio_encoder.encoders.1.self_attn.linear_q_k_v.weight     | Audio_Encoder Encoders 1 Self_Attn Linear_Q_K_V (W)     | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|   29 | audio_encoder.encoders.1.self_attn.linear_q_k_v.bias       | Audio_Encoder Encoders 1 Self_Attn Linear_Q_K_V (B)     | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|   30 | audio_encoder.encoders.1.self_attn.fsmn_block.weight       | Audio_Encoder Encoders 1 Self_Attn Fsmn_Block (W)       | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|   31 | audio_encoder.encoders.1.feed_forward.w_1.weight           | Audio_Encoder Encoders 1 Feed_Forward W_1 (W)           | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   32 | audio_encoder.encoders.1.feed_forward.w_1.bias             | Audio_Encoder Encoders 1 Feed_Forward W_1 (B)           | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   33 | audio_encoder.encoders.1.feed_forward.w_2.weight           | Audio_Encoder Encoders 1 Feed_Forward W_2 (W)           | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   34 | audio_encoder.encoders.1.feed_forward.w_2.bias             | Audio_Encoder Encoders 1 Feed_Forward W_2 (B)           | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   35 | audio_encoder.encoders.1.norm1.weight                      | Audio_Encoder Encoders 1 Norm1 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   36 | audio_encoder.encoders.1.norm1.bias                        | Audio_Encoder Encoders 1 Norm1 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   37 | audio_encoder.encoders.1.norm2.weight                      | Audio_Encoder Encoders 1 Norm2 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   38 | audio_encoder.encoders.1.norm2.bias                        | Audio_Encoder Encoders 1 Norm2 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   39 | audio_encoder.encoders.2.self_attn.linear_out.weight       | Audio_Encoder Encoders 2 Self_Attn Linear_Out (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   40 | audio_encoder.encoders.2.self_attn.linear_out.bias         | Audio_Encoder Encoders 2 Self_Attn Linear_Out (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   41 | audio_encoder.encoders.2.self_attn.linear_q_k_v.weight     | Audio_Encoder Encoders 2 Self_Attn Linear_Q_K_V (W)     | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|   42 | audio_encoder.encoders.2.self_attn.linear_q_k_v.bias       | Audio_Encoder Encoders 2 Self_Attn Linear_Q_K_V (B)     | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|   43 | audio_encoder.encoders.2.self_attn.fsmn_block.weight       | Audio_Encoder Encoders 2 Self_Attn Fsmn_Block (W)       | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|   44 | audio_encoder.encoders.2.feed_forward.w_1.weight           | Audio_Encoder Encoders 2 Feed_Forward W_1 (W)           | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   45 | audio_encoder.encoders.2.feed_forward.w_1.bias             | Audio_Encoder Encoders 2 Feed_Forward W_1 (B)           | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   46 | audio_encoder.encoders.2.feed_forward.w_2.weight           | Audio_Encoder Encoders 2 Feed_Forward W_2 (W)           | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   47 | audio_encoder.encoders.2.feed_forward.w_2.bias             | Audio_Encoder Encoders 2 Feed_Forward W_2 (B)           | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   48 | audio_encoder.encoders.2.norm1.weight                      | Audio_Encoder Encoders 2 Norm1 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   49 | audio_encoder.encoders.2.norm1.bias                        | Audio_Encoder Encoders 2 Norm1 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   50 | audio_encoder.encoders.2.norm2.weight                      | Audio_Encoder Encoders 2 Norm2 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   51 | audio_encoder.encoders.2.norm2.bias                        | Audio_Encoder Encoders 2 Norm2 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   52 | audio_encoder.encoders.3.self_attn.linear_out.weight       | Audio_Encoder Encoders 3 Self_Attn Linear_Out (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   53 | audio_encoder.encoders.3.self_attn.linear_out.bias         | Audio_Encoder Encoders 3 Self_Attn Linear_Out (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   54 | audio_encoder.encoders.3.self_attn.linear_q_k_v.weight     | Audio_Encoder Encoders 3 Self_Attn Linear_Q_K_V (W)     | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|   55 | audio_encoder.encoders.3.self_attn.linear_q_k_v.bias       | Audio_Encoder Encoders 3 Self_Attn Linear_Q_K_V (B)     | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|   56 | audio_encoder.encoders.3.self_attn.fsmn_block.weight       | Audio_Encoder Encoders 3 Self_Attn Fsmn_Block (W)       | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|   57 | audio_encoder.encoders.3.feed_forward.w_1.weight           | Audio_Encoder Encoders 3 Feed_Forward W_1 (W)           | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   58 | audio_encoder.encoders.3.feed_forward.w_1.bias             | Audio_Encoder Encoders 3 Feed_Forward W_1 (B)           | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   59 | audio_encoder.encoders.3.feed_forward.w_2.weight           | Audio_Encoder Encoders 3 Feed_Forward W_2 (W)           | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   60 | audio_encoder.encoders.3.feed_forward.w_2.bias             | Audio_Encoder Encoders 3 Feed_Forward W_2 (B)           | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   61 | audio_encoder.encoders.3.norm1.weight                      | Audio_Encoder Encoders 3 Norm1 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   62 | audio_encoder.encoders.3.norm1.bias                        | Audio_Encoder Encoders 3 Norm1 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   63 | audio_encoder.encoders.3.norm2.weight                      | Audio_Encoder Encoders 3 Norm2 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   64 | audio_encoder.encoders.3.norm2.bias                        | Audio_Encoder Encoders 3 Norm2 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   65 | audio_encoder.encoders.4.self_attn.linear_out.weight       | Audio_Encoder Encoders 4 Self_Attn Linear_Out (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   66 | audio_encoder.encoders.4.self_attn.linear_out.bias         | Audio_Encoder Encoders 4 Self_Attn Linear_Out (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   67 | audio_encoder.encoders.4.self_attn.linear_q_k_v.weight     | Audio_Encoder Encoders 4 Self_Attn Linear_Q_K_V (W)     | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|   68 | audio_encoder.encoders.4.self_attn.linear_q_k_v.bias       | Audio_Encoder Encoders 4 Self_Attn Linear_Q_K_V (B)     | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|   69 | audio_encoder.encoders.4.self_attn.fsmn_block.weight       | Audio_Encoder Encoders 4 Self_Attn Fsmn_Block (W)       | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|   70 | audio_encoder.encoders.4.feed_forward.w_1.weight           | Audio_Encoder Encoders 4 Feed_Forward W_1 (W)           | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   71 | audio_encoder.encoders.4.feed_forward.w_1.bias             | Audio_Encoder Encoders 4 Feed_Forward W_1 (B)           | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   72 | audio_encoder.encoders.4.feed_forward.w_2.weight           | Audio_Encoder Encoders 4 Feed_Forward W_2 (W)           | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   73 | audio_encoder.encoders.4.feed_forward.w_2.bias             | Audio_Encoder Encoders 4 Feed_Forward W_2 (B)           | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   74 | audio_encoder.encoders.4.norm1.weight                      | Audio_Encoder Encoders 4 Norm1 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   75 | audio_encoder.encoders.4.norm1.bias                        | Audio_Encoder Encoders 4 Norm1 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   76 | audio_encoder.encoders.4.norm2.weight                      | Audio_Encoder Encoders 4 Norm2 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   77 | audio_encoder.encoders.4.norm2.bias                        | Audio_Encoder Encoders 4 Norm2 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   78 | audio_encoder.encoders.5.self_attn.linear_out.weight       | Audio_Encoder Encoders 5 Self_Attn Linear_Out (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   79 | audio_encoder.encoders.5.self_attn.linear_out.bias         | Audio_Encoder Encoders 5 Self_Attn Linear_Out (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   80 | audio_encoder.encoders.5.self_attn.linear_q_k_v.weight     | Audio_Encoder Encoders 5 Self_Attn Linear_Q_K_V (W)     | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|   81 | audio_encoder.encoders.5.self_attn.linear_q_k_v.bias       | Audio_Encoder Encoders 5 Self_Attn Linear_Q_K_V (B)     | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|   82 | audio_encoder.encoders.5.self_attn.fsmn_block.weight       | Audio_Encoder Encoders 5 Self_Attn Fsmn_Block (W)       | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|   83 | audio_encoder.encoders.5.feed_forward.w_1.weight           | Audio_Encoder Encoders 5 Feed_Forward W_1 (W)           | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   84 | audio_encoder.encoders.5.feed_forward.w_1.bias             | Audio_Encoder Encoders 5 Feed_Forward W_1 (B)           | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   85 | audio_encoder.encoders.5.feed_forward.w_2.weight           | Audio_Encoder Encoders 5 Feed_Forward W_2 (W)           | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   86 | audio_encoder.encoders.5.feed_forward.w_2.bias             | Audio_Encoder Encoders 5 Feed_Forward W_2 (B)           | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   87 | audio_encoder.encoders.5.norm1.weight                      | Audio_Encoder Encoders 5 Norm1 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   88 | audio_encoder.encoders.5.norm1.bias                        | Audio_Encoder Encoders 5 Norm1 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   89 | audio_encoder.encoders.5.norm2.weight                      | Audio_Encoder Encoders 5 Norm2 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   90 | audio_encoder.encoders.5.norm2.bias                        | Audio_Encoder Encoders 5 Norm2 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   91 | audio_encoder.encoders.6.self_attn.linear_out.weight       | Audio_Encoder Encoders 6 Self_Attn Linear_Out (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|   92 | audio_encoder.encoders.6.self_attn.linear_out.bias         | Audio_Encoder Encoders 6 Self_Attn Linear_Out (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|   93 | audio_encoder.encoders.6.self_attn.linear_q_k_v.weight     | Audio_Encoder Encoders 6 Self_Attn Linear_Q_K_V (W)     | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|   94 | audio_encoder.encoders.6.self_attn.linear_q_k_v.bias       | Audio_Encoder Encoders 6 Self_Attn Linear_Q_K_V (B)     | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|   95 | audio_encoder.encoders.6.self_attn.fsmn_block.weight       | Audio_Encoder Encoders 6 Self_Attn Fsmn_Block (W)       | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|   96 | audio_encoder.encoders.6.feed_forward.w_1.weight           | Audio_Encoder Encoders 6 Feed_Forward W_1 (W)           | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|   97 | audio_encoder.encoders.6.feed_forward.w_1.bias             | Audio_Encoder Encoders 6 Feed_Forward W_1 (B)           | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|   98 | audio_encoder.encoders.6.feed_forward.w_2.weight           | Audio_Encoder Encoders 6 Feed_Forward W_2 (W)           | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|   99 | audio_encoder.encoders.6.feed_forward.w_2.bias             | Audio_Encoder Encoders 6 Feed_Forward W_2 (B)           | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  100 | audio_encoder.encoders.6.norm1.weight                      | Audio_Encoder Encoders 6 Norm1 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  101 | audio_encoder.encoders.6.norm1.bias                        | Audio_Encoder Encoders 6 Norm1 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  102 | audio_encoder.encoders.6.norm2.weight                      | Audio_Encoder Encoders 6 Norm2 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  103 | audio_encoder.encoders.6.norm2.bias                        | Audio_Encoder Encoders 6 Norm2 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  104 | audio_encoder.encoders.7.self_attn.linear_out.weight       | Audio_Encoder Encoders 7 Self_Attn Linear_Out (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  105 | audio_encoder.encoders.7.self_attn.linear_out.bias         | Audio_Encoder Encoders 7 Self_Attn Linear_Out (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  106 | audio_encoder.encoders.7.self_attn.linear_q_k_v.weight     | Audio_Encoder Encoders 7 Self_Attn Linear_Q_K_V (W)     | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  107 | audio_encoder.encoders.7.self_attn.linear_q_k_v.bias       | Audio_Encoder Encoders 7 Self_Attn Linear_Q_K_V (B)     | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  108 | audio_encoder.encoders.7.self_attn.fsmn_block.weight       | Audio_Encoder Encoders 7 Self_Attn Fsmn_Block (W)       | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  109 | audio_encoder.encoders.7.feed_forward.w_1.weight           | Audio_Encoder Encoders 7 Feed_Forward W_1 (W)           | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  110 | audio_encoder.encoders.7.feed_forward.w_1.bias             | Audio_Encoder Encoders 7 Feed_Forward W_1 (B)           | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  111 | audio_encoder.encoders.7.feed_forward.w_2.weight           | Audio_Encoder Encoders 7 Feed_Forward W_2 (W)           | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  112 | audio_encoder.encoders.7.feed_forward.w_2.bias             | Audio_Encoder Encoders 7 Feed_Forward W_2 (B)           | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  113 | audio_encoder.encoders.7.norm1.weight                      | Audio_Encoder Encoders 7 Norm1 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  114 | audio_encoder.encoders.7.norm1.bias                        | Audio_Encoder Encoders 7 Norm1 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  115 | audio_encoder.encoders.7.norm2.weight                      | Audio_Encoder Encoders 7 Norm2 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  116 | audio_encoder.encoders.7.norm2.bias                        | Audio_Encoder Encoders 7 Norm2 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  117 | audio_encoder.encoders.8.self_attn.linear_out.weight       | Audio_Encoder Encoders 8 Self_Attn Linear_Out (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  118 | audio_encoder.encoders.8.self_attn.linear_out.bias         | Audio_Encoder Encoders 8 Self_Attn Linear_Out (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  119 | audio_encoder.encoders.8.self_attn.linear_q_k_v.weight     | Audio_Encoder Encoders 8 Self_Attn Linear_Q_K_V (W)     | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  120 | audio_encoder.encoders.8.self_attn.linear_q_k_v.bias       | Audio_Encoder Encoders 8 Self_Attn Linear_Q_K_V (B)     | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  121 | audio_encoder.encoders.8.self_attn.fsmn_block.weight       | Audio_Encoder Encoders 8 Self_Attn Fsmn_Block (W)       | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  122 | audio_encoder.encoders.8.feed_forward.w_1.weight           | Audio_Encoder Encoders 8 Feed_Forward W_1 (W)           | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  123 | audio_encoder.encoders.8.feed_forward.w_1.bias             | Audio_Encoder Encoders 8 Feed_Forward W_1 (B)           | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  124 | audio_encoder.encoders.8.feed_forward.w_2.weight           | Audio_Encoder Encoders 8 Feed_Forward W_2 (W)           | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  125 | audio_encoder.encoders.8.feed_forward.w_2.bias             | Audio_Encoder Encoders 8 Feed_Forward W_2 (B)           | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  126 | audio_encoder.encoders.8.norm1.weight                      | Audio_Encoder Encoders 8 Norm1 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  127 | audio_encoder.encoders.8.norm1.bias                        | Audio_Encoder Encoders 8 Norm1 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  128 | audio_encoder.encoders.8.norm2.weight                      | Audio_Encoder Encoders 8 Norm2 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  129 | audio_encoder.encoders.8.norm2.bias                        | Audio_Encoder Encoders 8 Norm2 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  130 | audio_encoder.encoders.9.self_attn.linear_out.weight       | Audio_Encoder Encoders 9 Self_Attn Linear_Out (W)       | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  131 | audio_encoder.encoders.9.self_attn.linear_out.bias         | Audio_Encoder Encoders 9 Self_Attn Linear_Out (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  132 | audio_encoder.encoders.9.self_attn.linear_q_k_v.weight     | Audio_Encoder Encoders 9 Self_Attn Linear_Q_K_V (W)     | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  133 | audio_encoder.encoders.9.self_attn.linear_q_k_v.bias       | Audio_Encoder Encoders 9 Self_Attn Linear_Q_K_V (B)     | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  134 | audio_encoder.encoders.9.self_attn.fsmn_block.weight       | Audio_Encoder Encoders 9 Self_Attn Fsmn_Block (W)       | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  135 | audio_encoder.encoders.9.feed_forward.w_1.weight           | Audio_Encoder Encoders 9 Feed_Forward W_1 (W)           | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  136 | audio_encoder.encoders.9.feed_forward.w_1.bias             | Audio_Encoder Encoders 9 Feed_Forward W_1 (B)           | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  137 | audio_encoder.encoders.9.feed_forward.w_2.weight           | Audio_Encoder Encoders 9 Feed_Forward W_2 (W)           | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  138 | audio_encoder.encoders.9.feed_forward.w_2.bias             | Audio_Encoder Encoders 9 Feed_Forward W_2 (B)           | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  139 | audio_encoder.encoders.9.norm1.weight                      | Audio_Encoder Encoders 9 Norm1 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  140 | audio_encoder.encoders.9.norm1.bias                        | Audio_Encoder Encoders 9 Norm1 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  141 | audio_encoder.encoders.9.norm2.weight                      | Audio_Encoder Encoders 9 Norm2 (W)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  142 | audio_encoder.encoders.9.norm2.bias                        | Audio_Encoder Encoders 9 Norm2 (B)                      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  143 | audio_encoder.encoders.10.self_attn.linear_out.weight      | Audio_Encoder Encoders 10 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  144 | audio_encoder.encoders.10.self_attn.linear_out.bias        | Audio_Encoder Encoders 10 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  145 | audio_encoder.encoders.10.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 10 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  146 | audio_encoder.encoders.10.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 10 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  147 | audio_encoder.encoders.10.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 10 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  148 | audio_encoder.encoders.10.feed_forward.w_1.weight          | Audio_Encoder Encoders 10 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  149 | audio_encoder.encoders.10.feed_forward.w_1.bias            | Audio_Encoder Encoders 10 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  150 | audio_encoder.encoders.10.feed_forward.w_2.weight          | Audio_Encoder Encoders 10 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  151 | audio_encoder.encoders.10.feed_forward.w_2.bias            | Audio_Encoder Encoders 10 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  152 | audio_encoder.encoders.10.norm1.weight                     | Audio_Encoder Encoders 10 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  153 | audio_encoder.encoders.10.norm1.bias                       | Audio_Encoder Encoders 10 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  154 | audio_encoder.encoders.10.norm2.weight                     | Audio_Encoder Encoders 10 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  155 | audio_encoder.encoders.10.norm2.bias                       | Audio_Encoder Encoders 10 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  156 | audio_encoder.encoders.11.self_attn.linear_out.weight      | Audio_Encoder Encoders 11 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  157 | audio_encoder.encoders.11.self_attn.linear_out.bias        | Audio_Encoder Encoders 11 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  158 | audio_encoder.encoders.11.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 11 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  159 | audio_encoder.encoders.11.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 11 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  160 | audio_encoder.encoders.11.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 11 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  161 | audio_encoder.encoders.11.feed_forward.w_1.weight          | Audio_Encoder Encoders 11 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  162 | audio_encoder.encoders.11.feed_forward.w_1.bias            | Audio_Encoder Encoders 11 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  163 | audio_encoder.encoders.11.feed_forward.w_2.weight          | Audio_Encoder Encoders 11 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  164 | audio_encoder.encoders.11.feed_forward.w_2.bias            | Audio_Encoder Encoders 11 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  165 | audio_encoder.encoders.11.norm1.weight                     | Audio_Encoder Encoders 11 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  166 | audio_encoder.encoders.11.norm1.bias                       | Audio_Encoder Encoders 11 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  167 | audio_encoder.encoders.11.norm2.weight                     | Audio_Encoder Encoders 11 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  168 | audio_encoder.encoders.11.norm2.bias                       | Audio_Encoder Encoders 11 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  169 | audio_encoder.encoders.12.self_attn.linear_out.weight      | Audio_Encoder Encoders 12 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  170 | audio_encoder.encoders.12.self_attn.linear_out.bias        | Audio_Encoder Encoders 12 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  171 | audio_encoder.encoders.12.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 12 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  172 | audio_encoder.encoders.12.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 12 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  173 | audio_encoder.encoders.12.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 12 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  174 | audio_encoder.encoders.12.feed_forward.w_1.weight          | Audio_Encoder Encoders 12 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  175 | audio_encoder.encoders.12.feed_forward.w_1.bias            | Audio_Encoder Encoders 12 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  176 | audio_encoder.encoders.12.feed_forward.w_2.weight          | Audio_Encoder Encoders 12 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  177 | audio_encoder.encoders.12.feed_forward.w_2.bias            | Audio_Encoder Encoders 12 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  178 | audio_encoder.encoders.12.norm1.weight                     | Audio_Encoder Encoders 12 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  179 | audio_encoder.encoders.12.norm1.bias                       | Audio_Encoder Encoders 12 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  180 | audio_encoder.encoders.12.norm2.weight                     | Audio_Encoder Encoders 12 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  181 | audio_encoder.encoders.12.norm2.bias                       | Audio_Encoder Encoders 12 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  182 | audio_encoder.encoders.13.self_attn.linear_out.weight      | Audio_Encoder Encoders 13 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  183 | audio_encoder.encoders.13.self_attn.linear_out.bias        | Audio_Encoder Encoders 13 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  184 | audio_encoder.encoders.13.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 13 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  185 | audio_encoder.encoders.13.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 13 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  186 | audio_encoder.encoders.13.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 13 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  187 | audio_encoder.encoders.13.feed_forward.w_1.weight          | Audio_Encoder Encoders 13 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  188 | audio_encoder.encoders.13.feed_forward.w_1.bias            | Audio_Encoder Encoders 13 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  189 | audio_encoder.encoders.13.feed_forward.w_2.weight          | Audio_Encoder Encoders 13 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  190 | audio_encoder.encoders.13.feed_forward.w_2.bias            | Audio_Encoder Encoders 13 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  191 | audio_encoder.encoders.13.norm1.weight                     | Audio_Encoder Encoders 13 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  192 | audio_encoder.encoders.13.norm1.bias                       | Audio_Encoder Encoders 13 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  193 | audio_encoder.encoders.13.norm2.weight                     | Audio_Encoder Encoders 13 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  194 | audio_encoder.encoders.13.norm2.bias                       | Audio_Encoder Encoders 13 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  195 | audio_encoder.encoders.14.self_attn.linear_out.weight      | Audio_Encoder Encoders 14 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  196 | audio_encoder.encoders.14.self_attn.linear_out.bias        | Audio_Encoder Encoders 14 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  197 | audio_encoder.encoders.14.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 14 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  198 | audio_encoder.encoders.14.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 14 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  199 | audio_encoder.encoders.14.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 14 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  200 | audio_encoder.encoders.14.feed_forward.w_1.weight          | Audio_Encoder Encoders 14 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  201 | audio_encoder.encoders.14.feed_forward.w_1.bias            | Audio_Encoder Encoders 14 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  202 | audio_encoder.encoders.14.feed_forward.w_2.weight          | Audio_Encoder Encoders 14 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  203 | audio_encoder.encoders.14.feed_forward.w_2.bias            | Audio_Encoder Encoders 14 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  204 | audio_encoder.encoders.14.norm1.weight                     | Audio_Encoder Encoders 14 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  205 | audio_encoder.encoders.14.norm1.bias                       | Audio_Encoder Encoders 14 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  206 | audio_encoder.encoders.14.norm2.weight                     | Audio_Encoder Encoders 14 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  207 | audio_encoder.encoders.14.norm2.bias                       | Audio_Encoder Encoders 14 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  208 | audio_encoder.encoders.15.self_attn.linear_out.weight      | Audio_Encoder Encoders 15 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  209 | audio_encoder.encoders.15.self_attn.linear_out.bias        | Audio_Encoder Encoders 15 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  210 | audio_encoder.encoders.15.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 15 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  211 | audio_encoder.encoders.15.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 15 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  212 | audio_encoder.encoders.15.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 15 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  213 | audio_encoder.encoders.15.feed_forward.w_1.weight          | Audio_Encoder Encoders 15 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  214 | audio_encoder.encoders.15.feed_forward.w_1.bias            | Audio_Encoder Encoders 15 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  215 | audio_encoder.encoders.15.feed_forward.w_2.weight          | Audio_Encoder Encoders 15 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  216 | audio_encoder.encoders.15.feed_forward.w_2.bias            | Audio_Encoder Encoders 15 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  217 | audio_encoder.encoders.15.norm1.weight                     | Audio_Encoder Encoders 15 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  218 | audio_encoder.encoders.15.norm1.bias                       | Audio_Encoder Encoders 15 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  219 | audio_encoder.encoders.15.norm2.weight                     | Audio_Encoder Encoders 15 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  220 | audio_encoder.encoders.15.norm2.bias                       | Audio_Encoder Encoders 15 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  221 | audio_encoder.encoders.16.self_attn.linear_out.weight      | Audio_Encoder Encoders 16 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  222 | audio_encoder.encoders.16.self_attn.linear_out.bias        | Audio_Encoder Encoders 16 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  223 | audio_encoder.encoders.16.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 16 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  224 | audio_encoder.encoders.16.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 16 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  225 | audio_encoder.encoders.16.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 16 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  226 | audio_encoder.encoders.16.feed_forward.w_1.weight          | Audio_Encoder Encoders 16 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  227 | audio_encoder.encoders.16.feed_forward.w_1.bias            | Audio_Encoder Encoders 16 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  228 | audio_encoder.encoders.16.feed_forward.w_2.weight          | Audio_Encoder Encoders 16 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  229 | audio_encoder.encoders.16.feed_forward.w_2.bias            | Audio_Encoder Encoders 16 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  230 | audio_encoder.encoders.16.norm1.weight                     | Audio_Encoder Encoders 16 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  231 | audio_encoder.encoders.16.norm1.bias                       | Audio_Encoder Encoders 16 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  232 | audio_encoder.encoders.16.norm2.weight                     | Audio_Encoder Encoders 16 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  233 | audio_encoder.encoders.16.norm2.bias                       | Audio_Encoder Encoders 16 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  234 | audio_encoder.encoders.17.self_attn.linear_out.weight      | Audio_Encoder Encoders 17 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  235 | audio_encoder.encoders.17.self_attn.linear_out.bias        | Audio_Encoder Encoders 17 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  236 | audio_encoder.encoders.17.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 17 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  237 | audio_encoder.encoders.17.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 17 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  238 | audio_encoder.encoders.17.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 17 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  239 | audio_encoder.encoders.17.feed_forward.w_1.weight          | Audio_Encoder Encoders 17 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  240 | audio_encoder.encoders.17.feed_forward.w_1.bias            | Audio_Encoder Encoders 17 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  241 | audio_encoder.encoders.17.feed_forward.w_2.weight          | Audio_Encoder Encoders 17 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  242 | audio_encoder.encoders.17.feed_forward.w_2.bias            | Audio_Encoder Encoders 17 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  243 | audio_encoder.encoders.17.norm1.weight                     | Audio_Encoder Encoders 17 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  244 | audio_encoder.encoders.17.norm1.bias                       | Audio_Encoder Encoders 17 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  245 | audio_encoder.encoders.17.norm2.weight                     | Audio_Encoder Encoders 17 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  246 | audio_encoder.encoders.17.norm2.bias                       | Audio_Encoder Encoders 17 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  247 | audio_encoder.encoders.18.self_attn.linear_out.weight      | Audio_Encoder Encoders 18 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  248 | audio_encoder.encoders.18.self_attn.linear_out.bias        | Audio_Encoder Encoders 18 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  249 | audio_encoder.encoders.18.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 18 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  250 | audio_encoder.encoders.18.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 18 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  251 | audio_encoder.encoders.18.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 18 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  252 | audio_encoder.encoders.18.feed_forward.w_1.weight          | Audio_Encoder Encoders 18 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  253 | audio_encoder.encoders.18.feed_forward.w_1.bias            | Audio_Encoder Encoders 18 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  254 | audio_encoder.encoders.18.feed_forward.w_2.weight          | Audio_Encoder Encoders 18 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  255 | audio_encoder.encoders.18.feed_forward.w_2.bias            | Audio_Encoder Encoders 18 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  256 | audio_encoder.encoders.18.norm1.weight                     | Audio_Encoder Encoders 18 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  257 | audio_encoder.encoders.18.norm1.bias                       | Audio_Encoder Encoders 18 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  258 | audio_encoder.encoders.18.norm2.weight                     | Audio_Encoder Encoders 18 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  259 | audio_encoder.encoders.18.norm2.bias                       | Audio_Encoder Encoders 18 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  260 | audio_encoder.encoders.19.self_attn.linear_out.weight      | Audio_Encoder Encoders 19 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  261 | audio_encoder.encoders.19.self_attn.linear_out.bias        | Audio_Encoder Encoders 19 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  262 | audio_encoder.encoders.19.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 19 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  263 | audio_encoder.encoders.19.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 19 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  264 | audio_encoder.encoders.19.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 19 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  265 | audio_encoder.encoders.19.feed_forward.w_1.weight          | Audio_Encoder Encoders 19 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  266 | audio_encoder.encoders.19.feed_forward.w_1.bias            | Audio_Encoder Encoders 19 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  267 | audio_encoder.encoders.19.feed_forward.w_2.weight          | Audio_Encoder Encoders 19 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  268 | audio_encoder.encoders.19.feed_forward.w_2.bias            | Audio_Encoder Encoders 19 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  269 | audio_encoder.encoders.19.norm1.weight                     | Audio_Encoder Encoders 19 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  270 | audio_encoder.encoders.19.norm1.bias                       | Audio_Encoder Encoders 19 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  271 | audio_encoder.encoders.19.norm2.weight                     | Audio_Encoder Encoders 19 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  272 | audio_encoder.encoders.19.norm2.bias                       | Audio_Encoder Encoders 19 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  273 | audio_encoder.encoders.20.self_attn.linear_out.weight      | Audio_Encoder Encoders 20 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  274 | audio_encoder.encoders.20.self_attn.linear_out.bias        | Audio_Encoder Encoders 20 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  275 | audio_encoder.encoders.20.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 20 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  276 | audio_encoder.encoders.20.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 20 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  277 | audio_encoder.encoders.20.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 20 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  278 | audio_encoder.encoders.20.feed_forward.w_1.weight          | Audio_Encoder Encoders 20 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  279 | audio_encoder.encoders.20.feed_forward.w_1.bias            | Audio_Encoder Encoders 20 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  280 | audio_encoder.encoders.20.feed_forward.w_2.weight          | Audio_Encoder Encoders 20 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  281 | audio_encoder.encoders.20.feed_forward.w_2.bias            | Audio_Encoder Encoders 20 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  282 | audio_encoder.encoders.20.norm1.weight                     | Audio_Encoder Encoders 20 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  283 | audio_encoder.encoders.20.norm1.bias                       | Audio_Encoder Encoders 20 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  284 | audio_encoder.encoders.20.norm2.weight                     | Audio_Encoder Encoders 20 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  285 | audio_encoder.encoders.20.norm2.bias                       | Audio_Encoder Encoders 20 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  286 | audio_encoder.encoders.21.self_attn.linear_out.weight      | Audio_Encoder Encoders 21 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  287 | audio_encoder.encoders.21.self_attn.linear_out.bias        | Audio_Encoder Encoders 21 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  288 | audio_encoder.encoders.21.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 21 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  289 | audio_encoder.encoders.21.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 21 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  290 | audio_encoder.encoders.21.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 21 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  291 | audio_encoder.encoders.21.feed_forward.w_1.weight          | Audio_Encoder Encoders 21 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  292 | audio_encoder.encoders.21.feed_forward.w_1.bias            | Audio_Encoder Encoders 21 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  293 | audio_encoder.encoders.21.feed_forward.w_2.weight          | Audio_Encoder Encoders 21 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  294 | audio_encoder.encoders.21.feed_forward.w_2.bias            | Audio_Encoder Encoders 21 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  295 | audio_encoder.encoders.21.norm1.weight                     | Audio_Encoder Encoders 21 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  296 | audio_encoder.encoders.21.norm1.bias                       | Audio_Encoder Encoders 21 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  297 | audio_encoder.encoders.21.norm2.weight                     | Audio_Encoder Encoders 21 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  298 | audio_encoder.encoders.21.norm2.bias                       | Audio_Encoder Encoders 21 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  299 | audio_encoder.encoders.22.self_attn.linear_out.weight      | Audio_Encoder Encoders 22 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  300 | audio_encoder.encoders.22.self_attn.linear_out.bias        | Audio_Encoder Encoders 22 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  301 | audio_encoder.encoders.22.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 22 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  302 | audio_encoder.encoders.22.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 22 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  303 | audio_encoder.encoders.22.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 22 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  304 | audio_encoder.encoders.22.feed_forward.w_1.weight          | Audio_Encoder Encoders 22 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  305 | audio_encoder.encoders.22.feed_forward.w_1.bias            | Audio_Encoder Encoders 22 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  306 | audio_encoder.encoders.22.feed_forward.w_2.weight          | Audio_Encoder Encoders 22 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  307 | audio_encoder.encoders.22.feed_forward.w_2.bias            | Audio_Encoder Encoders 22 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  308 | audio_encoder.encoders.22.norm1.weight                     | Audio_Encoder Encoders 22 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  309 | audio_encoder.encoders.22.norm1.bias                       | Audio_Encoder Encoders 22 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  310 | audio_encoder.encoders.22.norm2.weight                     | Audio_Encoder Encoders 22 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  311 | audio_encoder.encoders.22.norm2.bias                       | Audio_Encoder Encoders 22 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  312 | audio_encoder.encoders.23.self_attn.linear_out.weight      | Audio_Encoder Encoders 23 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  313 | audio_encoder.encoders.23.self_attn.linear_out.bias        | Audio_Encoder Encoders 23 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  314 | audio_encoder.encoders.23.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 23 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  315 | audio_encoder.encoders.23.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 23 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  316 | audio_encoder.encoders.23.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 23 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  317 | audio_encoder.encoders.23.feed_forward.w_1.weight          | Audio_Encoder Encoders 23 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  318 | audio_encoder.encoders.23.feed_forward.w_1.bias            | Audio_Encoder Encoders 23 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  319 | audio_encoder.encoders.23.feed_forward.w_2.weight          | Audio_Encoder Encoders 23 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  320 | audio_encoder.encoders.23.feed_forward.w_2.bias            | Audio_Encoder Encoders 23 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  321 | audio_encoder.encoders.23.norm1.weight                     | Audio_Encoder Encoders 23 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  322 | audio_encoder.encoders.23.norm1.bias                       | Audio_Encoder Encoders 23 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  323 | audio_encoder.encoders.23.norm2.weight                     | Audio_Encoder Encoders 23 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  324 | audio_encoder.encoders.23.norm2.bias                       | Audio_Encoder Encoders 23 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  325 | audio_encoder.encoders.24.self_attn.linear_out.weight      | Audio_Encoder Encoders 24 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  326 | audio_encoder.encoders.24.self_attn.linear_out.bias        | Audio_Encoder Encoders 24 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  327 | audio_encoder.encoders.24.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 24 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  328 | audio_encoder.encoders.24.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 24 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  329 | audio_encoder.encoders.24.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 24 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  330 | audio_encoder.encoders.24.feed_forward.w_1.weight          | Audio_Encoder Encoders 24 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  331 | audio_encoder.encoders.24.feed_forward.w_1.bias            | Audio_Encoder Encoders 24 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  332 | audio_encoder.encoders.24.feed_forward.w_2.weight          | Audio_Encoder Encoders 24 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  333 | audio_encoder.encoders.24.feed_forward.w_2.bias            | Audio_Encoder Encoders 24 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  334 | audio_encoder.encoders.24.norm1.weight                     | Audio_Encoder Encoders 24 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  335 | audio_encoder.encoders.24.norm1.bias                       | Audio_Encoder Encoders 24 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  336 | audio_encoder.encoders.24.norm2.weight                     | Audio_Encoder Encoders 24 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  337 | audio_encoder.encoders.24.norm2.bias                       | Audio_Encoder Encoders 24 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  338 | audio_encoder.encoders.25.self_attn.linear_out.weight      | Audio_Encoder Encoders 25 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  339 | audio_encoder.encoders.25.self_attn.linear_out.bias        | Audio_Encoder Encoders 25 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  340 | audio_encoder.encoders.25.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 25 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  341 | audio_encoder.encoders.25.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 25 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  342 | audio_encoder.encoders.25.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 25 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  343 | audio_encoder.encoders.25.feed_forward.w_1.weight          | Audio_Encoder Encoders 25 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  344 | audio_encoder.encoders.25.feed_forward.w_1.bias            | Audio_Encoder Encoders 25 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  345 | audio_encoder.encoders.25.feed_forward.w_2.weight          | Audio_Encoder Encoders 25 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  346 | audio_encoder.encoders.25.feed_forward.w_2.bias            | Audio_Encoder Encoders 25 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  347 | audio_encoder.encoders.25.norm1.weight                     | Audio_Encoder Encoders 25 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  348 | audio_encoder.encoders.25.norm1.bias                       | Audio_Encoder Encoders 25 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  349 | audio_encoder.encoders.25.norm2.weight                     | Audio_Encoder Encoders 25 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  350 | audio_encoder.encoders.25.norm2.bias                       | Audio_Encoder Encoders 25 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  351 | audio_encoder.encoders.26.self_attn.linear_out.weight      | Audio_Encoder Encoders 26 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  352 | audio_encoder.encoders.26.self_attn.linear_out.bias        | Audio_Encoder Encoders 26 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  353 | audio_encoder.encoders.26.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 26 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  354 | audio_encoder.encoders.26.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 26 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  355 | audio_encoder.encoders.26.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 26 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  356 | audio_encoder.encoders.26.feed_forward.w_1.weight          | Audio_Encoder Encoders 26 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  357 | audio_encoder.encoders.26.feed_forward.w_1.bias            | Audio_Encoder Encoders 26 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  358 | audio_encoder.encoders.26.feed_forward.w_2.weight          | Audio_Encoder Encoders 26 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  359 | audio_encoder.encoders.26.feed_forward.w_2.bias            | Audio_Encoder Encoders 26 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  360 | audio_encoder.encoders.26.norm1.weight                     | Audio_Encoder Encoders 26 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  361 | audio_encoder.encoders.26.norm1.bias                       | Audio_Encoder Encoders 26 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  362 | audio_encoder.encoders.26.norm2.weight                     | Audio_Encoder Encoders 26 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  363 | audio_encoder.encoders.26.norm2.bias                       | Audio_Encoder Encoders 26 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  364 | audio_encoder.encoders.27.self_attn.linear_out.weight      | Audio_Encoder Encoders 27 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  365 | audio_encoder.encoders.27.self_attn.linear_out.bias        | Audio_Encoder Encoders 27 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  366 | audio_encoder.encoders.27.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 27 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  367 | audio_encoder.encoders.27.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 27 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  368 | audio_encoder.encoders.27.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 27 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  369 | audio_encoder.encoders.27.feed_forward.w_1.weight          | Audio_Encoder Encoders 27 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  370 | audio_encoder.encoders.27.feed_forward.w_1.bias            | Audio_Encoder Encoders 27 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  371 | audio_encoder.encoders.27.feed_forward.w_2.weight          | Audio_Encoder Encoders 27 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  372 | audio_encoder.encoders.27.feed_forward.w_2.bias            | Audio_Encoder Encoders 27 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  373 | audio_encoder.encoders.27.norm1.weight                     | Audio_Encoder Encoders 27 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  374 | audio_encoder.encoders.27.norm1.bias                       | Audio_Encoder Encoders 27 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  375 | audio_encoder.encoders.27.norm2.weight                     | Audio_Encoder Encoders 27 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  376 | audio_encoder.encoders.27.norm2.bias                       | Audio_Encoder Encoders 27 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  377 | audio_encoder.encoders.28.self_attn.linear_out.weight      | Audio_Encoder Encoders 28 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  378 | audio_encoder.encoders.28.self_attn.linear_out.bias        | Audio_Encoder Encoders 28 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  379 | audio_encoder.encoders.28.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 28 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  380 | audio_encoder.encoders.28.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 28 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  381 | audio_encoder.encoders.28.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 28 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  382 | audio_encoder.encoders.28.feed_forward.w_1.weight          | Audio_Encoder Encoders 28 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  383 | audio_encoder.encoders.28.feed_forward.w_1.bias            | Audio_Encoder Encoders 28 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  384 | audio_encoder.encoders.28.feed_forward.w_2.weight          | Audio_Encoder Encoders 28 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  385 | audio_encoder.encoders.28.feed_forward.w_2.bias            | Audio_Encoder Encoders 28 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  386 | audio_encoder.encoders.28.norm1.weight                     | Audio_Encoder Encoders 28 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  387 | audio_encoder.encoders.28.norm1.bias                       | Audio_Encoder Encoders 28 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  388 | audio_encoder.encoders.28.norm2.weight                     | Audio_Encoder Encoders 28 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  389 | audio_encoder.encoders.28.norm2.bias                       | Audio_Encoder Encoders 28 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  390 | audio_encoder.encoders.29.self_attn.linear_out.weight      | Audio_Encoder Encoders 29 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  391 | audio_encoder.encoders.29.self_attn.linear_out.bias        | Audio_Encoder Encoders 29 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  392 | audio_encoder.encoders.29.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 29 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  393 | audio_encoder.encoders.29.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 29 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  394 | audio_encoder.encoders.29.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 29 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  395 | audio_encoder.encoders.29.feed_forward.w_1.weight          | Audio_Encoder Encoders 29 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  396 | audio_encoder.encoders.29.feed_forward.w_1.bias            | Audio_Encoder Encoders 29 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  397 | audio_encoder.encoders.29.feed_forward.w_2.weight          | Audio_Encoder Encoders 29 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  398 | audio_encoder.encoders.29.feed_forward.w_2.bias            | Audio_Encoder Encoders 29 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  399 | audio_encoder.encoders.29.norm1.weight                     | Audio_Encoder Encoders 29 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  400 | audio_encoder.encoders.29.norm1.bias                       | Audio_Encoder Encoders 29 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  401 | audio_encoder.encoders.29.norm2.weight                     | Audio_Encoder Encoders 29 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  402 | audio_encoder.encoders.29.norm2.bias                       | Audio_Encoder Encoders 29 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  403 | audio_encoder.encoders.30.self_attn.linear_out.weight      | Audio_Encoder Encoders 30 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  404 | audio_encoder.encoders.30.self_attn.linear_out.bias        | Audio_Encoder Encoders 30 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  405 | audio_encoder.encoders.30.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 30 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  406 | audio_encoder.encoders.30.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 30 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  407 | audio_encoder.encoders.30.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 30 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  408 | audio_encoder.encoders.30.feed_forward.w_1.weight          | Audio_Encoder Encoders 30 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  409 | audio_encoder.encoders.30.feed_forward.w_1.bias            | Audio_Encoder Encoders 30 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  410 | audio_encoder.encoders.30.feed_forward.w_2.weight          | Audio_Encoder Encoders 30 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  411 | audio_encoder.encoders.30.feed_forward.w_2.bias            | Audio_Encoder Encoders 30 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  412 | audio_encoder.encoders.30.norm1.weight                     | Audio_Encoder Encoders 30 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  413 | audio_encoder.encoders.30.norm1.bias                       | Audio_Encoder Encoders 30 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  414 | audio_encoder.encoders.30.norm2.weight                     | Audio_Encoder Encoders 30 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  415 | audio_encoder.encoders.30.norm2.bias                       | Audio_Encoder Encoders 30 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  416 | audio_encoder.encoders.31.self_attn.linear_out.weight      | Audio_Encoder Encoders 31 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  417 | audio_encoder.encoders.31.self_attn.linear_out.bias        | Audio_Encoder Encoders 31 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  418 | audio_encoder.encoders.31.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 31 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  419 | audio_encoder.encoders.31.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 31 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  420 | audio_encoder.encoders.31.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 31 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  421 | audio_encoder.encoders.31.feed_forward.w_1.weight          | Audio_Encoder Encoders 31 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  422 | audio_encoder.encoders.31.feed_forward.w_1.bias            | Audio_Encoder Encoders 31 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  423 | audio_encoder.encoders.31.feed_forward.w_2.weight          | Audio_Encoder Encoders 31 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  424 | audio_encoder.encoders.31.feed_forward.w_2.bias            | Audio_Encoder Encoders 31 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  425 | audio_encoder.encoders.31.norm1.weight                     | Audio_Encoder Encoders 31 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  426 | audio_encoder.encoders.31.norm1.bias                       | Audio_Encoder Encoders 31 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  427 | audio_encoder.encoders.31.norm2.weight                     | Audio_Encoder Encoders 31 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  428 | audio_encoder.encoders.31.norm2.bias                       | Audio_Encoder Encoders 31 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  429 | audio_encoder.encoders.32.self_attn.linear_out.weight      | Audio_Encoder Encoders 32 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  430 | audio_encoder.encoders.32.self_attn.linear_out.bias        | Audio_Encoder Encoders 32 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  431 | audio_encoder.encoders.32.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 32 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  432 | audio_encoder.encoders.32.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 32 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  433 | audio_encoder.encoders.32.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 32 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  434 | audio_encoder.encoders.32.feed_forward.w_1.weight          | Audio_Encoder Encoders 32 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  435 | audio_encoder.encoders.32.feed_forward.w_1.bias            | Audio_Encoder Encoders 32 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  436 | audio_encoder.encoders.32.feed_forward.w_2.weight          | Audio_Encoder Encoders 32 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  437 | audio_encoder.encoders.32.feed_forward.w_2.bias            | Audio_Encoder Encoders 32 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  438 | audio_encoder.encoders.32.norm1.weight                     | Audio_Encoder Encoders 32 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  439 | audio_encoder.encoders.32.norm1.bias                       | Audio_Encoder Encoders 32 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  440 | audio_encoder.encoders.32.norm2.weight                     | Audio_Encoder Encoders 32 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  441 | audio_encoder.encoders.32.norm2.bias                       | Audio_Encoder Encoders 32 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  442 | audio_encoder.encoders.33.self_attn.linear_out.weight      | Audio_Encoder Encoders 33 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  443 | audio_encoder.encoders.33.self_attn.linear_out.bias        | Audio_Encoder Encoders 33 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  444 | audio_encoder.encoders.33.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 33 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  445 | audio_encoder.encoders.33.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 33 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  446 | audio_encoder.encoders.33.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 33 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  447 | audio_encoder.encoders.33.feed_forward.w_1.weight          | Audio_Encoder Encoders 33 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  448 | audio_encoder.encoders.33.feed_forward.w_1.bias            | Audio_Encoder Encoders 33 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  449 | audio_encoder.encoders.33.feed_forward.w_2.weight          | Audio_Encoder Encoders 33 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  450 | audio_encoder.encoders.33.feed_forward.w_2.bias            | Audio_Encoder Encoders 33 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  451 | audio_encoder.encoders.33.norm1.weight                     | Audio_Encoder Encoders 33 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  452 | audio_encoder.encoders.33.norm1.bias                       | Audio_Encoder Encoders 33 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  453 | audio_encoder.encoders.33.norm2.weight                     | Audio_Encoder Encoders 33 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  454 | audio_encoder.encoders.33.norm2.bias                       | Audio_Encoder Encoders 33 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  455 | audio_encoder.encoders.34.self_attn.linear_out.weight      | Audio_Encoder Encoders 34 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  456 | audio_encoder.encoders.34.self_attn.linear_out.bias        | Audio_Encoder Encoders 34 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  457 | audio_encoder.encoders.34.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 34 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  458 | audio_encoder.encoders.34.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 34 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  459 | audio_encoder.encoders.34.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 34 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  460 | audio_encoder.encoders.34.feed_forward.w_1.weight          | Audio_Encoder Encoders 34 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  461 | audio_encoder.encoders.34.feed_forward.w_1.bias            | Audio_Encoder Encoders 34 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  462 | audio_encoder.encoders.34.feed_forward.w_2.weight          | Audio_Encoder Encoders 34 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  463 | audio_encoder.encoders.34.feed_forward.w_2.bias            | Audio_Encoder Encoders 34 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  464 | audio_encoder.encoders.34.norm1.weight                     | Audio_Encoder Encoders 34 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  465 | audio_encoder.encoders.34.norm1.bias                       | Audio_Encoder Encoders 34 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  466 | audio_encoder.encoders.34.norm2.weight                     | Audio_Encoder Encoders 34 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  467 | audio_encoder.encoders.34.norm2.bias                       | Audio_Encoder Encoders 34 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  468 | audio_encoder.encoders.35.self_attn.linear_out.weight      | Audio_Encoder Encoders 35 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  469 | audio_encoder.encoders.35.self_attn.linear_out.bias        | Audio_Encoder Encoders 35 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  470 | audio_encoder.encoders.35.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 35 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  471 | audio_encoder.encoders.35.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 35 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  472 | audio_encoder.encoders.35.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 35 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  473 | audio_encoder.encoders.35.feed_forward.w_1.weight          | Audio_Encoder Encoders 35 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  474 | audio_encoder.encoders.35.feed_forward.w_1.bias            | Audio_Encoder Encoders 35 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  475 | audio_encoder.encoders.35.feed_forward.w_2.weight          | Audio_Encoder Encoders 35 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  476 | audio_encoder.encoders.35.feed_forward.w_2.bias            | Audio_Encoder Encoders 35 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  477 | audio_encoder.encoders.35.norm1.weight                     | Audio_Encoder Encoders 35 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  478 | audio_encoder.encoders.35.norm1.bias                       | Audio_Encoder Encoders 35 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  479 | audio_encoder.encoders.35.norm2.weight                     | Audio_Encoder Encoders 35 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  480 | audio_encoder.encoders.35.norm2.bias                       | Audio_Encoder Encoders 35 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  481 | audio_encoder.encoders.36.self_attn.linear_out.weight      | Audio_Encoder Encoders 36 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  482 | audio_encoder.encoders.36.self_attn.linear_out.bias        | Audio_Encoder Encoders 36 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  483 | audio_encoder.encoders.36.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 36 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  484 | audio_encoder.encoders.36.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 36 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  485 | audio_encoder.encoders.36.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 36 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  486 | audio_encoder.encoders.36.feed_forward.w_1.weight          | Audio_Encoder Encoders 36 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  487 | audio_encoder.encoders.36.feed_forward.w_1.bias            | Audio_Encoder Encoders 36 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  488 | audio_encoder.encoders.36.feed_forward.w_2.weight          | Audio_Encoder Encoders 36 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  489 | audio_encoder.encoders.36.feed_forward.w_2.bias            | Audio_Encoder Encoders 36 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  490 | audio_encoder.encoders.36.norm1.weight                     | Audio_Encoder Encoders 36 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  491 | audio_encoder.encoders.36.norm1.bias                       | Audio_Encoder Encoders 36 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  492 | audio_encoder.encoders.36.norm2.weight                     | Audio_Encoder Encoders 36 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  493 | audio_encoder.encoders.36.norm2.bias                       | Audio_Encoder Encoders 36 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  494 | audio_encoder.encoders.37.self_attn.linear_out.weight      | Audio_Encoder Encoders 37 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  495 | audio_encoder.encoders.37.self_attn.linear_out.bias        | Audio_Encoder Encoders 37 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  496 | audio_encoder.encoders.37.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 37 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  497 | audio_encoder.encoders.37.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 37 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  498 | audio_encoder.encoders.37.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 37 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  499 | audio_encoder.encoders.37.feed_forward.w_1.weight          | Audio_Encoder Encoders 37 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  500 | audio_encoder.encoders.37.feed_forward.w_1.bias            | Audio_Encoder Encoders 37 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  501 | audio_encoder.encoders.37.feed_forward.w_2.weight          | Audio_Encoder Encoders 37 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  502 | audio_encoder.encoders.37.feed_forward.w_2.bias            | Audio_Encoder Encoders 37 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  503 | audio_encoder.encoders.37.norm1.weight                     | Audio_Encoder Encoders 37 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  504 | audio_encoder.encoders.37.norm1.bias                       | Audio_Encoder Encoders 37 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  505 | audio_encoder.encoders.37.norm2.weight                     | Audio_Encoder Encoders 37 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  506 | audio_encoder.encoders.37.norm2.bias                       | Audio_Encoder Encoders 37 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  507 | audio_encoder.encoders.38.self_attn.linear_out.weight      | Audio_Encoder Encoders 38 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  508 | audio_encoder.encoders.38.self_attn.linear_out.bias        | Audio_Encoder Encoders 38 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  509 | audio_encoder.encoders.38.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 38 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  510 | audio_encoder.encoders.38.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 38 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  511 | audio_encoder.encoders.38.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 38 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  512 | audio_encoder.encoders.38.feed_forward.w_1.weight          | Audio_Encoder Encoders 38 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  513 | audio_encoder.encoders.38.feed_forward.w_1.bias            | Audio_Encoder Encoders 38 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  514 | audio_encoder.encoders.38.feed_forward.w_2.weight          | Audio_Encoder Encoders 38 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  515 | audio_encoder.encoders.38.feed_forward.w_2.bias            | Audio_Encoder Encoders 38 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  516 | audio_encoder.encoders.38.norm1.weight                     | Audio_Encoder Encoders 38 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  517 | audio_encoder.encoders.38.norm1.bias                       | Audio_Encoder Encoders 38 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  518 | audio_encoder.encoders.38.norm2.weight                     | Audio_Encoder Encoders 38 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  519 | audio_encoder.encoders.38.norm2.bias                       | Audio_Encoder Encoders 38 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  520 | audio_encoder.encoders.39.self_attn.linear_out.weight      | Audio_Encoder Encoders 39 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  521 | audio_encoder.encoders.39.self_attn.linear_out.bias        | Audio_Encoder Encoders 39 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  522 | audio_encoder.encoders.39.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 39 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  523 | audio_encoder.encoders.39.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 39 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  524 | audio_encoder.encoders.39.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 39 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  525 | audio_encoder.encoders.39.feed_forward.w_1.weight          | Audio_Encoder Encoders 39 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  526 | audio_encoder.encoders.39.feed_forward.w_1.bias            | Audio_Encoder Encoders 39 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  527 | audio_encoder.encoders.39.feed_forward.w_2.weight          | Audio_Encoder Encoders 39 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  528 | audio_encoder.encoders.39.feed_forward.w_2.bias            | Audio_Encoder Encoders 39 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  529 | audio_encoder.encoders.39.norm1.weight                     | Audio_Encoder Encoders 39 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  530 | audio_encoder.encoders.39.norm1.bias                       | Audio_Encoder Encoders 39 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  531 | audio_encoder.encoders.39.norm2.weight                     | Audio_Encoder Encoders 39 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  532 | audio_encoder.encoders.39.norm2.bias                       | Audio_Encoder Encoders 39 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  533 | audio_encoder.encoders.40.self_attn.linear_out.weight      | Audio_Encoder Encoders 40 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  534 | audio_encoder.encoders.40.self_attn.linear_out.bias        | Audio_Encoder Encoders 40 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  535 | audio_encoder.encoders.40.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 40 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  536 | audio_encoder.encoders.40.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 40 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  537 | audio_encoder.encoders.40.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 40 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  538 | audio_encoder.encoders.40.feed_forward.w_1.weight          | Audio_Encoder Encoders 40 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  539 | audio_encoder.encoders.40.feed_forward.w_1.bias            | Audio_Encoder Encoders 40 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  540 | audio_encoder.encoders.40.feed_forward.w_2.weight          | Audio_Encoder Encoders 40 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  541 | audio_encoder.encoders.40.feed_forward.w_2.bias            | Audio_Encoder Encoders 40 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  542 | audio_encoder.encoders.40.norm1.weight                     | Audio_Encoder Encoders 40 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  543 | audio_encoder.encoders.40.norm1.bias                       | Audio_Encoder Encoders 40 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  544 | audio_encoder.encoders.40.norm2.weight                     | Audio_Encoder Encoders 40 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  545 | audio_encoder.encoders.40.norm2.bias                       | Audio_Encoder Encoders 40 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  546 | audio_encoder.encoders.41.self_attn.linear_out.weight      | Audio_Encoder Encoders 41 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  547 | audio_encoder.encoders.41.self_attn.linear_out.bias        | Audio_Encoder Encoders 41 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  548 | audio_encoder.encoders.41.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 41 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  549 | audio_encoder.encoders.41.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 41 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  550 | audio_encoder.encoders.41.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 41 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  551 | audio_encoder.encoders.41.feed_forward.w_1.weight          | Audio_Encoder Encoders 41 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  552 | audio_encoder.encoders.41.feed_forward.w_1.bias            | Audio_Encoder Encoders 41 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  553 | audio_encoder.encoders.41.feed_forward.w_2.weight          | Audio_Encoder Encoders 41 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  554 | audio_encoder.encoders.41.feed_forward.w_2.bias            | Audio_Encoder Encoders 41 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  555 | audio_encoder.encoders.41.norm1.weight                     | Audio_Encoder Encoders 41 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  556 | audio_encoder.encoders.41.norm1.bias                       | Audio_Encoder Encoders 41 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  557 | audio_encoder.encoders.41.norm2.weight                     | Audio_Encoder Encoders 41 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  558 | audio_encoder.encoders.41.norm2.bias                       | Audio_Encoder Encoders 41 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  559 | audio_encoder.encoders.42.self_attn.linear_out.weight      | Audio_Encoder Encoders 42 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  560 | audio_encoder.encoders.42.self_attn.linear_out.bias        | Audio_Encoder Encoders 42 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  561 | audio_encoder.encoders.42.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 42 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  562 | audio_encoder.encoders.42.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 42 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  563 | audio_encoder.encoders.42.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 42 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  564 | audio_encoder.encoders.42.feed_forward.w_1.weight          | Audio_Encoder Encoders 42 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  565 | audio_encoder.encoders.42.feed_forward.w_1.bias            | Audio_Encoder Encoders 42 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  566 | audio_encoder.encoders.42.feed_forward.w_2.weight          | Audio_Encoder Encoders 42 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  567 | audio_encoder.encoders.42.feed_forward.w_2.bias            | Audio_Encoder Encoders 42 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  568 | audio_encoder.encoders.42.norm1.weight                     | Audio_Encoder Encoders 42 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  569 | audio_encoder.encoders.42.norm1.bias                       | Audio_Encoder Encoders 42 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  570 | audio_encoder.encoders.42.norm2.weight                     | Audio_Encoder Encoders 42 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  571 | audio_encoder.encoders.42.norm2.bias                       | Audio_Encoder Encoders 42 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  572 | audio_encoder.encoders.43.self_attn.linear_out.weight      | Audio_Encoder Encoders 43 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  573 | audio_encoder.encoders.43.self_attn.linear_out.bias        | Audio_Encoder Encoders 43 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  574 | audio_encoder.encoders.43.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 43 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  575 | audio_encoder.encoders.43.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 43 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  576 | audio_encoder.encoders.43.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 43 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  577 | audio_encoder.encoders.43.feed_forward.w_1.weight          | Audio_Encoder Encoders 43 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  578 | audio_encoder.encoders.43.feed_forward.w_1.bias            | Audio_Encoder Encoders 43 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  579 | audio_encoder.encoders.43.feed_forward.w_2.weight          | Audio_Encoder Encoders 43 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  580 | audio_encoder.encoders.43.feed_forward.w_2.bias            | Audio_Encoder Encoders 43 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  581 | audio_encoder.encoders.43.norm1.weight                     | Audio_Encoder Encoders 43 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  582 | audio_encoder.encoders.43.norm1.bias                       | Audio_Encoder Encoders 43 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  583 | audio_encoder.encoders.43.norm2.weight                     | Audio_Encoder Encoders 43 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  584 | audio_encoder.encoders.43.norm2.bias                       | Audio_Encoder Encoders 43 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  585 | audio_encoder.encoders.44.self_attn.linear_out.weight      | Audio_Encoder Encoders 44 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  586 | audio_encoder.encoders.44.self_attn.linear_out.bias        | Audio_Encoder Encoders 44 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  587 | audio_encoder.encoders.44.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 44 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  588 | audio_encoder.encoders.44.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 44 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  589 | audio_encoder.encoders.44.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 44 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  590 | audio_encoder.encoders.44.feed_forward.w_1.weight          | Audio_Encoder Encoders 44 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  591 | audio_encoder.encoders.44.feed_forward.w_1.bias            | Audio_Encoder Encoders 44 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  592 | audio_encoder.encoders.44.feed_forward.w_2.weight          | Audio_Encoder Encoders 44 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  593 | audio_encoder.encoders.44.feed_forward.w_2.bias            | Audio_Encoder Encoders 44 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  594 | audio_encoder.encoders.44.norm1.weight                     | Audio_Encoder Encoders 44 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  595 | audio_encoder.encoders.44.norm1.bias                       | Audio_Encoder Encoders 44 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  596 | audio_encoder.encoders.44.norm2.weight                     | Audio_Encoder Encoders 44 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  597 | audio_encoder.encoders.44.norm2.bias                       | Audio_Encoder Encoders 44 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  598 | audio_encoder.encoders.45.self_attn.linear_out.weight      | Audio_Encoder Encoders 45 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  599 | audio_encoder.encoders.45.self_attn.linear_out.bias        | Audio_Encoder Encoders 45 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  600 | audio_encoder.encoders.45.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 45 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  601 | audio_encoder.encoders.45.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 45 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  602 | audio_encoder.encoders.45.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 45 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  603 | audio_encoder.encoders.45.feed_forward.w_1.weight          | Audio_Encoder Encoders 45 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  604 | audio_encoder.encoders.45.feed_forward.w_1.bias            | Audio_Encoder Encoders 45 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  605 | audio_encoder.encoders.45.feed_forward.w_2.weight          | Audio_Encoder Encoders 45 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  606 | audio_encoder.encoders.45.feed_forward.w_2.bias            | Audio_Encoder Encoders 45 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  607 | audio_encoder.encoders.45.norm1.weight                     | Audio_Encoder Encoders 45 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  608 | audio_encoder.encoders.45.norm1.bias                       | Audio_Encoder Encoders 45 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  609 | audio_encoder.encoders.45.norm2.weight                     | Audio_Encoder Encoders 45 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  610 | audio_encoder.encoders.45.norm2.bias                       | Audio_Encoder Encoders 45 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  611 | audio_encoder.encoders.46.self_attn.linear_out.weight      | Audio_Encoder Encoders 46 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  612 | audio_encoder.encoders.46.self_attn.linear_out.bias        | Audio_Encoder Encoders 46 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  613 | audio_encoder.encoders.46.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 46 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  614 | audio_encoder.encoders.46.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 46 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  615 | audio_encoder.encoders.46.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 46 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  616 | audio_encoder.encoders.46.feed_forward.w_1.weight          | Audio_Encoder Encoders 46 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  617 | audio_encoder.encoders.46.feed_forward.w_1.bias            | Audio_Encoder Encoders 46 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  618 | audio_encoder.encoders.46.feed_forward.w_2.weight          | Audio_Encoder Encoders 46 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  619 | audio_encoder.encoders.46.feed_forward.w_2.bias            | Audio_Encoder Encoders 46 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  620 | audio_encoder.encoders.46.norm1.weight                     | Audio_Encoder Encoders 46 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  621 | audio_encoder.encoders.46.norm1.bias                       | Audio_Encoder Encoders 46 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  622 | audio_encoder.encoders.46.norm2.weight                     | Audio_Encoder Encoders 46 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  623 | audio_encoder.encoders.46.norm2.bias                       | Audio_Encoder Encoders 46 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  624 | audio_encoder.encoders.47.self_attn.linear_out.weight      | Audio_Encoder Encoders 47 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  625 | audio_encoder.encoders.47.self_attn.linear_out.bias        | Audio_Encoder Encoders 47 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  626 | audio_encoder.encoders.47.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 47 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  627 | audio_encoder.encoders.47.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 47 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  628 | audio_encoder.encoders.47.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 47 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  629 | audio_encoder.encoders.47.feed_forward.w_1.weight          | Audio_Encoder Encoders 47 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  630 | audio_encoder.encoders.47.feed_forward.w_1.bias            | Audio_Encoder Encoders 47 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  631 | audio_encoder.encoders.47.feed_forward.w_2.weight          | Audio_Encoder Encoders 47 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  632 | audio_encoder.encoders.47.feed_forward.w_2.bias            | Audio_Encoder Encoders 47 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  633 | audio_encoder.encoders.47.norm1.weight                     | Audio_Encoder Encoders 47 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  634 | audio_encoder.encoders.47.norm1.bias                       | Audio_Encoder Encoders 47 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  635 | audio_encoder.encoders.47.norm2.weight                     | Audio_Encoder Encoders 47 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  636 | audio_encoder.encoders.47.norm2.bias                       | Audio_Encoder Encoders 47 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  637 | audio_encoder.encoders.48.self_attn.linear_out.weight      | Audio_Encoder Encoders 48 Self_Attn Linear_Out (W)      | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  638 | audio_encoder.encoders.48.self_attn.linear_out.bias        | Audio_Encoder Encoders 48 Self_Attn Linear_Out (B)      | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  639 | audio_encoder.encoders.48.self_attn.linear_q_k_v.weight    | Audio_Encoder Encoders 48 Self_Attn Linear_Q_K_V (W)    | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  640 | audio_encoder.encoders.48.self_attn.linear_q_k_v.bias      | Audio_Encoder Encoders 48 Self_Attn Linear_Q_K_V (B)    | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  641 | audio_encoder.encoders.48.self_attn.fsmn_block.weight      | Audio_Encoder Encoders 48 Self_Attn Fsmn_Block (W)      | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  642 | audio_encoder.encoders.48.feed_forward.w_1.weight          | Audio_Encoder Encoders 48 Feed_Forward W_1 (W)          | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  643 | audio_encoder.encoders.48.feed_forward.w_1.bias            | Audio_Encoder Encoders 48 Feed_Forward W_1 (B)          | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  644 | audio_encoder.encoders.48.feed_forward.w_2.weight          | Audio_Encoder Encoders 48 Feed_Forward W_2 (W)          | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  645 | audio_encoder.encoders.48.feed_forward.w_2.bias            | Audio_Encoder Encoders 48 Feed_Forward W_2 (B)          | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  646 | audio_encoder.encoders.48.norm1.weight                     | Audio_Encoder Encoders 48 Norm1 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  647 | audio_encoder.encoders.48.norm1.bias                       | Audio_Encoder Encoders 48 Norm1 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  648 | audio_encoder.encoders.48.norm2.weight                     | Audio_Encoder Encoders 48 Norm2 (W)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  649 | audio_encoder.encoders.48.norm2.bias                       | Audio_Encoder Encoders 48 Norm2 (B)                     | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  650 | audio_encoder.tp_encoders.0.self_attn.linear_out.weight    | Audio_Encoder Tp_Encoders 0 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  651 | audio_encoder.tp_encoders.0.self_attn.linear_out.bias      | Audio_Encoder Tp_Encoders 0 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  652 | audio_encoder.tp_encoders.0.self_attn.linear_q_k_v.weight  | Audio_Encoder Tp_Encoders 0 Self_Attn Linear_Q_K_V (W)  | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  653 | audio_encoder.tp_encoders.0.self_attn.linear_q_k_v.bias    | Audio_Encoder Tp_Encoders 0 Self_Attn Linear_Q_K_V (B)  | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  654 | audio_encoder.tp_encoders.0.self_attn.fsmn_block.weight    | Audio_Encoder Tp_Encoders 0 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  655 | audio_encoder.tp_encoders.0.feed_forward.w_1.weight        | Audio_Encoder Tp_Encoders 0 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  656 | audio_encoder.tp_encoders.0.feed_forward.w_1.bias          | Audio_Encoder Tp_Encoders 0 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  657 | audio_encoder.tp_encoders.0.feed_forward.w_2.weight        | Audio_Encoder Tp_Encoders 0 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  658 | audio_encoder.tp_encoders.0.feed_forward.w_2.bias          | Audio_Encoder Tp_Encoders 0 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  659 | audio_encoder.tp_encoders.0.norm1.weight                   | Audio_Encoder Tp_Encoders 0 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  660 | audio_encoder.tp_encoders.0.norm1.bias                     | Audio_Encoder Tp_Encoders 0 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  661 | audio_encoder.tp_encoders.0.norm2.weight                   | Audio_Encoder Tp_Encoders 0 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  662 | audio_encoder.tp_encoders.0.norm2.bias                     | Audio_Encoder Tp_Encoders 0 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  663 | audio_encoder.tp_encoders.1.self_attn.linear_out.weight    | Audio_Encoder Tp_Encoders 1 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  664 | audio_encoder.tp_encoders.1.self_attn.linear_out.bias      | Audio_Encoder Tp_Encoders 1 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  665 | audio_encoder.tp_encoders.1.self_attn.linear_q_k_v.weight  | Audio_Encoder Tp_Encoders 1 Self_Attn Linear_Q_K_V (W)  | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  666 | audio_encoder.tp_encoders.1.self_attn.linear_q_k_v.bias    | Audio_Encoder Tp_Encoders 1 Self_Attn Linear_Q_K_V (B)  | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  667 | audio_encoder.tp_encoders.1.self_attn.fsmn_block.weight    | Audio_Encoder Tp_Encoders 1 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  668 | audio_encoder.tp_encoders.1.feed_forward.w_1.weight        | Audio_Encoder Tp_Encoders 1 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  669 | audio_encoder.tp_encoders.1.feed_forward.w_1.bias          | Audio_Encoder Tp_Encoders 1 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  670 | audio_encoder.tp_encoders.1.feed_forward.w_2.weight        | Audio_Encoder Tp_Encoders 1 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  671 | audio_encoder.tp_encoders.1.feed_forward.w_2.bias          | Audio_Encoder Tp_Encoders 1 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  672 | audio_encoder.tp_encoders.1.norm1.weight                   | Audio_Encoder Tp_Encoders 1 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  673 | audio_encoder.tp_encoders.1.norm1.bias                     | Audio_Encoder Tp_Encoders 1 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  674 | audio_encoder.tp_encoders.1.norm2.weight                   | Audio_Encoder Tp_Encoders 1 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  675 | audio_encoder.tp_encoders.1.norm2.bias                     | Audio_Encoder Tp_Encoders 1 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  676 | audio_encoder.tp_encoders.2.self_attn.linear_out.weight    | Audio_Encoder Tp_Encoders 2 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  677 | audio_encoder.tp_encoders.2.self_attn.linear_out.bias      | Audio_Encoder Tp_Encoders 2 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  678 | audio_encoder.tp_encoders.2.self_attn.linear_q_k_v.weight  | Audio_Encoder Tp_Encoders 2 Self_Attn Linear_Q_K_V (W)  | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  679 | audio_encoder.tp_encoders.2.self_attn.linear_q_k_v.bias    | Audio_Encoder Tp_Encoders 2 Self_Attn Linear_Q_K_V (B)  | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  680 | audio_encoder.tp_encoders.2.self_attn.fsmn_block.weight    | Audio_Encoder Tp_Encoders 2 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  681 | audio_encoder.tp_encoders.2.feed_forward.w_1.weight        | Audio_Encoder Tp_Encoders 2 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  682 | audio_encoder.tp_encoders.2.feed_forward.w_1.bias          | Audio_Encoder Tp_Encoders 2 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  683 | audio_encoder.tp_encoders.2.feed_forward.w_2.weight        | Audio_Encoder Tp_Encoders 2 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  684 | audio_encoder.tp_encoders.2.feed_forward.w_2.bias          | Audio_Encoder Tp_Encoders 2 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  685 | audio_encoder.tp_encoders.2.norm1.weight                   | Audio_Encoder Tp_Encoders 2 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  686 | audio_encoder.tp_encoders.2.norm1.bias                     | Audio_Encoder Tp_Encoders 2 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  687 | audio_encoder.tp_encoders.2.norm2.weight                   | Audio_Encoder Tp_Encoders 2 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  688 | audio_encoder.tp_encoders.2.norm2.bias                     | Audio_Encoder Tp_Encoders 2 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  689 | audio_encoder.tp_encoders.3.self_attn.linear_out.weight    | Audio_Encoder Tp_Encoders 3 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  690 | audio_encoder.tp_encoders.3.self_attn.linear_out.bias      | Audio_Encoder Tp_Encoders 3 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  691 | audio_encoder.tp_encoders.3.self_attn.linear_q_k_v.weight  | Audio_Encoder Tp_Encoders 3 Self_Attn Linear_Q_K_V (W)  | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  692 | audio_encoder.tp_encoders.3.self_attn.linear_q_k_v.bias    | Audio_Encoder Tp_Encoders 3 Self_Attn Linear_Q_K_V (B)  | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  693 | audio_encoder.tp_encoders.3.self_attn.fsmn_block.weight    | Audio_Encoder Tp_Encoders 3 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  694 | audio_encoder.tp_encoders.3.feed_forward.w_1.weight        | Audio_Encoder Tp_Encoders 3 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  695 | audio_encoder.tp_encoders.3.feed_forward.w_1.bias          | Audio_Encoder Tp_Encoders 3 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  696 | audio_encoder.tp_encoders.3.feed_forward.w_2.weight        | Audio_Encoder Tp_Encoders 3 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  697 | audio_encoder.tp_encoders.3.feed_forward.w_2.bias          | Audio_Encoder Tp_Encoders 3 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  698 | audio_encoder.tp_encoders.3.norm1.weight                   | Audio_Encoder Tp_Encoders 3 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  699 | audio_encoder.tp_encoders.3.norm1.bias                     | Audio_Encoder Tp_Encoders 3 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  700 | audio_encoder.tp_encoders.3.norm2.weight                   | Audio_Encoder Tp_Encoders 3 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  701 | audio_encoder.tp_encoders.3.norm2.bias                     | Audio_Encoder Tp_Encoders 3 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  702 | audio_encoder.tp_encoders.4.self_attn.linear_out.weight    | Audio_Encoder Tp_Encoders 4 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  703 | audio_encoder.tp_encoders.4.self_attn.linear_out.bias      | Audio_Encoder Tp_Encoders 4 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  704 | audio_encoder.tp_encoders.4.self_attn.linear_q_k_v.weight  | Audio_Encoder Tp_Encoders 4 Self_Attn Linear_Q_K_V (W)  | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  705 | audio_encoder.tp_encoders.4.self_attn.linear_q_k_v.bias    | Audio_Encoder Tp_Encoders 4 Self_Attn Linear_Q_K_V (B)  | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  706 | audio_encoder.tp_encoders.4.self_attn.fsmn_block.weight    | Audio_Encoder Tp_Encoders 4 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  707 | audio_encoder.tp_encoders.4.feed_forward.w_1.weight        | Audio_Encoder Tp_Encoders 4 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  708 | audio_encoder.tp_encoders.4.feed_forward.w_1.bias          | Audio_Encoder Tp_Encoders 4 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  709 | audio_encoder.tp_encoders.4.feed_forward.w_2.weight        | Audio_Encoder Tp_Encoders 4 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  710 | audio_encoder.tp_encoders.4.feed_forward.w_2.bias          | Audio_Encoder Tp_Encoders 4 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  711 | audio_encoder.tp_encoders.4.norm1.weight                   | Audio_Encoder Tp_Encoders 4 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  712 | audio_encoder.tp_encoders.4.norm1.bias                     | Audio_Encoder Tp_Encoders 4 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  713 | audio_encoder.tp_encoders.4.norm2.weight                   | Audio_Encoder Tp_Encoders 4 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  714 | audio_encoder.tp_encoders.4.norm2.bias                     | Audio_Encoder Tp_Encoders 4 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  715 | audio_encoder.tp_encoders.5.self_attn.linear_out.weight    | Audio_Encoder Tp_Encoders 5 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  716 | audio_encoder.tp_encoders.5.self_attn.linear_out.bias      | Audio_Encoder Tp_Encoders 5 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  717 | audio_encoder.tp_encoders.5.self_attn.linear_q_k_v.weight  | Audio_Encoder Tp_Encoders 5 Self_Attn Linear_Q_K_V (W)  | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  718 | audio_encoder.tp_encoders.5.self_attn.linear_q_k_v.bias    | Audio_Encoder Tp_Encoders 5 Self_Attn Linear_Q_K_V (B)  | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  719 | audio_encoder.tp_encoders.5.self_attn.fsmn_block.weight    | Audio_Encoder Tp_Encoders 5 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  720 | audio_encoder.tp_encoders.5.feed_forward.w_1.weight        | Audio_Encoder Tp_Encoders 5 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  721 | audio_encoder.tp_encoders.5.feed_forward.w_1.bias          | Audio_Encoder Tp_Encoders 5 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  722 | audio_encoder.tp_encoders.5.feed_forward.w_2.weight        | Audio_Encoder Tp_Encoders 5 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  723 | audio_encoder.tp_encoders.5.feed_forward.w_2.bias          | Audio_Encoder Tp_Encoders 5 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  724 | audio_encoder.tp_encoders.5.norm1.weight                   | Audio_Encoder Tp_Encoders 5 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  725 | audio_encoder.tp_encoders.5.norm1.bias                     | Audio_Encoder Tp_Encoders 5 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  726 | audio_encoder.tp_encoders.5.norm2.weight                   | Audio_Encoder Tp_Encoders 5 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  727 | audio_encoder.tp_encoders.5.norm2.bias                     | Audio_Encoder Tp_Encoders 5 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  728 | audio_encoder.tp_encoders.6.self_attn.linear_out.weight    | Audio_Encoder Tp_Encoders 6 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  729 | audio_encoder.tp_encoders.6.self_attn.linear_out.bias      | Audio_Encoder Tp_Encoders 6 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  730 | audio_encoder.tp_encoders.6.self_attn.linear_q_k_v.weight  | Audio_Encoder Tp_Encoders 6 Self_Attn Linear_Q_K_V (W)  | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  731 | audio_encoder.tp_encoders.6.self_attn.linear_q_k_v.bias    | Audio_Encoder Tp_Encoders 6 Self_Attn Linear_Q_K_V (B)  | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  732 | audio_encoder.tp_encoders.6.self_attn.fsmn_block.weight    | Audio_Encoder Tp_Encoders 6 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  733 | audio_encoder.tp_encoders.6.feed_forward.w_1.weight        | Audio_Encoder Tp_Encoders 6 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  734 | audio_encoder.tp_encoders.6.feed_forward.w_1.bias          | Audio_Encoder Tp_Encoders 6 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  735 | audio_encoder.tp_encoders.6.feed_forward.w_2.weight        | Audio_Encoder Tp_Encoders 6 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  736 | audio_encoder.tp_encoders.6.feed_forward.w_2.bias          | Audio_Encoder Tp_Encoders 6 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  737 | audio_encoder.tp_encoders.6.norm1.weight                   | Audio_Encoder Tp_Encoders 6 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  738 | audio_encoder.tp_encoders.6.norm1.bias                     | Audio_Encoder Tp_Encoders 6 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  739 | audio_encoder.tp_encoders.6.norm2.weight                   | Audio_Encoder Tp_Encoders 6 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  740 | audio_encoder.tp_encoders.6.norm2.bias                     | Audio_Encoder Tp_Encoders 6 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  741 | audio_encoder.tp_encoders.7.self_attn.linear_out.weight    | Audio_Encoder Tp_Encoders 7 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  742 | audio_encoder.tp_encoders.7.self_attn.linear_out.bias      | Audio_Encoder Tp_Encoders 7 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  743 | audio_encoder.tp_encoders.7.self_attn.linear_q_k_v.weight  | Audio_Encoder Tp_Encoders 7 Self_Attn Linear_Q_K_V (W)  | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  744 | audio_encoder.tp_encoders.7.self_attn.linear_q_k_v.bias    | Audio_Encoder Tp_Encoders 7 Self_Attn Linear_Q_K_V (B)  | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  745 | audio_encoder.tp_encoders.7.self_attn.fsmn_block.weight    | Audio_Encoder Tp_Encoders 7 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  746 | audio_encoder.tp_encoders.7.feed_forward.w_1.weight        | Audio_Encoder Tp_Encoders 7 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  747 | audio_encoder.tp_encoders.7.feed_forward.w_1.bias          | Audio_Encoder Tp_Encoders 7 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  748 | audio_encoder.tp_encoders.7.feed_forward.w_2.weight        | Audio_Encoder Tp_Encoders 7 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  749 | audio_encoder.tp_encoders.7.feed_forward.w_2.bias          | Audio_Encoder Tp_Encoders 7 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  750 | audio_encoder.tp_encoders.7.norm1.weight                   | Audio_Encoder Tp_Encoders 7 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  751 | audio_encoder.tp_encoders.7.norm1.bias                     | Audio_Encoder Tp_Encoders 7 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  752 | audio_encoder.tp_encoders.7.norm2.weight                   | Audio_Encoder Tp_Encoders 7 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  753 | audio_encoder.tp_encoders.7.norm2.bias                     | Audio_Encoder Tp_Encoders 7 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  754 | audio_encoder.tp_encoders.8.self_attn.linear_out.weight    | Audio_Encoder Tp_Encoders 8 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  755 | audio_encoder.tp_encoders.8.self_attn.linear_out.bias      | Audio_Encoder Tp_Encoders 8 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  756 | audio_encoder.tp_encoders.8.self_attn.linear_q_k_v.weight  | Audio_Encoder Tp_Encoders 8 Self_Attn Linear_Q_K_V (W)  | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  757 | audio_encoder.tp_encoders.8.self_attn.linear_q_k_v.bias    | Audio_Encoder Tp_Encoders 8 Self_Attn Linear_Q_K_V (B)  | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  758 | audio_encoder.tp_encoders.8.self_attn.fsmn_block.weight    | Audio_Encoder Tp_Encoders 8 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  759 | audio_encoder.tp_encoders.8.feed_forward.w_1.weight        | Audio_Encoder Tp_Encoders 8 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  760 | audio_encoder.tp_encoders.8.feed_forward.w_1.bias          | Audio_Encoder Tp_Encoders 8 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  761 | audio_encoder.tp_encoders.8.feed_forward.w_2.weight        | Audio_Encoder Tp_Encoders 8 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  762 | audio_encoder.tp_encoders.8.feed_forward.w_2.bias          | Audio_Encoder Tp_Encoders 8 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  763 | audio_encoder.tp_encoders.8.norm1.weight                   | Audio_Encoder Tp_Encoders 8 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  764 | audio_encoder.tp_encoders.8.norm1.bias                     | Audio_Encoder Tp_Encoders 8 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  765 | audio_encoder.tp_encoders.8.norm2.weight                   | Audio_Encoder Tp_Encoders 8 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  766 | audio_encoder.tp_encoders.8.norm2.bias                     | Audio_Encoder Tp_Encoders 8 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  767 | audio_encoder.tp_encoders.9.self_attn.linear_out.weight    | Audio_Encoder Tp_Encoders 9 Self_Attn Linear_Out (W)    | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  768 | audio_encoder.tp_encoders.9.self_attn.linear_out.bias      | Audio_Encoder Tp_Encoders 9 Self_Attn Linear_Out (B)    | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  769 | audio_encoder.tp_encoders.9.self_attn.linear_q_k_v.weight  | Audio_Encoder Tp_Encoders 9 Self_Attn Linear_Q_K_V (W)  | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  770 | audio_encoder.tp_encoders.9.self_attn.linear_q_k_v.bias    | Audio_Encoder Tp_Encoders 9 Self_Attn Linear_Q_K_V (B)  | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  771 | audio_encoder.tp_encoders.9.self_attn.fsmn_block.weight    | Audio_Encoder Tp_Encoders 9 Self_Attn Fsmn_Block (W)    | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  772 | audio_encoder.tp_encoders.9.feed_forward.w_1.weight        | Audio_Encoder Tp_Encoders 9 Feed_Forward W_1 (W)        | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  773 | audio_encoder.tp_encoders.9.feed_forward.w_1.bias          | Audio_Encoder Tp_Encoders 9 Feed_Forward W_1 (B)        | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  774 | audio_encoder.tp_encoders.9.feed_forward.w_2.weight        | Audio_Encoder Tp_Encoders 9 Feed_Forward W_2 (W)        | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  775 | audio_encoder.tp_encoders.9.feed_forward.w_2.bias          | Audio_Encoder Tp_Encoders 9 Feed_Forward W_2 (B)        | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  776 | audio_encoder.tp_encoders.9.norm1.weight                   | Audio_Encoder Tp_Encoders 9 Norm1 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  777 | audio_encoder.tp_encoders.9.norm1.bias                     | Audio_Encoder Tp_Encoders 9 Norm1 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  778 | audio_encoder.tp_encoders.9.norm2.weight                   | Audio_Encoder Tp_Encoders 9 Norm2 (W)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  779 | audio_encoder.tp_encoders.9.norm2.bias                     | Audio_Encoder Tp_Encoders 9 Norm2 (B)                   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  780 | audio_encoder.tp_encoders.10.self_attn.linear_out.weight   | Audio_Encoder Tp_Encoders 10 Self_Attn Linear_Out (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  781 | audio_encoder.tp_encoders.10.self_attn.linear_out.bias     | Audio_Encoder Tp_Encoders 10 Self_Attn Linear_Out (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  782 | audio_encoder.tp_encoders.10.self_attn.linear_q_k_v.weight | Audio_Encoder Tp_Encoders 10 Self_Attn Linear_Q_K_V (W) | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  783 | audio_encoder.tp_encoders.10.self_attn.linear_q_k_v.bias   | Audio_Encoder Tp_Encoders 10 Self_Attn Linear_Q_K_V (B) | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  784 | audio_encoder.tp_encoders.10.self_attn.fsmn_block.weight   | Audio_Encoder Tp_Encoders 10 Self_Attn Fsmn_Block (W)   | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  785 | audio_encoder.tp_encoders.10.feed_forward.w_1.weight       | Audio_Encoder Tp_Encoders 10 Feed_Forward W_1 (W)       | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  786 | audio_encoder.tp_encoders.10.feed_forward.w_1.bias         | Audio_Encoder Tp_Encoders 10 Feed_Forward W_1 (B)       | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  787 | audio_encoder.tp_encoders.10.feed_forward.w_2.weight       | Audio_Encoder Tp_Encoders 10 Feed_Forward W_2 (W)       | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  788 | audio_encoder.tp_encoders.10.feed_forward.w_2.bias         | Audio_Encoder Tp_Encoders 10 Feed_Forward W_2 (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  789 | audio_encoder.tp_encoders.10.norm1.weight                  | Audio_Encoder Tp_Encoders 10 Norm1 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  790 | audio_encoder.tp_encoders.10.norm1.bias                    | Audio_Encoder Tp_Encoders 10 Norm1 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  791 | audio_encoder.tp_encoders.10.norm2.weight                  | Audio_Encoder Tp_Encoders 10 Norm2 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  792 | audio_encoder.tp_encoders.10.norm2.bias                    | Audio_Encoder Tp_Encoders 10 Norm2 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  793 | audio_encoder.tp_encoders.11.self_attn.linear_out.weight   | Audio_Encoder Tp_Encoders 11 Self_Attn Linear_Out (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  794 | audio_encoder.tp_encoders.11.self_attn.linear_out.bias     | Audio_Encoder Tp_Encoders 11 Self_Attn Linear_Out (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  795 | audio_encoder.tp_encoders.11.self_attn.linear_q_k_v.weight | Audio_Encoder Tp_Encoders 11 Self_Attn Linear_Q_K_V (W) | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  796 | audio_encoder.tp_encoders.11.self_attn.linear_q_k_v.bias   | Audio_Encoder Tp_Encoders 11 Self_Attn Linear_Q_K_V (B) | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  797 | audio_encoder.tp_encoders.11.self_attn.fsmn_block.weight   | Audio_Encoder Tp_Encoders 11 Self_Attn Fsmn_Block (W)   | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  798 | audio_encoder.tp_encoders.11.feed_forward.w_1.weight       | Audio_Encoder Tp_Encoders 11 Feed_Forward W_1 (W)       | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  799 | audio_encoder.tp_encoders.11.feed_forward.w_1.bias         | Audio_Encoder Tp_Encoders 11 Feed_Forward W_1 (B)       | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  800 | audio_encoder.tp_encoders.11.feed_forward.w_2.weight       | Audio_Encoder Tp_Encoders 11 Feed_Forward W_2 (W)       | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  801 | audio_encoder.tp_encoders.11.feed_forward.w_2.bias         | Audio_Encoder Tp_Encoders 11 Feed_Forward W_2 (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  802 | audio_encoder.tp_encoders.11.norm1.weight                  | Audio_Encoder Tp_Encoders 11 Norm1 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  803 | audio_encoder.tp_encoders.11.norm1.bias                    | Audio_Encoder Tp_Encoders 11 Norm1 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  804 | audio_encoder.tp_encoders.11.norm2.weight                  | Audio_Encoder Tp_Encoders 11 Norm2 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  805 | audio_encoder.tp_encoders.11.norm2.bias                    | Audio_Encoder Tp_Encoders 11 Norm2 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  806 | audio_encoder.tp_encoders.12.self_attn.linear_out.weight   | Audio_Encoder Tp_Encoders 12 Self_Attn Linear_Out (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  807 | audio_encoder.tp_encoders.12.self_attn.linear_out.bias     | Audio_Encoder Tp_Encoders 12 Self_Attn Linear_Out (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  808 | audio_encoder.tp_encoders.12.self_attn.linear_q_k_v.weight | Audio_Encoder Tp_Encoders 12 Self_Attn Linear_Q_K_V (W) | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  809 | audio_encoder.tp_encoders.12.self_attn.linear_q_k_v.bias   | Audio_Encoder Tp_Encoders 12 Self_Attn Linear_Q_K_V (B) | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  810 | audio_encoder.tp_encoders.12.self_attn.fsmn_block.weight   | Audio_Encoder Tp_Encoders 12 Self_Attn Fsmn_Block (W)   | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  811 | audio_encoder.tp_encoders.12.feed_forward.w_1.weight       | Audio_Encoder Tp_Encoders 12 Feed_Forward W_1 (W)       | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  812 | audio_encoder.tp_encoders.12.feed_forward.w_1.bias         | Audio_Encoder Tp_Encoders 12 Feed_Forward W_1 (B)       | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  813 | audio_encoder.tp_encoders.12.feed_forward.w_2.weight       | Audio_Encoder Tp_Encoders 12 Feed_Forward W_2 (W)       | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  814 | audio_encoder.tp_encoders.12.feed_forward.w_2.bias         | Audio_Encoder Tp_Encoders 12 Feed_Forward W_2 (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  815 | audio_encoder.tp_encoders.12.norm1.weight                  | Audio_Encoder Tp_Encoders 12 Norm1 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  816 | audio_encoder.tp_encoders.12.norm1.bias                    | Audio_Encoder Tp_Encoders 12 Norm1 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  817 | audio_encoder.tp_encoders.12.norm2.weight                  | Audio_Encoder Tp_Encoders 12 Norm2 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  818 | audio_encoder.tp_encoders.12.norm2.bias                    | Audio_Encoder Tp_Encoders 12 Norm2 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  819 | audio_encoder.tp_encoders.13.self_attn.linear_out.weight   | Audio_Encoder Tp_Encoders 13 Self_Attn Linear_Out (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  820 | audio_encoder.tp_encoders.13.self_attn.linear_out.bias     | Audio_Encoder Tp_Encoders 13 Self_Attn Linear_Out (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  821 | audio_encoder.tp_encoders.13.self_attn.linear_q_k_v.weight | Audio_Encoder Tp_Encoders 13 Self_Attn Linear_Q_K_V (W) | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  822 | audio_encoder.tp_encoders.13.self_attn.linear_q_k_v.bias   | Audio_Encoder Tp_Encoders 13 Self_Attn Linear_Q_K_V (B) | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  823 | audio_encoder.tp_encoders.13.self_attn.fsmn_block.weight   | Audio_Encoder Tp_Encoders 13 Self_Attn Fsmn_Block (W)   | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  824 | audio_encoder.tp_encoders.13.feed_forward.w_1.weight       | Audio_Encoder Tp_Encoders 13 Feed_Forward W_1 (W)       | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  825 | audio_encoder.tp_encoders.13.feed_forward.w_1.bias         | Audio_Encoder Tp_Encoders 13 Feed_Forward W_1 (B)       | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  826 | audio_encoder.tp_encoders.13.feed_forward.w_2.weight       | Audio_Encoder Tp_Encoders 13 Feed_Forward W_2 (W)       | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  827 | audio_encoder.tp_encoders.13.feed_forward.w_2.bias         | Audio_Encoder Tp_Encoders 13 Feed_Forward W_2 (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  828 | audio_encoder.tp_encoders.13.norm1.weight                  | Audio_Encoder Tp_Encoders 13 Norm1 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  829 | audio_encoder.tp_encoders.13.norm1.bias                    | Audio_Encoder Tp_Encoders 13 Norm1 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  830 | audio_encoder.tp_encoders.13.norm2.weight                  | Audio_Encoder Tp_Encoders 13 Norm2 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  831 | audio_encoder.tp_encoders.13.norm2.bias                    | Audio_Encoder Tp_Encoders 13 Norm2 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  832 | audio_encoder.tp_encoders.14.self_attn.linear_out.weight   | Audio_Encoder Tp_Encoders 14 Self_Attn Linear_Out (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  833 | audio_encoder.tp_encoders.14.self_attn.linear_out.bias     | Audio_Encoder Tp_Encoders 14 Self_Attn Linear_Out (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  834 | audio_encoder.tp_encoders.14.self_attn.linear_q_k_v.weight | Audio_Encoder Tp_Encoders 14 Self_Attn Linear_Q_K_V (W) | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  835 | audio_encoder.tp_encoders.14.self_attn.linear_q_k_v.bias   | Audio_Encoder Tp_Encoders 14 Self_Attn Linear_Q_K_V (B) | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  836 | audio_encoder.tp_encoders.14.self_attn.fsmn_block.weight   | Audio_Encoder Tp_Encoders 14 Self_Attn Fsmn_Block (W)   | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  837 | audio_encoder.tp_encoders.14.feed_forward.w_1.weight       | Audio_Encoder Tp_Encoders 14 Feed_Forward W_1 (W)       | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  838 | audio_encoder.tp_encoders.14.feed_forward.w_1.bias         | Audio_Encoder Tp_Encoders 14 Feed_Forward W_1 (B)       | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  839 | audio_encoder.tp_encoders.14.feed_forward.w_2.weight       | Audio_Encoder Tp_Encoders 14 Feed_Forward W_2 (W)       | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  840 | audio_encoder.tp_encoders.14.feed_forward.w_2.bias         | Audio_Encoder Tp_Encoders 14 Feed_Forward W_2 (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  841 | audio_encoder.tp_encoders.14.norm1.weight                  | Audio_Encoder Tp_Encoders 14 Norm1 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  842 | audio_encoder.tp_encoders.14.norm1.bias                    | Audio_Encoder Tp_Encoders 14 Norm1 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  843 | audio_encoder.tp_encoders.14.norm2.weight                  | Audio_Encoder Tp_Encoders 14 Norm2 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  844 | audio_encoder.tp_encoders.14.norm2.bias                    | Audio_Encoder Tp_Encoders 14 Norm2 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  845 | audio_encoder.tp_encoders.15.self_attn.linear_out.weight   | Audio_Encoder Tp_Encoders 15 Self_Attn Linear_Out (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  846 | audio_encoder.tp_encoders.15.self_attn.linear_out.bias     | Audio_Encoder Tp_Encoders 15 Self_Attn Linear_Out (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  847 | audio_encoder.tp_encoders.15.self_attn.linear_q_k_v.weight | Audio_Encoder Tp_Encoders 15 Self_Attn Linear_Q_K_V (W) | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  848 | audio_encoder.tp_encoders.15.self_attn.linear_q_k_v.bias   | Audio_Encoder Tp_Encoders 15 Self_Attn Linear_Q_K_V (B) | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  849 | audio_encoder.tp_encoders.15.self_attn.fsmn_block.weight   | Audio_Encoder Tp_Encoders 15 Self_Attn Fsmn_Block (W)   | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  850 | audio_encoder.tp_encoders.15.feed_forward.w_1.weight       | Audio_Encoder Tp_Encoders 15 Feed_Forward W_1 (W)       | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  851 | audio_encoder.tp_encoders.15.feed_forward.w_1.bias         | Audio_Encoder Tp_Encoders 15 Feed_Forward W_1 (B)       | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  852 | audio_encoder.tp_encoders.15.feed_forward.w_2.weight       | Audio_Encoder Tp_Encoders 15 Feed_Forward W_2 (W)       | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  853 | audio_encoder.tp_encoders.15.feed_forward.w_2.bias         | Audio_Encoder Tp_Encoders 15 Feed_Forward W_2 (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  854 | audio_encoder.tp_encoders.15.norm1.weight                  | Audio_Encoder Tp_Encoders 15 Norm1 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  855 | audio_encoder.tp_encoders.15.norm1.bias                    | Audio_Encoder Tp_Encoders 15 Norm1 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  856 | audio_encoder.tp_encoders.15.norm2.weight                  | Audio_Encoder Tp_Encoders 15 Norm2 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  857 | audio_encoder.tp_encoders.15.norm2.bias                    | Audio_Encoder Tp_Encoders 15 Norm2 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  858 | audio_encoder.tp_encoders.16.self_attn.linear_out.weight   | Audio_Encoder Tp_Encoders 16 Self_Attn Linear_Out (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  859 | audio_encoder.tp_encoders.16.self_attn.linear_out.bias     | Audio_Encoder Tp_Encoders 16 Self_Attn Linear_Out (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  860 | audio_encoder.tp_encoders.16.self_attn.linear_q_k_v.weight | Audio_Encoder Tp_Encoders 16 Self_Attn Linear_Q_K_V (W) | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  861 | audio_encoder.tp_encoders.16.self_attn.linear_q_k_v.bias   | Audio_Encoder Tp_Encoders 16 Self_Attn Linear_Q_K_V (B) | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  862 | audio_encoder.tp_encoders.16.self_attn.fsmn_block.weight   | Audio_Encoder Tp_Encoders 16 Self_Attn Fsmn_Block (W)   | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  863 | audio_encoder.tp_encoders.16.feed_forward.w_1.weight       | Audio_Encoder Tp_Encoders 16 Feed_Forward W_1 (W)       | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  864 | audio_encoder.tp_encoders.16.feed_forward.w_1.bias         | Audio_Encoder Tp_Encoders 16 Feed_Forward W_1 (B)       | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  865 | audio_encoder.tp_encoders.16.feed_forward.w_2.weight       | Audio_Encoder Tp_Encoders 16 Feed_Forward W_2 (W)       | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  866 | audio_encoder.tp_encoders.16.feed_forward.w_2.bias         | Audio_Encoder Tp_Encoders 16 Feed_Forward W_2 (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  867 | audio_encoder.tp_encoders.16.norm1.weight                  | Audio_Encoder Tp_Encoders 16 Norm1 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  868 | audio_encoder.tp_encoders.16.norm1.bias                    | Audio_Encoder Tp_Encoders 16 Norm1 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  869 | audio_encoder.tp_encoders.16.norm2.weight                  | Audio_Encoder Tp_Encoders 16 Norm2 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  870 | audio_encoder.tp_encoders.16.norm2.bias                    | Audio_Encoder Tp_Encoders 16 Norm2 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  871 | audio_encoder.tp_encoders.17.self_attn.linear_out.weight   | Audio_Encoder Tp_Encoders 17 Self_Attn Linear_Out (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  872 | audio_encoder.tp_encoders.17.self_attn.linear_out.bias     | Audio_Encoder Tp_Encoders 17 Self_Attn Linear_Out (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  873 | audio_encoder.tp_encoders.17.self_attn.linear_q_k_v.weight | Audio_Encoder Tp_Encoders 17 Self_Attn Linear_Q_K_V (W) | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  874 | audio_encoder.tp_encoders.17.self_attn.linear_q_k_v.bias   | Audio_Encoder Tp_Encoders 17 Self_Attn Linear_Q_K_V (B) | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  875 | audio_encoder.tp_encoders.17.self_attn.fsmn_block.weight   | Audio_Encoder Tp_Encoders 17 Self_Attn Fsmn_Block (W)   | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  876 | audio_encoder.tp_encoders.17.feed_forward.w_1.weight       | Audio_Encoder Tp_Encoders 17 Feed_Forward W_1 (W)       | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  877 | audio_encoder.tp_encoders.17.feed_forward.w_1.bias         | Audio_Encoder Tp_Encoders 17 Feed_Forward W_1 (B)       | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  878 | audio_encoder.tp_encoders.17.feed_forward.w_2.weight       | Audio_Encoder Tp_Encoders 17 Feed_Forward W_2 (W)       | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  879 | audio_encoder.tp_encoders.17.feed_forward.w_2.bias         | Audio_Encoder Tp_Encoders 17 Feed_Forward W_2 (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  880 | audio_encoder.tp_encoders.17.norm1.weight                  | Audio_Encoder Tp_Encoders 17 Norm1 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  881 | audio_encoder.tp_encoders.17.norm1.bias                    | Audio_Encoder Tp_Encoders 17 Norm1 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  882 | audio_encoder.tp_encoders.17.norm2.weight                  | Audio_Encoder Tp_Encoders 17 Norm2 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  883 | audio_encoder.tp_encoders.17.norm2.bias                    | Audio_Encoder Tp_Encoders 17 Norm2 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  884 | audio_encoder.tp_encoders.18.self_attn.linear_out.weight   | Audio_Encoder Tp_Encoders 18 Self_Attn Linear_Out (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  885 | audio_encoder.tp_encoders.18.self_attn.linear_out.bias     | Audio_Encoder Tp_Encoders 18 Self_Attn Linear_Out (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  886 | audio_encoder.tp_encoders.18.self_attn.linear_q_k_v.weight | Audio_Encoder Tp_Encoders 18 Self_Attn Linear_Q_K_V (W) | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  887 | audio_encoder.tp_encoders.18.self_attn.linear_q_k_v.bias   | Audio_Encoder Tp_Encoders 18 Self_Attn Linear_Q_K_V (B) | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  888 | audio_encoder.tp_encoders.18.self_attn.fsmn_block.weight   | Audio_Encoder Tp_Encoders 18 Self_Attn Fsmn_Block (W)   | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  889 | audio_encoder.tp_encoders.18.feed_forward.w_1.weight       | Audio_Encoder Tp_Encoders 18 Feed_Forward W_1 (W)       | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  890 | audio_encoder.tp_encoders.18.feed_forward.w_1.bias         | Audio_Encoder Tp_Encoders 18 Feed_Forward W_1 (B)       | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  891 | audio_encoder.tp_encoders.18.feed_forward.w_2.weight       | Audio_Encoder Tp_Encoders 18 Feed_Forward W_2 (W)       | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  892 | audio_encoder.tp_encoders.18.feed_forward.w_2.bias         | Audio_Encoder Tp_Encoders 18 Feed_Forward W_2 (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  893 | audio_encoder.tp_encoders.18.norm1.weight                  | Audio_Encoder Tp_Encoders 18 Norm1 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  894 | audio_encoder.tp_encoders.18.norm1.bias                    | Audio_Encoder Tp_Encoders 18 Norm1 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  895 | audio_encoder.tp_encoders.18.norm2.weight                  | Audio_Encoder Tp_Encoders 18 Norm2 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  896 | audio_encoder.tp_encoders.18.norm2.bias                    | Audio_Encoder Tp_Encoders 18 Norm2 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  897 | audio_encoder.tp_encoders.19.self_attn.linear_out.weight   | Audio_Encoder Tp_Encoders 19 Self_Attn Linear_Out (W)   | (~262K)    262144 |  512 x    512 x   1 x 1 | BF16 |
|  898 | audio_encoder.tp_encoders.19.self_attn.linear_out.bias     | Audio_Encoder Tp_Encoders 19 Self_Attn Linear_Out (B)   | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  899 | audio_encoder.tp_encoders.19.self_attn.linear_q_k_v.weight | Audio_Encoder Tp_Encoders 19 Self_Attn Linear_Q_K_V (W) | (~786K)    786432 |  512 x   1536 x   1 x 1 | BF16 |
|  900 | audio_encoder.tp_encoders.19.self_attn.linear_q_k_v.bias   | Audio_Encoder Tp_Encoders 19 Self_Attn Linear_Q_K_V (B) | (  ~2K)      1536 | 1536 x      1 x   1 x 1 | BF16 |
|  901 | audio_encoder.tp_encoders.19.self_attn.fsmn_block.weight   | Audio_Encoder Tp_Encoders 19 Self_Attn Fsmn_Block (W)   | (  ~6K)      5632 |   11 x      1 x 512 x 1 | F16  |
|  902 | audio_encoder.tp_encoders.19.feed_forward.w_1.weight       | Audio_Encoder Tp_Encoders 19 Feed_Forward W_1 (W)       | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  903 | audio_encoder.tp_encoders.19.feed_forward.w_1.bias         | Audio_Encoder Tp_Encoders 19 Feed_Forward W_1 (B)       | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  904 | audio_encoder.tp_encoders.19.feed_forward.w_2.weight       | Audio_Encoder Tp_Encoders 19 Feed_Forward W_2 (W)       | (  ~1M)   1048576 | 2048 x    512 x   1 x 1 | BF16 |
|  905 | audio_encoder.tp_encoders.19.feed_forward.w_2.bias         | Audio_Encoder Tp_Encoders 19 Feed_Forward W_2 (B)       | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  906 | audio_encoder.tp_encoders.19.norm1.weight                  | Audio_Encoder Tp_Encoders 19 Norm1 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  907 | audio_encoder.tp_encoders.19.norm1.bias                    | Audio_Encoder Tp_Encoders 19 Norm1 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  908 | audio_encoder.tp_encoders.19.norm2.weight                  | Audio_Encoder Tp_Encoders 19 Norm2 (W)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  909 | audio_encoder.tp_encoders.19.norm2.bias                    | Audio_Encoder Tp_Encoders 19 Norm2 (B)                  | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  910 | audio_encoder.after_norm.weight                            | Audio_Encoder After_Norm (W)                            | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  911 | audio_encoder.after_norm.bias                              | Audio_Encoder After_Norm (B)                            | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  912 | audio_encoder.tp_norm.weight                               | Audio_Encoder Tp_Norm (W)                               | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  913 | audio_encoder.tp_norm.bias                                 | Audio_Encoder Tp_Norm (B)                               | (  512)       512 |  512 x      1 x   1 x 1 | BF16 |
|  914 | audio_adaptor.linear1.weight                               | Audio_Adaptor Linear1 (W)                               | (  ~1M)   1048576 |  512 x   2048 x   1 x 1 | BF16 |
|  915 | audio_adaptor.linear1.bias                                 | Audio_Adaptor Linear1 (B)                               | (  ~2K)      2048 | 2048 x      1 x   1 x 1 | BF16 |
|  916 | audio_adaptor.linear2.weight                               | Audio_Adaptor Linear2 (W)                               | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
|  917 | audio_adaptor.linear2.bias                                 | Audio_Adaptor Linear2 (B)                               | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  918 | audio_adaptor.blocks.0.self_attn.linear_q.weight           | Audio_Adaptor Blocks 0 Self_Attn Linear_Q (W)           | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  919 | audio_adaptor.blocks.0.self_attn.linear_q.bias             | Audio_Adaptor Blocks 0 Self_Attn Linear_Q (B)           | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  920 | audio_adaptor.blocks.0.self_attn.linear_k.weight           | Audio_Adaptor Blocks 0 Self_Attn Linear_K (W)           | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  921 | audio_adaptor.blocks.0.self_attn.linear_k.bias             | Audio_Adaptor Blocks 0 Self_Attn Linear_K (B)           | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  922 | audio_adaptor.blocks.0.self_attn.linear_v.weight           | Audio_Adaptor Blocks 0 Self_Attn Linear_V (W)           | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  923 | audio_adaptor.blocks.0.self_attn.linear_v.bias             | Audio_Adaptor Blocks 0 Self_Attn Linear_V (B)           | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  924 | audio_adaptor.blocks.0.self_attn.linear_out.weight         | Audio_Adaptor Blocks 0 Self_Attn Linear_Out (W)         | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  925 | audio_adaptor.blocks.0.self_attn.linear_out.bias           | Audio_Adaptor Blocks 0 Self_Attn Linear_Out (B)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  926 | audio_adaptor.blocks.0.feed_forward.w_1.weight             | Audio_Adaptor Blocks 0 Feed_Forward W_1 (W)             | (~262K)    262144 | 1024 x    256 x   1 x 1 | BF16 |
|  927 | audio_adaptor.blocks.0.feed_forward.w_1.bias               | Audio_Adaptor Blocks 0 Feed_Forward W_1 (B)             | (  256)       256 |  256 x      1 x   1 x 1 | BF16 |
|  928 | audio_adaptor.blocks.0.feed_forward.w_2.weight             | Audio_Adaptor Blocks 0 Feed_Forward W_2 (W)             | (~262K)    262144 |  256 x   1024 x   1 x 1 | BF16 |
|  929 | audio_adaptor.blocks.0.feed_forward.w_2.bias               | Audio_Adaptor Blocks 0 Feed_Forward W_2 (B)             | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  930 | audio_adaptor.blocks.0.norm1.weight                        | Audio_Adaptor Blocks 0 Norm1 (W)                        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  931 | audio_adaptor.blocks.0.norm1.bias                          | Audio_Adaptor Blocks 0 Norm1 (B)                        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  932 | audio_adaptor.blocks.0.norm2.weight                        | Audio_Adaptor Blocks 0 Norm2 (W)                        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  933 | audio_adaptor.blocks.0.norm2.bias                          | Audio_Adaptor Blocks 0 Norm2 (B)                        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  934 | audio_adaptor.blocks.1.self_attn.linear_q.weight           | Audio_Adaptor Blocks 1 Self_Attn Linear_Q (W)           | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  935 | audio_adaptor.blocks.1.self_attn.linear_q.bias             | Audio_Adaptor Blocks 1 Self_Attn Linear_Q (B)           | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  936 | audio_adaptor.blocks.1.self_attn.linear_k.weight           | Audio_Adaptor Blocks 1 Self_Attn Linear_K (W)           | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  937 | audio_adaptor.blocks.1.self_attn.linear_k.bias             | Audio_Adaptor Blocks 1 Self_Attn Linear_K (B)           | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  938 | audio_adaptor.blocks.1.self_attn.linear_v.weight           | Audio_Adaptor Blocks 1 Self_Attn Linear_V (W)           | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  939 | audio_adaptor.blocks.1.self_attn.linear_v.bias             | Audio_Adaptor Blocks 1 Self_Attn Linear_V (B)           | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  940 | audio_adaptor.blocks.1.self_attn.linear_out.weight         | Audio_Adaptor Blocks 1 Self_Attn Linear_Out (W)         | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  941 | audio_adaptor.blocks.1.self_attn.linear_out.bias           | Audio_Adaptor Blocks 1 Self_Attn Linear_Out (B)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  942 | audio_adaptor.blocks.1.feed_forward.w_1.weight             | Audio_Adaptor Blocks 1 Feed_Forward W_1 (W)             | (~262K)    262144 | 1024 x    256 x   1 x 1 | BF16 |
|  943 | audio_adaptor.blocks.1.feed_forward.w_1.bias               | Audio_Adaptor Blocks 1 Feed_Forward W_1 (B)             | (  256)       256 |  256 x      1 x   1 x 1 | BF16 |
|  944 | audio_adaptor.blocks.1.feed_forward.w_2.weight             | Audio_Adaptor Blocks 1 Feed_Forward W_2 (W)             | (~262K)    262144 |  256 x   1024 x   1 x 1 | BF16 |
|  945 | audio_adaptor.blocks.1.feed_forward.w_2.bias               | Audio_Adaptor Blocks 1 Feed_Forward W_2 (B)             | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  946 | audio_adaptor.blocks.1.norm1.weight                        | Audio_Adaptor Blocks 1 Norm1 (W)                        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  947 | audio_adaptor.blocks.1.norm1.bias                          | Audio_Adaptor Blocks 1 Norm1 (B)                        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  948 | audio_adaptor.blocks.1.norm2.weight                        | Audio_Adaptor Blocks 1 Norm2 (W)                        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  949 | audio_adaptor.blocks.1.norm2.bias                          | Audio_Adaptor Blocks 1 Norm2 (B)                        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  950 | llm.lm_head.weight                                         | Llm Lm_Head (W)                                         | (~156M) 155582464 | 1024 x 151936 x   1 x 1 | BF16 |
|  951 | llm.model.embed_tokens.weight                              | Llm Model Embed_Tokens (W)                              | (~156M) 155582464 | 1024 x 151936 x   1 x 1 | BF16 |
|  952 | llm.model.layers.0.input_layernorm.weight                  | Llm Model Layers 0 Input_Layernorm (W)                  | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  953 | llm.model.layers.0.mlp.down_proj.weight                    | Llm Model Layers 0 Mlp Down_Proj (W)                    | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
|  954 | llm.model.layers.0.mlp.gate_proj.weight                    | Llm Model Layers 0 Mlp Gate_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
|  955 | llm.model.layers.0.mlp.up_proj.weight                      | Llm Model Layers 0 Mlp Up_Proj (W)                      | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
|  956 | llm.model.layers.0.post_attention_layernorm.weight         | Llm Model Layers 0 Post_Attention_Layernorm (W)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  957 | llm.model.layers.0.self_attn.k_norm.weight                 | Llm Model Layers 0 Self_Attn K_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
|  958 | llm.model.layers.0.self_attn.k_proj.weight                 | Llm Model Layers 0 Self_Attn K_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  959 | llm.model.layers.0.self_attn.o_proj.weight                 | Llm Model Layers 0 Self_Attn O_Proj (W)                 | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
|  960 | llm.model.layers.0.self_attn.q_norm.weight                 | Llm Model Layers 0 Self_Attn Q_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
|  961 | llm.model.layers.0.self_attn.q_proj.weight                 | Llm Model Layers 0 Self_Attn Q_Proj (W)                 | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
|  962 | llm.model.layers.0.self_attn.v_proj.weight                 | Llm Model Layers 0 Self_Attn V_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  963 | llm.model.layers.1.input_layernorm.weight                  | Llm Model Layers 1 Input_Layernorm (W)                  | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  964 | llm.model.layers.1.mlp.down_proj.weight                    | Llm Model Layers 1 Mlp Down_Proj (W)                    | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
|  965 | llm.model.layers.1.mlp.gate_proj.weight                    | Llm Model Layers 1 Mlp Gate_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
|  966 | llm.model.layers.1.mlp.up_proj.weight                      | Llm Model Layers 1 Mlp Up_Proj (W)                      | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
|  967 | llm.model.layers.1.post_attention_layernorm.weight         | Llm Model Layers 1 Post_Attention_Layernorm (W)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  968 | llm.model.layers.1.self_attn.k_norm.weight                 | Llm Model Layers 1 Self_Attn K_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
|  969 | llm.model.layers.1.self_attn.k_proj.weight                 | Llm Model Layers 1 Self_Attn K_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  970 | llm.model.layers.1.self_attn.o_proj.weight                 | Llm Model Layers 1 Self_Attn O_Proj (W)                 | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
|  971 | llm.model.layers.1.self_attn.q_norm.weight                 | Llm Model Layers 1 Self_Attn Q_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
|  972 | llm.model.layers.1.self_attn.q_proj.weight                 | Llm Model Layers 1 Self_Attn Q_Proj (W)                 | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
|  973 | llm.model.layers.1.self_attn.v_proj.weight                 | Llm Model Layers 1 Self_Attn V_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  974 | llm.model.layers.10.input_layernorm.weight                 | Llm Model Layers 10 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  975 | llm.model.layers.10.mlp.down_proj.weight                   | Llm Model Layers 10 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
|  976 | llm.model.layers.10.mlp.gate_proj.weight                   | Llm Model Layers 10 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
|  977 | llm.model.layers.10.mlp.up_proj.weight                     | Llm Model Layers 10 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
|  978 | llm.model.layers.10.post_attention_layernorm.weight        | Llm Model Layers 10 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  979 | llm.model.layers.10.self_attn.k_norm.weight                | Llm Model Layers 10 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
|  980 | llm.model.layers.10.self_attn.k_proj.weight                | Llm Model Layers 10 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  981 | llm.model.layers.10.self_attn.o_proj.weight                | Llm Model Layers 10 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
|  982 | llm.model.layers.10.self_attn.q_norm.weight                | Llm Model Layers 10 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
|  983 | llm.model.layers.10.self_attn.q_proj.weight                | Llm Model Layers 10 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
|  984 | llm.model.layers.10.self_attn.v_proj.weight                | Llm Model Layers 10 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  985 | llm.model.layers.11.input_layernorm.weight                 | Llm Model Layers 11 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  986 | llm.model.layers.11.mlp.down_proj.weight                   | Llm Model Layers 11 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
|  987 | llm.model.layers.11.mlp.gate_proj.weight                   | Llm Model Layers 11 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
|  988 | llm.model.layers.11.mlp.up_proj.weight                     | Llm Model Layers 11 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
|  989 | llm.model.layers.11.post_attention_layernorm.weight        | Llm Model Layers 11 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  990 | llm.model.layers.11.self_attn.k_norm.weight                | Llm Model Layers 11 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
|  991 | llm.model.layers.11.self_attn.k_proj.weight                | Llm Model Layers 11 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  992 | llm.model.layers.11.self_attn.o_proj.weight                | Llm Model Layers 11 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
|  993 | llm.model.layers.11.self_attn.q_norm.weight                | Llm Model Layers 11 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
|  994 | llm.model.layers.11.self_attn.q_proj.weight                | Llm Model Layers 11 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
|  995 | llm.model.layers.11.self_attn.v_proj.weight                | Llm Model Layers 11 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
|  996 | llm.model.layers.12.input_layernorm.weight                 | Llm Model Layers 12 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
|  997 | llm.model.layers.12.mlp.down_proj.weight                   | Llm Model Layers 12 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
|  998 | llm.model.layers.12.mlp.gate_proj.weight                   | Llm Model Layers 12 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
|  999 | llm.model.layers.12.mlp.up_proj.weight                     | Llm Model Layers 12 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1000 | llm.model.layers.12.post_attention_layernorm.weight        | Llm Model Layers 12 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1001 | llm.model.layers.12.self_attn.k_norm.weight                | Llm Model Layers 12 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1002 | llm.model.layers.12.self_attn.k_proj.weight                | Llm Model Layers 12 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1003 | llm.model.layers.12.self_attn.o_proj.weight                | Llm Model Layers 12 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1004 | llm.model.layers.12.self_attn.q_norm.weight                | Llm Model Layers 12 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1005 | llm.model.layers.12.self_attn.q_proj.weight                | Llm Model Layers 12 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1006 | llm.model.layers.12.self_attn.v_proj.weight                | Llm Model Layers 12 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1007 | llm.model.layers.13.input_layernorm.weight                 | Llm Model Layers 13 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1008 | llm.model.layers.13.mlp.down_proj.weight                   | Llm Model Layers 13 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1009 | llm.model.layers.13.mlp.gate_proj.weight                   | Llm Model Layers 13 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1010 | llm.model.layers.13.mlp.up_proj.weight                     | Llm Model Layers 13 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1011 | llm.model.layers.13.post_attention_layernorm.weight        | Llm Model Layers 13 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1012 | llm.model.layers.13.self_attn.k_norm.weight                | Llm Model Layers 13 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1013 | llm.model.layers.13.self_attn.k_proj.weight                | Llm Model Layers 13 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1014 | llm.model.layers.13.self_attn.o_proj.weight                | Llm Model Layers 13 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1015 | llm.model.layers.13.self_attn.q_norm.weight                | Llm Model Layers 13 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1016 | llm.model.layers.13.self_attn.q_proj.weight                | Llm Model Layers 13 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1017 | llm.model.layers.13.self_attn.v_proj.weight                | Llm Model Layers 13 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1018 | llm.model.layers.14.input_layernorm.weight                 | Llm Model Layers 14 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1019 | llm.model.layers.14.mlp.down_proj.weight                   | Llm Model Layers 14 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1020 | llm.model.layers.14.mlp.gate_proj.weight                   | Llm Model Layers 14 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1021 | llm.model.layers.14.mlp.up_proj.weight                     | Llm Model Layers 14 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1022 | llm.model.layers.14.post_attention_layernorm.weight        | Llm Model Layers 14 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1023 | llm.model.layers.14.self_attn.k_norm.weight                | Llm Model Layers 14 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1024 | llm.model.layers.14.self_attn.k_proj.weight                | Llm Model Layers 14 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1025 | llm.model.layers.14.self_attn.o_proj.weight                | Llm Model Layers 14 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1026 | llm.model.layers.14.self_attn.q_norm.weight                | Llm Model Layers 14 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1027 | llm.model.layers.14.self_attn.q_proj.weight                | Llm Model Layers 14 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1028 | llm.model.layers.14.self_attn.v_proj.weight                | Llm Model Layers 14 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1029 | llm.model.layers.15.input_layernorm.weight                 | Llm Model Layers 15 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1030 | llm.model.layers.15.mlp.down_proj.weight                   | Llm Model Layers 15 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1031 | llm.model.layers.15.mlp.gate_proj.weight                   | Llm Model Layers 15 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1032 | llm.model.layers.15.mlp.up_proj.weight                     | Llm Model Layers 15 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1033 | llm.model.layers.15.post_attention_layernorm.weight        | Llm Model Layers 15 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1034 | llm.model.layers.15.self_attn.k_norm.weight                | Llm Model Layers 15 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1035 | llm.model.layers.15.self_attn.k_proj.weight                | Llm Model Layers 15 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1036 | llm.model.layers.15.self_attn.o_proj.weight                | Llm Model Layers 15 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1037 | llm.model.layers.15.self_attn.q_norm.weight                | Llm Model Layers 15 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1038 | llm.model.layers.15.self_attn.q_proj.weight                | Llm Model Layers 15 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1039 | llm.model.layers.15.self_attn.v_proj.weight                | Llm Model Layers 15 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1040 | llm.model.layers.16.input_layernorm.weight                 | Llm Model Layers 16 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1041 | llm.model.layers.16.mlp.down_proj.weight                   | Llm Model Layers 16 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1042 | llm.model.layers.16.mlp.gate_proj.weight                   | Llm Model Layers 16 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1043 | llm.model.layers.16.mlp.up_proj.weight                     | Llm Model Layers 16 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1044 | llm.model.layers.16.post_attention_layernorm.weight        | Llm Model Layers 16 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1045 | llm.model.layers.16.self_attn.k_norm.weight                | Llm Model Layers 16 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1046 | llm.model.layers.16.self_attn.k_proj.weight                | Llm Model Layers 16 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1047 | llm.model.layers.16.self_attn.o_proj.weight                | Llm Model Layers 16 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1048 | llm.model.layers.16.self_attn.q_norm.weight                | Llm Model Layers 16 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1049 | llm.model.layers.16.self_attn.q_proj.weight                | Llm Model Layers 16 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1050 | llm.model.layers.16.self_attn.v_proj.weight                | Llm Model Layers 16 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1051 | llm.model.layers.17.input_layernorm.weight                 | Llm Model Layers 17 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1052 | llm.model.layers.17.mlp.down_proj.weight                   | Llm Model Layers 17 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1053 | llm.model.layers.17.mlp.gate_proj.weight                   | Llm Model Layers 17 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1054 | llm.model.layers.17.mlp.up_proj.weight                     | Llm Model Layers 17 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1055 | llm.model.layers.17.post_attention_layernorm.weight        | Llm Model Layers 17 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1056 | llm.model.layers.17.self_attn.k_norm.weight                | Llm Model Layers 17 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1057 | llm.model.layers.17.self_attn.k_proj.weight                | Llm Model Layers 17 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1058 | llm.model.layers.17.self_attn.o_proj.weight                | Llm Model Layers 17 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1059 | llm.model.layers.17.self_attn.q_norm.weight                | Llm Model Layers 17 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1060 | llm.model.layers.17.self_attn.q_proj.weight                | Llm Model Layers 17 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1061 | llm.model.layers.17.self_attn.v_proj.weight                | Llm Model Layers 17 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1062 | llm.model.layers.18.input_layernorm.weight                 | Llm Model Layers 18 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1063 | llm.model.layers.18.mlp.down_proj.weight                   | Llm Model Layers 18 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1064 | llm.model.layers.18.mlp.gate_proj.weight                   | Llm Model Layers 18 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1065 | llm.model.layers.18.mlp.up_proj.weight                     | Llm Model Layers 18 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1066 | llm.model.layers.18.post_attention_layernorm.weight        | Llm Model Layers 18 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1067 | llm.model.layers.18.self_attn.k_norm.weight                | Llm Model Layers 18 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1068 | llm.model.layers.18.self_attn.k_proj.weight                | Llm Model Layers 18 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1069 | llm.model.layers.18.self_attn.o_proj.weight                | Llm Model Layers 18 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1070 | llm.model.layers.18.self_attn.q_norm.weight                | Llm Model Layers 18 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1071 | llm.model.layers.18.self_attn.q_proj.weight                | Llm Model Layers 18 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1072 | llm.model.layers.18.self_attn.v_proj.weight                | Llm Model Layers 18 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1073 | llm.model.layers.19.input_layernorm.weight                 | Llm Model Layers 19 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1074 | llm.model.layers.19.mlp.down_proj.weight                   | Llm Model Layers 19 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1075 | llm.model.layers.19.mlp.gate_proj.weight                   | Llm Model Layers 19 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1076 | llm.model.layers.19.mlp.up_proj.weight                     | Llm Model Layers 19 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1077 | llm.model.layers.19.post_attention_layernorm.weight        | Llm Model Layers 19 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1078 | llm.model.layers.19.self_attn.k_norm.weight                | Llm Model Layers 19 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1079 | llm.model.layers.19.self_attn.k_proj.weight                | Llm Model Layers 19 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1080 | llm.model.layers.19.self_attn.o_proj.weight                | Llm Model Layers 19 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1081 | llm.model.layers.19.self_attn.q_norm.weight                | Llm Model Layers 19 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1082 | llm.model.layers.19.self_attn.q_proj.weight                | Llm Model Layers 19 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1083 | llm.model.layers.19.self_attn.v_proj.weight                | Llm Model Layers 19 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1084 | llm.model.layers.2.input_layernorm.weight                  | Llm Model Layers 2 Input_Layernorm (W)                  | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1085 | llm.model.layers.2.mlp.down_proj.weight                    | Llm Model Layers 2 Mlp Down_Proj (W)                    | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1086 | llm.model.layers.2.mlp.gate_proj.weight                    | Llm Model Layers 2 Mlp Gate_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1087 | llm.model.layers.2.mlp.up_proj.weight                      | Llm Model Layers 2 Mlp Up_Proj (W)                      | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1088 | llm.model.layers.2.post_attention_layernorm.weight         | Llm Model Layers 2 Post_Attention_Layernorm (W)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1089 | llm.model.layers.2.self_attn.k_norm.weight                 | Llm Model Layers 2 Self_Attn K_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1090 | llm.model.layers.2.self_attn.k_proj.weight                 | Llm Model Layers 2 Self_Attn K_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1091 | llm.model.layers.2.self_attn.o_proj.weight                 | Llm Model Layers 2 Self_Attn O_Proj (W)                 | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1092 | llm.model.layers.2.self_attn.q_norm.weight                 | Llm Model Layers 2 Self_Attn Q_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1093 | llm.model.layers.2.self_attn.q_proj.weight                 | Llm Model Layers 2 Self_Attn Q_Proj (W)                 | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1094 | llm.model.layers.2.self_attn.v_proj.weight                 | Llm Model Layers 2 Self_Attn V_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1095 | llm.model.layers.20.input_layernorm.weight                 | Llm Model Layers 20 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1096 | llm.model.layers.20.mlp.down_proj.weight                   | Llm Model Layers 20 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1097 | llm.model.layers.20.mlp.gate_proj.weight                   | Llm Model Layers 20 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1098 | llm.model.layers.20.mlp.up_proj.weight                     | Llm Model Layers 20 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1099 | llm.model.layers.20.post_attention_layernorm.weight        | Llm Model Layers 20 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1100 | llm.model.layers.20.self_attn.k_norm.weight                | Llm Model Layers 20 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1101 | llm.model.layers.20.self_attn.k_proj.weight                | Llm Model Layers 20 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1102 | llm.model.layers.20.self_attn.o_proj.weight                | Llm Model Layers 20 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1103 | llm.model.layers.20.self_attn.q_norm.weight                | Llm Model Layers 20 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1104 | llm.model.layers.20.self_attn.q_proj.weight                | Llm Model Layers 20 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1105 | llm.model.layers.20.self_attn.v_proj.weight                | Llm Model Layers 20 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1106 | llm.model.layers.21.input_layernorm.weight                 | Llm Model Layers 21 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1107 | llm.model.layers.21.mlp.down_proj.weight                   | Llm Model Layers 21 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1108 | llm.model.layers.21.mlp.gate_proj.weight                   | Llm Model Layers 21 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1109 | llm.model.layers.21.mlp.up_proj.weight                     | Llm Model Layers 21 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1110 | llm.model.layers.21.post_attention_layernorm.weight        | Llm Model Layers 21 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1111 | llm.model.layers.21.self_attn.k_norm.weight                | Llm Model Layers 21 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1112 | llm.model.layers.21.self_attn.k_proj.weight                | Llm Model Layers 21 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1113 | llm.model.layers.21.self_attn.o_proj.weight                | Llm Model Layers 21 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1114 | llm.model.layers.21.self_attn.q_norm.weight                | Llm Model Layers 21 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1115 | llm.model.layers.21.self_attn.q_proj.weight                | Llm Model Layers 21 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1116 | llm.model.layers.21.self_attn.v_proj.weight                | Llm Model Layers 21 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1117 | llm.model.layers.22.input_layernorm.weight                 | Llm Model Layers 22 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1118 | llm.model.layers.22.mlp.down_proj.weight                   | Llm Model Layers 22 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1119 | llm.model.layers.22.mlp.gate_proj.weight                   | Llm Model Layers 22 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1120 | llm.model.layers.22.mlp.up_proj.weight                     | Llm Model Layers 22 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1121 | llm.model.layers.22.post_attention_layernorm.weight        | Llm Model Layers 22 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1122 | llm.model.layers.22.self_attn.k_norm.weight                | Llm Model Layers 22 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1123 | llm.model.layers.22.self_attn.k_proj.weight                | Llm Model Layers 22 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1124 | llm.model.layers.22.self_attn.o_proj.weight                | Llm Model Layers 22 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1125 | llm.model.layers.22.self_attn.q_norm.weight                | Llm Model Layers 22 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1126 | llm.model.layers.22.self_attn.q_proj.weight                | Llm Model Layers 22 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1127 | llm.model.layers.22.self_attn.v_proj.weight                | Llm Model Layers 22 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1128 | llm.model.layers.23.input_layernorm.weight                 | Llm Model Layers 23 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1129 | llm.model.layers.23.mlp.down_proj.weight                   | Llm Model Layers 23 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1130 | llm.model.layers.23.mlp.gate_proj.weight                   | Llm Model Layers 23 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1131 | llm.model.layers.23.mlp.up_proj.weight                     | Llm Model Layers 23 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1132 | llm.model.layers.23.post_attention_layernorm.weight        | Llm Model Layers 23 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1133 | llm.model.layers.23.self_attn.k_norm.weight                | Llm Model Layers 23 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1134 | llm.model.layers.23.self_attn.k_proj.weight                | Llm Model Layers 23 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1135 | llm.model.layers.23.self_attn.o_proj.weight                | Llm Model Layers 23 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1136 | llm.model.layers.23.self_attn.q_norm.weight                | Llm Model Layers 23 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1137 | llm.model.layers.23.self_attn.q_proj.weight                | Llm Model Layers 23 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1138 | llm.model.layers.23.self_attn.v_proj.weight                | Llm Model Layers 23 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1139 | llm.model.layers.24.input_layernorm.weight                 | Llm Model Layers 24 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1140 | llm.model.layers.24.mlp.down_proj.weight                   | Llm Model Layers 24 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1141 | llm.model.layers.24.mlp.gate_proj.weight                   | Llm Model Layers 24 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1142 | llm.model.layers.24.mlp.up_proj.weight                     | Llm Model Layers 24 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1143 | llm.model.layers.24.post_attention_layernorm.weight        | Llm Model Layers 24 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1144 | llm.model.layers.24.self_attn.k_norm.weight                | Llm Model Layers 24 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1145 | llm.model.layers.24.self_attn.k_proj.weight                | Llm Model Layers 24 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1146 | llm.model.layers.24.self_attn.o_proj.weight                | Llm Model Layers 24 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1147 | llm.model.layers.24.self_attn.q_norm.weight                | Llm Model Layers 24 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1148 | llm.model.layers.24.self_attn.q_proj.weight                | Llm Model Layers 24 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1149 | llm.model.layers.24.self_attn.v_proj.weight                | Llm Model Layers 24 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1150 | llm.model.layers.25.input_layernorm.weight                 | Llm Model Layers 25 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1151 | llm.model.layers.25.mlp.down_proj.weight                   | Llm Model Layers 25 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1152 | llm.model.layers.25.mlp.gate_proj.weight                   | Llm Model Layers 25 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1153 | llm.model.layers.25.mlp.up_proj.weight                     | Llm Model Layers 25 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1154 | llm.model.layers.25.post_attention_layernorm.weight        | Llm Model Layers 25 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1155 | llm.model.layers.25.self_attn.k_norm.weight                | Llm Model Layers 25 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1156 | llm.model.layers.25.self_attn.k_proj.weight                | Llm Model Layers 25 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1157 | llm.model.layers.25.self_attn.o_proj.weight                | Llm Model Layers 25 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1158 | llm.model.layers.25.self_attn.q_norm.weight                | Llm Model Layers 25 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1159 | llm.model.layers.25.self_attn.q_proj.weight                | Llm Model Layers 25 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1160 | llm.model.layers.25.self_attn.v_proj.weight                | Llm Model Layers 25 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1161 | llm.model.layers.26.input_layernorm.weight                 | Llm Model Layers 26 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1162 | llm.model.layers.26.mlp.down_proj.weight                   | Llm Model Layers 26 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1163 | llm.model.layers.26.mlp.gate_proj.weight                   | Llm Model Layers 26 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1164 | llm.model.layers.26.mlp.up_proj.weight                     | Llm Model Layers 26 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1165 | llm.model.layers.26.post_attention_layernorm.weight        | Llm Model Layers 26 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1166 | llm.model.layers.26.self_attn.k_norm.weight                | Llm Model Layers 26 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1167 | llm.model.layers.26.self_attn.k_proj.weight                | Llm Model Layers 26 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1168 | llm.model.layers.26.self_attn.o_proj.weight                | Llm Model Layers 26 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1169 | llm.model.layers.26.self_attn.q_norm.weight                | Llm Model Layers 26 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1170 | llm.model.layers.26.self_attn.q_proj.weight                | Llm Model Layers 26 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1171 | llm.model.layers.26.self_attn.v_proj.weight                | Llm Model Layers 26 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1172 | llm.model.layers.27.input_layernorm.weight                 | Llm Model Layers 27 Input_Layernorm (W)                 | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1173 | llm.model.layers.27.mlp.down_proj.weight                   | Llm Model Layers 27 Mlp Down_Proj (W)                   | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1174 | llm.model.layers.27.mlp.gate_proj.weight                   | Llm Model Layers 27 Mlp Gate_Proj (W)                   | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1175 | llm.model.layers.27.mlp.up_proj.weight                     | Llm Model Layers 27 Mlp Up_Proj (W)                     | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1176 | llm.model.layers.27.post_attention_layernorm.weight        | Llm Model Layers 27 Post_Attention_Layernorm (W)        | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1177 | llm.model.layers.27.self_attn.k_norm.weight                | Llm Model Layers 27 Self_Attn K_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1178 | llm.model.layers.27.self_attn.k_proj.weight                | Llm Model Layers 27 Self_Attn K_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1179 | llm.model.layers.27.self_attn.o_proj.weight                | Llm Model Layers 27 Self_Attn O_Proj (W)                | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1180 | llm.model.layers.27.self_attn.q_norm.weight                | Llm Model Layers 27 Self_Attn Q_Norm (W)                | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1181 | llm.model.layers.27.self_attn.q_proj.weight                | Llm Model Layers 27 Self_Attn Q_Proj (W)                | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1182 | llm.model.layers.27.self_attn.v_proj.weight                | Llm Model Layers 27 Self_Attn V_Proj (W)                | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1183 | llm.model.layers.3.input_layernorm.weight                  | Llm Model Layers 3 Input_Layernorm (W)                  | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1184 | llm.model.layers.3.mlp.down_proj.weight                    | Llm Model Layers 3 Mlp Down_Proj (W)                    | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1185 | llm.model.layers.3.mlp.gate_proj.weight                    | Llm Model Layers 3 Mlp Gate_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1186 | llm.model.layers.3.mlp.up_proj.weight                      | Llm Model Layers 3 Mlp Up_Proj (W)                      | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1187 | llm.model.layers.3.post_attention_layernorm.weight         | Llm Model Layers 3 Post_Attention_Layernorm (W)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1188 | llm.model.layers.3.self_attn.k_norm.weight                 | Llm Model Layers 3 Self_Attn K_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1189 | llm.model.layers.3.self_attn.k_proj.weight                 | Llm Model Layers 3 Self_Attn K_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1190 | llm.model.layers.3.self_attn.o_proj.weight                 | Llm Model Layers 3 Self_Attn O_Proj (W)                 | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1191 | llm.model.layers.3.self_attn.q_norm.weight                 | Llm Model Layers 3 Self_Attn Q_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1192 | llm.model.layers.3.self_attn.q_proj.weight                 | Llm Model Layers 3 Self_Attn Q_Proj (W)                 | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1193 | llm.model.layers.3.self_attn.v_proj.weight                 | Llm Model Layers 3 Self_Attn V_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1194 | llm.model.layers.4.input_layernorm.weight                  | Llm Model Layers 4 Input_Layernorm (W)                  | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1195 | llm.model.layers.4.mlp.down_proj.weight                    | Llm Model Layers 4 Mlp Down_Proj (W)                    | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1196 | llm.model.layers.4.mlp.gate_proj.weight                    | Llm Model Layers 4 Mlp Gate_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1197 | llm.model.layers.4.mlp.up_proj.weight                      | Llm Model Layers 4 Mlp Up_Proj (W)                      | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1198 | llm.model.layers.4.post_attention_layernorm.weight         | Llm Model Layers 4 Post_Attention_Layernorm (W)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1199 | llm.model.layers.4.self_attn.k_norm.weight                 | Llm Model Layers 4 Self_Attn K_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1200 | llm.model.layers.4.self_attn.k_proj.weight                 | Llm Model Layers 4 Self_Attn K_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1201 | llm.model.layers.4.self_attn.o_proj.weight                 | Llm Model Layers 4 Self_Attn O_Proj (W)                 | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1202 | llm.model.layers.4.self_attn.q_norm.weight                 | Llm Model Layers 4 Self_Attn Q_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1203 | llm.model.layers.4.self_attn.q_proj.weight                 | Llm Model Layers 4 Self_Attn Q_Proj (W)                 | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1204 | llm.model.layers.4.self_attn.v_proj.weight                 | Llm Model Layers 4 Self_Attn V_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1205 | llm.model.layers.5.input_layernorm.weight                  | Llm Model Layers 5 Input_Layernorm (W)                  | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1206 | llm.model.layers.5.mlp.down_proj.weight                    | Llm Model Layers 5 Mlp Down_Proj (W)                    | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1207 | llm.model.layers.5.mlp.gate_proj.weight                    | Llm Model Layers 5 Mlp Gate_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1208 | llm.model.layers.5.mlp.up_proj.weight                      | Llm Model Layers 5 Mlp Up_Proj (W)                      | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1209 | llm.model.layers.5.post_attention_layernorm.weight         | Llm Model Layers 5 Post_Attention_Layernorm (W)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1210 | llm.model.layers.5.self_attn.k_norm.weight                 | Llm Model Layers 5 Self_Attn K_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1211 | llm.model.layers.5.self_attn.k_proj.weight                 | Llm Model Layers 5 Self_Attn K_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1212 | llm.model.layers.5.self_attn.o_proj.weight                 | Llm Model Layers 5 Self_Attn O_Proj (W)                 | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1213 | llm.model.layers.5.self_attn.q_norm.weight                 | Llm Model Layers 5 Self_Attn Q_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1214 | llm.model.layers.5.self_attn.q_proj.weight                 | Llm Model Layers 5 Self_Attn Q_Proj (W)                 | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1215 | llm.model.layers.5.self_attn.v_proj.weight                 | Llm Model Layers 5 Self_Attn V_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1216 | llm.model.layers.6.input_layernorm.weight                  | Llm Model Layers 6 Input_Layernorm (W)                  | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1217 | llm.model.layers.6.mlp.down_proj.weight                    | Llm Model Layers 6 Mlp Down_Proj (W)                    | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1218 | llm.model.layers.6.mlp.gate_proj.weight                    | Llm Model Layers 6 Mlp Gate_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1219 | llm.model.layers.6.mlp.up_proj.weight                      | Llm Model Layers 6 Mlp Up_Proj (W)                      | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1220 | llm.model.layers.6.post_attention_layernorm.weight         | Llm Model Layers 6 Post_Attention_Layernorm (W)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1221 | llm.model.layers.6.self_attn.k_norm.weight                 | Llm Model Layers 6 Self_Attn K_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1222 | llm.model.layers.6.self_attn.k_proj.weight                 | Llm Model Layers 6 Self_Attn K_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1223 | llm.model.layers.6.self_attn.o_proj.weight                 | Llm Model Layers 6 Self_Attn O_Proj (W)                 | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1224 | llm.model.layers.6.self_attn.q_norm.weight                 | Llm Model Layers 6 Self_Attn Q_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1225 | llm.model.layers.6.self_attn.q_proj.weight                 | Llm Model Layers 6 Self_Attn Q_Proj (W)                 | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1226 | llm.model.layers.6.self_attn.v_proj.weight                 | Llm Model Layers 6 Self_Attn V_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1227 | llm.model.layers.7.input_layernorm.weight                  | Llm Model Layers 7 Input_Layernorm (W)                  | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1228 | llm.model.layers.7.mlp.down_proj.weight                    | Llm Model Layers 7 Mlp Down_Proj (W)                    | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1229 | llm.model.layers.7.mlp.gate_proj.weight                    | Llm Model Layers 7 Mlp Gate_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1230 | llm.model.layers.7.mlp.up_proj.weight                      | Llm Model Layers 7 Mlp Up_Proj (W)                      | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1231 | llm.model.layers.7.post_attention_layernorm.weight         | Llm Model Layers 7 Post_Attention_Layernorm (W)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1232 | llm.model.layers.7.self_attn.k_norm.weight                 | Llm Model Layers 7 Self_Attn K_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1233 | llm.model.layers.7.self_attn.k_proj.weight                 | Llm Model Layers 7 Self_Attn K_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1234 | llm.model.layers.7.self_attn.o_proj.weight                 | Llm Model Layers 7 Self_Attn O_Proj (W)                 | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1235 | llm.model.layers.7.self_attn.q_norm.weight                 | Llm Model Layers 7 Self_Attn Q_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1236 | llm.model.layers.7.self_attn.q_proj.weight                 | Llm Model Layers 7 Self_Attn Q_Proj (W)                 | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1237 | llm.model.layers.7.self_attn.v_proj.weight                 | Llm Model Layers 7 Self_Attn V_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1238 | llm.model.layers.8.input_layernorm.weight                  | Llm Model Layers 8 Input_Layernorm (W)                  | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1239 | llm.model.layers.8.mlp.down_proj.weight                    | Llm Model Layers 8 Mlp Down_Proj (W)                    | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1240 | llm.model.layers.8.mlp.gate_proj.weight                    | Llm Model Layers 8 Mlp Gate_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1241 | llm.model.layers.8.mlp.up_proj.weight                      | Llm Model Layers 8 Mlp Up_Proj (W)                      | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1242 | llm.model.layers.8.post_attention_layernorm.weight         | Llm Model Layers 8 Post_Attention_Layernorm (W)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1243 | llm.model.layers.8.self_attn.k_norm.weight                 | Llm Model Layers 8 Self_Attn K_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1244 | llm.model.layers.8.self_attn.k_proj.weight                 | Llm Model Layers 8 Self_Attn K_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1245 | llm.model.layers.8.self_attn.o_proj.weight                 | Llm Model Layers 8 Self_Attn O_Proj (W)                 | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1246 | llm.model.layers.8.self_attn.q_norm.weight                 | Llm Model Layers 8 Self_Attn Q_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1247 | llm.model.layers.8.self_attn.q_proj.weight                 | Llm Model Layers 8 Self_Attn Q_Proj (W)                 | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1248 | llm.model.layers.8.self_attn.v_proj.weight                 | Llm Model Layers 8 Self_Attn V_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1249 | llm.model.layers.9.input_layernorm.weight                  | Llm Model Layers 9 Input_Layernorm (W)                  | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1250 | llm.model.layers.9.mlp.down_proj.weight                    | Llm Model Layers 9 Mlp Down_Proj (W)                    | (  ~3M)   3145728 | 3072 x   1024 x   1 x 1 | BF16 |
| 1251 | llm.model.layers.9.mlp.gate_proj.weight                    | Llm Model Layers 9 Mlp Gate_Proj (W)                    | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1252 | llm.model.layers.9.mlp.up_proj.weight                      | Llm Model Layers 9 Mlp Up_Proj (W)                      | (  ~3M)   3145728 | 1024 x   3072 x   1 x 1 | BF16 |
| 1253 | llm.model.layers.9.post_attention_layernorm.weight         | Llm Model Layers 9 Post_Attention_Layernorm (W)         | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |
| 1254 | llm.model.layers.9.self_attn.k_norm.weight                 | Llm Model Layers 9 Self_Attn K_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1255 | llm.model.layers.9.self_attn.k_proj.weight                 | Llm Model Layers 9 Self_Attn K_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1256 | llm.model.layers.9.self_attn.o_proj.weight                 | Llm Model Layers 9 Self_Attn O_Proj (W)                 | (  ~2M)   2097152 | 2048 x   1024 x   1 x 1 | BF16 |
| 1257 | llm.model.layers.9.self_attn.q_norm.weight                 | Llm Model Layers 9 Self_Attn Q_Norm (W)                 | (  128)       128 |  128 x      1 x   1 x 1 | BF16 |
| 1258 | llm.model.layers.9.self_attn.q_proj.weight                 | Llm Model Layers 9 Self_Attn Q_Proj (W)                 | (  ~2M)   2097152 | 1024 x   2048 x   1 x 1 | BF16 |
| 1259 | llm.model.layers.9.self_attn.v_proj.weight                 | Llm Model Layers 9 Self_Attn V_Proj (W)                 | (  ~1M)   1048576 | 1024 x   1024 x   1 x 1 | BF16 |
| 1260 | llm.model.norm.weight                                      | Llm Model Norm (W)                                      | (  ~1K)      1024 | 1024 x      1 x   1 x 1 | BF16 |

- Total elements in base: (~985M) 985374304
- Percentage of total elements: 100.00%



