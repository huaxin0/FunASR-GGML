// funasr/model/loader.cpp
// 模型加载实现
//
// 这里是 "GGUF tensor name" 和 "结构体字段" 的映射关系所在。
// GGUF 里的名字是转换脚本写入时决定的（不可改），
// 结构体字段名是我们的内部命名（随便叫）。
// 这个文件负责把两者连接起来。
//
#include "model/loader.hpp"
#include <cstdio>
#include <string>

namespace funasr {

// ============================================================
// 加载 Encoder 单层
// prefix 示例:
//   "audio_encoder.encoders0.0"   (encoder0)
//   "audio_encoder.encoders.0"    (main encoder layer 0)
//   "audio_encoder.tp_encoders.0" (tp encoder layer 0)
// ============================================================
void ModelLoader::load_encoder_layer(
    GGUFReader& reader,
    EncoderLayerWeights& layer,
    const std::string& prefix
) {
    // Attention Q/K/V/Out
    layer.attn_q_w   = reader.require_tensor(prefix + ".self_attn.linear_q.weight");
    layer.attn_q_b   = reader.require_tensor(prefix + ".self_attn.linear_q.bias");
    layer.attn_k_w   = reader.require_tensor(prefix + ".self_attn.linear_k.weight");
    layer.attn_k_b   = reader.require_tensor(prefix + ".self_attn.linear_k.bias");
    layer.attn_v_w   = reader.require_tensor(prefix + ".self_attn.linear_v.weight");
    layer.attn_v_b   = reader.require_tensor(prefix + ".self_attn.linear_v.bias");
    layer.attn_out_w = reader.require_tensor(prefix + ".self_attn.linear_out.weight");
    layer.attn_out_b = reader.require_tensor(prefix + ".self_attn.linear_out.bias");

    // FSMN
    layer.fsmn_w     = reader.require_tensor(prefix + ".self_attn.fsmn_block.weight");

    // FFN
    layer.ffn_w1     = reader.require_tensor(prefix + ".feed_forward.w_1.weight");
    layer.ffn_b1     = reader.require_tensor(prefix + ".feed_forward.w_1.bias");
    layer.ffn_w2     = reader.require_tensor(prefix + ".feed_forward.w_2.weight");
    layer.ffn_b2     = reader.require_tensor(prefix + ".feed_forward.w_2.bias");

    // Layer Norm
    layer.norm1_w    = reader.require_tensor(prefix + ".norm1.weight");
    layer.norm1_b    = reader.require_tensor(prefix + ".norm1.bias");
    layer.norm2_w    = reader.require_tensor(prefix + ".norm2.weight");
    layer.norm2_b    = reader.require_tensor(prefix + ".norm2.bias");
}

// ============================================================
// 加载 Adaptor Block
// GGUF name: audio_adaptor.blocks.{i}.self_attn.linear_q.weight ...
// ============================================================
void ModelLoader::load_adaptor_block(
    GGUFReader& reader,
    AdaptorBlockWeights& block,
    int block_idx
) {
    std::string prefix = "audio_adaptor.blocks." + std::to_string(block_idx);

    // Attention Q/K/V/Out
    block.attn_q_w   = reader.require_tensor(prefix + ".self_attn.linear_q.weight");
    block.attn_q_b   = reader.require_tensor(prefix + ".self_attn.linear_q.bias");
    block.attn_k_w   = reader.require_tensor(prefix + ".self_attn.linear_k.weight");
    block.attn_k_b   = reader.require_tensor(prefix + ".self_attn.linear_k.bias");
    block.attn_v_w   = reader.require_tensor(prefix + ".self_attn.linear_v.weight");
    block.attn_v_b   = reader.require_tensor(prefix + ".self_attn.linear_v.bias");
    block.attn_out_w = reader.require_tensor(prefix + ".self_attn.linear_out.weight");
    block.attn_out_b = reader.require_tensor(prefix + ".self_attn.linear_out.bias");

    // FFN (中间维度 256)
    block.ffn_w1     = reader.require_tensor(prefix + ".feed_forward.w_1.weight");
    block.ffn_b1     = reader.require_tensor(prefix + ".feed_forward.w_1.bias");
    block.ffn_w2     = reader.require_tensor(prefix + ".feed_forward.w_2.weight");
    block.ffn_b2     = reader.require_tensor(prefix + ".feed_forward.w_2.bias");

    // Layer Norm
    block.norm1_w    = reader.require_tensor(prefix + ".norm1.weight");
    block.norm1_b    = reader.require_tensor(prefix + ".norm1.bias");
    block.norm2_w    = reader.require_tensor(prefix + ".norm2.weight");
    block.norm2_b    = reader.require_tensor(prefix + ".norm2.bias");
}

// ============================================================
// 加载 LLM 单层
// GGUF name: llm.model.layers.{i}.self_attn.q_proj.weight ...
// ============================================================
void ModelLoader::load_llm_layer(
    GGUFReader& reader,
    LLMLayerWeights& layer,
    int layer_idx
) {
    std::string prefix = "llm.model.layers." + std::to_string(layer_idx);

    // GQA Attention
    layer.q_proj_w       = reader.require_tensor(prefix + ".self_attn.q_proj.weight");
    layer.k_proj_w       = reader.require_tensor(prefix + ".self_attn.k_proj.weight");
    layer.v_proj_w       = reader.require_tensor(prefix + ".self_attn.v_proj.weight");
    layer.o_proj_w       = reader.require_tensor(prefix + ".self_attn.o_proj.weight");
    layer.q_norm_w       = reader.require_tensor(prefix + ".self_attn.q_norm.weight");
    layer.k_norm_w       = reader.require_tensor(prefix + ".self_attn.k_norm.weight");

    // SwiGLU MLP
    layer.gate_proj_w    = reader.require_tensor(prefix + ".mlp.gate_proj.weight");
    layer.up_proj_w      = reader.require_tensor(prefix + ".mlp.up_proj.weight");
    layer.down_proj_w    = reader.require_tensor(prefix + ".mlp.down_proj.weight");

    // RMS Norm
    layer.input_norm_w     = reader.require_tensor(prefix + ".input_layernorm.weight");
    layer.post_attn_norm_w = reader.require_tensor(prefix + ".post_attention_layernorm.weight");
}

// ============================================================
// 加载 Config（从 GGUF metadata）
// ============================================================
bool ModelLoader::load_config(GGUFReader& reader, ModelConfig& config) {
    if (!reader.is_open()) {
        printf("[Loader] ERROR: reader not open\n");
        return false;
    }
    return config.load_from_gguf(reader.gguf_ctx());
}

// ============================================================
// 加载 Encoder
// ============================================================
bool ModelLoader::load_encoder(
    GGUFReader& reader,
    const EncoderConfig& cfg,
    EncoderWeights& weights
) {
    printf("[Loader] Loading Audio Encoder...\n");
    reader.clear_errors();

    // encoder0 (特殊第一层，input 560→512)
    load_encoder_layer(reader, weights.encoder0, "audio_encoder.encoders0.0");

    // main encoders (49 层)
    int n_main = cfg.num_main_blocks();
    weights.main_layers.resize(n_main);
    for (int i = 0; i < n_main; i++) {
        load_encoder_layer(reader, weights.main_layers[i],
                           "audio_encoder.encoders." + std::to_string(i));
    }

    // tp encoders (20 层)
    weights.tp_layers.resize(cfg.tp_blocks);
    for (int i = 0; i < cfg.tp_blocks; i++) {
        load_encoder_layer(reader, weights.tp_layers[i],
                           "audio_encoder.tp_encoders." + std::to_string(i));
    }

    // 顶层 norm
    weights.after_norm_w = reader.require_tensor("audio_encoder.after_norm.weight");
    weights.after_norm_b = reader.require_tensor("audio_encoder.after_norm.bias");
    weights.tp_norm_w    = reader.require_tensor("audio_encoder.tp_norm.weight");
    weights.tp_norm_b    = reader.require_tensor("audio_encoder.tp_norm.bias");

    if (reader.has_errors()) {
        printf("[Loader] Encoder: %zu tensor(s) missing\n", reader.missing_tensors().size());
        reader.print_errors();
        return false;
    }

    printf("[Loader] Encoder loaded: %d tensors [OK]\n", weights.tensor_count());
    return true;
}

// ============================================================
// 加载 Adaptor
// ============================================================
bool ModelLoader::load_adaptor(
    GGUFReader& reader,
    const AdaptorConfig& cfg,
    AdaptorWeights& weights
) {
    printf("[Loader] Loading Audio Adaptor...\n");
    reader.clear_errors();

    // Linear layers
    weights.linear1_w = reader.require_tensor("audio_adaptor.linear1.weight");
    weights.linear1_b = reader.require_tensor("audio_adaptor.linear1.bias");
    weights.linear2_w = reader.require_tensor("audio_adaptor.linear2.weight");
    weights.linear2_b = reader.require_tensor("audio_adaptor.linear2.bias");

    // Attention blocks
    weights.blocks.resize(cfg.num_blocks);
    for (int i = 0; i < cfg.num_blocks; i++) {
        load_adaptor_block(reader, weights.blocks[i], i);
    }

    if (reader.has_errors()) {
        printf("[Loader] Adaptor: %zu tensor(s) missing\n", reader.missing_tensors().size());
        reader.print_errors();
        return false;
    }

    printf("[Loader] Adaptor loaded: %d tensors [OK]\n", weights.tensor_count());
    return true;
}

// ============================================================
// 加载 LLM Decoder
// ============================================================
bool ModelLoader::load_llm(
    GGUFReader& reader,
    const LLMConfig& cfg,
    LLMWeights& weights
) {
    printf("[Loader] Loading LLM Decoder...\n");
    reader.clear_errors();

    // Embedding
    weights.embed_tokens = reader.require_tensor("llm.model.embed_tokens.weight");

    // 28 Layers
    weights.layers.resize(cfg.block_count);
    for (int i = 0; i < cfg.block_count; i++) {
        load_llm_layer(reader, weights.layers[i], i);
    }

    // Final norm + LM head
    weights.model_norm_w = reader.require_tensor("llm.model.norm.weight");
    weights.lm_head_w    = reader.require_tensor("llm.lm_head.weight");

    if (reader.has_errors()) {
        printf("[Loader] LLM: %zu tensor(s) missing\n", reader.missing_tensors().size());
        reader.print_errors();
        return false;
    }

    printf("[Loader] LLM loaded: %d tensors [OK]\n", weights.tensor_count());
    return true;
}

// ============================================================
// 一站式加载
// ============================================================
bool ModelLoader::load(GGUFReader& reader, FunASRModel& model) {
    printf("\n========================================\n");
    printf("  Loading FunASR Model\n");
    printf("========================================\n");

    if (!reader.is_open()) {
        printf("[Loader] ERROR: reader not open\n");
        return false;
    }

    printf("File: %s\n", reader.path().c_str());
    printf("Tensors: %d, KV pairs: %d\n\n", reader.tensor_count(), reader.kv_count());

    // Step 1: Config
    if (!load_config(reader, model.config)) {
        printf("[Loader] Config loading failed\n");
        return false;
    }

    // Step 2: Encoder
    if (!load_encoder(reader, model.config.encoder, model.encoder)) {
        return false;
    }

    // Step 3: Adaptor
    if (!load_adaptor(reader, model.config.adaptor, model.adaptor)) {
        return false;
    }

    // Step 4: LLM
    if (!load_llm(reader, model.config.llm, model.llm)) {
        return false;
    }

    // Summary
    printf("\n========================================\n");
    printf("  Loading Complete!\n");
    printf("========================================\n");
    model.print_summary();

    // 验证总数
    int expected = reader.tensor_count();
    int actual = model.tensor_count();
    if (actual != expected) {
        printf("[Loader] WARNING: loaded %d tensors, GGUF has %d\n", actual, expected);
    }

    return true;
}

} // namespace funasr