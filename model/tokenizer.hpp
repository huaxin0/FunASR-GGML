// funasr/model/tokenizer.hpp
// BPE Tokenizer — 从 GGUF metadata 加载，GPT-2 style byte encoding
//
// 设计原则：
//   1. 从 GGUFReader 加载（复用已打开的文件，不重复 IO）
//   2. 完整实现 encode (text→ids) 和 decode (ids→text)
//   3. 特殊 token (<|im_start|> 等) 在 encode 时自动识别
//   4. 内部状态全 private，对外只暴露必要接口
//
#ifndef FUNASR_MODEL_TOKENIZER_HPP
#define FUNASR_MODEL_TOKENIZER_HPP

#include "core/gguf_reader.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <cstdint>

namespace funasr {

class Tokenizer {
public:
    Tokenizer();

    // ============================================================
    // 加载（从 GGUFReader，复用已打开的 GGUF context）
    // ============================================================
    bool load(GGUFReader& reader);

    // ============================================================
    // 编码：text → token ids
    // 自动处理特殊 token (<|im_start|>, <|im_end|> 等)
    // ============================================================
    std::vector<int> encode(const std::string& text) const;

    // ============================================================
    // 解码：token ids → text
    // 跳过特殊 token，处理 GPT-2 byte encoding
    // ============================================================
    std::string decode(const std::vector<int>& ids) const;

    // 解码单个 token
    std::string decode_token(int id) const;

    // ============================================================
    // 查询接口
    // ============================================================
    int  eos_id()     const { return eos_id_; }
    int  bos_id()     const { return bos_id_; }
    int  pad_id()     const { return pad_id_; }
    int  vocab_size() const { return static_cast<int>(tokens_.size()); }
    bool is_loaded()  const { return loaded_; }

    // 获取 token 文本（调试用）
    const std::string& token_text(int id) const;

private:
    // ===== 数据 =====
    std::vector<std::string> tokens_;                  // id → token string
    std::unordered_map<std::string, int> token_to_id_; // token string → id

    // BPE merges：(pair_first, pair_second) → merge rank
    struct PairHash {
        size_t operator()(const std::pair<std::wstring, std::wstring>& p) const {
            auto h1 = std::hash<std::wstring>{}(p.first);
            auto h2 = std::hash<std::wstring>{}(p.second);
            return h1 ^ (h2 * 2654435761u);  // 比 XOR 好的 hash combine
        }
    };
    using BPERanks = std::unordered_map<
        std::pair<std::wstring, std::wstring>, int, PairHash>;
    BPERanks bpe_ranks_;

    // GPT-2 byte-to-unicode 映射
    std::unordered_map<uint8_t, wchar_t> b2u_;
    std::unordered_map<wchar_t, uint8_t> u2b_;

    // 特殊 token
    int eos_id_ = 151645;
    int bos_id_ = 151643;
    int pad_id_ = -1;

    bool loaded_ = false;

    // ===== 内部方法 =====

    // 初始化 GPT-2 byte-to-unicode 映射表
    void init_byte_mappings();

    // BPE 核心算法：将 wstring 拆分成 BPE token 序列
    void bpe(const std::wstring& token, std::vector<std::wstring>& result) const;

    // 编码普通文本段（不含特殊 token）
    std::vector<int> encode_text(const std::string& text) const;

    // ===== UTF-8 ↔ wstring 转换 =====
    static std::wstring utf8_to_wstring(const std::string& str);
    static std::string  wstring_to_utf8(const std::wstring& str);
};

} // namespace funasr

#endif // FUNASR_MODEL_TOKENIZER_HPP