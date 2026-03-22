// funasr/model/tokenizer.cpp
// BPE Tokenizer 实现
//
#include "model/tokenizer.hpp"
#include <gguf.h>
#include <cstdio>
#include <cstring>
#include <climits>
#include <regex>
#include <algorithm>

namespace funasr {

// ============================================================
// UTF-8 ↔ wstring 手动转换（不依赖 deprecated codecvt）
// ============================================================
std::wstring Tokenizer::utf8_to_wstring(const std::string& str) {
    std::wstring result;
    size_t i = 0;
    while (i < str.size()) {
        uint32_t cp = 0;
        unsigned char c = static_cast<unsigned char>(str[i]);

        if (c < 0x80) {
            cp = c;
            i += 1;
        } else if ((c >> 5) == 0x06) {
            if (i + 1 >= str.size()) break;
            cp = (c & 0x1F) << 6;
            cp |= (static_cast<unsigned char>(str[i + 1]) & 0x3F);
            i += 2;
        } else if ((c >> 4) == 0x0E) {
            if (i + 2 >= str.size()) break;
            cp = (c & 0x0F) << 12;
            cp |= (static_cast<unsigned char>(str[i + 1]) & 0x3F) << 6;
            cp |= (static_cast<unsigned char>(str[i + 2]) & 0x3F);
            i += 3;
        } else if ((c >> 3) == 0x1E) {
            if (i + 3 >= str.size()) break;
            cp = (c & 0x07) << 18;
            cp |= (static_cast<unsigned char>(str[i + 1]) & 0x3F) << 12;
            cp |= (static_cast<unsigned char>(str[i + 2]) & 0x3F) << 6;
            cp |= (static_cast<unsigned char>(str[i + 3]) & 0x3F);
            i += 4;
        } else {
            i += 1; // skip invalid byte
            continue;
        }
        result.push_back(static_cast<wchar_t>(cp));
    }
    return result;
}

std::string Tokenizer::wstring_to_utf8(const std::wstring& str) {
    std::string result;
    for (wchar_t wc : str) {
        uint32_t cp = static_cast<uint32_t>(wc);
        if (cp < 0x80) {
            result.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            result.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            result.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            result.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return result;
}

// ============================================================
// 构造函数
// ============================================================
Tokenizer::Tokenizer() {
    init_byte_mappings();
}

// ============================================================
// GPT-2 byte-to-unicode 映射
// 将 0-255 的每个字节映射到一个可打印的 unicode 字符
// 这是 GPT-2 / Qwen 系列 tokenizer 的标准做法
// ============================================================
void Tokenizer::init_byte_mappings() {
    b2u_.clear();
    u2b_.clear();

    // 可打印的 ASCII + Latin-1 范围直接映射
    auto insert_range = [this](int start, int end) {
        for (int c = start; c <= end; c++) {
            b2u_[static_cast<uint8_t>(c)] = static_cast<wchar_t>(c);
        }
    };

    insert_range('!', '~');     // 0x21 - 0x7E
    insert_range(0xA1, 0xAC);   // ¡ to ¬
    insert_range(0xAE, 0xFF);   // ® to ÿ

    // 剩余字节映射到 256+ 的 unicode 字符
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (b2u_.find(static_cast<uint8_t>(b)) == b2u_.end()) {
            b2u_[static_cast<uint8_t>(b)] = static_cast<wchar_t>(256 + n);
            n++;
        }
    }

    // 反向映射
    for (auto& [byte_val, wchar_val] : b2u_) {
        u2b_[wchar_val] = byte_val;
    }
}

// ============================================================
// 从 GGUFReader 加载
// ============================================================
bool Tokenizer::load(GGUFReader& reader) {
    if (!reader.is_open()) {
        printf("[Tokenizer] ERROR: reader not open\n");
        return false;
    }

    struct gguf_context* ctx = reader.gguf_ctx();

    // 1. 读取 tokens
    int tokens_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tokens_key < 0) {
        printf("[Tokenizer] ERROR: tokenizer.ggml.tokens not found\n");
        return false;
    }

    int n_tokens = gguf_get_arr_n(ctx, tokens_key);
    tokens_.resize(n_tokens);
    token_to_id_.clear();
    token_to_id_.reserve(n_tokens);

    for (int i = 0; i < n_tokens; i++) {
        const char* s = gguf_get_arr_str(ctx, tokens_key, i);
        tokens_[i] = s ? s : "";
        token_to_id_[tokens_[i]] = i;
    }

    // 2. 读取特殊 token IDs
    int idx;

    idx = gguf_find_key(ctx, "tokenizer.ggml.eos_token_id");
    if (idx >= 0) eos_id_ = static_cast<int>(gguf_get_val_u32(ctx, idx));

    idx = gguf_find_key(ctx, "tokenizer.ggml.bos_token_id");
    if (idx >= 0) bos_id_ = static_cast<int>(gguf_get_val_u32(ctx, idx));

    idx = gguf_find_key(ctx, "tokenizer.ggml.padding_token_id");
    if (idx >= 0) pad_id_ = static_cast<int>(gguf_get_val_u32(ctx, idx));

    // 3. 读取 BPE merges
    int merges_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (merges_key >= 0) {
        int n_merges = gguf_get_arr_n(ctx, merges_key);
        bpe_ranks_.reserve(n_merges);

        for (int i = 0; i < n_merges; i++) {
            const char* merge_str = gguf_get_arr_str(ctx, merges_key, i);
            if (!merge_str) continue;

            std::string s = merge_str;
            size_t space = s.find(' ');
            if (space != std::string::npos) {
                std::wstring first  = utf8_to_wstring(s.substr(0, space));
                std::wstring second = utf8_to_wstring(s.substr(space + 1));
                bpe_ranks_[{first, second}] = i;
            }
        }

        printf("[Tokenizer] Loaded %d merges\n", n_merges);
    } else {
        printf("[Tokenizer] WARNING: no BPE merges found\n");
    }

    loaded_ = true;
    printf("[Tokenizer] Loaded %d tokens, eos=%d, bos=%d\n",
           n_tokens, eos_id_, bos_id_);
    return true;
}

// ============================================================
// BPE 核心算法
// 输入: unicode 字符序列（经过 byte-to-unicode 转换后的）
// 输出: BPE token 序列
// ============================================================
void Tokenizer::bpe(const std::wstring& token, std::vector<std::wstring>& result) const {
    result.clear();

    if (token.size() <= 1) {
        result.push_back(token);
        return;
    }

    // 初始化：每个字符作为一个 token
    std::vector<std::wstring> word;
    word.reserve(token.size());
    for (wchar_t c : token) {
        word.push_back(std::wstring(1, c));
    }

    // 反复找优先级最高（rank 最小）的相邻 pair 并合并
    while (word.size() > 1) {
        int min_rank = INT_MAX;
        int min_idx = -1;

        for (size_t i = 0; i + 1 < word.size(); i++) {
            auto it = bpe_ranks_.find({word[i], word[i + 1]});
            if (it != bpe_ranks_.end() && it->second < min_rank) {
                min_rank = it->second;
                min_idx = static_cast<int>(i);
            }
        }

        if (min_idx < 0) break;  // 没有可合并的 pair

        // 合并
        word[min_idx] = word[min_idx] + word[min_idx + 1];
        word.erase(word.begin() + min_idx + 1);
    }

    result = std::move(word);
}

// ============================================================
// 编码普通文本段（不含特殊 token）
// GPT-2 style: regex 分词 → byte-to-unicode → BPE → lookup
// ============================================================
std::vector<int> Tokenizer::encode_text(const std::string& text) const {
    std::vector<int> ids;

    if (bpe_ranks_.empty()) {
        // 没有 merges，退化为字符级编码
        for (unsigned char c : text) {
            auto it = b2u_.find(c);
            if (it != b2u_.end()) {
                std::string s = wstring_to_utf8(std::wstring(1, it->second));
                auto tok_it = token_to_id_.find(s);
                if (tok_it != token_to_id_.end()) {
                    ids.push_back(tok_it->second);
                }
            }
        }
        return ids;
    }

    // GPT-2 regex pattern
    std::regex re(
        "('s|'t|'re|'ve|'m|'ll|'d"
        "| ?[[:alpha:]]+"
        "| ?[[:digit:]]+"
        "| ?[^\\s\\w]+"
        "|\\s+)"
    );

    std::string input = text;
    std::smatch match;

    while (std::regex_search(input, match, re)) {
        std::string segment = match.str(0);
        input = match.suffix().str();

        // bytes → unicode characters
        std::wstring wtoken;
        for (unsigned char c : segment) {
            auto it = b2u_.find(c);
            if (it != b2u_.end()) {
                wtoken.push_back(it->second);
            }
        }

        // BPE
        std::vector<std::wstring> bpe_tokens;
        bpe(wtoken, bpe_tokens);

        // lookup token → id
        for (const auto& ws : bpe_tokens) {
            std::string s = wstring_to_utf8(ws);
            auto it = token_to_id_.find(s);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            }
        }
    }

    return ids;
}

// ============================================================
// 编码（含特殊 token 处理）
// ============================================================
std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;

    // 特殊 token 表（按长度降序，确保最长匹配优先）
    struct SpecialToken {
        const char* text;
        int id;
    };
    const SpecialToken special_tokens[] = {
        {"<|im_start|>", 151644},
        {"<|endoftext|>", 151643},
        {"<|im_end|>",   151645},
    };

    size_t pos = 0;

    while (pos < text.size()) {
        // 检查当前位置是否匹配特殊 token
        bool found_special = false;
        for (const auto& st : special_tokens) {
            size_t len = std::strlen(st.text);
            if (text.compare(pos, len, st.text) == 0) {
                ids.push_back(st.id);
                pos += len;
                found_special = true;
                break;
            }
        }
        if (found_special) continue;

        // 找到下一个特殊 token 的位置
        size_t next_special = std::string::npos;
        for (const auto& st : special_tokens) {
            size_t found = text.find(st.text, pos);
            if (found != std::string::npos && found < next_special) {
                next_special = found;
            }
        }

        // 截取到下一个特殊 token 之前的普通文本
        std::string segment;
        if (next_special != std::string::npos) {
            segment = text.substr(pos, next_special - pos);
            pos = next_special;
        } else {
            segment = text.substr(pos);
            pos = text.size();
        }

        // 编码普通文本
        if (!segment.empty()) {
            auto seg_ids = encode_text(segment);
            ids.insert(ids.end(), seg_ids.begin(), seg_ids.end());
        }
    }

    return ids;
}

// ============================================================
// 解码单个 token
// GPT-2 style: token string → unicode chars → byte mapping
// ============================================================
std::string Tokenizer::decode_token(int id) const {
    if (id < 0 || id >= static_cast<int>(tokens_.size())) return "";

    const std::string& token = tokens_[id];

    // 跳过特殊 token（<|...|> 格式）
    if (token.size() >= 4 && token[0] == '<' && token[1] == '|') {
        return "";
    }

    // 将 token 的 unicode 字符映射回原始字节
    std::wstring w = utf8_to_wstring(token);
    std::string result;
    result.reserve(w.size());

    for (wchar_t c : w) {
        auto it = u2b_.find(c);
        if (it != u2b_.end()) {
            result.push_back(static_cast<char>(it->second));
        } else {
            // 不在映射表里，直接转 UTF-8
            std::string utf8_char = wstring_to_utf8(std::wstring(1, c));
            result += utf8_char;
        }
    }

    return result;
}

// ============================================================
// 解码 token 序列
// ============================================================
std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;

    for (int id : ids) {
        // 跳过特殊 token
        if (id == eos_id_ || id == bos_id_ || id == pad_id_) {
            continue;
        }
        result += decode_token(id);
    }

    return result;
}

// ============================================================
// 获取 token 文本
// ============================================================
const std::string& Tokenizer::token_text(int id) const {
    static const std::string empty;
    if (id < 0 || id >= static_cast<int>(tokens_.size())) return empty;
    return tokens_[id];
}

} // namespace funasr