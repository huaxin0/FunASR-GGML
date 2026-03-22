// Fbank 特征提取器 — 纯 CPU 计算，不涉及 GGML
//
// 数据流:
//   WAV (16kHz PCM) → float audio
//     → 分帧 → 去直流 → 预加重 → Hamming窗 → FFT → 功率谱
//     → Mel 滤波(80bins) → log 压缩
//     → LFR (m=7, n=6): 7帧拼接, 步长6
//     → 输出 [T, 560] F32 (行优先)
//
// 所有参数从 FrontendConfig 读取，不硬编码。
//
#ifndef FUNASR_COMPUTE_FBANK_HPP
#define FUNASR_COMPUTE_FBANK_HPP

#include "core/config.hpp"
#include <vector>
#include <string>
#include <cmath>
namespace funasr {

class FbankProcessor {
public:
    // 用 Config 初始化
    explicit FbankProcessor(const FrontendConfig& config);

    // ============================================================
    // 核心接口：float 音频 → fbank+LFR 特征
    //
    // audio:  归一化到 ±1.0 的 float 音频数据
    // n_samples: 采样点数
    // fbank_out: 输出特征，行优先 [T × D]
    // T: 输出帧数 (LFR 后)
    // D: 输出维度 (= num_mels × lfr_m = 560)
    //
    // 内部会做 ×32768 upscale（Kaldi 惯例）
    // ============================================================
    bool process(const float* audio, size_t n_samples,
                 std::vector<float>& fbank_out, int& T, int& D);

    // ============================================================
    // 便捷接口：从 WAV 文件直接处理
    // ============================================================
    bool process_wav(const std::string& wav_path,
                     std::vector<float>& fbank_out, int& T, int& D);

    // 只读 WAV，不做 fbank（用于需要原始音频的场景）
    bool read_wav(const std::string& wav_path,
                  std::vector<float>& audio, int& sample_rate);

    // 访问配置
    const FrontendConfig& config() const { return config_; }

private:
    FrontendConfig config_;

    // 预计算的窗和滤波器
    std::vector<float> window_;                        // Hamming 窗 [frame_len]
    std::vector<std::vector<float>> mel_filterbank_;   // [num_mels][n_fft/2+1]

    // FFT 参数
    int n_fft_ = 512;
    int frame_len_;    // = sample_rate * frame_length / 1000
    int frame_shift_;  // = sample_rate * frame_shift / 1000

    float preemphasis_ = 0.97f;

    // 初始化
    void init_window();
    void init_mel_filterbank();

    // Mel 频率转换 (HTK 标准)
    static float hz_to_mel(float hz) { return 1127.0f * std::log(1.0f + hz / 700.0f); }
    static float mel_to_hz(float mel) { return 700.0f * (std::exp(mel / 1127.0f) - 1.0f); }

    // FFT (迭代 Cooley-Tukey)
    void fft(std::vector<float>& real, std::vector<float>& imag) const;

    // 功率谱
    void compute_power_spectrum(const std::vector<float>& frame,
                                std::vector<float>& power_spec) const;

    // 单帧 Fbank (去直流 → 预加重 → 窗 → FFT → Mel → log)
    void compute_frame_fbank(const float* frame_raw,
                             std::vector<float>& mel_out) const;

    // LFR 下采样
    void apply_lfr(const std::vector<std::vector<float>>& fbank_frames,
                   std::vector<float>& lfr_out, int& T, int& D) const;
};

} // namespace funasr

#endif // FUNASR_COMPUTE_FBANK_HPP