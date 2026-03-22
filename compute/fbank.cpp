// Fbank 特征提取实现
//
#include "compute/fbank.hpp"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace funasr {

// ============================================================
// 构造函数
// ============================================================
FbankProcessor::FbankProcessor(const FrontendConfig& config)
    : config_(config)
{
    frame_len_   = config_.sample_rate * config_.frame_length / 1000;  // 400
    frame_shift_ = config_.sample_rate * config_.frame_shift  / 1000;  // 160

    init_window();
    init_mel_filterbank();
}

// ============================================================
// Hamming 窗初始化
// ============================================================
void FbankProcessor::init_window() {
    window_.resize(frame_len_);
    double a = 2.0 * M_PI / (frame_len_ - 1);
    for (int i = 0; i < frame_len_; i++) {
        window_[i] = static_cast<float>(0.54 - 0.46 * std::cos(a * i));
    }
}

// ============================================================
// Mel 滤波器组初始化 (HTK 标准)
// ============================================================
void FbankProcessor::init_mel_filterbank() {
    int n_fft_bins = n_fft_ / 2 + 1;  // 257
    float mel_low  = hz_to_mel(20.0f);
    float mel_high = hz_to_mel(config_.sample_rate / 2.0f);

    // 均匀分布的 mel 点
    std::vector<float> mel_points(config_.num_mels + 2);
    for (int i = 0; i < config_.num_mels + 2; i++) {
        mel_points[i] = mel_low + i * (mel_high - mel_low) / (config_.num_mels + 1);
    }

    // 频率 bin 对应的 Hz
    std::vector<float> bin_freqs(n_fft_bins);
    for (int i = 0; i < n_fft_bins; i++) {
        bin_freqs[i] = (config_.sample_rate / 2.0f) * i / (n_fft_bins - 1);
    }

    // 构建三角滤波器
    mel_filterbank_.resize(config_.num_mels);
    for (int m = 0; m < config_.num_mels; m++) {
        mel_filterbank_[m].resize(n_fft_bins, 0.0f);
        float left_mel   = mel_points[m];
        float center_mel = mel_points[m + 1];
        float right_mel  = mel_points[m + 2];

        for (int k = 0; k < n_fft_bins; k++) {
            float mel = hz_to_mel(bin_freqs[k]);
            if (mel > left_mel && mel < right_mel) {
                if (mel <= center_mel) {
                    mel_filterbank_[m][k] = (mel - left_mel) / (center_mel - left_mel);
                } else {
                    mel_filterbank_[m][k] = (right_mel - mel) / (right_mel - center_mel);
                }
            }
        }
    }
}

// ============================================================
// FFT (迭代 Cooley-Tukey, bit-reversal)
// ============================================================
void FbankProcessor::fft(std::vector<float>& real, std::vector<float>& imag) const {
    int n = static_cast<int>(real.size());
    if (n <= 1) return;

    // Bit-reversal permutation
    int j = 0;
    for (int i = 1; i < n - 1; i++) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    // Butterfly
    for (int len = 2; len <= n; len *= 2) {
        float angle = -2.0f * static_cast<float>(M_PI) / len;
        float wpr = std::cos(angle), wpi = std::sin(angle);
        for (int i = 0; i < n; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                int u = i + k, v = i + k + len / 2;
                float tr = wr * real[v] - wi * imag[v];
                float ti = wr * imag[v] + wi * real[v];
                real[v] = real[u] - tr;
                imag[v] = imag[u] - ti;
                real[u] = real[u] + tr;
                imag[u] = imag[u] + ti;
                float new_wr = wr * wpr - wi * wpi;
                wi = wr * wpi + wi * wpr;
                wr = new_wr;
            }
        }
    }
}

// ============================================================
// 功率谱
// ============================================================
void FbankProcessor::compute_power_spectrum(
    const std::vector<float>& frame,
    std::vector<float>& power_spec
) const {
    std::vector<float> real(n_fft_, 0.0f), imag(n_fft_, 0.0f);
    for (size_t i = 0; i < frame.size() && i < static_cast<size_t>(n_fft_); i++) {
        real[i] = frame[i];
    }
    fft(real, imag);

    int n_bins = n_fft_ / 2 + 1;
    power_spec.resize(n_bins);
    for (int k = 0; k < n_bins; k++) {
        power_spec[k] = real[k] * real[k] + imag[k] * imag[k];
    }
}

// ============================================================
// 单帧 Fbank 计算
// frame_raw: 指向 frame_len_ 个 float 的指针（已 upscale 的音频）
// mel_out: 输出 num_mels 个 log-mel 值
// ============================================================
void FbankProcessor::compute_frame_fbank(
    const float* frame_raw,
    std::vector<float>& mel_out
) const {
    std::vector<float> frame(frame_len_);

    // 1. 提取帧
    std::memcpy(frame.data(), frame_raw, frame_len_ * sizeof(float));

    // 2. 去直流偏移
    float mean_val = 0.0f;
    for (int j = 0; j < frame_len_; j++) {
        mean_val += frame[j];
    }
    mean_val /= frame_len_;
    for (int j = 0; j < frame_len_; j++) {
        frame[j] -= mean_val;
    }

    // 3. 预加重 (从后往前)
    for (int j = frame_len_ - 1; j >= 1; j--) {
        frame[j] -= preemphasis_ * frame[j - 1];
    }
    frame[0] -= preemphasis_ * frame[0];

    // 4. 加窗
    for (int j = 0; j < frame_len_; j++) {
        frame[j] *= window_[j];
    }

    // 5. 功率谱
    std::vector<float> power_spec;
    compute_power_spectrum(frame, power_spec);

    // 6. Mel 滤波 + log 压缩
    mel_out.resize(config_.num_mels);
    for (int m = 0; m < config_.num_mels; m++) {
        float sum = 0.0f;
        for (size_t k = 0; k < power_spec.size(); k++) {
            sum += power_spec[k] * mel_filterbank_[m][k];
        }
        if (sum < 1e-10f) sum = 1e-10f;
        mel_out[m] = std::log(sum);
    }
}

// ============================================================
// LFR 下采样
// 输入: fbank_frames[num_frames][num_mels]
// 输出: lfr_out [T * D] 行优先
// ============================================================
void FbankProcessor::apply_lfr(
    const std::vector<std::vector<float>>& fbank_frames,
    std::vector<float>& lfr_out,
    int& T, int& D
) const {
    int num_frames = static_cast<int>(fbank_frames.size());
    int lfr_m = config_.lfr_m;
    int lfr_n = config_.lfr_n;
    D = config_.num_mels * lfr_m;  // 80 × 7 = 560

    // 计算 LFR 帧数
    T = 0;
    for (int i = 0; i + lfr_m <= num_frames; i += lfr_n) {
        T++;
    }

    if (T <= 0) {
        lfr_out.clear();
        return;
    }

    lfr_out.resize(T * D);
    int out_idx = 0;
    for (int i = 0; i + lfr_m <= num_frames; i += lfr_n) {
        for (int j = 0; j < lfr_m; j++) {
            for (int d = 0; d < config_.num_mels; d++) {
                lfr_out[out_idx++] = fbank_frames[i + j][d];
            }
        }
    }
}

// ============================================================
// 核心接口：float 音频 → fbank+LFR 特征
// ============================================================
bool FbankProcessor::process(
    const float* audio, size_t n_samples,
    std::vector<float>& fbank_out, int& T, int& D
) {
    if (!audio || n_samples == 0) {
        T = 0; D = 0;
        return false;
    }

    // Upscale (Kaldi 惯例: float ±1.0 → int16 范围)
    std::vector<float> upscaled(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        upscaled[i] = audio[i] * 32768.0f;
    }

    // 分帧计算 fbank
    int num_frames = 1 + static_cast<int>((n_samples - frame_len_) / frame_shift_);
    if (num_frames <= 0) {
        T = 0; D = 0;
        return false;
    }

    std::vector<std::vector<float>> fbank_frames(num_frames);
    for (int i = 0; i < num_frames; i++) {
        int start = i * frame_shift_;
        compute_frame_fbank(upscaled.data() + start, fbank_frames[i]);
    }

    // LFR
    apply_lfr(fbank_frames, fbank_out, T, D);

    return T > 0;
}

// ============================================================
// WAV 文件读取 (16-bit PCM)
// ============================================================
bool FbankProcessor::read_wav(
    const std::string& wav_path,
    std::vector<float>& audio,
    int& sample_rate
) {
    FILE* fp = fopen(wav_path.c_str(), "rb");
    if (!fp) {
        printf("[Fbank] ERROR: cannot open '%s'\n", wav_path.c_str());
        return false;
    }

    // RIFF header
    char riff[4];
    if (fread(riff, 1, 4, fp) != 4 || strncmp(riff, "RIFF", 4) != 0) {
        fclose(fp); return false;
    }

    uint32_t file_size;
    if (fread(&file_size, 4, 1, fp) != 1) { fclose(fp); return false; }

    char wave[4];
    if (fread(wave, 1, 4, fp) != 4 || strncmp(wave, "WAVE", 4) != 0) {
        fclose(fp); return false;
    }

    // Parse chunks
    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t wav_sample_rate = 0, data_size = 0;

    while (!feof(fp)) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, fp) != 4) break;
        if (fread(&chunk_size, 4, 1, fp) != 1) break;

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            if (fread(&audio_format, 2, 1, fp) != 1) break;
            if (fread(&num_channels, 2, 1, fp) != 1) break;
            if (fread(&wav_sample_rate, 4, 1, fp) != 1) break;
            fseek(fp, 6, SEEK_CUR);  // skip byte_rate + block_align
            if (fread(&bits_per_sample, 2, 1, fp) != 1) break;
            if (chunk_size > 16) {
                fseek(fp, chunk_size - 16, SEEK_CUR);
            }
        } else if (strncmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            break;
        } else {
            fseek(fp, chunk_size, SEEK_CUR);
        }
    }

    if (audio_format != 1 || bits_per_sample != 16) {
        printf("[Fbank] ERROR: only 16-bit PCM WAV supported\n");
        fclose(fp);
        return false;
    }

    sample_rate = static_cast<int>(wav_sample_rate);
    int num_samples = data_size / (num_channels * (bits_per_sample / 8));

    std::vector<int16_t> raw(num_samples * num_channels);
    size_t read_count = fread(raw.data(), sizeof(int16_t), num_samples * num_channels, fp);
    fclose(fp);

    if (static_cast<int>(read_count) != num_samples * num_channels) {
        return false;
    }

    // int16 → float ±1.0，多声道取平均
    audio.resize(num_samples);
    for (int i = 0; i < num_samples; i++) {
        float sum = 0.0f;
        for (int c = 0; c < num_channels; c++) {
            sum += raw[i * num_channels + c] / 32768.0f;
        }
        audio[i] = sum / num_channels;
    }

    return true;
}

// ============================================================
// 便捷接口：WAV → fbank
// ============================================================
bool FbankProcessor::process_wav(
    const std::string& wav_path,
    std::vector<float>& fbank_out, int& T, int& D
) {
    std::vector<float> audio;
    int wav_sr;
    if (!read_wav(wav_path, audio, wav_sr)) {
        return false;
    }

    if (wav_sr != config_.sample_rate) {
        printf("[Fbank] WARNING: WAV sample rate %d != expected %d, no resampling\n",
               wav_sr, config_.sample_rate);
    }

    return process(audio.data(), audio.size(), fbank_out, T, D);
}

} // namespace funasr