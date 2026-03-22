// funasr/pipeline/audio_capture.hpp
// 麦克风捕获封装 (miniaudio)
//
// miniaudio 是 header-only，回调在音频线程中执行。
// 这个类只负责捕获，不做任何处理。
//
#ifndef FUNASR_PIPELINE_AUDIO_CAPTURE_HPP
#define FUNASR_PIPELINE_AUDIO_CAPTURE_HPP

#include <functional>
#include <cstdio>
#include <cstdint>

// miniaudio forward declare — 实际 include 在 .cpp 中
struct ma_device;

namespace funasr {

// 音频数据回调: samples 是 float ±1.0, count 是采样点数
using AudioCallback = std::function<void(const float* samples, size_t count)>;

class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();

    // 禁用拷贝
    AudioCapture(const AudioCapture&) = delete;
    AudioCapture& operator=(const AudioCapture&) = delete;

    // 初始化设备
    bool init(int sample_rate = 16000);

    // 开始/停止捕获
    bool start();
    void stop();

    // 设置数据回调（在 init 之前调用）
    void set_callback(AudioCallback cb) { callback_ = std::move(cb); }

    bool is_running() const { return running_; }

    // miniaudio 回调入口（内部使用）
    void on_audio_data(const float* samples, uint32_t frame_count);

private:
    ma_device* device_ = nullptr;
    AudioCallback callback_;
    bool running_ = false;
    int sample_rate_ = 16000;
};

} // namespace funasr

#endif // FUNASR_PIPELINE_AUDIO_CAPTURE_HPP