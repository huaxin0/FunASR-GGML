// funasr/pipeline/audio_capture.cpp
// 麦克风捕获实现
//
// MINIAUDIO_IMPLEMENTATION 只在这一个文件中定义
//
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "pipeline/audio_capture.hpp"
#include <cstdio>

namespace funasr {

// miniaudio 回调（C 函数，转发到 AudioCapture 实例）
static void ma_data_callback(ma_device* device, void* output, const void* input, ma_uint32 frame_count) {
    (void)output;
    auto* capture = static_cast<AudioCapture*>(device->pUserData);
    if (capture && input) {
        capture->on_audio_data(static_cast<const float*>(input), frame_count);
    }
}

AudioCapture::AudioCapture() = default;

AudioCapture::~AudioCapture() {
    stop();
    if (device_) {
        ma_device_uninit(device_);
        delete device_;
        device_ = nullptr;
    }
}

bool AudioCapture::init(int sample_rate) {
    sample_rate_ = sample_rate;

    device_ = new ma_device();

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format   = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate       = static_cast<ma_uint32>(sample_rate_);
    config.dataCallback     = ma_data_callback;
    config.pUserData        = this;

    if (ma_device_init(nullptr, &config, device_) != MA_SUCCESS) {
        printf("[AudioCapture] Failed to init device\n");
        delete device_;
        device_ = nullptr;
        return false;
    }

    printf("[AudioCapture] Device initialized: %dHz, mono, f32\n", sample_rate_);
    return true;
}

bool AudioCapture::start() {
    if (!device_) {
        printf("[AudioCapture] Not initialized\n");
        return false;
    }

    if (ma_device_start(device_) != MA_SUCCESS) {
        printf("[AudioCapture] Failed to start\n");
        return false;
    }

    running_ = true;
    printf("[AudioCapture] Recording started\n");
    return true;
}

void AudioCapture::stop() {
    if (device_ && running_) {
        ma_device_stop(device_);
        running_ = false;
        printf("[AudioCapture] Recording stopped\n");
    }
}

void AudioCapture::on_audio_data(const float* samples, uint32_t frame_count) {
    if (callback_) {
        callback_(samples, static_cast<size_t>(frame_count));
    }
}

} // namespace funasr