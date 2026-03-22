// funasr/pipeline/realtime.hpp
// 实时识别器 — VAD 检测 + 语音分段 + 推理
//
// 工作流:
//   麦克风回调 → feed_audio() → 内部 buffer
//   处理线程 (100ms 一次):
//     1. 校准噪声基底 (前 2 秒)
//     2. VAD: 能量 > 阈值 → 语音
//     3. 语音段结束 (静音超时或超长) → 触发识别
//     4. 回调输出结果
//
#ifndef FUNASR_PIPELINE_REALTIME_HPP
#define FUNASR_PIPELINE_REALTIME_HPP

#include "pipeline/recognizer.hpp"
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>

namespace funasr {

// 实时配置
struct RealtimeConfig {
    // VAD 参数
    int   check_interval_ms    = 100;    // 检查间隔
    int   calibration_frames   = 20;     // 校准帧数 (× interval = 2秒)
    float threshold_factor     = 3.0f;   // 语音阈值 = noise_floor × factor
    float min_threshold        = 0.00001f; // 最低阈值
    int   min_speech_ms        = 1000;   // 最短语音段 (ms)
    int   max_speech_ms        = 5000;   // 最长语音段 (ms)
    int   silence_trigger_ms   = 600;    // 静音多久触发识别 (ms)
    int   tail_trim_ms         = 150;    // 裁剪尾部静音 (ms)

    // 推理配置
    InferenceConfig inference;
};

// 每句识别结果回调
using UtteranceCallback = std::function<void(
    int utterance_id,
    const std::string& text,
    float audio_sec,
    float inference_ms
)>;

class RealtimeRecognizer {
public:
    explicit RealtimeRecognizer(Recognizer& recognizer)
        : recognizer_(recognizer) {}

    ~RealtimeRecognizer() { stop(); }

    RealtimeRecognizer(const RealtimeRecognizer&) = delete;
    RealtimeRecognizer& operator=(const RealtimeRecognizer&) = delete;

    // ============================================================
    // 启动处理线程
    // ============================================================
    bool start(const RealtimeConfig& config = RealtimeConfig(),
               UtteranceCallback callback = nullptr)
    {
        if (running_) return false;

        config_ = config;
        callback_ = std::move(callback);
        running_ = true;
        utterance_count_ = 0;

        process_thread_ = std::thread(&RealtimeRecognizer::process_loop, this);
        return true;
    }

    // ============================================================
    // 停止
    // ============================================================
    void stop() {
        running_ = false;
        if (process_thread_.joinable()) {
            process_thread_.join();
        }
    }

    // ============================================================
    // 喂入音频数据（从麦克风回调调用，线程安全）
    // ============================================================
    void feed_audio(const float* samples, size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        audio_buffer_.insert(audio_buffer_.end(), samples, samples + count);
    }

    bool is_running()      const { return running_; }
    int  utterance_count()  const { return utterance_count_; }
    long total_audio_ms()   const { return total_audio_ms_; }
    long total_inference_ms() const { return total_inference_ms_; }

private:
    Recognizer& recognizer_;
    RealtimeConfig config_;
    UtteranceCallback callback_;

    // 音频 buffer (线程安全)
    std::vector<float> audio_buffer_;
    std::mutex mutex_;

    // 处理线程
    std::thread process_thread_;
    std::atomic<bool> running_{false};

    // VAD 状态
    enum class State { CALIBRATING, IDLE, SPEECH, SILENCE };

    int utterance_count_ = 0;
    long total_audio_ms_ = 0;
    long total_inference_ms_ = 0;

    // ============================================================
    // 能量计算
    // ============================================================
    static float compute_energy(const std::vector<float>& audio) {
        if (audio.empty()) return 0.0f;
        float energy = 0.0f;
        for (float s : audio) energy += s * s;
        return energy / audio.size();
    }

    // ============================================================
    // 处理循环
    // ============================================================
    void process_loop() {
        State state = State::CALIBRATING;
        float noise_floor = 0.0f;
        float speech_threshold = 0.0f;
        int calibration_count = 0;

        std::vector<float> speech_buffer;
        int silence_ms = 0;

        const int sample_rate = 16000;
        const int min_samples = sample_rate * config_.min_speech_ms / 1000;
        const int max_samples = sample_rate * config_.max_speech_ms / 1000;

        printf("[Realtime] Calibrating noise floor...\n");

        while (running_) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(config_.check_interval_ms));

            // 取出 chunk
            std::vector<float> chunk;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                chunk = std::move(audio_buffer_);
                audio_buffer_.clear();
            }

            if (chunk.empty()) continue;

            float energy = compute_energy(chunk);

            // ===== 校准阶段 =====
            if (state == State::CALIBRATING) {
                noise_floor += energy;
                calibration_count++;

                if (calibration_count >= config_.calibration_frames) {
                    noise_floor /= calibration_count;
                    speech_threshold = std::max(
                        noise_floor * config_.threshold_factor,
                        config_.min_threshold);

                    printf("[Realtime] Calibrated: noise=%.2e, threshold=%.2e\n",
                           noise_floor, speech_threshold);
                    printf("[Realtime] Ready! Start speaking... (Ctrl+C to quit)\n\n");
                    state = State::IDLE;
                }
                continue;
            }

            bool has_speech = (energy > speech_threshold);

            // ===== 状态机 =====
            switch (state) {
                case State::IDLE:
                    if (has_speech) {
                        state = State::SPEECH;
                        speech_buffer.clear();
                        speech_buffer.insert(speech_buffer.end(),
                                             chunk.begin(), chunk.end());
                        silence_ms = 0;
                        printf("\r[listening...]");
                        fflush(stdout);
                    }
                    break;

                case State::SPEECH:
                    speech_buffer.insert(speech_buffer.end(),
                                         chunk.begin(), chunk.end());
                    if (!has_speech) {
                        silence_ms += config_.check_interval_ms;
                        state = State::SILENCE;
                    }
                    break;

                case State::SILENCE:
                    speech_buffer.insert(speech_buffer.end(),
                                         chunk.begin(), chunk.end());
                    if (has_speech) {
                        silence_ms = 0;
                        state = State::SPEECH;
                    } else {
                        silence_ms += config_.check_interval_ms;
                    }
                    break;

                default:
                    break;
            }

            // ===== 触发识别 =====
            bool should_recognize = false;
            if ((state == State::SPEECH || state == State::SILENCE)
                && static_cast<int>(speech_buffer.size()) >= min_samples)
            {
                if (silence_ms >= config_.silence_trigger_ms
                    || static_cast<int>(speech_buffer.size()) >= max_samples)
                {
                    should_recognize = true;
                }
            }

            if (should_recognize) {
                printf("\r              \r");  // 清除 [listening...]

                // 裁剪尾部静音
                int remove_samples = (silence_ms - config_.tail_trim_ms)
                                   * sample_rate / 1000;
                if (remove_samples > 0
                    && static_cast<int>(speech_buffer.size()) > remove_samples + min_samples)
                {
                    speech_buffer.resize(speech_buffer.size() - remove_samples);
                }

                float audio_sec = speech_buffer.size() / static_cast<float>(sample_rate);

                // 推理
                auto t_start = std::chrono::high_resolution_clock::now();

                auto result = recognizer_.transcribe_audio(
                    speech_buffer.data(), speech_buffer.size(),
                    config_.inference);

                auto t_end = std::chrono::high_resolution_clock::now();
                float ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_end - t_start).count();

                // 过滤无效结果
                if (!result.text.empty()
                    && result.text.find("sil") == std::string::npos
                    && result.text != "the")
                {
                    utterance_count_++;
                    total_audio_ms_ += static_cast<long>(audio_sec * 1000);
                    total_inference_ms_ += static_cast<long>(ms);

                    if (callback_) {
                        callback_(utterance_count_, result.text, audio_sec, ms);
                    } else {
                        printf("[%d] %s  (%.1fs, %.0fms, RTF=%.2f)\n",
                               utterance_count_, result.text.c_str(),
                               audio_sec, ms, ms / (audio_sec * 1000));
                    }
                }

                // 重置
                state = State::IDLE;
                speech_buffer.clear();
                silence_ms = 0;
            }
        }
    }
};

} // namespace funasr

#endif // FUNASR_PIPELINE_REALTIME_HPP