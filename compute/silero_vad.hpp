#ifndef FUNASR_COMPUTE_SILERO_VAD_HPP
#define FUNASR_COMPUTE_SILERO_VAD_HPP

#include <memory>
#include <string>
#include <vector>
#include <cstddef>

namespace funasr {

struct VadSegment {
    float start_sec = 0.0f;
    float end_sec = 0.0f;
};

struct SileroVadParams {
    float threshold = 0.5f;
    int min_speech_duration_ms = 250;
    int min_silence_duration_ms = 100;
    float max_speech_duration_s = 30.0f;
    int speech_pad_ms = 30;
    float samples_overlap = 0.1f;
};

class SileroVAD {
public:
    SileroVAD();
    ~SileroVAD();

    bool init(const std::string& model_path);

    std::vector<VadSegment> detect(const float* audio, size_t n_samples,
                                   int sample_rate = 16000,
                                   const SileroVadParams& params = {});

    bool is_initialized() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace funasr

#endif // FUNASR_COMPUTE_SILERO_VAD_HPP
