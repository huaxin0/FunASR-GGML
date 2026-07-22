// funasr/pipeline/chunking.hpp
// Offline/long-audio chunk planning helpers.
//
// These helpers only decide ASR chunk boundaries. They do not run inference and
// do not split final subtitle text.
#ifndef FUNASR_PIPELINE_CHUNKING_HPP
#define FUNASR_PIPELINE_CHUNKING_HPP

#include "compute/silero_vad.hpp"
#include <cstddef>
#include <vector>

namespace funasr {

struct AudioChunk {
    size_t start_sample = 0;
    size_t end_sample = 0;
    size_t infer_start_sample = 0;
    size_t infer_end_sample = 0;
    float start_sec = 0.0f;
    float end_sec = 0.0f;
};

float selected_audio_duration_seconds(
    const std::vector<AudioChunk>& chunks,
    int sample_rate);

std::vector<AudioChunk> split_audio_by_window(
    const std::vector<float>& samples,
    int sample_rate,
    int chunk_sec);

std::vector<AudioChunk> split_audio_by_energy_vad(
    const std::vector<float>& samples,
    int sample_rate,
    int max_segment_sec,
    int min_silence_ms,
    int segment_pad_ms,
    float threshold);

std::vector<AudioChunk> convert_silero_segments_to_chunks(
    const std::vector<VadSegment>& vad_segments,
    size_t total_samples,
    int sample_rate,
    float samples_overlap);

std::vector<AudioChunk> split_chunks_by_max_duration(
    const std::vector<AudioChunk>& input,
    size_t total_samples,
    int sample_rate,
    int max_segment_sec);

} // namespace funasr

#endif // FUNASR_PIPELINE_CHUNKING_HPP
