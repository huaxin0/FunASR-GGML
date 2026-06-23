#include "pipeline/chunking.hpp"

#include <algorithm>
#include <cmath>

namespace funasr {
namespace {

void add_audio_chunk(std::vector<AudioChunk>& chunks,
                     size_t start_sample, size_t end_sample,
                     size_t infer_start_sample, size_t infer_end_sample,
                     size_t total_samples, int sample_rate) {
    constexpr int min_segment_ms = 500;
    start_sample = std::min(start_sample, total_samples);
    end_sample = std::min(end_sample, total_samples);
    infer_start_sample = std::min(infer_start_sample, total_samples);
    infer_end_sample = std::min(infer_end_sample, total_samples);

    size_t min_samples = static_cast<size_t>(sample_rate) * min_segment_ms / 1000;
    if (end_sample <= start_sample || end_sample - start_sample < min_samples) {
        return;
    }
    if (infer_end_sample <= infer_start_sample) {
        infer_start_sample = start_sample;
        infer_end_sample = end_sample;
    }

    AudioChunk chunk;
    chunk.start_sample = start_sample;
    chunk.end_sample = end_sample;
    chunk.infer_start_sample = infer_start_sample;
    chunk.infer_end_sample = infer_end_sample;
    chunk.start_sec = static_cast<float>(start_sample) / sample_rate;
    chunk.end_sec = static_cast<float>(end_sample) / sample_rate;
    chunks.push_back(chunk);
}

} // namespace

std::vector<AudioChunk> split_audio_by_window(
    const std::vector<float>& samples,
    int sample_rate,
    int chunk_sec) {
    std::vector<AudioChunk> chunks;
    if (samples.empty() || sample_rate <= 0 || chunk_sec <= 0) {
        return chunks;
    }

    size_t chunk_samples = static_cast<size_t>(sample_rate) * chunk_sec;
    for (size_t start = 0; start < samples.size(); start += chunk_samples) {
        size_t end = std::min(samples.size(), start + chunk_samples);
        add_audio_chunk(chunks, start, end, start, end, samples.size(), sample_rate);
    }
    return chunks;
}

std::vector<AudioChunk> split_audio_by_energy_vad(
    const std::vector<float>& samples,
    int sample_rate,
    int max_segment_sec,
    int min_silence_ms,
    int segment_pad_ms,
    float threshold) {
    std::vector<AudioChunk> chunks;
    if (samples.empty() || sample_rate <= 0) {
        return chunks;
    }

    constexpr int frame_ms = 10;
    constexpr int min_segment_ms = 500;
    int frame_samples = sample_rate * frame_ms / 1000;
    int pad_frames = std::max(0, segment_pad_ms) / frame_ms;
    int min_silence_frames = std::max(1, min_silence_ms / frame_ms);
    int max_segment_frames = std::max(1, max_segment_sec * 1000 / frame_ms);
    int min_segment_frames = std::max(1, min_segment_ms / frame_ms);

    int n_frames = static_cast<int>((samples.size() + frame_samples - 1) / frame_samples);
    std::vector<float> rms(static_cast<size_t>(n_frames), 0.0f);
    for (int i = 0; i < n_frames; i++) {
        size_t start = static_cast<size_t>(i) * frame_samples;
        size_t end = std::min(samples.size(), start + frame_samples);
        double sum = 0.0;
        for (size_t j = start; j < end; j++) {
            sum += static_cast<double>(samples[j]) * samples[j];
        }
        if (end > start) {
            rms[static_cast<size_t>(i)] = static_cast<float>(
                std::sqrt(sum / static_cast<double>(end - start)));
        }
    }

    bool in_segment = false;
    int seg_start_frame = 0;
    int last_speech_frame = 0;
    int silence_frames = 0;

    auto frame_to_sample = [&](int frame) -> size_t {
        return std::min(samples.size(), static_cast<size_t>(std::max(0, frame)) * frame_samples);
    };

    auto finish_segment = [&](int end_frame) {
        if (end_frame - seg_start_frame + 1 < min_segment_frames) {
            return;
        }
        int core_start = seg_start_frame;
        int core_end = std::min(n_frames, end_frame + 1);
        int infer_start = std::max(0, core_start - pad_frames);
        int infer_end = std::min(n_frames, core_end + pad_frames);
        add_audio_chunk(chunks,
                        frame_to_sample(core_start),
                        frame_to_sample(core_end),
                        frame_to_sample(infer_start),
                        frame_to_sample(infer_end),
                        samples.size(), sample_rate);
    };

    for (int i = 0; i < n_frames; i++) {
        bool speech = rms[static_cast<size_t>(i)] > threshold;
        if (speech) {
            if (!in_segment) {
                in_segment = true;
                seg_start_frame = i;
                silence_frames = 0;
            }
            last_speech_frame = i;
            silence_frames = 0;
        } else if (in_segment) {
            silence_frames++;
        }

        if (in_segment && i - seg_start_frame + 1 >= max_segment_frames) {
            int search_begin = std::min(i, seg_start_frame + min_segment_frames);
            int split_frame = search_begin;
            float min_rms = rms[static_cast<size_t>(split_frame)];
            for (int j = search_begin; j <= i; j++) {
                if (rms[static_cast<size_t>(j)] < min_rms) {
                    min_rms = rms[static_cast<size_t>(j)];
                    split_frame = j;
                }
            }
            finish_segment(split_frame);
            in_segment = true;
            seg_start_frame = std::min(i, split_frame + 1);
            last_speech_frame = i;
            silence_frames = speech ? 0 : 1;
        } else if (in_segment && silence_frames >= min_silence_frames) {
            finish_segment(last_speech_frame);
            in_segment = false;
            silence_frames = 0;
        }
    }

    if (in_segment) {
        finish_segment(last_speech_frame);
    }

    if (chunks.empty()) {
        add_audio_chunk(chunks, 0, samples.size(), 0, samples.size(),
                        samples.size(), sample_rate);
    }
    return chunks;
}

std::vector<AudioChunk> convert_silero_segments_to_chunks(
    const std::vector<VadSegment>& vad_segments,
    size_t total_samples,
    int sample_rate,
    float samples_overlap) {
    std::vector<AudioChunk> chunks;
    chunks.reserve(vad_segments.size());
    if (sample_rate <= 0) {
        return chunks;
    }

    size_t overlap_samples = static_cast<size_t>(
        std::max(0.0f, samples_overlap) * sample_rate + 0.5f);
    for (size_t i = 0; i < vad_segments.size(); i++) {
        const auto& vad_segment = vad_segments[i];
        float start_sec = std::max(0.0f, vad_segment.start_sec);
        float end_sec = std::max(start_sec, vad_segment.end_sec);

        size_t start_sample = std::min(
            total_samples,
            static_cast<size_t>(start_sec * sample_rate + 0.5f));
        size_t end_sample = std::min(
            total_samples,
            static_cast<size_t>(end_sec * sample_rate + 0.5f));
        if (end_sample <= start_sample) {
            continue;
        }

        AudioChunk chunk;
        chunk.start_sample = start_sample;
        chunk.end_sample = end_sample;
        chunk.infer_start_sample = start_sample;
        chunk.infer_end_sample = i + 1 < vad_segments.size()
            ? std::min(total_samples, end_sample + overlap_samples)
            : end_sample;
        chunk.start_sec = static_cast<float>(start_sample) / sample_rate;
        chunk.end_sec = static_cast<float>(end_sample) / sample_rate;
        chunks.push_back(chunk);
    }
    return chunks;
}

std::vector<AudioChunk> split_chunks_by_max_duration(
    const std::vector<AudioChunk>& input,
    size_t total_samples,
    int sample_rate,
    int max_segment_sec) {
    if (input.empty() || sample_rate <= 0 || max_segment_sec <= 0) {
        return input;
    }

    const size_t max_samples = static_cast<size_t>(sample_rate) * max_segment_sec;
    std::vector<AudioChunk> output;
    output.reserve(input.size());

    for (const auto& chunk : input) {
        size_t infer_len = chunk.infer_end_sample > chunk.infer_start_sample
            ? chunk.infer_end_sample - chunk.infer_start_sample
            : 0;
        if (infer_len <= max_samples) {
            output.push_back(chunk);
            continue;
        }

        size_t cursor = chunk.start_sample;
        size_t end = std::max(chunk.end_sample, chunk.start_sample);
        while (cursor < end) {
            size_t chunk_end = std::min(end, cursor + max_samples);
            add_audio_chunk(output,
                            cursor,
                            chunk_end,
                            cursor,
                            chunk_end,
                            total_samples,
                            sample_rate);
            cursor = chunk_end;
        }
    }

    return output;
}

} // namespace funasr
