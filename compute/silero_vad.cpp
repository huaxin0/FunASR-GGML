// Silero VAD ggml implementation adapted from whisper.cpp (MIT).
// Source: https://github.com/ggml-org/whisper.cpp

#include "compute/silero_vad.hpp"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace funasr {

namespace {

constexpr int SILERO_SAMPLE_RATE = 16000;
constexpr int VAD_MAX_NODES = 4096;

#define VAD_LOG_ERROR(...) std::fprintf(stderr, __VA_ARGS__)
#define VAD_LOG_INFO(...)  std::fprintf(stderr, __VA_ARGS__)

template<typename T>
bool read_value(std::ifstream& in, T& dest) {
    in.read(reinterpret_cast<char*>(&dest), sizeof(T));
    return static_cast<bool>(in);
}

struct VadHParams {
    int32_t n_encoder_layers = 0;
    std::vector<int32_t> encoder_in_channels;
    std::vector<int32_t> encoder_out_channels;
    std::vector<int32_t> kernel_sizes;
    int32_t lstm_input_size = 0;
    int32_t lstm_hidden_size = 0;
    int32_t final_conv_in = 0;
    int32_t final_conv_out = 0;
};

struct VadModel {
    std::string type;
    std::string version;
    VadHParams hparams;

    ggml_tensor* stft_forward_basis = nullptr;

    ggml_tensor* encoder_0_weight = nullptr;
    ggml_tensor* encoder_0_bias = nullptr;
    ggml_tensor* encoder_1_weight = nullptr;
    ggml_tensor* encoder_1_bias = nullptr;
    ggml_tensor* encoder_2_weight = nullptr;
    ggml_tensor* encoder_2_bias = nullptr;
    ggml_tensor* encoder_3_weight = nullptr;
    ggml_tensor* encoder_3_bias = nullptr;

    ggml_tensor* lstm_ih_weight = nullptr;
    ggml_tensor* lstm_ih_bias = nullptr;
    ggml_tensor* lstm_hh_weight = nullptr;
    ggml_tensor* lstm_hh_bias = nullptr;

    ggml_tensor* final_conv_weight = nullptr;
    ggml_tensor* final_conv_bias = nullptr;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int n_loaded = 0;
    std::map<std::string, ggml_tensor*> tensors;
};

struct VadSegmentCs {
    int64_t start = 0;
    int64_t end = 0;
};

struct VadSegments {
    std::vector<VadSegmentCs> data;
};

struct VadSched {
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;
};

size_t sched_size(VadSched& allocr) {
    if (!allocr.sched) return allocr.meta.size();
    size_t size = allocr.meta.size();
    for (int i = 0; i < ggml_backend_sched_get_n_backends(allocr.sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(allocr.sched, i);
        size += ggml_backend_sched_get_buffer_size(allocr.sched, backend);
    }
    return size;
}

int cs_to_samples(int64_t cs) {
    return static_cast<int>((cs / 100.0) * SILERO_SAMPLE_RATE + 0.5);
}

int64_t samples_to_cs(int samples) {
    return static_cast<int64_t>((samples / static_cast<double>(SILERO_SAMPLE_RATE)) * 100.0 + 0.5);
}

ggml_tensor* build_stft_layer(ggml_context* ctx, const VadModel& model, ggml_tensor* cur) {
    ggml_tensor* padded = ggml_pad_reflect_1d(ctx, cur, 64, 64);
    ggml_tensor* stft = ggml_conv_1d(
        ctx, model.stft_forward_basis, padded, model.hparams.lstm_input_size, 0, 1);

    const int cutoff = static_cast<int>(model.stft_forward_basis->ne[2] / 2);
    ggml_tensor* real_part = ggml_view_2d(ctx, stft, 4, cutoff, stft->nb[1], 0);
    ggml_tensor* img_part = ggml_view_2d(ctx, stft, 4, cutoff, stft->nb[1],
                                         static_cast<size_t>(cutoff) * stft->nb[1]);

    ggml_tensor* real_squared = ggml_mul(ctx, real_part, real_part);
    ggml_tensor* img_squared = ggml_mul(ctx, img_part, img_part);
    ggml_tensor* sum_squares = ggml_add(ctx, real_squared, img_squared);
    return ggml_sqrt(ctx, sum_squares);
}

ggml_tensor* build_encoder_layer(ggml_context* ctx, const VadModel& model, ggml_tensor* cur) {
    cur = ggml_conv_1d(ctx, model.encoder_0_weight, cur, 1, 1, 1);
    cur = ggml_add(ctx, cur, ggml_reshape_3d(ctx, model.encoder_0_bias, 1, 128, 1));
    cur = ggml_relu(ctx, cur);

    cur = ggml_conv_1d(ctx, model.encoder_1_weight, cur, 2, 1, 1);
    cur = ggml_add(ctx, cur, ggml_reshape_3d(ctx, model.encoder_1_bias, 1, 64, 1));
    cur = ggml_relu(ctx, cur);

    cur = ggml_conv_1d(ctx, model.encoder_2_weight, cur, 2, 1, 1);
    cur = ggml_add(ctx, cur, ggml_reshape_3d(ctx, model.encoder_2_bias, 1, 64, 1));
    cur = ggml_relu(ctx, cur);

    cur = ggml_conv_1d(ctx, model.encoder_3_weight, cur, 1, 1, 1);
    cur = ggml_add(ctx, cur, ggml_reshape_3d(ctx, model.encoder_3_bias, 1, 128, 1));
    cur = ggml_relu(ctx, cur);

    return cur;
}

} // namespace

struct SileroVAD::Impl {
    int64_t t_vad_us = 0;
    int n_window = 0;
    int n_context = 0;
    int n_threads = 4;

    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t state_buffer = nullptr;
    std::vector<uint8_t> state_ctx_buf;
    VadSched sched;

    VadModel model;
    std::string path_model;
    ggml_tensor* h_state = nullptr;
    ggml_tensor* c_state = nullptr;
    std::vector<float> probs;

    bool load_model_file(const std::string& path);

    ~Impl() {
        if (state_buffer) ggml_backend_buffer_free(state_buffer);
        if (model.buffer) ggml_backend_buffer_free(model.buffer);
        if (model.ctx) ggml_free(model.ctx);
        if (sched.sched) ggml_backend_sched_free(sched.sched);
        if (backend) ggml_backend_free(backend);
    }

    ggml_tensor* build_lstm_layer(ggml_context* ctx, ggml_tensor* cur, ggml_cgraph* gf) {
        const int hdim = model.hparams.lstm_hidden_size;
        ggml_tensor* x_t = ggml_transpose(ctx, cur);

        ggml_tensor* inp_gate = ggml_mul_mat(ctx, model.lstm_ih_weight, x_t);
        inp_gate = ggml_add(ctx, inp_gate, model.lstm_ih_bias);

        ggml_tensor* hid_gate = ggml_mul_mat(ctx, model.lstm_hh_weight, h_state);
        hid_gate = ggml_add(ctx, hid_gate, model.lstm_hh_bias);

        ggml_tensor* out_gate = ggml_add(ctx, inp_gate, hid_gate);
        const size_t hdim_size = ggml_row_size(out_gate->type, hdim);

        ggml_tensor* i_t = ggml_sigmoid(ctx, ggml_view_1d(ctx, out_gate, hdim, 0 * hdim_size));
        ggml_tensor* f_t = ggml_sigmoid(ctx, ggml_view_1d(ctx, out_gate, hdim, 1 * hdim_size));
        ggml_tensor* g_t = ggml_tanh(ctx, ggml_view_1d(ctx, out_gate, hdim, 2 * hdim_size));
        ggml_tensor* o_t = ggml_sigmoid(ctx, ggml_view_1d(ctx, out_gate, hdim, 3 * hdim_size));

        ggml_tensor* c_out = ggml_add(ctx,
            ggml_mul(ctx, f_t, c_state),
            ggml_mul(ctx, i_t, g_t));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, c_out, c_state));

        ggml_tensor* out = ggml_mul(ctx, o_t, ggml_tanh(ctx, c_out));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, out, h_state));

        return out;
    }

    ggml_cgraph* build_graph() {
        ggml_init_params params = {
            sched.meta.size(),
            sched.meta.data(),
            true,
        };

        ggml_context* ctx = ggml_init(params);
        ggml_cgraph* gf = ggml_new_graph(ctx);

        ggml_tensor* frame = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_window, 1);
        ggml_set_name(frame, "frame");
        ggml_set_input(frame);

        ggml_tensor* cur = build_stft_layer(ctx, model, frame);
        cur = build_encoder_layer(ctx, model, cur);
        cur = ggml_view_2d(ctx, cur, 1, 128, cur->nb[1], 0);
        cur = build_lstm_layer(ctx, cur, gf);
        cur = ggml_relu(ctx, cur);
        cur = ggml_conv_1d(ctx, model.final_conv_weight, cur, 1, 0, 1);
        cur = ggml_add(ctx, cur, model.final_conv_bias);
        cur = ggml_sigmoid(ctx, cur);
        ggml_set_name(cur, "prob");
        ggml_set_output(cur);

        ggml_build_forward_expand(gf, cur);
        ggml_free(ctx);
        return gf;
    }

    bool init_context() {
        backend = ggml_backend_cpu_init();
        if (!backend) {
            VAD_LOG_ERROR("[SileroVAD] failed to initialize CPU backend\n");
            return false;
        }

        auto* reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto* set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (set_threads) {
            set_threads(backend, n_threads);
        }

        const int32_t lstm_hidden_size = model.hparams.lstm_hidden_size;
        state_ctx_buf.resize(2u * ggml_tensor_overhead());

        ggml_init_params params = {
            state_ctx_buf.size(),
            state_ctx_buf.data(),
            true,
        };

        ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            VAD_LOG_ERROR("[SileroVAD] failed to init LSTM state context\n");
            return false;
        }

        h_state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, lstm_hidden_size);
        ggml_set_name(h_state, "h_state");
        c_state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, lstm_hidden_size);
        ggml_set_name(c_state, "c_state");

        state_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_free(ctx);
        if (!state_buffer) {
            VAD_LOG_ERROR("[SileroVAD] failed to allocate VAD state\n");
            return false;
        }

        sched.sched = ggml_backend_sched_new(&backend, nullptr, 1, VAD_MAX_NODES, false, true);
        sched.meta.resize(ggml_tensor_overhead() * VAD_MAX_NODES + ggml_graph_overhead());
        if (!ggml_backend_sched_alloc_graph(sched.sched, build_graph())) {
            VAD_LOG_ERROR("[SileroVAD] failed to allocate compute buffer\n");
            return false;
        }
        ggml_backend_sched_reset(sched.sched);

        VAD_LOG_INFO("[SileroVAD] compute buffer: %.2f MB\n", sched_size(sched) / 1e6);
        return true;
    }

    void reset_state() {
        if (state_buffer) ggml_backend_buffer_clear(state_buffer, 0);
    }

    bool detect_speech_no_reset(const float* samples, int n_samples) {
        int n_chunks = n_samples / n_window;
        if (n_samples % n_window != 0) n_chunks++;

        probs.assign(static_cast<size_t>(n_chunks), 0.0f);
        std::vector<float> window(static_cast<size_t>(n_window), 0.0f);

        ggml_cgraph* gf = build_graph();
        if (!ggml_backend_sched_alloc_graph(sched.sched, gf)) {
            VAD_LOG_ERROR("[SileroVAD] failed to allocate graph\n");
            return false;
        }

        ggml_tensor* frame = ggml_graph_get_tensor(gf, "frame");
        ggml_tensor* prob = ggml_graph_get_tensor(gf, "prob");
        if (!frame || !prob) {
            VAD_LOG_ERROR("[SileroVAD] failed to find graph tensors\n");
            ggml_backend_sched_reset(sched.sched);
            return false;
        }

        const int64_t t_start = ggml_time_us();
        for (int i = 0; i < n_chunks; i++) {
            const int idx_start = i * n_window;
            const int idx_end = std::min(idx_start + n_window, n_samples);
            const int chunk_len = idx_end - idx_start;

            std::fill(window.begin(), window.end(), 0.0f);
            if (chunk_len > 0) {
                std::copy(samples + idx_start, samples + idx_end, window.begin());
            }

            ggml_backend_tensor_set(frame, window.data(), 0,
                                    ggml_nelements(frame) * sizeof(float));

            if (ggml_backend_sched_graph_compute(sched.sched, gf) != GGML_STATUS_SUCCESS) {
                VAD_LOG_ERROR("[SileroVAD] graph compute failed\n");
                ggml_backend_sched_reset(sched.sched);
                return false;
            }

            ggml_backend_tensor_get(prob, &probs[static_cast<size_t>(i)], 0, sizeof(float));
        }

        t_vad_us += ggml_time_us() - t_start;
        ggml_backend_sched_reset(sched.sched);
        return true;
    }

    bool detect_speech(const float* samples, int n_samples) {
        reset_state();
        return detect_speech_no_reset(samples, n_samples);
    }

    std::unique_ptr<VadSegments> segments_from_probs(const SileroVadParams& params) {
        const int n_probs = static_cast<int>(probs.size());
        const float threshold = params.threshold;
        const int sample_rate = SILERO_SAMPLE_RATE;
        const int min_silence_samples = sample_rate * params.min_silence_duration_ms / 1000;
        const int audio_length_samples = n_probs * n_window;
        const int min_speech_samples = sample_rate * params.min_speech_duration_ms / 1000;
        const int speech_pad_samples = sample_rate * params.speech_pad_ms / 1000;

        int max_speech_samples;
        if (params.max_speech_duration_s > 100000.0f) {
            max_speech_samples = INT32_MAX / 2;
        } else {
            int64_t temp = static_cast<int64_t>(sample_rate) *
                         static_cast<int64_t>(params.max_speech_duration_s) -
                         n_window - 2 * speech_pad_samples;
            max_speech_samples = temp > INT32_MAX ? INT32_MAX / 2 : static_cast<int>(temp);
            if (max_speech_samples < 0) max_speech_samples = INT32_MAX / 2;
        }

        const int min_silence_samples_at_max_speech = sample_rate * 98 / 1000;
        float neg_threshold = threshold - 0.15f;
        if (neg_threshold < 0.01f) neg_threshold = 0.01f;

        struct SpeechSegment {
            int start;
            int end;
        };

        std::vector<SpeechSegment> speeches;
        speeches.reserve(256);

        bool is_speech_segment = false;
        int temp_end = 0;
        int prev_end = 0;
        int next_start = 0;
        int curr_speech_start = 0;
        bool has_curr_speech = false;

        for (int i = 0; i < n_probs; i++) {
            const float curr_prob = probs[static_cast<size_t>(i)];
            const int curr_sample = n_window * i;

            if (curr_prob >= threshold && temp_end) {
                temp_end = 0;
                if (next_start < prev_end) next_start = curr_sample;
            }

            if (curr_prob >= threshold && !is_speech_segment) {
                is_speech_segment = true;
                curr_speech_start = curr_sample;
                has_curr_speech = true;
                continue;
            }

            if (is_speech_segment && curr_sample - curr_speech_start > max_speech_samples) {
                if (prev_end) {
                    speeches.push_back({curr_speech_start, prev_end});
                    has_curr_speech = true;

                    if (next_start < prev_end) {
                        is_speech_segment = false;
                        has_curr_speech = false;
                    } else {
                        curr_speech_start = next_start;
                    }
                    prev_end = next_start = temp_end = 0;
                } else {
                    speeches.push_back({curr_speech_start, curr_sample});
                    prev_end = next_start = temp_end = 0;
                    is_speech_segment = false;
                    has_curr_speech = false;
                    continue;
                }
            }

            if (curr_prob < neg_threshold && is_speech_segment) {
                if (!temp_end) temp_end = curr_sample;
                if (curr_sample - temp_end > min_silence_samples_at_max_speech) {
                    prev_end = temp_end;
                }

                if (curr_sample - temp_end < min_silence_samples) {
                    continue;
                }

                if (temp_end - curr_speech_start > min_speech_samples) {
                    speeches.push_back({curr_speech_start, temp_end});
                }

                prev_end = next_start = temp_end = 0;
                is_speech_segment = false;
                has_curr_speech = false;
            }
        }

        if (has_curr_speech && audio_length_samples - curr_speech_start > min_speech_samples) {
            speeches.push_back({curr_speech_start, audio_length_samples});
        }

        if (speeches.size() > 1) {
            for (int i = 0; i < static_cast<int>(speeches.size()) - 1; i++) {
                const int max_merge_gap_samples = sample_rate * 200 / 1000;
                if (speeches[static_cast<size_t>(i + 1)].start -
                    speeches[static_cast<size_t>(i)].end < max_merge_gap_samples) {
                    speeches[static_cast<size_t>(i)].end = speeches[static_cast<size_t>(i + 1)].end;
                    speeches.erase(speeches.begin() + i + 1);
                    i--;
                }
            }
        }

        for (int i = 0; i < static_cast<int>(speeches.size()); i++) {
            if (speeches[static_cast<size_t>(i)].end -
                speeches[static_cast<size_t>(i)].start < min_speech_samples) {
                speeches.erase(speeches.begin() + i);
                i--;
            }
        }

        auto out = std::make_unique<VadSegments>();
        out->data.resize(speeches.size());

        for (int i = 0; i < static_cast<int>(speeches.size()); i++) {
            auto& speech = speeches[static_cast<size_t>(i)];
            if (i == 0) {
                speech.start = speech.start > speech_pad_samples
                    ? speech.start - speech_pad_samples
                    : 0;
            }

            if (i < static_cast<int>(speeches.size()) - 1) {
                auto& next = speeches[static_cast<size_t>(i + 1)];
                int silence_duration = next.start - speech.end;
                if (silence_duration < 2 * speech_pad_samples) {
                    speech.end += silence_duration / 2;
                    next.start = next.start > silence_duration / 2
                        ? next.start - silence_duration / 2
                        : 0;
                } else {
                    speech.end = speech.end + speech_pad_samples < audio_length_samples
                        ? speech.end + speech_pad_samples
                        : audio_length_samples;
                    next.start = next.start > speech_pad_samples
                        ? next.start - speech_pad_samples
                        : 0;
                }
            } else {
                speech.end = speech.end + speech_pad_samples < audio_length_samples
                    ? speech.end + speech_pad_samples
                    : audio_length_samples;
            }

            out->data[static_cast<size_t>(i)].start = samples_to_cs(speech.start);
            out->data[static_cast<size_t>(i)].end = samples_to_cs(speech.end);
        }

        return out;
    }

    std::unique_ptr<VadSegments> segments_from_samples(const SileroVadParams& params,
                                                       const float* samples,
                                                       int n_samples) {
        if (!detect_speech(samples, n_samples)) {
            return nullptr;
        }
        return segments_from_probs(params);
    }
};

bool SileroVAD::Impl::load_model_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        VAD_LOG_ERROR("[SileroVAD] failed to open model: %s\n", path.c_str());
        return false;
    }

    uint32_t magic = 0;
    if (!read_value(in, magic) || magic != GGML_FILE_MAGIC) {
        VAD_LOG_ERROR("[SileroVAD] invalid model magic\n");
        return false;
    }

    auto& model = this->model;
    auto& h = model.hparams;

    int32_t str_len = 0;
    if (!read_value(in, str_len) || str_len <= 0 || str_len > 1024) {
        VAD_LOG_ERROR("[SileroVAD] invalid model type length\n");
        return false;
    }
    std::vector<char> type_buf(static_cast<size_t>(str_len));
    in.read(type_buf.data(), str_len);
    if (!in) return false;
    model.type.assign(type_buf.begin(), type_buf.end());

    int32_t major = 0, minor = 0, patch = 0;
    if (!read_value(in, major) || !read_value(in, minor) || !read_value(in, patch)) {
        return false;
    }
    model.version = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);

    if (!read_value(in, this->n_window) || !read_value(in, this->n_context)) return false;
    if (!read_value(in, h.n_encoder_layers) || h.n_encoder_layers <= 0) return false;

    h.encoder_in_channels.resize(static_cast<size_t>(h.n_encoder_layers));
    h.encoder_out_channels.resize(static_cast<size_t>(h.n_encoder_layers));
    h.kernel_sizes.resize(static_cast<size_t>(h.n_encoder_layers));
    for (int32_t i = 0; i < h.n_encoder_layers; i++) {
        if (!read_value(in, h.encoder_in_channels[static_cast<size_t>(i)]) ||
            !read_value(in, h.encoder_out_channels[static_cast<size_t>(i)]) ||
            !read_value(in, h.kernel_sizes[static_cast<size_t>(i)])) {
            return false;
        }
    }

    if (!read_value(in, h.lstm_input_size) ||
        !read_value(in, h.lstm_hidden_size) ||
        !read_value(in, h.final_conv_in) ||
        !read_value(in, h.final_conv_out)) {
        return false;
    }

    if (h.n_encoder_layers != 4 || this->n_window <= 0 || h.lstm_hidden_size <= 0) {
        VAD_LOG_ERROR("[SileroVAD] unsupported VAD model architecture\n");
        return false;
    }

    constexpr size_t n_tensors = 4 * 2 + 4 + 2 + 1;
    ggml_init_params ctx_params = {
        n_tensors * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    model.ctx = ggml_init(ctx_params);
    if (!model.ctx) return false;

    auto add_tensor = [&](const char* name, ggml_tensor* tensor) {
        model.tensors[name] = tensor;
        return tensor;
    };

    model.stft_forward_basis = add_tensor(
        "_model.stft.forward_basis_buffer",
        ggml_new_tensor_3d(model.ctx, GGML_TYPE_F16, 256, 1, 258));

    model.encoder_0_weight = add_tensor(
        "_model.encoder.0.reparam_conv.weight",
        ggml_new_tensor_3d(model.ctx, GGML_TYPE_F16, h.kernel_sizes[0], h.encoder_in_channels[0], h.encoder_out_channels[0]));
    model.encoder_0_bias = add_tensor(
        "_model.encoder.0.reparam_conv.bias",
        ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, h.encoder_out_channels[0]));
    model.encoder_1_weight = add_tensor(
        "_model.encoder.1.reparam_conv.weight",
        ggml_new_tensor_3d(model.ctx, GGML_TYPE_F16, h.kernel_sizes[1], h.encoder_in_channels[1], h.encoder_out_channels[1]));
    model.encoder_1_bias = add_tensor(
        "_model.encoder.1.reparam_conv.bias",
        ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, h.encoder_out_channels[1]));
    model.encoder_2_weight = add_tensor(
        "_model.encoder.2.reparam_conv.weight",
        ggml_new_tensor_3d(model.ctx, GGML_TYPE_F16, h.kernel_sizes[2], h.encoder_in_channels[2], h.encoder_out_channels[2]));
    model.encoder_2_bias = add_tensor(
        "_model.encoder.2.reparam_conv.bias",
        ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, h.encoder_out_channels[2]));
    model.encoder_3_weight = add_tensor(
        "_model.encoder.3.reparam_conv.weight",
        ggml_new_tensor_3d(model.ctx, GGML_TYPE_F16, h.kernel_sizes[3], h.encoder_in_channels[3], h.encoder_out_channels[3]));
    model.encoder_3_bias = add_tensor(
        "_model.encoder.3.reparam_conv.bias",
        ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, h.encoder_out_channels[3]));

    const int hstate_dim = h.lstm_hidden_size * 4;
    model.lstm_ih_weight = add_tensor(
        "_model.decoder.rnn.weight_ih",
        ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, h.lstm_hidden_size, hstate_dim));
    model.lstm_hh_weight = add_tensor(
        "_model.decoder.rnn.weight_hh",
        ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, h.lstm_hidden_size, hstate_dim));
    model.lstm_ih_bias = add_tensor(
        "_model.decoder.rnn.bias_ih",
        ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, hstate_dim));
    model.lstm_hh_bias = add_tensor(
        "_model.decoder.rnn.bias_hh",
        ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, hstate_dim));

    model.final_conv_weight = add_tensor(
        "_model.decoder.decoder.2.weight",
        ggml_new_tensor_2d(model.ctx, GGML_TYPE_F16, h.final_conv_in, 1));
    model.final_conv_bias = add_tensor(
        "_model.decoder.decoder.2.bias",
        ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 1));

    model.buffer = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx, ggml_backend_cpu_buffer_type());
    if (!model.buffer) {
        VAD_LOG_ERROR("[SileroVAD] failed to allocate model tensors\n");
        return false;
    }

    size_t total_size = 0;
    while (in.peek() != EOF) {
        int32_t n_dims = 0;
        int32_t length = 0;
        int32_t ttype = 0;
        if (!read_value(in, n_dims)) break;
        if (!read_value(in, length) || !read_value(in, ttype)) return false;
        if (n_dims < 0 || n_dims > 4 || length <= 0 || length > 4096) {
            VAD_LOG_ERROR("[SileroVAD] invalid tensor header\n");
            return false;
        }

        int32_t ne[4] = {1, 1, 1, 1};
        int64_t nelements = 1;
        for (int i = 0; i < n_dims; i++) {
            if (!read_value(in, ne[i])) return false;
            nelements *= ne[i];
        }

        std::vector<char> name_buf(static_cast<size_t>(length));
        in.read(name_buf.data(), length);
        if (!in) return false;
        std::string name(name_buf.begin(), name_buf.end());

        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            VAD_LOG_ERROR("[SileroVAD] unknown tensor '%s'\n", name.c_str());
            return false;
        }
        ggml_tensor* tensor = it->second;

        if (ggml_nelements(tensor) != nelements ||
            (n_dims > 0 && tensor->ne[0] != ne[0]) ||
            (n_dims > 1 && tensor->ne[1] != ne[1]) ||
            (n_dims > 2 && tensor->ne[2] != ne[2])) {
            VAD_LOG_ERROR("[SileroVAD] tensor '%s' has unexpected shape\n", name.c_str());
            return false;
        }

        const ggml_type file_type = static_cast<ggml_type>(ttype);
        const size_t expected_bytes =
            (static_cast<size_t>(nelements) * ggml_type_size(file_type)) / ggml_blck_size(file_type);
        if (expected_bytes != ggml_nbytes(tensor)) {
            VAD_LOG_ERROR("[SileroVAD] tensor '%s' has unexpected byte size\n", name.c_str());
            return false;
        }

        in.read(reinterpret_cast<char*>(tensor->data), ggml_nbytes(tensor));
        if (!in) return false;
        total_size += ggml_nbytes(tensor);
        model.n_loaded++;
    }

    if (model.n_loaded != static_cast<int>(model.tensors.size())) {
        VAD_LOG_ERROR("[SileroVAD] not all tensors loaded, expected %zu got %d\n",
                      model.tensors.size(), model.n_loaded);
        return false;
    }

    VAD_LOG_INFO("[SileroVAD] loaded %s v%s, model size %.2f MB\n",
                 model.type.c_str(), model.version.c_str(), total_size / 1e6);
    return true;
}

SileroVAD::SileroVAD()
    : impl_(std::make_unique<Impl>()) {}

SileroVAD::~SileroVAD() = default;

bool SileroVAD::init(const std::string& model_path) {
    impl_ = std::make_unique<Impl>();
    if (!impl_->load_model_file(model_path)) {
        impl_.reset();
        return false;
    }
    impl_->path_model = model_path;
    if (!impl_->init_context()) {
        impl_.reset();
        return false;
    }
    return true;
}

std::vector<VadSegment> SileroVAD::detect(const float* audio, size_t n_samples,
                                          int sample_rate,
                                          const SileroVadParams& params) {
    std::vector<VadSegment> result;
    if (!is_initialized() || !audio || n_samples == 0) {
        return result;
    }
    if (sample_rate != SILERO_SAMPLE_RATE) {
        VAD_LOG_ERROR("[SileroVAD] expected 16 kHz audio, got %d Hz\n", sample_rate);
        return result;
    }
    if (n_samples > static_cast<size_t>(INT32_MAX)) {
        VAD_LOG_ERROR("[SileroVAD] audio is too long for one VAD pass\n");
        return result;
    }

    auto segments = impl_->segments_from_samples(
        params, audio, static_cast<int>(n_samples));
    if (!segments) {
        return result;
    }

    result.reserve(segments->data.size());
    for (const auto& seg : segments->data) {
        result.push_back({
            static_cast<float>(seg.start) / 100.0f,
            static_cast<float>(seg.end) / 100.0f,
        });
    }
    return result;
}

bool SileroVAD::is_initialized() const {
    return impl_ && impl_->backend && impl_->state_buffer && impl_->model.buffer;
}

} // namespace funasr
