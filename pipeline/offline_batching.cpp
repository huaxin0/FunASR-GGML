#include "pipeline/offline_batching.hpp"
#include "pipeline/recognizer.hpp"

#include <algorithm>
#include <chrono>

namespace funasr {

void ContinuousKVSlotPool::reset(int num_slots, int ctx_size) {
    capacity_ = std::max(0, num_slots);
    ctx_size_ = std::max(0, ctx_size);
    in_use_.assign(static_cast<size_t>(capacity_), false);

    std::queue<int> empty;
    std::swap(free_slots_, empty);
    for (int i = 0; i < capacity_; i++) {
        free_slots_.push(i);
    }
}

bool ContinuousKVSlotPool::has_free_slot() const {
    return !free_slots_.empty();
}

bool ContinuousKVSlotPool::has_capacity(int max_tokens) const {
    (void)max_tokens;
    return has_free_slot();
}

KVHandle ContinuousKVSlotPool::acquire(int request_id) {
    return acquire(request_id, ctx_size_);
}

KVHandle ContinuousKVSlotPool::acquire(int request_id, int max_tokens) {
    (void)max_tokens;
    if (free_slots_.empty()) {
        return {};
    }

    int slot = free_slots_.front();
    free_slots_.pop();
    in_use_[static_cast<size_t>(slot)] = true;
    return KVHandle{request_id, slot, ctx_size_};
}

void ContinuousKVSlotPool::release(const KVHandle& handle) {
    if (!handle.valid() || handle.slot_id >= capacity_) {
        return;
    }

    size_t slot = static_cast<size_t>(handle.slot_id);
    if (!in_use_[slot]) {
        return;
    }

    in_use_[slot] = false;
    free_slots_.push(handle.slot_id);
}

void PagedKVBlockPool::reset(int num_blocks, int block_size) {
    capacity_ = std::max(0, num_blocks);
    block_size_ = std::max(1, block_size);

    std::queue<int> empty;
    std::swap(free_blocks_, empty);
    for (int i = 0; i < capacity_; i++) {
        free_blocks_.push(i);
    }
}

int PagedKVBlockPool::blocks_needed(int max_tokens) const {
    if (max_tokens <= 0) {
        return 0;
    }
    return (max_tokens + block_size_ - 1) / block_size_;
}

bool PagedKVBlockPool::has_capacity(int max_tokens) const {
    return static_cast<int>(free_blocks_.size()) >= blocks_needed(max_tokens);
}

KVHandle PagedKVBlockPool::acquire(int request_id, int max_tokens) {
    const int need = blocks_needed(max_tokens);
    if (need <= 0 || static_cast<int>(free_blocks_.size()) < need) {
        return {};
    }

    KVHandle handle;
    handle.request_id = request_id;
    handle.slot_id = -1;
    handle.ctx_size = need * block_size_;
    handle.block_size = block_size_;
    handle.block_table.reserve(static_cast<size_t>(need));

    for (int i = 0; i < need; i++) {
        int block = free_blocks_.front();
        free_blocks_.pop();
        handle.block_table.push_back(block);
    }
    return handle;
}

void PagedKVBlockPool::release(const KVHandle& handle) {
    for (int block : handle.block_table) {
        if (block >= 0 && block < capacity_) {
            free_blocks_.push(block);
        }
    }
}

OfflineBatchTranscriber::OfflineBatchTranscriber(Recognizer& recognizer)
    : recognizer_(recognizer)
{}

void OfflineBatchTranscriber::reset_stats(
    size_t total_chunks,
    const OfflineBatchConfig& config) {
    last_stats_ = OfflineBatchStats{};
    last_stats_.total_chunks = static_cast<int>(total_chunks);
    last_stats_.batch_size = std::max(1, config.batch_size);
    last_stats_.use_gpu = config.use_gpu;
    last_stats_.use_paged_kv = config.use_paged_kv;
    last_stats_.kv_block_size = config.kv_block_size;
}

void OfflineBatchTranscriber::update_peak_blocks(const KVPool& kv_pool) {
    const int used = kv_pool.capacity() - kv_pool.free_count();
    last_stats_.peak_blocks_in_use = std::max(last_stats_.peak_blocks_in_use, used);
}

namespace {

int argmax_logits(const std::vector<float>& logits) {
    float max_val = -1e30f;
    int max_idx = 0;
    for (size_t i = 0; i < logits.size(); i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = static_cast<int>(i);
        }
    }
    return max_idx;
}

bool tail_repeats(const std::vector<int>& tokens, int ngram, int repeats) {
    const int need = ngram * repeats;
    if (ngram <= 0 || repeats <= 1 || static_cast<int>(tokens.size()) < need) {
        return false;
    }
    const int start = static_cast<int>(tokens.size()) - need;
    for (int r = 1; r < repeats; r++) {
        for (int i = 0; i < ngram; i++) {
            if (tokens[static_cast<size_t>(start + i)] !=
                tokens[static_cast<size_t>(start + r * ngram + i)]) {
                return false;
            }
        }
    }
    return true;
}

bool has_repetition_loop_local(const std::vector<int>& tokens) {
    for (int ngram = 4; ngram <= 16; ngram++) {
        if (tail_repeats(tokens, ngram, 4)) {
            return true;
        }
    }
    return false;
}

int estimate_kv_budget_tokens(const AudioChunk& chunk,
                              const std::vector<float>& samples,
                              int sample_rate,
                              const OfflineBatchConfig& config) {
    if (!config.use_paged_kv) {
        return config.ctx_size;
    }

    const size_t begin = std::min(chunk.infer_start_sample, samples.size());
    const size_t end = std::min(chunk.infer_end_sample, samples.size());
    const float seconds = sample_rate > 0 && end > begin
        ? static_cast<float>(end - begin) / static_cast<float>(sample_rate)
        : 0.0f;

    // FunASR adaptor output is about 16-17 embeddings/sec for 16 kHz audio in
    // the common 30s window case. Use a conservative upper estimate so paged KV
    // does not allocate a full ctx slot while still leaving headroom for prompt
    // tokens, frontend boundary effects, and decode.
    const int estimated_prefill = static_cast<int>(seconds * 25.0f) + 128;
    const int budget = estimated_prefill + config.max_tokens;
    return std::max(1, std::min(config.ctx_size, budget));
}

} // namespace

std::vector<OfflineChunkResult> OfflineBatchTranscriber::transcribe(
    const std::vector<float>& samples,
    int sample_rate,
    const std::vector<AudioChunk>& chunks,
    const OfflineBatchConfig& config,
    OfflineProgressCallback progress) {
    reset_stats(chunks.size(), config);

    std::vector<OfflineChunkResult> results;
    results.reserve(chunks.size());

    if (config.use_gpu) {
        return transcribe_continuous_gpu(samples, sample_rate, chunks, config, progress);
    }

    std::unique_ptr<KVPool> kv_pool;
    if (config.use_paged_kv) {
        int num_blocks = config.kv_num_blocks;
        if (num_blocks <= 0) {
            num_blocks = std::max(1, config.batch_size)
                       * ((config.ctx_size + config.kv_block_size - 1) / config.kv_block_size);
        }
        kv_pool = std::make_unique<PagedKVBlockPool>(num_blocks, config.kv_block_size);
        last_stats_.kv_block_capacity = num_blocks;
        printf("[OfflineBatch] Paged KV scheduler pool: blocks=%d block_size=%d\n",
               num_blocks, config.kv_block_size);
        if (config.use_gpu) {
            printf("[OfflineBatch] WARNING: paged attention kernel is not wired yet; "
                   "GPU execution will use serial compatibility until block-table attention lands.\n");
        }
    } else {
        kv_pool = std::make_unique<ContinuousKVSlotPool>(
            std::max(1, config.batch_size), config.ctx_size);
    }

    int completed = 0;
    size_t next_request = 0;

    while (next_request < chunks.size()) {
        std::vector<OfflineRequest> active;
        active.reserve(static_cast<size_t>(std::max(1, config.batch_size)));

        while (next_request < chunks.size() &&
               static_cast<int>(active.size()) < std::max(1, config.batch_size)) {
            int request_id = static_cast<int>(next_request);
            const int kv_budget = estimate_kv_budget_tokens(
                chunks[next_request], samples, sample_rate, config);
            if (!kv_pool->has_capacity(kv_budget)) {
                break;
            }
            KVHandle handle = kv_pool->acquire(request_id, kv_budget);
            if (!handle.valid()) {
                break;
            }
            if (handle.paged()) {
                update_peak_blocks(*kv_pool);
            }

            OfflineRequest request;
            request.id = request_id;
            request.phase = OfflineRequestPhase::WaitingPrefill;
            request.kv = handle;
            request.max_tokens = config.max_tokens;
            request.start_sec = chunks[next_request].start_sec;
            request.end_sec = chunks[next_request].end_sec;
            active.push_back(request);
            next_request++;
        }

        // Multiple active request states are admitted together. Compute is
        // still serial, but each request now goes through prepare -> LLM run,
        // which gives later batched prefill/decode a clean insertion point.
        bool can_use_gpu_batch = config.use_gpu && active.size() > 1 &&
                                 !active.front().kv.paged();

        if (can_use_gpu_batch) {
            InferenceConfig inference;
            inference.max_new_tokens = config.max_tokens;
            inference.n_threads = config.n_threads;
            inference.kv_cache_size = config.ctx_size;
            inference.use_gpu = config.use_gpu;
            inference.gpu_id = config.gpu_id;

            std::vector<AudioSpan> batch_audio;
            std::vector<int> slot_ids;
            batch_audio.reserve(active.size());
            slot_ids.reserve(active.size());

            for (auto& request : active) {
                const AudioChunk& chunk = chunks[static_cast<size_t>(request.id)];
                size_t begin = std::min(chunk.infer_start_sample, samples.size());
                size_t end = std::min(chunk.infer_end_sample, samples.size());
                if (end <= begin) {
                    batch_audio.push_back(AudioSpan{});
                } else {
                    batch_audio.push_back(AudioSpan{samples.data() + begin, end - begin});
                }
                slot_ids.push_back(request.kv.slot_id);
                request.phase = OfflineRequestPhase::Decoding;
            }

            std::vector<InferenceResult> asr_results =
                recognizer_.transcribe_audio_batch_gpu(batch_audio, slot_ids, inference);

            for (size_t i = 0; i < active.size(); i++) {
                auto& request = active[i];
                const AudioChunk& chunk = chunks[static_cast<size_t>(request.id)];
                const InferenceResult& asr = asr_results[i];

                OfflineChunkResult result;
                result.id = request.id;
                result.slot_id = request.kv.slot_id;
                result.start_sec = chunk.start_sec;
                result.end_sec = chunk.end_sec;
                result.text = asr.text;
                result.ok = !asr.text.empty();
                result.encoder_ms = asr.encoder_ms;
                result.prefill_ms = asr.prefill_ms;
                result.decode_ms = asr.decode_ms;
                result.total_ms = asr.total_ms;
                result.prefill_tokens = asr.prefill_tokens;
                result.decode_tokens = asr.decode_tokens;

                request.text = result.text;
                request.phase = result.ok ? OfflineRequestPhase::Finished
                                          : OfflineRequestPhase::Failed;

                results.push_back(result);
                kv_pool->release(request.kv);

                if (progress) {
                    progress(++completed, static_cast<int>(chunks.size()), result);
                }
            }
        } else {
            for (auto& request : active) {
                request.phase = OfflineRequestPhase::Decoding;
                OfflineChunkResult result = run_one(
                    samples, sample_rate, request,
                    chunks[static_cast<size_t>(request.id)], config, request.id);
                result.id = request.id;
                result.slot_id = request.kv.slot_id;

                request.text = result.text;
                request.phase = result.ok ? OfflineRequestPhase::Finished
                                          : OfflineRequestPhase::Failed;

                results.push_back(result);
                kv_pool->release(request.kv);

                if (progress) {
                    progress(++completed, static_cast<int>(chunks.size()), result);
                }
            }
        }
    }

    std::sort(results.begin(), results.end(),
              [](const OfflineChunkResult& a, const OfflineChunkResult& b) {
                  return a.id < b.id;
              });
    last_stats_.sync_fallback_total();
    return results;
}

std::vector<OfflineChunkResult> OfflineBatchTranscriber::transcribe_continuous_gpu(
    const std::vector<float>& samples,
    int sample_rate,
    const std::vector<AudioChunk>& chunks,
    const OfflineBatchConfig& config,
    OfflineProgressCallback progress) {
    (void)sample_rate;

    reset_stats(chunks.size(), config);

    std::vector<OfflineChunkResult> results;
    results.reserve(chunks.size());

    std::unique_ptr<KVPool> kv_pool;
    if (config.use_paged_kv) {
        int num_blocks = config.kv_num_blocks;
        if (num_blocks <= 0) {
            num_blocks = std::max(1, config.batch_size)
                       * std::max(1, config.ctx_size / std::max(1, config.kv_block_size));
        }
        kv_pool = std::make_unique<PagedKVBlockPool>(num_blocks, config.kv_block_size);
        last_stats_.kv_block_capacity = num_blocks;
        printf("[OfflineBatch] GPU paged KV scheduler: blocks=%d block_size=%d\n",
               num_blocks, config.kv_block_size);
    } else {
        kv_pool = std::make_unique<ContinuousKVSlotPool>(
            std::max(1, config.batch_size), config.ctx_size);
    }
    std::vector<OfflineRequest> active;
    active.reserve(static_cast<size_t>(std::max(1, config.batch_size)));

    InferenceConfig inference;
    inference.max_new_tokens = config.max_tokens;
    inference.n_threads = config.n_threads;
    inference.kv_cache_size = config.ctx_size;
    inference.use_gpu = true;
    inference.gpu_id = config.gpu_id;

    const int eos_id = recognizer_.tokenizer().eos_id();
    size_t next_request = 0;
    int completed = 0;

    auto emit_finished = [&](OfflineRequest& request) {
        OfflineChunkResult result;
        result.id = request.id;
        result.slot_id = request.kv.slot_id;
        result.start_sec = request.start_sec;
        result.end_sec = request.end_sec;
        result.text = recognizer_.tokenizer().decode(request.tokens);
        result.ok = !result.text.empty();
        result.encoder_ms = request.encoder_ms;
        result.prefill_ms = request.prefill_ms;
        result.decode_ms = request.decode_ms;
        result.decode_tokens = static_cast<int>(request.tokens.size());
        result.prefill_tokens = request.prefill_tokens;
        result.total_ms = result.encoder_ms + result.prefill_ms + result.decode_ms;
        results.push_back(result);
        kv_pool->release(request.kv);
        if (progress) {
            progress(++completed, static_cast<int>(chunks.size()), results.back());
        } else {
            completed++;
        }
    };

    auto admit_one = [&]() -> bool {
        if (next_request >= chunks.size() ||
            static_cast<int>(active.size()) >= std::max(1, config.batch_size)) {
            return false;
        }

        int request_id = static_cast<int>(next_request);
        const int kv_budget = estimate_kv_budget_tokens(
            chunks[next_request], samples, sample_rate, config);
        last_stats_.admit_attempts++;
        if (!kv_pool->has_capacity(kv_budget)) {
            last_stats_.admit_no_kv_capacity++;
            return false;
        }

        KVHandle handle = kv_pool->acquire(request_id, kv_budget);
        if (!handle.valid()) {
            return false;
        }
        if (handle.paged()) {
            update_peak_blocks(*kv_pool);
        }

        const AudioChunk& chunk = chunks[next_request];
        size_t begin = std::min(chunk.infer_start_sample, samples.size());
        size_t end = std::min(chunk.infer_end_sample, samples.size());

        OfflineRequest request;
        request.id = request_id;
        request.phase = OfflineRequestPhase::Prefilling;
        request.kv = handle;
        request.max_tokens = config.max_tokens;
        request.start_sec = chunk.start_sec;
        request.end_sec = chunk.end_sec;

        if (end > begin) {
            GPUPrefillState prefill;
            auto prefill_t0 = std::chrono::high_resolution_clock::now();
            if (handle.paged()) {
                prefill = recognizer_.gpu_prefill_audio_paged(
                    AudioSpan{samples.data() + begin, end - begin},
                    request_id, handle.block_table, handle.block_size, inference);
            } else {
                prefill = recognizer_.gpu_prefill_audio_slot(
                    AudioSpan{samples.data() + begin, end - begin},
                    request_id, handle.slot_id, inference);
            }
            auto prefill_t1 = std::chrono::high_resolution_clock::now();
            last_stats_.prefill_wall_ms +=
                std::chrono::duration<double, std::milli>(prefill_t1 - prefill_t0).count();
            if (prefill.ok) {
                request.phase = OfflineRequestPhase::Decoding;
                request.n_past = prefill.n_past;
                request.logits = std::move(prefill.logits);
                request.next_token = argmax_logits(request.logits);
                if (handle.paged()) {
                    request.logits.clear();
                    request.logits.shrink_to_fit();
                }
                request.encoder_ms = prefill.stats.encoder_ms;
                request.prefill_ms = prefill.stats.prefill_ms;
                request.prefill_tokens = prefill.stats.prefill_tokens;
                active.push_back(std::move(request));
                last_stats_.admit_success++;
                next_request++;
                return true;
            }
        }

        request.phase = OfflineRequestPhase::Failed;
        emit_finished(request);
        next_request++;
        return true;
    };

    auto admit_round = [&]() -> int {
        int admitted = 0;
        while (next_request < chunks.size() &&
               static_cast<int>(active.size()) < std::max(1, config.batch_size)) {
            if (!admit_one()) {
                break;
            }
            admitted++;
        }
        if (admitted > 0) {
            last_stats_.admit_rounds++;
            last_stats_.admit_round_requests += admitted;
            last_stats_.max_admit_round_size =
                std::max(last_stats_.max_admit_round_size, admitted);
        }
        return admitted;
    };

    admit_round();

    while (!active.empty() || next_request < chunks.size()) {
        std::vector<GPUDecodeStepInput> decode_inputs;
        std::vector<size_t> decode_indices;
        decode_inputs.reserve(active.size());
        decode_indices.reserve(active.size());

        for (size_t i = 0; i < active.size(); i++) {
            OfflineRequest& request = active[i];
            if (request.phase != OfflineRequestPhase::Decoding) {
                continue;
            }
            if (static_cast<int>(request.tokens.size()) >= request.max_tokens ||
                request.n_past >= config.ctx_size) {
                request.phase = OfflineRequestPhase::Finished;
                continue;
            }

            int token = request.next_token >= 0
                ? request.next_token
                : argmax_logits(request.logits);
            request.next_token = -1;
            if (token == eos_id) {
                request.phase = OfflineRequestPhase::Finished;
                continue;
            }

            request.tokens.push_back(token);
            if (has_repetition_loop_local(request.tokens)) {
                printf("[OfflineBatch] WARNING: request %d stopped after detecting repetition loop\n",
                       request.id);
                request.phase = OfflineRequestPhase::Finished;
                continue;
            }

            GPUDecodeStepInput input;
            input.request_id = request.id;
            input.slot_id = request.kv.slot_id;
            input.n_past = request.n_past;
            input.token_id = token;
            input.block_table = request.kv.block_table;
            input.block_size = request.kv.block_size;
            decode_inputs.push_back(input);
            decode_indices.push_back(i);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<GPUDecodeStepOutput> decode_outputs;
        if (!decode_inputs.empty()) {
            last_stats_.decode_steps++;
            last_stats_.active_batch_sum += static_cast<long long>(decode_inputs.size());

            GPUDecodeDispatchStats dispatch_stats;
            decode_outputs = recognizer_.gpu_decode_step_slots(
                decode_inputs, inference, &dispatch_stats);
            last_stats_.fallback_reasons.token_id_fast_path_unavailable +=
                dispatch_stats.token_id_fast_path_unavailable;
            last_stats_.fallback_reasons.host_embedding_batch_unavailable +=
                dispatch_stats.host_embedding_batch_unavailable;
            last_stats_.fallback_reasons.serial_env_forced +=
                dispatch_stats.serial_env_forced;
            last_stats_.fallback_reasons.invalid_paged_input +=
                dispatch_stats.invalid_paged_input;
            if (decode_inputs.size() == 1) {
                last_stats_.fallback_reasons.single_request++;
            } else {
                last_stats_.decode_group_calls++;
            }
            last_stats_.sync_fallback_total();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        float step_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (!decode_inputs.empty()) {
            last_stats_.decode_dispatch_wall_ms +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        } else {
            last_stats_.scheduler_idle_steps++;
        }

        for (const auto& output : decode_outputs) {
            auto it = std::find_if(active.begin(), active.end(),
                [&](const OfflineRequest& request) {
                    return request.id == output.request_id;
                });
            if (it == active.end() || !output.ok) {
                continue;
            }
            it->n_past = output.n_past;
            it->next_token = output.next_token;
            if (!output.logits.empty()) {
                it->logits = output.logits;
            }
            it->decode_ms += step_ms;
            // This is wall time for the shared decode step. It is useful for
            // tracing but should not be summed as exclusive per-request time.
        }

        for (size_t i = active.size(); i > 0; i--) {
            size_t idx = i - 1;
            if (active[idx].phase == OfflineRequestPhase::Finished ||
                active[idx].phase == OfflineRequestPhase::Failed) {
                emit_finished(active[idx]);
                active.erase(active.begin() + static_cast<std::ptrdiff_t>(idx));
            }
        }

        admit_round();

        if (decode_inputs.empty() && active.empty() && next_request < chunks.size()) {
            if (admit_round() == 0) {
                break;
            }
        }
    }

    std::sort(results.begin(), results.end(),
              [](const OfflineChunkResult& a, const OfflineChunkResult& b) {
                  return a.id < b.id;
              });
    last_stats_.sync_fallback_total();
    return results;
}

OfflineChunkResult OfflineBatchTranscriber::run_one(
    const std::vector<float>& samples,
    int sample_rate,
    const OfflineRequest& request,
    const AudioChunk& chunk,
    const OfflineBatchConfig& config,
    int request_id) {
    (void)sample_rate;

    OfflineChunkResult result;
    result.id = request_id;
    result.slot_id = request.kv.slot_id;
    result.start_sec = chunk.start_sec;
    result.end_sec = chunk.end_sec;

    size_t begin = std::min(chunk.infer_start_sample, samples.size());
    size_t end = std::min(chunk.infer_end_sample, samples.size());
    if (end <= begin) {
        return result;
    }

    InferenceConfig inference;
    inference.max_new_tokens = config.max_tokens;
    inference.n_threads = config.n_threads;
    inference.kv_cache_size = config.ctx_size;
    inference.use_gpu = config.use_gpu;
    inference.gpu_id = config.gpu_id;

    PreparedLLMInput prepared = recognizer_.prepare_llm_input(
        samples.data() + begin,
        end - begin,
        inference);
    InferenceResult asr = recognizer_.run_prepared(prepared, inference);

    result.text = asr.text;
    result.ok = !asr.text.empty();
    result.encoder_ms = asr.encoder_ms;
    result.prefill_ms = asr.prefill_ms;
    result.decode_ms = asr.decode_ms;
    result.total_ms = asr.total_ms;
    result.prefill_tokens = asr.prefill_tokens;
    result.decode_tokens = asr.decode_tokens;
    return result;
}

} // namespace funasr
