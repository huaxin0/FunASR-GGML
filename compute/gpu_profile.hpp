#ifndef FUNASR_COMPUTE_GPU_PROFILE_HPP
#define FUNASR_COMPUTE_GPU_PROFILE_HPP

namespace funasr {

inline int paged_decode_graph_max_n_kv(
    int exact_max_n_kv,
    int max_blocks,
    int block_size,
    bool bucket_enabled) {
    if (!bucket_enabled || max_blocks <= 0 || block_size <= 0) {
        return exact_max_n_kv;
    }
    return max_blocks * block_size;
}

struct PagedDecodeProfile {
    long calls = 0;
    double build_ms = 0.0;
    double alloc_ms = 0.0;
    double set_input_ms = 0.0;
    double compute_ms = 0.0;
    double get_ms = 0.0;
    long paged_attn_calls = 0;
    double paged_attn_ms = 0.0;
    long graph_cache_probe_calls = 0;
    long shape_cache_probe_hits = 0;
    long param_cache_probe_hits = 0;
    long full_graph_cache_probe_hits = 0;
    long graph_cache_hits = 0;
    long graph_cache_misses = 0;

    double total_ms() const {
        return build_ms + alloc_ms + set_input_ms + compute_ms + get_ms;
    }

    double avg_total_ms() const {
        return calls > 0 ? total_ms() / static_cast<double>(calls) : 0.0;
    }

    double avg_paged_attn_ms() const {
        return paged_attn_calls > 0
            ? paged_attn_ms / static_cast<double>(paged_attn_calls)
            : 0.0;
    }

    double shape_cache_probe_hit_rate() const {
        return graph_cache_probe_calls > 0
            ? 100.0 * static_cast<double>(shape_cache_probe_hits) /
                  static_cast<double>(graph_cache_probe_calls)
            : 0.0;
    }

    double full_graph_cache_probe_hit_rate() const {
        return graph_cache_probe_calls > 0
            ? 100.0 * static_cast<double>(full_graph_cache_probe_hits) /
                  static_cast<double>(graph_cache_probe_calls)
            : 0.0;
    }

    double param_cache_probe_hit_rate() const {
        return graph_cache_probe_calls > 0
            ? 100.0 * static_cast<double>(param_cache_probe_hits) /
                  static_cast<double>(graph_cache_probe_calls)
            : 0.0;
    }

    double graph_cache_hit_rate() const {
        const long total = graph_cache_hits + graph_cache_misses;
        return total > 0
            ? 100.0 * static_cast<double>(graph_cache_hits) /
                  static_cast<double>(total)
            : 0.0;
    }
};

} // namespace funasr

#endif // FUNASR_COMPUTE_GPU_PROFILE_HPP
