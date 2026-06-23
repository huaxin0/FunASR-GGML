#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build-cuda/test_offline_batching"
MODEL="${ROOT_DIR}/FunAsr_q8.bin"
AUDIO="${ROOT_DIR}/outputs/video_asr/20260502_130430/media/source_16k.wav"
OUT_ROOT="${ROOT_DIR}/outputs/bench_mmq_streamk"
REPEAT=1
MAX_CHUNKS=""
BUILD=0

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --model <path>       Model file (default: ${MODEL})
  --audio <path>       Audio file (default: ${AUDIO})
  --bin <path>         test_offline_batching binary (default: ${BIN})
  --out <dir>          Output root directory (default: ${OUT_ROOT})
  --repeat <n>         Repeats per variant (default: ${REPEAT})
  --max-chunks <n>     Optional short benchmark chunk limit
  --build              Build test_offline_batching before running
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --audio)
            AUDIO="$2"
            shift 2
            ;;
        --bin)
            BIN="$2"
            shift 2
            ;;
        --out)
            OUT_ROOT="$2"
            shift 2
            ;;
        --repeat)
            REPEAT="$2"
            shift 2
            ;;
        --max-chunks)
            MAX_CHUNKS="$2"
            shift 2
            ;;
        --build)
            BUILD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "${MODEL}" ]]; then
    echo "Model not found: ${MODEL}" >&2
    exit 1
fi

if [[ ! -f "${AUDIO}" ]]; then
    echo "Audio not found: ${AUDIO}" >&2
    exit 1
fi

if [[ "${BUILD}" == "1" ]]; then
    cmake --build "${ROOT_DIR}/build-cuda" --target test_offline_batching -j"$(nproc)"
fi

if [[ ! -x "${BIN}" ]]; then
    echo "Binary not found or not executable: ${BIN}" >&2
    echo "Run with --build or build it manually first." >&2
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_ROOT}/${STAMP}"
mkdir -p "${OUT_DIR}"

SUMMARY="${OUT_DIR}/summary.tsv"
META="${OUT_DIR}/meta.txt"

cat > "${META}" <<EOF
root=${ROOT_DIR}
bin=${BIN}
model=${MODEL}
audio=${AUDIO}
repeat=${REPEAT}
max_chunks=${MAX_CHUNKS}
date=$(date -Is)
EOF

printf "variant\trepeat\twall_ms\trtf\tprefill_wall_ms\tdecode_dispatch_ms\tavg_decode_step_ms\tpaged_compute_ms\tpaged_total_ms\ttokens_s\tgrouped_calls\tfallback_calls\tcache_hit_rate\n" > "${SUMMARY}"

COMMON_ENV=(
    FUNASR_PAGED_KV_WRITE_OP=1
    FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1
    FUNASR_PAGED_DECODE_GRAPH_CACHE=1
)

COMMON_ARGS=(
    "${MODEL}"
    "${AUDIO}"
    --gpu
    --kv-mode paged
    --batch-size 12
    --ctx-size 4096
    --kv-block-size 128
    --chunk-mode window
    --chunk-sec 30
    --max-tokens 220
)

if [[ -n "${MAX_CHUNKS}" ]]; then
    COMMON_ARGS+=(--max-chunks "${MAX_CHUNKS}")
fi

run_variant() {
    local variant="$1"
    local extra_env="$2"
    local repeat_id="$3"
    local log_file="${OUT_DIR}/${variant}_r${repeat_id}.log"

    echo "=== Running ${variant} repeat ${repeat_id}/${REPEAT} ==="
    echo "log=${log_file}"

    if [[ -n "${extra_env}" ]]; then
        env "${COMMON_ENV[@]}" ${extra_env} "${BIN}" "${COMMON_ARGS[@]}" 2>&1 | tee "${log_file}"
    else
        env "${COMMON_ENV[@]}" "${BIN}" "${COMMON_ARGS[@]}" 2>&1 | tee "${log_file}"
    fi

    awk -v variant="${variant}" -v repeat_id="${repeat_id}" '
        function field_value(key, value, i) {
            for (i = 1; i <= NF; i++) {
                if (index($i, key "=") == 1) {
                    value = $i;
                    sub("^[^=]+=", "", value);
                    gsub(/ms|%/, "", value);
                    return value;
                }
            }
            return "";
        }
        /wall total=/ {
            wall_ms = field_value("total");
            rtf = field_value("rtf");
        }
        /scheduler:/ {
            grouped = field_value("grouped_calls");
            fallback = field_value("fallback_calls");
        }
        /scheduler_profile:/ {
            prefill = field_value("prefill_wall");
            decode = field_value("decode_dispatch");
            avg_step = field_value("avg_decode_step");
        }
        /throughput:/ {
            tokens = field_value("tokens/s");
        }
        /paged_profile:/ {
            paged_compute = field_value("compute");
            paged_total = field_value("total");
        }
        /paged_graph_cache_probe:/ {
            cache = field_value("cache_hit_rate");
        }
        END {
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                variant, repeat_id, wall_ms, rtf, prefill, decode, avg_step,
                paged_compute, paged_total, tokens, grouped, fallback, cache;
        }
    ' "${log_file}" >> "${SUMMARY}"
}

for repeat_id in $(seq 1 "${REPEAT}"); do
    run_variant "baseline" "" "${repeat_id}"
    run_variant "disable_all" "FUNASR_DISABLE_MMQ_STREAM_K=1" "${repeat_id}"
    run_variant "min_ncols_8" "FUNASR_MMQ_STREAM_K_MIN_NCOLS=8" "${repeat_id}"
    run_variant "min_ncols_16" "FUNASR_MMQ_STREAM_K_MIN_NCOLS=16" "${repeat_id}"
    run_variant "min_ncols_32" "FUNASR_MMQ_STREAM_K_MIN_NCOLS=32" "${repeat_id}"
done

echo
echo "=== Summary ==="
cat "${SUMMARY}"
echo
echo "Logs written to: ${OUT_DIR}"
