#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build-cuda/test_offline_batching"
MODEL="${ROOT_DIR}/FunAsr_q8.bin"
AUDIO="${ROOT_DIR}/outputs/video_asr/20260502_130430/media/source_16k.wav"
OUT_ROOT="${ROOT_DIR}/outputs/bench_paged_kv_pool_concurrency"
REPEAT=1
WARMUP=0
MAX_CHUNKS=""
NUM_BLOCKS=160
BATCH_LIST="12,16,24,32"
BUILD=0

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --model <path>       Model file (default: ${MODEL})
  --audio <path>       Audio file (default: ${AUDIO})
  --bin <path>         Benchmark binary (default: ${BIN})
  --out <dir>          Output root (default: ${OUT_ROOT})
  --repeat <n>         Measured repeats per batch (default: ${REPEAT})
  --warmup <n>         Unrecorded repeats per batch (default: ${WARMUP})
  --max-chunks <n>     Optional selected chunk limit
  --blocks <n>         Fixed global KV block count (default: ${NUM_BLOCKS})
  --batches <csv>      Batch sizes to run (default: ${BATCH_LIST})
  --build              Build test_offline_batching first
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --audio) AUDIO="$2"; shift 2 ;;
        --bin) BIN="$2"; shift 2 ;;
        --out) OUT_ROOT="$2"; shift 2 ;;
        --repeat) REPEAT="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --max-chunks) MAX_CHUNKS="$2"; shift 2 ;;
        --blocks) NUM_BLOCKS="$2"; shift 2 ;;
        --batches) BATCH_LIST="$2"; shift 2 ;;
        --build) BUILD=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

for value in "${REPEAT}" "${NUM_BLOCKS}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "repeat and blocks must be positive integers" >&2
        exit 1
    }
done
[[ "${WARMUP}" =~ ^[0-9]+$ ]] || {
    echo "warmup must be a non-negative integer" >&2
    exit 1
}
if [[ -n "${MAX_CHUNKS}" && ! "${MAX_CHUNKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "max-chunks must be a positive integer" >&2
    exit 1
fi
IFS=',' read -r -a BATCHES <<< "${BATCH_LIST}"
[[ "${#BATCHES[@]}" -gt 0 ]] || { echo "batches cannot be empty" >&2; exit 1; }
for batch in "${BATCHES[@]}"; do
    [[ "${batch}" =~ ^[1-9][0-9]*$ ]] || {
        echo "batches must be comma-separated positive integers" >&2
        exit 1
    }
done

[[ -f "${MODEL}" ]] || { echo "Model not found: ${MODEL}" >&2; exit 1; }
[[ -f "${AUDIO}" ]] || { echo "Audio not found: ${AUDIO}" >&2; exit 1; }
if [[ "${BUILD}" == "1" ]]; then
    cmake --build "${ROOT_DIR}/build-cuda" -j2 --target test_offline_batching
fi
[[ -x "${BIN}" ]] || { echo "Binary not executable: ${BIN}" >&2; exit 1; }

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
warmup=${WARMUP}
max_chunks=${MAX_CHUNKS}
blocks=${NUM_BLOCKS}
batches=${BATCH_LIST}
block_size=128
date=$(date -Is)
EOF

printf "batch\trepeat\tok\twall_ms\trtf\ttokens_s\tavg_active\tno_kv\tpeak_blocks\tcapacity\tfinal_free\townership_errors\tpeak_vram_mib\tprefill_wall_ms\tdecode_dispatch_ms\tavg_decode_step_ms\tgraph_hit_rate\n" > "${SUMMARY}"

COMMON_ENV=(
    FUNASR_PAGED_KV_WRITE_OP=1
    FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1
    FUNASR_PAGED_DECODE_GRAPH_CACHE=1
)

COMMON_ARGS=(
    "${MODEL}" "${AUDIO}"
    --gpu --kv-mode paged --ctx-size 4096
    --kv-block-size 128 --kv-num-blocks "${NUM_BLOCKS}"
    --chunk-mode window --chunk-sec 30 --max-tokens 220
    --prefix-kv-cache off --dynamic-kv-blocks on
)
if [[ -n "${MAX_CHUNKS}" ]]; then
    COMMON_ARGS+=(--max-chunks "${MAX_CHUNKS}")
fi

SAMPLER_PID=""
stop_sampler() {
    if [[ -n "${SAMPLER_PID}" ]]; then
        kill "${SAMPLER_PID}" 2>/dev/null || true
        wait "${SAMPLER_PID}" 2>/dev/null || true
        SAMPLER_PID=""
    fi
}
trap stop_sampler EXIT INT TERM

start_sampler() {
    local samples_file="$1"
    (
        while true; do
            nvidia-smi -i 0 \
                --query-gpu=memory.used \
                --format=csv,noheader,nounits 2>/dev/null || true
            sleep 0.1
        done
    ) > "${samples_file}" &
    SAMPLER_PID=$!
}

run_once() {
    local batch="$1"
    local repeat_id="$2"
    local record="$3"
    local label="b${batch}_${repeat_id}"
    local log_file="${OUT_DIR}/${label}.log"
    local samples_file="${OUT_DIR}/${label}_vram.txt"

    echo "=== batch=${batch} blocks=${NUM_BLOCKS} repeat=${repeat_id} ==="
    start_sampler "${samples_file}"
    set +e
    env "${COMMON_ENV[@]}" "${BIN}" "${COMMON_ARGS[@]}" \
        --batch-size "${batch}" 2>&1 | tee "${log_file}"
    local run_status=${PIPESTATUS[0]}
    set -e
    stop_sampler
    if [[ "${run_status}" -ne 0 ]]; then
        echo "Benchmark failed: batch=${batch} repeat=${repeat_id}" >&2
        return "${run_status}"
    fi

    if [[ "${record}" != "1" ]]; then
        return 0
    fi

    local peak_vram
    peak_vram="$(awk 'NF && $1 + 0 > max { max = $1 + 0 } END { print max + 0 }' "${samples_file}")"
    awk -v batch="${batch}" -v repeat_id="${repeat_id}" -v peak_vram="${peak_vram}" '
        function value(key, out, i) {
            for (i = 1; i <= NF; i++) {
                if (index($i, key "=") == 1) {
                    out = $i
                    sub("^[^=]+=", "", out)
                    gsub(/ms|%/, "", out)
                    return out
                }
            }
            return ""
        }
        /\[OfflineTest\] ok=/ { ok = value("ok") }
        /wall total=/ { wall = value("total"); rtf = value("rtf") }
        /scheduler:/ {
            avg_active = value("avg_active")
            peak_pair = value("blocks_peak")
            split(peak_pair, block_parts, "/")
            peak_blocks = block_parts[1]
            capacity = block_parts[2]
        }
        /scheduler_profile:/ {
            no_kv = value("no_kv")
            prefill = value("prefill_wall")
            decode = value("decode_dispatch")
            avg_step = value("avg_decode_step")
        }
        /throughput:/ { tokens = value("tokens/s") }
        /paged_kv_runtime:/ {
            final_pair = value("final_free")
            split(final_pair, free_parts, "/")
            final_free = free_parts[1]
            errors = value("ownership_errors")
        }
        /paged_graph_cache_probe:/ { graph_hit = value("cache_hit_rate") }
        END {
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                batch, repeat_id, ok, wall, rtf, tokens, avg_active, no_kv,
                peak_blocks, capacity, final_free, errors, peak_vram,
                prefill, decode, avg_step, graph_hit
        }
    ' "${log_file}" >> "${SUMMARY}"
}

for batch in "${BATCHES[@]}"; do
    for warmup_id in $(seq 1 "${WARMUP}"); do
        run_once "${batch}" "warmup${warmup_id}" 0
    done
    for repeat_id in $(seq 1 "${REPEAT}"); do
        run_once "${batch}" "${repeat_id}" 1
    done
done

echo
echo "=== Summary ==="
cat "${SUMMARY}"
echo
echo "Logs written to: ${OUT_DIR}"
