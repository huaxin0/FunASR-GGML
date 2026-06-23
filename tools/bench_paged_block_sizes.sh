#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  tools/bench_paged_block_sizes.sh <model.bin> <audio.wav/mp3/flac> [options]

Options:
  --bin <path>             test_offline_batching binary (default: ./build-cuda/test_offline_batching)
  --build-dir <path>       CMake build dir used for optional build (default: build-cuda)
  --log-dir <path>         Directory for per-run logs (default: outputs/bench_paged_blocks/<timestamp>)
  --sizes "<list>"         Space-separated KV block sizes (default: "32 64 96 128")
  --batch-size <n>         Offline scheduler slots (default: 12)
  --ctx-size <n>           KV context size (default: 4096)
  --chunk-sec <n>          Window chunk seconds (default: 30)
  --max-tokens <n>         Max generated tokens (default: 220)
  --gpu-id <n>             CUDA device id (default: 0)
  --no-build               Do not build test_offline_batching before running
  --summarize <log-dir>    Rebuild summary.tsv from existing block_*.log files and exit
  --extra "<args>"         Extra args appended to each test_offline_batching run
  -h, --help               Show this help

Example:
  tools/bench_paged_block_sizes.sh FunAsr_q8.bin outputs/video_asr/20260502_130430/media/source_16k.wav \
    --batch-size 12 --ctx-size 4096 --sizes "32 64 96 128"
USAGE
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

if [[ "${1:-}" == "--summarize" ]]; then
    if [[ $# -ne 2 ]]; then
        usage
        exit 1
    fi
    log_dir="$2"
    if [[ ! -d "$log_dir" ]]; then
        echo "Log directory not found: $log_dir" >&2
        exit 1
    fi
    shift 2
else
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage
    exit 1
fi

model_path="$1"
audio_path="$2"
shift 2
fi

bin_path="./build-cuda/test_offline_batching"
build_dir="build-cuda"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="${log_dir:-outputs/bench_paged_blocks/${timestamp}}"
sizes="32 64 96 128"
batch_size=12
ctx_size=4096
chunk_sec=30
max_tokens=220
gpu_id=0
do_build=1
extra_args=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin)
            bin_path="$2"
            shift 2
            ;;
        --build-dir)
            build_dir="$2"
            shift 2
            ;;
        --log-dir)
            log_dir="$2"
            shift 2
            ;;
        --sizes)
            sizes="$2"
            shift 2
            ;;
        --batch-size)
            batch_size="$2"
            shift 2
            ;;
        --ctx-size)
            ctx_size="$2"
            shift 2
            ;;
        --chunk-sec)
            chunk_sec="$2"
            shift 2
            ;;
        --max-tokens)
            max_tokens="$2"
            shift 2
            ;;
        --gpu-id)
            gpu_id="$2"
            shift 2
            ;;
        --no-build)
            do_build=0
            shift
            ;;
        --extra)
            extra_args="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

extract_value() {
    local marker="$1"
    local key="$2"
    local file="$3"
    awk -v marker="$marker" -v key="$key" '
        index($0, marker) > 0 {
            for (i = 1; i <= NF; i++) {
                split($i, parts, "=")
                if (parts[1] == key) {
                    value = parts[2]
                    sub(/ms$/, "", value)
                    print value
                    exit
                }
            }
        }
    ' "$file"
}

write_summary_header() {
    printf "block_size\tstatus\twall_ms\trtf\taudio_sec_per_s\ttokens_per_s\tavg_active\tfallback_calls\tcompute_ms\ttotal_step_ms\tlog\n"
}

summarize_log() {
    local block_size="$1"
    local status="$2"
    local log_path="$3"

    local wall_ms rtf audio_sps tokens_s avg_active fallback_calls compute_ms total_step_ms
    wall_ms="$(awk '/\[OfflineTest\] wall total=/{sub(/.*wall total=/, ""); sub(/ms.*/, ""); print; exit}' "$log_path")"
    rtf="$(extract_value "[OfflineTest] throughput:" "rtf" "$log_path")"
    audio_sps="$(extract_value "[OfflineTest] throughput:" "audio_sec/s" "$log_path")"
    tokens_s="$(extract_value "[OfflineTest] throughput:" "tokens/s" "$log_path")"
    avg_active="$(extract_value "[OfflineTest] scheduler:" "avg_active" "$log_path")"
    fallback_calls="$(extract_value "[OfflineTest] scheduler:" "fallback_calls" "$log_path")"
    compute_ms="$(extract_value "[OfflineTest] paged_profile:" "compute" "$log_path")"
    total_step_ms="$(extract_value "[OfflineTest] paged_profile:" "total" "$log_path")"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$block_size" "$status" "${wall_ms:-NA}" "${rtf:-NA}" \
        "${audio_sps:-NA}" "${tokens_s:-NA}" "${avg_active:-NA}" \
        "${fallback_calls:-NA}" "${compute_ms:-NA}" "${total_step_ms:-NA}" \
        "$log_path"
}

if [[ -z "${model_path:-}" ]]; then
    summary_path="${log_dir}/summary.tsv"
    write_summary_header | tee "$summary_path"
    for log_path in "$log_dir"/block_*.log; do
        [[ -e "$log_path" ]] || continue
        block_size="$(basename "$log_path" .log)"
        block_size="${block_size#block_}"
        if rg -q "\\[OfflineTest\\] ok=" "$log_path"; then
            status=0
        else
            status=1
        fi
        summarize_log "$block_size" "$status" "$log_path" | tee -a "$summary_path"
    done
    echo "Summary: $summary_path"
    exit 0
fi

if [[ ! -f "$model_path" ]]; then
    echo "Model not found: $model_path" >&2
    exit 1
fi

if [[ ! -f "$audio_path" ]]; then
    echo "Audio not found: $audio_path" >&2
    exit 1
fi

if [[ "$do_build" -eq 1 ]]; then
    cmake --build "$build_dir" --target test_offline_batching -j"$(nproc)"
fi

mkdir -p "$log_dir"
summary_path="${log_dir}/summary.tsv"

write_summary_header | tee "$summary_path"

for block_size in $sizes; do
    log_path="${log_dir}/block_${block_size}.log"
    echo
    echo "========== block_size=${block_size} =========="

    gpu_args=()
    if [[ "$gpu_id" != "0" ]]; then
        gpu_args=(--gpu-id "$gpu_id")
    fi

    cmd=(
        "$bin_path" "$model_path" "$audio_path"
        --gpu
        "${gpu_args[@]}"
        --kv-mode paged
        --kv-block-size "$block_size"
        --batch-size "$batch_size"
        --ctx-size "$ctx_size"
        --chunk-mode window
        --chunk-sec "$chunk_sec"
        --max-tokens "$max_tokens"
    )

    echo "[Bench] command: ${cmd[*]} ${extra_args}" | tee "$log_path"
    set +e
    # shellcheck disable=SC2086
    "${cmd[@]}" $extra_args 2>&1 | tee -a "$log_path"
    status=${PIPESTATUS[0]}
    set -e

    summarize_log "$block_size" "$status" "$log_path" | tee -a "$summary_path"
done

echo
echo "Summary: $summary_path"
