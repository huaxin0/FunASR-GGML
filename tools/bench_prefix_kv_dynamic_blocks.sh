#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build-cuda/test_offline_batching"
MODEL="${ROOT_DIR}/FunAsr_q8.bin"
AUDIO="${ROOT_DIR}/outputs/video_asr/20260502_130430/media/source_16k.wav"
OUT_ROOT="${ROOT_DIR}/outputs/bench_prefix_kv_dynamic"
REPEAT=1
WARMUP=0
MAX_CHUNKS=24

usage() {
    cat <<EOF
Usage: $0 [options]
  --model <path>       Model file
  --audio <path>       Audio file
  --bin <path>         Benchmark binary
  --out <dir>          Output root
  --repeat <n>         Measured repeats per variant (default: ${REPEAT})
  --warmup <n>         Unrecorded repeats per variant (default: ${WARMUP})
  --max-chunks <n>     Selected chunk limit (default: ${MAX_CHUNKS})
  --build              Build the benchmark binary first
EOF
}

BUILD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --audio) AUDIO="$2"; shift 2 ;;
        --bin) BIN="$2"; shift 2 ;;
        --out) OUT_ROOT="$2"; shift 2 ;;
        --repeat) REPEAT="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --max-chunks) MAX_CHUNKS="$2"; shift 2 ;;
        --build) BUILD=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
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
printf "variant\trepeat\tok\twall_ms\trtf\ttokens_s\tpeak_blocks\twaste_rate\tprefix_hits\tcow\tprefill_append\tdecode_append\townership_errors\tfinal_free\n" > "${SUMMARY}"

COMMON_ENV=(
    FUNASR_PAGED_KV_WRITE_OP=1
    FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1
    FUNASR_PAGED_DECODE_GRAPH_CACHE=1
)
COMMON_ARGS=(
    "${MODEL}" "${AUDIO}"
    --gpu --kv-mode paged --batch-size 12 --ctx-size 4096
    --kv-block-size 128 --chunk-mode window --chunk-sec 30
    --max-tokens 220 --max-chunks "${MAX_CHUNKS}"
)

run_once() {
    local variant="$1" prefix="$2" dynamic="$3" repeat_id="$4" log_file="$5"
    env "${COMMON_ENV[@]}" "${BIN}" "${COMMON_ARGS[@]}" \
        --prefix-kv-cache "${prefix}" --dynamic-kv-blocks "${dynamic}" \
        2>&1 | tee "${log_file}"

    awk -v variant="${variant}" -v repeat_id="${repeat_id}" '
        function value(key, out, i) {
            for (i = 1; i <= NF; i++) if (index($i, key "=") == 1) {
                out = $i; sub("^[^=]+=", "", out); gsub(/ms|%/, "", out); return out
            }
            return ""
        }
        /\[OfflineTest\] ok=/ { ok = value("ok") }
        /wall total=/ { wall = value("total"); rtf = value("rtf") }
        /throughput:/ { tokens = value("tokens/s") }
        /scheduler:/ { peak = value("blocks_peak") }
        /paged_kv_waste:/ { waste = value("waste_rate") }
        /paged_kv_runtime:/ {
            hits = value("prefix_hits"); cow = value("cow_copies")
            pfappend = value("prefill_appends"); decappend = value("decode_appends")
            errors = value("ownership_errors"); finalfree = value("final_free")
        }
        END { printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
            variant, repeat_id, ok, wall, rtf, tokens, peak, waste, hits, cow,
            pfappend, decappend, errors, finalfree }
    ' "${log_file}" >> "${SUMMARY}"
}

run_variant() {
    local variant="$1" prefix="$2" dynamic="$3"
    for warmup_id in $(seq 1 "${WARMUP}"); do
        echo "=== Warmup ${variant} ${warmup_id}/${WARMUP} ==="
        run_once "${variant}" "${prefix}" "${dynamic}" "warmup${warmup_id}" \
            "${OUT_DIR}/${variant}_warmup${warmup_id}.log"
        sed -i '$d' "${SUMMARY}"
    done
    for repeat_id in $(seq 1 "${REPEAT}"); do
        echo "=== Running ${variant} ${repeat_id}/${REPEAT} ==="
        run_once "${variant}" "${prefix}" "${dynamic}" "${repeat_id}" \
            "${OUT_DIR}/${variant}_r${repeat_id}.log"
    done
}

run_variant static_no_cache off off
run_variant dynamic_no_cache off on
run_variant static_prefix_cache on off
run_variant dynamic_prefix_cache on on

echo
echo "=== Summary ==="
cat "${SUMMARY}"
echo
echo "Logs written to: ${OUT_DIR}"
