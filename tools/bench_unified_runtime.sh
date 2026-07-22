#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build-cuda/test_offline_batching"
MODEL="${ROOT_DIR}/FunAsr_q8.bin"
AUDIO="${ROOT_DIR}/outputs/video_asr/20260502_130430/media/source_16k.wav"
OUT_ROOT="${ROOT_DIR}/outputs/bench_unified_runtime"
PROFILES="graph1:40:192:1:off:4:off,optimized40:40:192:16:on:32:off,optimized48:48:224:16:on:32:off,optimized56:56:256:16:on:32:off,overlap56:56:256:16:on:32:on"
REPEAT=1
MAX_CHUNKS=""
BUILD=0

usage() {
    printf '%s\n' \
        "Usage: $0 [options]" \
        "  --model <path>       Model file" \
        "  --audio <path>       Long audio file" \
        "  --bin <path>         test_offline_batching binary" \
        "  --out <dir>          Output root" \
        "  --profiles <csv>     name:batch:blocks:graphs:prefetch:bucket:gpu_overlap" \
        "  --repeat <n>         Repeats per profile (default: ${REPEAT})" \
        "  --max-chunks <n>     Optional short-run chunk limit" \
        "  --build              Build the benchmark binary first"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --audio) AUDIO="$2"; shift 2 ;;
        --bin) BIN="$2"; shift 2 ;;
        --out) OUT_ROOT="$2"; shift 2 ;;
        --profiles) PROFILES="$2"; shift 2 ;;
        --repeat) REPEAT="$2"; shift 2 ;;
        --max-chunks) MAX_CHUNKS="$2"; shift 2 ;;
        --build) BUILD=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${REPEAT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "repeat must be positive" >&2
    exit 2
}
if [[ -n "${MAX_CHUNKS}" && ! "${MAX_CHUNKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "max-chunks must be positive" >&2
    exit 2
fi
[[ -f "${MODEL}" ]] || { echo "Model not found: ${MODEL}" >&2; exit 2; }
[[ -f "${AUDIO}" ]] || { echo "Audio not found: ${AUDIO}" >&2; exit 2; }
if [[ "${BUILD}" == "1" ]]; then
    cmake --build "${ROOT_DIR}/build-cuda" -j2 --target test_offline_batching
fi
[[ -x "${BIN}" ]] || { echo "Binary not executable: ${BIN}" >&2; exit 2; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_ROOT}/${STAMP}"
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.tsv"
printf 'profile\trepeat\tbatch\tblocks\tgraphs\tprefetch\tbucket\tgpu_overlap\tok\twall_ms\trtf\taudio_xrt\tavg_active\tpeak_blocks\tcompute_ms\tgraph_hit_rate\tentries\tevictions\tfbank_ms\twait_ms\tgpu_frontend_ms\tpadding_pct\townership_errors\n' > "${SUMMARY}"

IFS=',' read -r -a PROFILE_LIST <<< "${PROFILES}"
for profile in "${PROFILE_LIST[@]}"; do
    IFS=':' read -r name batch blocks graphs prefetch bucket gpu_overlap <<< "${profile}"
    gpu_overlap="${gpu_overlap:-off}"
    if [[ -z "${name}" || ! "${batch}" =~ ^[1-9][0-9]*$ ||
          ! "${blocks}" =~ ^[1-9][0-9]*$ ||
          ! "${graphs}" =~ ^[1-9][0-9]*$ ||
          ! "${bucket}" =~ ^[1-9][0-9]*$ ||
          ( "${prefetch}" != "on" && "${prefetch}" != "off" ) ||
          ( "${gpu_overlap}" != "on" && "${gpu_overlap}" != "off" ) ]]; then
        echo "Invalid profile: ${profile}" >&2
        exit 2
    fi

    for repeat in $(seq 1 "${REPEAT}"); do
        log="${OUT_DIR}/${name}_${repeat}.log"
        args=(
            "${MODEL}" "${AUDIO}"
            --gpu --kv-mode paged --ctx-size 4096
            --batch-size "${batch}"
            --kv-block-size 128 --kv-num-blocks "${blocks}"
            --chunk-mode window --chunk-sec 30 --max-tokens 220
            --prefix-kv-cache on --dynamic-kv-blocks on
            --unified-scheduler on --max-scheduled-tokens 1024
            --max-prefill-chunk-tokens 512 --max-frontend-requests 4
            --frontend-batching on --frontend-prefetch "${prefetch}"
            --gpu-frontend-overlap "${gpu_overlap}"
            --frontend-bucket-window "${bucket}"
            --mixed-graph-cache-entries "${graphs}"
        )
        if [[ -n "${MAX_CHUNKS}" ]]; then
            args+=(--max-chunks "${MAX_CHUNKS}")
        fi

        echo "=== ${name} repeat=${repeat} batch=${batch} blocks=${blocks} ==="
        FUNASR_PAGED_KV_WRITE_OP=1 \
        FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1 \
        FUNASR_PAGED_DECODE_GRAPH_CACHE=1 \
            "${BIN}" "${args[@]}" 2>&1 | tee "${log}"

        awk -v name="${name}" -v repeat="${repeat}" \
            -v batch="${batch}" -v blocks="${blocks}" \
            -v graphs="${graphs}" -v prefetch="${prefetch}" \
            -v bucket="${bucket}" -v gpu_overlap="${gpu_overlap}" '
            function value(key, out, i) {
                for (i = 1; i <= NF; ++i) {
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
            /throughput:/ { audio_xrt = value("audio_sec/s") }
            /scheduler:/ {
                avg_active = value("avg_active")
                peak_pair = value("blocks_peak")
                split(peak_pair, peak, "/")
                peak_blocks = peak[1]
            }
            /unified_profile:/ {
                compute = value("compute")
                graph_hit = value("hit_rate")
                entry_pair = value("entries_peak")
                split(entry_pair, entry, "/")
                entries = entry[1]
                evictions = value("evictions")
            }
            /frontend_batch_profile:/ {
                fbank = value("fbank")
                wait = value("wait")
                gpu_frontend = value("batch_wall")
                padding_pair = value("padding")
                split(padding_pair, padding, "/")
                padding_pct = padding[2] > 0 ? 100.0 * padding[1] / padding[2] : 0
            }
            /paged_kv_runtime:/ { errors = value("ownership_errors") }
            END {
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.2f\t%s\n",
                    name, repeat, batch, blocks, graphs, prefetch, bucket, gpu_overlap,
                    ok, wall, rtf, audio_xrt, avg_active, peak_blocks,
                    compute, graph_hit, entries, evictions, fbank, wait,
                    gpu_frontend, padding_pct, errors
            }
        ' "${log}" >> "${SUMMARY}"
    done
done

echo
echo "=== Summary ==="
column -t -s $'\t' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
echo "Logs written to: ${OUT_DIR}"
