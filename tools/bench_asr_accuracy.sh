#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-cuda"
MODEL="${ROOT_DIR}/FunAsr_q8.bin"
WAV_DIR="/home/hua/data_aishell/wav/test"
TRANSCRIPT="/home/hua/data_aishell/transcript/aishell_transcript_v0.8.txt"
OUTPUT_ROOT="${ROOT_DIR}/outputs/accuracy_benchmark"
LIMIT=0
MAX_CER=2.2
CORPUS_BATCH=256
BATCH_SIZE=48
KV_BLOCKS=112
FRONTEND_PREFETCH=on
GPU_FRONTEND_OVERLAP=off
FRONTEND_BUCKET_WINDOW=32
GRAPH_CACHE_ENTRIES=16
BUILD=1

usage() {
  cat <<EOF
Usage: tools/bench_asr_accuracy.sh [options]

Options:
  --model <path>          Q8 model (default: ${MODEL})
  --wav-dir <path>        Evaluation WAV directory (default: ${WAV_DIR})
  --transcript <path>     AISHELL transcript (default: ${TRANSCRIPT})
  --output-root <path>    Report root (default: ${OUTPUT_ROOT})
  --limit <n>             First N sorted files; 0 means full corpus
  --max-cer <percent>     CER release threshold (default: ${MAX_CER})
  --corpus-batch <n>      Files loaded per scheduler call (default: ${CORPUS_BATCH})
  --batch-size <n>        Scheduler concurrency (default: ${BATCH_SIZE})
  --kv-num-blocks <n>     Physical KV blocks (default: ${KV_BLOCKS})
  --frontend-prefetch <on|off> CPU Fbank prefetch (default: ${FRONTEND_PREFETCH})
  --gpu-frontend-overlap <on|off> Experimental GPU stream overlap (default: ${GPU_FRONTEND_OVERLAP})
  --frontend-bucket-window <n> Length lookahead (default: ${FRONTEND_BUCKET_WINDOW})
  --mixed-graph-cache-entries <n> Mixed graph LRU size (default: ${GRAPH_CACHE_ENTRIES})
  --no-build              Reuse existing test_accuracy_benchmark
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --wav-dir) WAV_DIR="$2"; shift 2 ;;
    --transcript) TRANSCRIPT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --max-cer) MAX_CER="$2"; shift 2 ;;
    --corpus-batch) CORPUS_BATCH="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --kv-num-blocks) KV_BLOCKS="$2"; shift 2 ;;
    --frontend-prefetch) FRONTEND_PREFETCH="$2"; shift 2 ;;
    --gpu-frontend-overlap) GPU_FRONTEND_OVERLAP="$2"; shift 2 ;;
    --frontend-bucket-window) FRONTEND_BUCKET_WINDOW="$2"; shift 2 ;;
    --mixed-graph-cache-entries) GRAPH_CACHE_ENTRIES="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for path in "$MODEL" "$WAV_DIR" "$TRANSCRIPT"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
done

if [[ "$BUILD" -eq 1 ]]; then
  cmake --build "$BUILD_DIR" --target test_accuracy_benchmark -j2
fi

RUN_DIR="${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
HYPOTHESES="${RUN_DIR}/hypotheses.tsv"
EXPECTED_IDS="${RUN_DIR}/expected_ids.txt"

runner=(
  "${BUILD_DIR}/test_accuracy_benchmark"
  "$MODEL" "$WAV_DIR" "$HYPOTHESES"
  --expected-ids "$EXPECTED_IDS"
  --corpus-batch "$CORPUS_BATCH"
  --batch-size "$BATCH_SIZE"
  --ctx-size 4096
  --max-tokens 100
  --threads 4
  --kv-block-size 128
  --kv-num-blocks "$KV_BLOCKS"
  --max-scheduled-tokens 1024
  --max-prefill-chunk-tokens 512
  --max-frontend-requests 4
  --frontend-batching on
  --frontend-prefetch "$FRONTEND_PREFETCH"
  --gpu-frontend-overlap "$GPU_FRONTEND_OVERLAP"
  --frontend-bucket-window "$FRONTEND_BUCKET_WINDOW"
  --mixed-graph-cache-entries "$GRAPH_CACHE_ENTRIES"
)
if [[ "$LIMIT" -gt 0 ]]; then
  runner+=(--limit "$LIMIT")
fi

echo "[AccuracySuite] run_dir=${RUN_DIR}"
FUNASR_PAGED_KV_WRITE_OP=1 \
FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1 \
FUNASR_PAGED_DECODE_GRAPH_CACHE=1 \
  "${runner[@]}" 2>&1 | tee "${RUN_DIR}/inference.log"

python3 "${ROOT_DIR}/tools/eval_cer.py" \
  "$HYPOTHESES" "$TRANSCRIPT" \
  --expected-ids "$EXPECTED_IDS" \
  --metrics cer \
  --gate \
  --max-cer "$MAX_CER" \
  --json-out "${RUN_DIR}/accuracy.json" \
  --markdown-out "${RUN_DIR}/accuracy.md" \
  2>&1 | tee "${RUN_DIR}/accuracy.log"

echo "[AccuracySuite] PASS reports=${RUN_DIR}"
