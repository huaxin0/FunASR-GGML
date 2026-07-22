#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGML_DIR="${ROOT_DIR}/third_party/ggml"
PATCH_FILE="${ROOT_DIR}/patches/ggml-funasr-runtime.patch"

if [[ ! -e "${GGML_DIR}/.git" ]]; then
    echo "GGML submodule is not initialized." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 2
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "Missing runtime patch: ${PATCH_FILE}" >&2
    exit 2
fi

if git -C "${GGML_DIR}" apply --reverse --check "${PATCH_FILE}" \
        >/dev/null 2>&1; then
    echo "FunASR GGML runtime patch is already applied."
    exit 0
fi

git -C "${GGML_DIR}" apply --check "${PATCH_FILE}"
git -C "${GGML_DIR}" apply "${PATCH_FILE}"
echo "Applied FunASR GGML runtime extensions."
