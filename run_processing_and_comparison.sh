#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ -f "${VENV_DIR}/bin/activate" && "${USE_ACTIVE_ENV:-0}" != "1" ]]; then
    source "${VENV_DIR}/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_OCR_ENGINES="pytesseract,chandra,ollama-gemma4-latest,ollama-gemma4-26b,lightonocr"
SELECTED_OCR_ENGINES="${OCR_ENGINES:-${DEFAULT_OCR_ENGINES}}"

"${PYTHON_BIN}" "${ROOT_DIR}/generate_fake_images.py"
printf 'Running OCR comparison engines: %s\n' "${SELECTED_OCR_ENGINES}"
"${PYTHON_BIN}" "${ROOT_DIR}/compare_ocr_engines.py" \
    --engines "${SELECTED_OCR_ENGINES}" \
    --ollama-base-url "${OLLAMA_BASE_URL:-http://localhost:11434}" \
    --ollama-timeout-seconds "${OLLAMA_TIMEOUT_SECONDS:-300}"

printf '\nProcessing and comparison finished. Outputs are in %s\n' "${ROOT_DIR}/artifacts"
