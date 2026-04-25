#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if ! command -v tesseract >/dev/null 2>&1; then
    printf 'Missing system dependency: tesseract\n' >&2
    printf 'Install it first, for example:\n' >&2
    printf '  Ubuntu/Debian: sudo apt-get install -y tesseract-ocr\n' >&2
    printf '  RHEL/Rocky:    sudo dnf install -y tesseract\n' >&2
    exit 1
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install "chandra-ocr[hf]" pillow numpy pydicom pytesseract

python "${ROOT_DIR}/generate_fake_images.py"
python "${ROOT_DIR}/compare_ocr_engines.py"

printf '\nPipeline finished. Outputs are in %s\n' "${ROOT_DIR}/artifacts"
