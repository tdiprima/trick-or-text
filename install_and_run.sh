#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install "chandra-ocr[hf]" pillow numpy pydicom

python "${ROOT_DIR}/generate_fake_images.py"
python "${ROOT_DIR}/run_chandra_ocr.py"

printf '\nPipeline finished. Outputs are in %s\n' "${ROOT_DIR}/artifacts"
