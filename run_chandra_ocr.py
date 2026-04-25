#!/usr/bin/env python3
"""Run Chandra OCR 2 against the generated PNG and DICOM fixtures."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "artifacts" / "inputs"
OUTPUT_DIR = ROOT / "artifacts" / "ocr"


def require_chandra() -> str:
    executable = shutil.which("chandra")
    if not executable:
        raise SystemExit(
            "Could not find `chandra` on PATH. Run ./install_and_run.sh or activate the venv first."
        )
    return executable


def normalize_to_uint8(pixels: np.ndarray) -> np.ndarray:
    arr = pixels.astype(np.float32)
    arr -= arr.min()
    peak = arr.max()
    if peak > 0:
        arr = arr / peak
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def dicom_to_png(dicom_path: Path, output_dir: Path) -> Path:
    ds = pydicom.dcmread(str(dicom_path))
    pixels = normalize_to_uint8(ds.pixel_array)
    png_path = output_dir / f"{dicom_path.stem}_for_chandra.png"
    Image.fromarray(pixels, mode="L").save(png_path, format="PNG")
    return png_path


def markdown_to_plain_text(markdown_text: str) -> str:
    lines: list[str] = []
    in_code_block = False

    for raw_line in markdown_text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        line = raw_line
        if not in_code_block:
            line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
            line = re.sub(r"^\s{0,3}>\s?", "", line)
            line = re.sub(r"^\s{0,3}[-*+]\s+", "", line)
            line = re.sub(r"^\s{0,3}\d+\.\s+", "", line)

        lines.append(line.rstrip())

    return "\n".join(lines).strip() + "\n"


def run_chandra(chandra_bin: str, source_path: Path, output_root: Path) -> str:
    target_dir = output_root / source_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        chandra_bin,
        str(source_path),
        str(target_dir),
        "--method",
        "hf",
        "--no-images",
    ]
    subprocess.run(cmd, check=True)

    markdown_path = target_dir / source_path.stem / f"{source_path.stem}.md"
    if not markdown_path.exists():
        raise FileNotFoundError(f"Expected Chandra output at {markdown_path}")

    recognized_markdown = markdown_path.read_text(encoding="utf-8")
    recognized_text = markdown_to_plain_text(recognized_markdown)
    (target_dir / "recognized.md").write_text(recognized_markdown, encoding="utf-8")
    (target_dir / "recognized.txt").write_text(recognized_text, encoding="utf-8")
    return recognized_text


def build_summary(results: dict[str, dict[str, str]], output_path: Path) -> None:
    payload = {
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory holding fake_ocr_hostile.png and fake_ocr_hostile.dcm",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for Chandra outputs and recognized text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chandra_bin = require_chandra()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    png_input = args.input_dir / "fake_ocr_hostile.png"
    dicom_input = args.input_dir / "fake_ocr_hostile.dcm"
    if not png_input.exists() or not dicom_input.exists():
        raise SystemExit("Input images are missing. Run generate_fake_images.py first.")

    converted_dir = args.output_dir / "converted_inputs"
    converted_dir.mkdir(parents=True, exist_ok=True)
    dicom_png = dicom_to_png(dicom_input, converted_dir)

    results: dict[str, dict[str, str]] = {}
    for label, source in {
        "png": png_input,
        "dicom": dicom_png,
    }.items():
        recognized = run_chandra(chandra_bin, source, args.output_dir)
        results[label] = {
            "source": str(source.relative_to(ROOT)),
            "recognized_text_path": str(
                (args.output_dir / source.stem / "recognized.txt").relative_to(ROOT)
            ),
            "recognized_text": recognized,
        }

    build_summary(results, args.output_dir / "summary.json")
    sys.stdout.write(json.dumps({"status": "ok", "output_dir": str(args.output_dir.relative_to(ROOT))}) + "\n")


if __name__ == "__main__":
    main()
