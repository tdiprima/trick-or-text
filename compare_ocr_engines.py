#!/usr/bin/env python3
"""Run pytesseract and Chandra OCR against generated fixtures and compare results."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Sequence

import pytesseract

from run_chandra_ocr import INPUT_DIR, OUTPUT_DIR, ROOT, dicom_to_png, require_chandra, run_chandra


GROUND_TRUTH_FILENAMES = {
    "png": "fake_ocr_hostile_png_ground_truth.txt",
    "dicom": "fake_ocr_hostile_dicom_ground_truth.txt",
}


def require_tesseract() -> str:
    executable = shutil.which("tesseract")
    if not executable:
        raise SystemExit(
            "Could not find `tesseract` on PATH. Install the system package first, then rerun."
        )
    return executable


def normalize_text(text: str) -> str:
    collapsed = " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())
    return unicodedata.normalize("NFKC", collapsed).strip()


def edit_stats(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, int]:
    rows = len(reference)
    cols = len(hypothesis)
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(cols + 1)] for _ in range(rows + 1)
    ]

    for i in range(1, rows + 1):
        distance, subs, ins, dels = dp[i - 1][0]
        dp[i][0] = (distance + 1, subs, ins, dels + 1)
    for j in range(1, cols + 1):
        distance, subs, ins, dels = dp[0][j - 1]
        dp[0][j] = (distance + 1, subs, ins + 1, dels)

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                candidates = [dp[i - 1][j - 1]]
            else:
                distance, subs, ins, dels = dp[i - 1][j - 1]
                candidates = [(distance + 1, subs + 1, ins, dels)]

            distance, subs, ins, dels = dp[i][j - 1]
            candidates.append((distance + 1, subs, ins + 1, dels))

            distance, subs, ins, dels = dp[i - 1][j]
            candidates.append((distance + 1, subs, ins, dels + 1))
            dp[i][j] = min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))

    distance, substitutions, insertions, deletions = dp[rows][cols]
    return {
        "distance": distance,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
    }


def accuracy(reference_length: int, distance: int) -> float:
    if reference_length == 0:
        return 100.0 if distance == 0 else 0.0
    return max(0.0, (1.0 - (distance / reference_length)) * 100.0)


def score_text(reference_text: str, recognized_text: str) -> dict[str, object]:
    normalized_reference = normalize_text(reference_text)
    normalized_recognized = normalize_text(recognized_text)
    char_stats = edit_stats(list(normalized_reference), list(normalized_recognized))
    reference_words = normalized_reference.split()
    recognized_words = normalized_recognized.split()
    word_stats = edit_stats(reference_words, recognized_words)
    return {
        "char_accuracy_percent": round(
            accuracy(len(normalized_reference), char_stats["distance"]),
            2,
        ),
        "word_accuracy_percent": round(
            accuracy(len(reference_words), word_stats["distance"]),
            2,
        ),
        "substitutions": word_stats["substitutions"],
        "char_distance": char_stats["distance"],
        "word_distance": word_stats["distance"],
        "word_insertions": word_stats["insertions"],
        "word_deletions": word_stats["deletions"],
        "reference_text": normalized_reference,
        "recognized_text": normalized_recognized,
    }


def run_pytesseract(image_path: Path, tesseract_cmd: str) -> str:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    return pytesseract.image_to_string(str(image_path), config="--psm 6")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_engine_summary(sample_metrics: dict[str, dict[str, object]]) -> dict[str, float | int]:
    char_total = 0.0
    word_total = 0.0
    substitutions = 0
    count = len(sample_metrics)
    for metrics in sample_metrics.values():
        char_total += float(metrics["char_accuracy_percent"])
        word_total += float(metrics["word_accuracy_percent"])
        substitutions += int(metrics["substitutions"])
    return {
        "average_char_accuracy_percent": round(char_total / count, 2),
        "average_word_accuracy_percent": round(word_total / count, 2),
        "total_substitutions": substitutions,
    }


def choose_winner(
    pytesseract_metrics: dict[str, object],
    chandra_metrics: dict[str, object],
) -> str:
    pytesseract_score = (
        float(pytesseract_metrics["char_accuracy_percent"]),
        float(pytesseract_metrics["word_accuracy_percent"]),
        -int(pytesseract_metrics["substitutions"]),
    )
    chandra_score = (
        float(chandra_metrics["char_accuracy_percent"]),
        float(chandra_metrics["word_accuracy_percent"]),
        -int(chandra_metrics["substitutions"]),
    )
    if pytesseract_score > chandra_score:
        return "pytesseract"
    if chandra_score > pytesseract_score:
        return "chandra"
    return "tie"


def write_human_report(results: dict[str, object], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("OCR comparison report")
    lines.append("")

    sample_wins = {"pytesseract": 0, "chandra": 0, "tie": 0}
    for label in ("png", "dicom"):
        sample = results["samples"][label]
        pytesseract_metrics = sample["engines"]["pytesseract"]["metrics"]
        chandra_metrics = sample["engines"]["chandra"]["metrics"]
        winner = choose_winner(pytesseract_metrics, chandra_metrics)
        sample_wins[winner] += 1
        lines.append(f"{label.upper()} sample")
        lines.append(
            "pytesseract: "
            f"{pytesseract_metrics['char_accuracy_percent']}% char accuracy, "
            f"{pytesseract_metrics['word_accuracy_percent']}% word accuracy, "
            f"{pytesseract_metrics['substitutions']} substitutions."
        )
        lines.append(
            "Chandra OCR: "
            f"{chandra_metrics['char_accuracy_percent']}% char accuracy, "
            f"{chandra_metrics['word_accuracy_percent']}% word accuracy, "
            f"{chandra_metrics['substitutions']} substitutions."
        )
        if winner == "tie":
            lines.append("Judgment: this sample is effectively a tie after the selected scoring.")
        else:
            lines.append(
                f"Judgment: {winner} did better on this sample because it kept higher accuracy "
                "while making fewer replacement mistakes."
            )
        lines.append("")

    pytesseract_overall = results["overall"]["pytesseract"]
    chandra_overall = results["overall"]["chandra"]
    overall_winner = choose_winner(
        {
            "char_accuracy_percent": pytesseract_overall["average_char_accuracy_percent"],
            "word_accuracy_percent": pytesseract_overall["average_word_accuracy_percent"],
            "substitutions": pytesseract_overall["total_substitutions"],
        },
        {
            "char_accuracy_percent": chandra_overall["average_char_accuracy_percent"],
            "word_accuracy_percent": chandra_overall["average_word_accuracy_percent"],
            "substitutions": chandra_overall["total_substitutions"],
        },
    )

    lines.append("Overall")
    lines.append(
        "pytesseract average: "
        f"{pytesseract_overall['average_char_accuracy_percent']}% char accuracy, "
        f"{pytesseract_overall['average_word_accuracy_percent']}% word accuracy, "
        f"{pytesseract_overall['total_substitutions']} substitutions."
    )
    lines.append(
        "Chandra OCR average: "
        f"{chandra_overall['average_char_accuracy_percent']}% char accuracy, "
        f"{chandra_overall['average_word_accuracy_percent']}% word accuracy, "
        f"{chandra_overall['total_substitutions']} substitutions."
    )
    if overall_winner == "tie":
        lines.append(
            "Final judgment: neither engine clearly won overall. Their average scores and "
            "substitution counts are effectively tied."
        )
    else:
        lines.append(
            f"Final judgment: {overall_winner} did better overall. It won "
            f"{sample_wins[overall_winner]} sample(s) and finished with the stronger combined "
            "accuracy and substitution profile."
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory holding generated fixtures and ground-truth text files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for OCR output artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chandra_bin = require_chandra()
    tesseract_cmd = require_tesseract()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    png_input = args.input_dir / "fake_ocr_hostile.png"
    dicom_input = args.input_dir / "fake_ocr_hostile.dcm"
    if not png_input.exists() or not dicom_input.exists():
        raise SystemExit("Input images are missing. Run generate_fake_images.py first.")

    converted_dir = args.output_dir / "converted_inputs"
    converted_dir.mkdir(parents=True, exist_ok=True)
    dicom_png = dicom_to_png(dicom_input, converted_dir)

    sources = {
        "png": png_input,
        "dicom": dicom_png,
    }
    results: dict[str, object] = {"samples": {}, "overall": {}}

    for label, source in sources.items():
        sample_dir = args.output_dir / "comparison" / label
        sample_dir.mkdir(parents=True, exist_ok=True)
        ground_truth_path = args.input_dir / GROUND_TRUTH_FILENAMES[label]
        ground_truth = ground_truth_path.read_text(encoding="utf-8")

        pytesseract_text = run_pytesseract(source, tesseract_cmd)
        pytesseract_path = sample_dir / "pytesseract_recognized.txt"
        pytesseract_path.write_text(pytesseract_text, encoding="utf-8")

        chandra_text = run_chandra(chandra_bin, source, args.output_dir / "chandra")
        chandra_recognized_path = args.output_dir / "chandra" / source.stem / "recognized.txt"

        results["samples"][label] = {
            "source": display_path(source),
            "ground_truth_path": display_path(ground_truth_path),
            "engines": {
                "pytesseract": {
                    "recognized_text_path": display_path(pytesseract_path),
                    "metrics": score_text(ground_truth, pytesseract_text),
                },
                "chandra": {
                    "recognized_text_path": display_path(chandra_recognized_path),
                    "metrics": score_text(ground_truth, chandra_text),
                },
            },
        }

    results["overall"] = {
        "pytesseract": build_engine_summary(
            {
                label: sample["engines"]["pytesseract"]["metrics"]
                for label, sample in results["samples"].items()
            }
        ),
        "chandra": build_engine_summary(
            {
                label: sample["engines"]["chandra"]["metrics"]
                for label, sample in results["samples"].items()
            }
        ),
    }

    summary_path = args.output_dir / "comparison_summary.json"
    report_path = args.output_dir / "comparison_report.txt"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_human_report(results, report_path)
    sys.stdout.write(
        json.dumps(
            {
                "status": "ok",
                "summary_path": str(summary_path.relative_to(ROOT)),
                "report_path": str(report_path.relative_to(ROOT)),
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
