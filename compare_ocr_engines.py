#!/usr/bin/env python3
"""Run registered OCR engines against generated fixtures and compare results."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import pytesseract

from run_chandra_ocr import (
    INPUT_DIR,
    OUTPUT_DIR,
    ROOT,
    dicom_to_png,
    markdown_to_plain_text,
    require_chandra,
    run_chandra,
)


GROUND_TRUTH_FILENAMES = {
    "png": "fake_ocr_hostile_png_ground_truth.txt",
    "dicom": "fake_ocr_hostile_dicom_ground_truth.txt",
}
AVAILABLE_ENGINE_KEYS = ("pytesseract", "chandra", "lightonocr")


@dataclass(frozen=True)
class OcrSample:
    label: str
    source: Path
    ground_truth_path: Path
    ground_truth: str


@dataclass(frozen=True)
class OcrOutput:
    recognized_text: str
    recognized_text_path: Path
    artifacts: dict[str, str]


class OcrEngine(Protocol):
    key: str
    display_name: str

    def prepare(self) -> None:
        """Load binaries, clients, or model weights needed by the engine."""

    def recognize(self, sample: OcrSample, output_dir: Path) -> OcrOutput:
        """Run OCR for one sample and return normalized output locations."""


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


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def seconds_since(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 3)


class PyTesseractEngine:
    key = "pytesseract"
    display_name = "pytesseract"

    def __init__(self) -> None:
        self.tesseract_cmd: str | None = None

    def prepare(self) -> None:
        self.tesseract_cmd = require_tesseract()

    def recognize(self, sample: OcrSample, output_dir: Path) -> OcrOutput:
        if self.tesseract_cmd is None:
            raise RuntimeError("pytesseract engine was not prepared")

        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        recognized_text = pytesseract.image_to_string(str(sample.source), config="--psm 6")

        target_dir = output_dir / "comparison" / sample.label / self.key
        target_dir.mkdir(parents=True, exist_ok=True)
        recognized_path = target_dir / "recognized.txt"
        recognized_path.write_text(recognized_text, encoding="utf-8")
        return OcrOutput(
            recognized_text=recognized_text,
            recognized_text_path=recognized_path,
            artifacts={},
        )


class ChandraEngine:
    key = "chandra"
    display_name = "Chandra OCR"

    def __init__(self) -> None:
        self.chandra_bin: str | None = None

    def prepare(self) -> None:
        self.chandra_bin = require_chandra()

    def recognize(self, sample: OcrSample, output_dir: Path) -> OcrOutput:
        if self.chandra_bin is None:
            raise RuntimeError("Chandra OCR engine was not prepared")

        recognized_text = run_chandra(self.chandra_bin, sample.source, output_dir / self.key)
        recognized_path = output_dir / self.key / sample.source.stem / "recognized.txt"
        return OcrOutput(
            recognized_text=recognized_text,
            recognized_text_path=recognized_path,
            artifacts={
                "recognized_markdown_path": display_path(
                    output_dir / self.key / sample.source.stem / "recognized.md"
                ),
            },
        )


class LightOnOcrEngine:
    key = "lightonocr"
    display_name = "LightOnOCR"

    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device: str | None = None
        self.dtype: object | None = None
        self.model: object | None = None
        self.processor: object | None = None
        self.torch: object | None = None

    def prepare(self) -> None:
        try:
            import torch
            from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
        except ImportError as exc:
            raise SystemExit(
                "LightOnOCR requires optional dependencies. Install them with "
                "`python -m pip install torch transformers pillow`, then rerun with "
                "`--engines pytesseract,chandra,lightonocr`."
            ) from exc

        self.torch = torch
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.dtype = torch.float32 if self.device == "mps" else torch.bfloat16
        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=self.dtype,
        ).to(self.device)
        self.processor = LightOnOcrProcessor.from_pretrained(self.model_id)

    def recognize(self, sample: OcrSample, output_dir: Path) -> OcrOutput:
        if (
            self.torch is None
            or self.device is None
            or self.dtype is None
            or self.model is None
            or self.processor is None
        ):
            raise RuntimeError("LightOnOCR engine was not prepared")

        from PIL import Image

        target_dir = output_dir / "comparison" / sample.label / self.key
        target_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(sample.source) as image:
            rgb_image = image.convert("RGB")
            conversation = [{"role": "user", "content": [{"type": "image", "image": rgb_image}]}]
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        inputs = {
            key: value.to(device=self.device, dtype=self.dtype)
            if value.is_floating_point()
            else value.to(self.device)
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        recognized_markdown = self.processor.decode(generated_ids, skip_special_tokens=True)
        recognized_text = markdown_to_plain_text(recognized_markdown)

        markdown_path = target_dir / "recognized.md"
        recognized_path = target_dir / "recognized.txt"
        markdown_path.write_text(recognized_markdown, encoding="utf-8")
        recognized_path.write_text(recognized_text, encoding="utf-8")
        return OcrOutput(
            recognized_text=recognized_text,
            recognized_text_path=recognized_path,
            artifacts={
                "recognized_markdown_path": display_path(markdown_path),
                "model_id": self.model_id,
                "device": self.device,
                "max_new_tokens": str(self.max_new_tokens),
            },
        )


def build_engine_registry(args: argparse.Namespace) -> dict[str, OcrEngine]:
    return {
        "pytesseract": PyTesseractEngine(),
        "chandra": ChandraEngine(),
        "lightonocr": LightOnOcrEngine(args.lighton_model_id, args.lighton_max_new_tokens),
    }


def parse_engine_selection(value: str, registry: dict[str, OcrEngine]) -> list[str]:
    if value.strip().lower() == "all":
        return list(registry)

    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        raise SystemExit(
            "Unknown OCR engine(s): "
            f"{', '.join(unknown)}. Available engines: {', '.join(registry)}."
        )
    if not selected:
        raise SystemExit("Select at least one OCR engine.")
    return selected


def comparison_score(metrics: dict[str, object], elapsed_seconds: float) -> tuple[float, float, int, float]:
    return (
        float(metrics["char_accuracy_percent"]),
        float(metrics["word_accuracy_percent"]),
        -int(metrics["substitutions"]),
        -elapsed_seconds,
    )


def rank_sample_engines(engine_results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    scored: list[tuple[str, tuple[float, float, int, float]]] = []
    for engine_key, result in engine_results.items():
        if result["status"] != "ok":
            continue
        scored.append(
            (
                engine_key,
                comparison_score(
                    result["metrics"],
                    float(result["timing"]["ocr_seconds"]),
                ),
            )
        )

    scored.sort(key=lambda item: item[1], reverse=True)
    ranking: list[dict[str, object]] = []
    for index, (engine_key, _) in enumerate(scored, start=1):
        result = engine_results[engine_key]
        ranking.append(
            {
                "rank": index,
                "engine": engine_key,
                "display_name": result["display_name"],
                "char_accuracy_percent": result["metrics"]["char_accuracy_percent"],
                "word_accuracy_percent": result["metrics"]["word_accuracy_percent"],
                "substitutions": result["metrics"]["substitutions"],
                "ocr_seconds": result["timing"]["ocr_seconds"],
            }
        )
    return ranking


def build_engine_summary(sample_results: dict[str, dict[str, object]]) -> dict[str, object]:
    char_total = 0.0
    word_total = 0.0
    substitutions = 0
    elapsed_total = 0.0
    successes = 0
    failures = 0

    for result in sample_results.values():
        if result["status"] != "ok":
            failures += 1
            continue
        metrics = result["metrics"]
        char_total += float(metrics["char_accuracy_percent"])
        word_total += float(metrics["word_accuracy_percent"])
        substitutions += int(metrics["substitutions"])
        elapsed_total += float(result["timing"]["ocr_seconds"])
        successes += 1

    summary: dict[str, object] = {
        "samples_succeeded": successes,
        "samples_failed": failures,
        "total_substitutions": substitutions,
        "total_ocr_seconds": round(elapsed_total, 3),
    }
    if successes == 0:
        summary.update(
            {
                "average_char_accuracy_percent": None,
                "average_word_accuracy_percent": None,
                "average_ocr_seconds": None,
            }
        )
    else:
        summary.update(
            {
                "average_char_accuracy_percent": round(char_total / successes, 2),
                "average_word_accuracy_percent": round(word_total / successes, 2),
                "average_ocr_seconds": round(elapsed_total / successes, 3),
            }
        )
    return summary


def rank_overall_engines(overall: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    scored: list[tuple[str, tuple[float, float, int, float]]] = []
    for engine_key, summary in overall.items():
        if summary["samples_succeeded"] == 0:
            continue
        scored.append(
            (
                engine_key,
                (
                    float(summary["average_char_accuracy_percent"]),
                    float(summary["average_word_accuracy_percent"]),
                    -int(summary["total_substitutions"]),
                    -float(summary["average_ocr_seconds"]),
                ),
            )
        )

    scored.sort(key=lambda item: item[1], reverse=True)
    ranking: list[dict[str, object]] = []
    for index, (engine_key, _) in enumerate(scored, start=1):
        summary = overall[engine_key]
        ranking.append(
            {
                "rank": index,
                "engine": engine_key,
                "display_name": summary["display_name"],
                "average_char_accuracy_percent": summary["average_char_accuracy_percent"],
                "average_word_accuracy_percent": summary["average_word_accuracy_percent"],
                "total_substitutions": summary["total_substitutions"],
                "average_ocr_seconds": summary["average_ocr_seconds"],
                "total_ocr_seconds": summary["total_ocr_seconds"],
            }
        )
    return ranking


def run_engine_for_sample(
    engine: OcrEngine,
    sample: OcrSample,
    output_dir: Path,
) -> dict[str, object]:
    started_at = time.perf_counter()
    try:
        output = engine.recognize(sample, output_dir)
    except Exception as exc:  # noqa: BLE001 - per-engine failures should not hide other results.
        return {
            "display_name": engine.display_name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "timing": {"ocr_seconds": seconds_since(started_at)},
        }

    elapsed_seconds = seconds_since(started_at)
    return {
        "display_name": engine.display_name,
        "status": "ok",
        "recognized_text_path": display_path(output.recognized_text_path),
        "artifacts": output.artifacts,
        "timing": {"ocr_seconds": elapsed_seconds},
        "metrics": score_text(sample.ground_truth, output.recognized_text),
    }


def write_human_report(results: dict[str, object], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("OCR comparison report")
    lines.append("")

    for label, sample in results["samples"].items():
        lines.append(f"{label.upper()} sample")
        for ranking in sample["ranking"]:
            lines.append(
                f"{ranking['rank']}. {ranking['display_name']} ({ranking['engine']}): "
                f"{ranking['char_accuracy_percent']}% char accuracy, "
                f"{ranking['word_accuracy_percent']}% word accuracy, "
                f"{ranking['substitutions']} substitutions, "
                f"{ranking['ocr_seconds']}s OCR time."
            )

        failed = [
            (engine_key, result)
            for engine_key, result in sample["engines"].items()
            if result["status"] != "ok"
        ]
        for engine_key, result in failed:
            lines.append(
                f"{result['display_name']} ({engine_key}) failed after "
                f"{result['timing']['ocr_seconds']}s: {result['error']}"
            )

        if sample["ranking"]:
            winner = sample["ranking"][0]
            lines.append(
                f"Judgment: {winner['display_name']} ranked first for this sample by "
                "character accuracy, then word accuracy, then fewer substitutions, then speed."
            )
        else:
            lines.append("Judgment: no selected engine completed this sample.")
        lines.append("")

    lines.append("Overall")
    for ranking in results["overall_ranking"]:
        lines.append(
            f"{ranking['rank']}. {ranking['display_name']} ({ranking['engine']}): "
            f"{ranking['average_char_accuracy_percent']}% average char accuracy, "
            f"{ranking['average_word_accuracy_percent']}% average word accuracy, "
            f"{ranking['total_substitutions']} total substitutions, "
            f"{ranking['average_ocr_seconds']}s average OCR time."
        )

    failed_overall = [
        (engine_key, summary)
        for engine_key, summary in results["overall"].items()
        if summary["samples_failed"]
    ]
    for engine_key, summary in failed_overall:
        lines.append(
            f"{summary['display_name']} ({engine_key}) failed on "
            f"{summary['samples_failed']} sample(s)."
        )

    if results["overall_ranking"]:
        winner = results["overall_ranking"][0]
        lines.append(
            f"Final judgment: {winner['display_name']} ranked first overall using the same "
            "accuracy, substitution, and speed ordering."
        )
    else:
        lines.append("Final judgment: no selected engine completed enough work to compare.")

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
    parser.add_argument(
        "--engines",
        default="pytesseract,chandra",
        help=(
            "Comma-separated OCR engine keys to run, or `all`. "
            f"Available: {', '.join(AVAILABLE_ENGINE_KEYS)}."
        ),
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="Print available OCR engine keys and exit.",
    )
    parser.add_argument(
        "--lighton-model-id",
        default="lightonai/LightOnOCR-1B-1025",
        help="Hugging Face model id used by the LightOnOCR engine.",
    )
    parser.add_argument(
        "--lighton-max-new-tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens for each LightOnOCR sample.",
    )
    return parser.parse_args()


def load_samples(input_dir: Path, output_dir: Path) -> dict[str, OcrSample]:
    png_input = input_dir / "fake_ocr_hostile.png"
    dicom_input = input_dir / "fake_ocr_hostile.dcm"
    if not png_input.exists() or not dicom_input.exists():
        raise SystemExit("Input images are missing. Run generate_fake_images.py first.")

    converted_dir = output_dir / "converted_inputs"
    converted_dir.mkdir(parents=True, exist_ok=True)
    dicom_png = dicom_to_png(dicom_input, converted_dir)

    sources = {
        "png": png_input,
        "dicom": dicom_png,
    }
    samples: dict[str, OcrSample] = {}
    for label, source in sources.items():
        ground_truth_path = input_dir / GROUND_TRUTH_FILENAMES[label]
        samples[label] = OcrSample(
            label=label,
            source=source,
            ground_truth_path=ground_truth_path,
            ground_truth=ground_truth_path.read_text(encoding="utf-8"),
        )
    return samples


def main() -> None:
    args = parse_args()
    registry = build_engine_registry(args)

    if args.list_engines:
        sys.stdout.write(json.dumps({"engines": list(registry)}, indent=2) + "\n")
        return

    selected_engine_keys = parse_engine_selection(args.engines, registry)
    selected_engines = [registry[key] for key in selected_engine_keys]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.input_dir, args.output_dir)
    results: dict[str, object] = {
        "engine_order": selected_engine_keys,
        "engines": {},
        "samples": {},
        "overall": {},
        "overall_ranking": [],
    }

    for engine in selected_engines:
        started_at = time.perf_counter()
        engine.prepare()
        results["engines"][engine.key] = {
            "display_name": engine.display_name,
            "setup_seconds": seconds_since(started_at),
        }

    for label, sample in samples.items():
        sample_results: dict[str, object] = {
            "source": display_path(sample.source),
            "ground_truth_path": display_path(sample.ground_truth_path),
            "engines": {},
            "ranking": [],
        }
        for engine in selected_engines:
            sample_results["engines"][engine.key] = run_engine_for_sample(
                engine,
                sample,
                args.output_dir,
            )
        sample_results["ranking"] = rank_sample_engines(sample_results["engines"])
        results["samples"][label] = sample_results

    for engine in selected_engines:
        per_sample = {
            label: sample["engines"][engine.key]
            for label, sample in results["samples"].items()
        }
        summary = build_engine_summary(per_sample)
        summary["display_name"] = engine.display_name
        summary["setup_seconds"] = results["engines"][engine.key]["setup_seconds"]
        results["overall"][engine.key] = summary

    results["overall_ranking"] = rank_overall_engines(results["overall"])

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
