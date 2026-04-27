# Trick or Text

OCR stress demo for Chandra OCR 2, pytesseract, and optional model adapters.

This repo creates two synthetic OCR-hostile images on Linux:

- a PNG
- a fake DICOM image

The text is intentionally packed with ambiguous glyphs such as `I`, `l`, `1`, `O`, `0`, `rn`, punctuation, and symbols so OCR quality is easier to inspect.

## Files

- `generate_fake_images.py`: builds the PNG and DICOM fixtures under `artifacts/inputs/`
- `run_chandra_ocr.py`: converts the DICOM into a PNG for OCR, runs Chandra OCR 2 with the HuggingFace backend, and writes what Chandra recognized
- `compare_ocr_engines.py`: runs selected OCR engines, scores each against ground truth, records OCR timing, and writes machine-readable plus human-readable comparisons
- `install_and_run.sh`: creates `.venv`, installs dependencies, generates the files, and runs the comparison pipeline

## Usage

Run the whole pipeline on Rocky Linux, RHEL, or Ubuntu:

```bash
./install_and_run.sh
```

Run a specific engine set:

```bash
OCR_ENGINES=pytesseract,chandra ./install_and_run.sh
```

List available engine keys:

```bash
python compare_ocr_engines.py --list-engines
```

Run the optional LightOnOCR adapter:

```bash
INSTALL_LIGHTONOCR=1 OCR_ENGINES=pytesseract,chandra,lightonocr ./install_and_run.sh
```

You can also run it directly after installing optional dependencies:

```bash
python -m pip install torch transformers pillow
python compare_ocr_engines.py --engines pytesseract,chandra,lightonocr
```

To add another model or OCR library, add a class in `compare_ocr_engines.py` with:

- `key`: stable CLI/output identifier
- `display_name`: readable report name
- `prepare()`: one-time setup such as loading binaries, API clients, or model weights
- `recognize(sample, output_dir)`: per-sample OCR call that returns `OcrOutput`

Then register it in `build_engine_registry()`. The comparison pipeline automatically handles timing, metrics, summaries, and rankings for registered engines.

## Output

After the run, check:

- `artifacts/inputs/` for the generated `fake_ocr_hostile.png` and `fake_ocr_hostile.dcm`
- `artifacts/ocr/chandra/*/recognized.txt` for the OCR text Chandra produced
- `artifacts/ocr/comparison/<sample>/<engine>/recognized.txt` for comparison outputs written by engine adapters
- `artifacts/ocr/comparison_summary.json` for machine-readable per-engine timing, text metrics, summaries, failures, and rankings
- `artifacts/ocr/comparison_report.txt` for the human-readable comparison and judgment

## Notes

- The DICOM file is synthetic and only meant to carry image pixels for this demo.
- The OCR step uses `chandra --method hf`, so the first run may download model weights and can take a while.
- The LightOnOCR adapter uses the Hugging Face `lightonai/LightOnOCR-1B-1025` model by default and may download large model weights on first use.
- The scripts assume `python3`, `venv`, a working `pip`, and the `tesseract` system binary are available on the Linux host.
