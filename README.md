# Trick or Text 🎃 🦇

OCR Stress Demo for Chandra OCR 2 and pytesseract

This repo creates two synthetic OCR-hostile images on Linux:

- a PNG
- a fake DICOM image

The text is intentionally packed with ambiguous glyphs such as `I`, `l`, `1`, `O`, `0`, `rn`, punctuation, and symbols so OCR quality is easier to inspect.

## Files

- `generate_fake_images.py`: builds the PNG and DICOM fixtures under `artifacts/inputs/`
- `run_chandra_ocr.py`: converts the DICOM into a PNG for OCR, runs Chandra OCR 2 with the HuggingFace backend, and writes what Chandra recognized
- `compare_ocr_engines.py`: runs both Chandra OCR and pytesseract, scores both against ground truth, and writes a readable comparison
- `install_and_run.sh`: creates `.venv`, installs dependencies, generates the files, and runs the comparison pipeline

## Usage

Run the whole pipeline on Rocky Linux, RHEL, or Ubuntu:

```bash
./install_and_run.sh
```

## Output

After the run, check:

- `artifacts/inputs/` for the generated `fake_ocr_hostile.png` and `fake_ocr_hostile.dcm`
- `artifacts/ocr/chandra/*/recognized.txt` for the OCR text Chandra produced
- `artifacts/ocr/comparison/*/pytesseract_recognized.txt` for the OCR text pytesseract produced
- `artifacts/ocr/comparison_summary.json` for machine-readable metrics for both engines
- `artifacts/ocr/comparison_report.txt` for the human-readable comparison and judgment

## Notes

- The DICOM file is synthetic and only meant to carry image pixels for this demo.
- The OCR step uses `chandra --method hf`, so the first run may download model weights and can take a while.
- The scripts assume `python3`, `venv`, a working `pip`, and the `tesseract` system binary are available on the Linux host.

<br>
