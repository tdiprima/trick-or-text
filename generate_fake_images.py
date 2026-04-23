#!/usr/bin/env python3
"""Generate deliberately OCR-hostile PNG and DICOM test images."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "inputs"
SEED = 1337
IMAGE_SIZE = (1400, 900)
BACKGROUND = 245
FOREGROUND = 28


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def hostile_lines() -> list[str]:
    return [
        "VVl1|Il! []{} () <> //\\\\ :: ;;",
        "rn m nn mm nm mn vv vw wu uv",
        "0OQDG 8B3 S5Z2 ?7/ T+ Y= Xx %",
        "lI1| !i ;: ., `~ ^* _- -- __",
        "#$&@ §¶ ±×÷ ~= != <= >= -> <-",
        "A4H? 7T7? M/WN/VV 1lI0OQ 5S$",
    ]


def make_base_image(lines: list[str], rng: random.Random) -> Image.Image:
    image = Image.new("L", IMAGE_SIZE, color=BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = load_font(40)
    body_font = load_font(34)

    draw.text((70, 60), "Synthetic OCR Stress Sample", fill=FOREGROUND, font=title_font)

    y = 150
    for line in lines:
        jitter = rng.randint(-8, 8)
        draw.text((90 + jitter, y), line, fill=FOREGROUND, font=body_font)
        y += 95

    for _ in range(26):
        x1 = rng.randint(0, IMAGE_SIZE[0] - 1)
        y1 = rng.randint(0, IMAGE_SIZE[1] - 1)
        x2 = x1 + rng.randint(-220, 220)
        y2 = y1 + rng.randint(-40, 40)
        shade = rng.randint(160, 225)
        draw.line((x1, y1, x2, y2), fill=shade, width=rng.randint(1, 3))

    for _ in range(750):
        x = rng.randint(0, IMAGE_SIZE[0] - 1)
        y = rng.randint(0, IMAGE_SIZE[1] - 1)
        shade = rng.randint(130, 255)
        image.putpixel((x, y), shade)

    image = image.rotate(rng.uniform(-2.2, 2.2), expand=False, fillcolor=BACKGROUND)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.9))

    arr = np.asarray(image, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, arr.shape[1], dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, arr.shape[0], dtype=np.float32)
    vignette = 1.0 - 0.09 * (xs[None, :] ** 2 + ys[:, None] ** 2)
    wave = 5.0 * np.sin(np.linspace(0, math.tau * 2, arr.shape[1], dtype=np.float32))
    arr = np.clip(arr * vignette + wave[None, :], 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def save_dicom(image: Image.Image, path: Path) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientID = "OCRPHANTOM"
    ds.PatientName = "OCR^Stress"
    ds.Modality = "OT"
    ds.SeriesDescription = "Synthetic OCR stress image"
    now = datetime.now(UTC)
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")

    pixels = np.asarray(image, dtype=np.uint8)
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelData = pixels.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)


def write_ground_truth(lines: list[str], path: Path) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(SEED)
    lines = hostile_lines()
    png_image = make_base_image(lines, rng)
    dicom_lines = list(reversed(lines))
    dicom_image = make_base_image(dicom_lines, rng)

    save_png(png_image, OUTPUT_DIR / "fake_ocr_hostile.png")
    save_dicom(dicom_image, OUTPUT_DIR / "fake_ocr_hostile.dcm")
    write_ground_truth(lines, OUTPUT_DIR / "fake_ocr_hostile_png_ground_truth.txt")
    write_ground_truth(dicom_lines, OUTPUT_DIR / "fake_ocr_hostile_dicom_ground_truth.txt")


if __name__ == "__main__":
    main()
