"""
Split annotated images + YOLO labels into train/val/test sets by session id.

Expected input:
    panel_detector/raw/<session>_<frame>.png
    panel_detector/raw/<session>_<frame>.txt   (YOLO label, same basename)

Edit TRAIN_SESSIONS / VAL_SESSIONS / TEST_SESSIONS below to match your data.
Run from anywhere:  python scripts/split_dataset.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
DATASET = ROOT / "dataset"

TRAIN_SESSIONS = {"s01", "s02", "s03", "s04", "s05", "s06"}
VAL_SESSIONS = {"s07"}
TEST_SESSIONS = {"s08"}

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def session_of(stem: str) -> str:
    # Filenames look like "s07_0042" -> session "s07".
    return stem.split("_", 1)[0]


def target_split(session: str) -> str | None:
    if session in TRAIN_SESSIONS:
        return "train"
    if session in VAL_SESSIONS:
        return "val"
    if session in TEST_SESSIONS:
        return "test"
    return None


def main() -> None:
    for split in ("train", "val", "test"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    moved = {"train": 0, "val": 0, "test": 0}
    skipped: list[str] = []

    for img in RAW.iterdir():
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = img.with_suffix(".txt")
        if not label.exists():
            skipped.append(f"no label: {img.name}")
            continue
        split = target_split(session_of(img.stem))
        if split is None:
            skipped.append(f"unknown session: {img.name}")
            continue
        shutil.copy2(img, DATASET / "images" / split / img.name)
        shutil.copy2(label, DATASET / "labels" / split / label.name)
        moved[split] += 1

    print("Copied:", moved)
    if skipped:
        print(f"Skipped {len(skipped)} files (showing up to 10):")
        for s in skipped[:10]:
            print(" ", s)


if __name__ == "__main__":
    main()
