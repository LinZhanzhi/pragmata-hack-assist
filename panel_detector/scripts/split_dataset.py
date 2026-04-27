"""
Split annotated images + YOLO labels into train/val/test sets.

Default behavior: split by session id from filename prefix
    s07_0042  ->  session "s07"

Sessions are routed via TRAIN_SESSIONS / VAL_SESSIONS / TEST_SESSIONS.

Temporary override (for v1 with only one session):
    FRAME_OVERRIDES maps a session -> dict of split -> set of frame numbers.
    Frames listed there go to that split regardless of the session sets.
    Anything else from that session falls back to the session-set rules.

Run:  python scripts/split_dataset.py
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

# Temporary: with only s01 captured, hand-pick val/test frames so the
# trainer has something to validate against.  Remove this block once
# real s07/s08 sessions exist.
FRAME_OVERRIDES: dict[str, dict[str, set[int]]] = {
    "s01": {
        "val": {5, 12, 19, 26, 33},   # 5 frames spread across the session
        "test": {9, 22, 36},          # 3 held-out frames
    },
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def parse_stem(stem: str) -> tuple[str, int | None]:
    """`s07_0042` -> ('s07', 42).  Returns (session, frame_or_None)."""
    parts = stem.split("_", 1)
    session = parts[0]
    frame: int | None = None
    if len(parts) == 2:
        try:
            frame = int(parts[1])
        except ValueError:
            frame = None
    return session, frame


def target_split(session: str, frame: int | None) -> str | None:
    overrides = FRAME_OVERRIDES.get(session)
    if overrides and frame is not None:
        for split, frames in overrides.items():
            if frame in frames:
                return split
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
        session, frame = parse_stem(img.stem)
        split = target_split(session, frame)
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
