"""
Train a YOLO detection model for the puzzle panel.  Single class:
`puzzle_panel`.  We previously trained a pose model (4 corner keypoints)
but switched to bbox-only because predict.py reconstructs the corners
from the bbox via fixed top/bottom edge slopes (more accurate than the
keypoint head on this dataset).

Usage:
    python scripts/train.py
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data.yaml"
RUNS_DIR = ROOT / "runs"

# Detect-nano: tiny, fast, plenty for a single class on a fixed UI element.
BASE_MODEL = "yolov8n.pt"

EPOCHS = 100
IMG_SIZE = 640
BATCH = 32
DEVICE = 0
WORKERS = 2
CACHE = "ram"


def main() -> None:
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        cache=CACHE,
        project=str(RUNS_DIR),
        name="panel_detector",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
