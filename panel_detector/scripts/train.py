"""
Train a YOLO pose model that detects the puzzle panel and its 4 corner
keypoints (TL, TR, BR, BL).  Single class: `puzzle_panel`.

Usage:
    python scripts/train.py
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data.yaml"
RUNS_DIR = ROOT / "runs"

# Pose-nano: cheap, perfect for a single object with 4 keypoints.
BASE_MODEL = "yolov8n-pose.pt"

EPOCHS = 100
IMG_SIZE = 640
BATCH = 16


def main() -> None:
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(RUNS_DIR),
        name="panel_detector",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
