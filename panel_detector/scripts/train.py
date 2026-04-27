"""
Train a YOLO detector for the single class `puzzle_panel`.

Usage:
    python scripts/train.py
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data.yaml"
RUNS_DIR = ROOT / "runs"

# Nano model is plenty for one large UI rectangle.
BASE_MODEL = "yolov8n.pt"

EPOCHS = 80
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
