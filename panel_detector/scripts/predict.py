"""
Run the trained panel detector on a single screenshot.
Saves a cropped puzzle-panel image next to the input.

Usage:
    python scripts/predict.py path/to/screenshot.png
    python scripts/predict.py path/to/screenshot.png --weights runs/panel_detector/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "panel_detector" / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--margin", type=int, default=4,
                        help="Pixels of padding to add around the detected box.")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    results = model.predict(source=str(args.image), conf=args.conf, verbose=False)

    if not results or len(results[0].boxes) == 0:
        print("No puzzle panel detected.")
        return

    # Pick the highest-confidence box.
    boxes = results[0].boxes
    best = int(boxes.conf.argmax().item())
    x1, y1, x2, y2 = boxes.xyxy[best].tolist()

    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    m = args.margin
    crop = img.crop((
        max(0, int(x1) - m),
        max(0, int(y1) - m),
        min(W, int(x2) + m),
        min(H, int(y2) + m),
    ))
    out = args.image.with_name(args.image.stem + "_panel.png")
    crop.save(out)
    print(f"Saved crop: {out} (conf={boxes.conf[best].item():.3f})")


if __name__ == "__main__":
    main()
