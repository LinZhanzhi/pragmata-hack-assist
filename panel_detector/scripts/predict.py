"""
Run the trained panel detector + keypoint model on a single screenshot.
Reads the 4 corner keypoints (TL, TR, BR, BL) and warps the trapezoidal
panel to a clean rectangle saved as <input>_panel.png.

Usage:
    python scripts/predict.py path/to/screenshot.png
    python scripts/predict.py path/to/screenshot.png --weights runs/panel_detector/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "panel_detector" / "weights" / "best.pt"

# Output canonical size of the warped panel.  Tweak as you like; this is
# the working canvas for downstream grid-line analysis and cell slicing.
OUT_W = 800
OUT_H = 800


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out-size", type=int, nargs=2, default=(OUT_W, OUT_H),
                        metavar=("W", "H"),
                        help="Output canvas size after perspective warp.")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    results = model.predict(source=str(args.image), conf=args.conf, verbose=False)

    if not results or len(results[0].boxes) == 0:
        print("No puzzle panel detected.")
        return

    res = results[0]
    boxes = res.boxes
    best = int(boxes.conf.argmax().item())

    if res.keypoints is None or len(res.keypoints.xy) <= best:
        print("Model did not return keypoints; check the trained weights.")
        return

    # Keypoints come back as a (num_objects, 4, 2) tensor in image pixels.
    kpts = res.keypoints.xy[best].cpu().numpy().astype(np.float32)
    if kpts.shape != (4, 2):
        print(f"Unexpected keypoint shape: {kpts.shape}")
        return

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Could not read image: {args.image}")
        return

    out_w, out_h = args.out_size
    dst = np.array([
        [0, 0],            # TL
        [out_w - 1, 0],    # TR
        [out_w - 1, out_h - 1],  # BR
        [0, out_h - 1],    # BL
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(kpts, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))

    out = args.image.with_name(args.image.stem + "_panel.png")
    cv2.imwrite(str(out), warped)
    print(f"Saved warped panel: {out} (conf={boxes.conf[best].item():.3f})")
    print(f"Corners (TL, TR, BR, BL): {kpts.tolist()}")


if __name__ == "__main__":
    main()
