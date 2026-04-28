"""
Batch-warp every raw frame to <stem>_panel.png. Loads YOLO once.

Usage:
    python panel_detector/scripts/batch_warp.py panel_detector/raw
    python panel_detector/scripts/batch_warp.py panel_detector/raw --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from predict import corners_from_bbox, OUT_W, OUT_H, DEFAULT_WEIGHTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out-size", type=int, nargs=2, default=(OUT_W, OUT_H))
    ap.add_argument("--force", action="store_true",
                    help="Re-warp even if <stem>_panel.png exists.")
    args = ap.parse_args()

    out_w, out_h = args.out_size
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    candidates = sorted(p for p in args.folder.glob("*.png")
                        if not p.stem.endswith("_panel"))
    todo = []
    for p in candidates:
        out = p.with_name(p.stem + "_panel.png")
        if out.exists() and not args.force:
            continue
        todo.append((p, out))

    print(f"{len(todo)} frame(s) to warp (skipping {len(candidates) - len(todo)} already done)")
    if not todo:
        return

    model = YOLO(str(args.weights))
    fail = []
    for img_path, out_path in todo:
        results = model.predict(source=str(img_path), conf=args.conf, verbose=False)
        if not results or len(results[0].boxes) == 0:
            print(f"[no detection] {img_path.name}")
            fail.append(img_path.name)
            continue
        boxes = results[0].boxes
        best = int(boxes.conf.argmax().item())
        xyxy = boxes.xyxy[best].cpu().numpy()
        kpts = corners_from_bbox(xyxy)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[bad read] {img_path.name}")
            fail.append(img_path.name)
            continue
        M = cv2.getPerspectiveTransform(kpts, dst)
        warped = cv2.warpPerspective(img, M, (out_w, out_h))
        cv2.imwrite(str(out_path), warped)
        print(f"OK   {img_path.name} -> {out_path.name} (conf={boxes.conf[best].item():.3f})")

    if fail:
        print(f"\n{len(fail)} failure(s):")
        for n in fail:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
