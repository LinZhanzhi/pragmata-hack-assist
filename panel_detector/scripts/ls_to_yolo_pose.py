"""
Convert a Label Studio JSON-MIN export to YOLO pose label files.

Each output `.txt` (one per image) contains a single line:

    <class_id> <cx> <cy> <w> <h>
        <tl_x> <tl_y> <v>
        <tr_x> <tr_y> <v>
        <br_x> <br_y> <v>
        <bl_x> <bl_y> <v>

All coords normalized to [0, 1].  Visibility flag is 2 (labeled & visible).

Image filenames imported into Label Studio carry an 8-hex-character UUID
prefix like `bdfb308a-s01_0009.png`; the converter strips that prefix so
the output `.txt` matches our canonical `s01_####.png` names in raw/.

Usage:
    python scripts/ls_to_yolo_pose.py path/to/project-export.json
    python scripts/ls_to_yolo_pose.py path/to/export.json --out raw
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"

CLASS_ID = 0
KP_ORDER = ["TL", "TR", "BR", "BL"]
UUID_PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")


def strip_uuid(name: str) -> str:
    return UUID_PREFIX.sub("", name)


def image_basename(item: dict[str, Any]) -> str:
    """Best-effort extraction of the original filename from an LS item."""
    img = item.get("image") or item.get("ocr") or ""
    name = Path(img).name
    return strip_uuid(name)


def first_list(item: dict[str, Any], *keys: str) -> list:
    for k in keys:
        v = item.get(k)
        if isinstance(v, list) and v:
            return v
    return []


def convert_item(item: dict[str, Any]) -> tuple[str, str] | None:
    """Return (image_basename, label_line) or None if item is unusable."""
    name = image_basename(item)
    if not name:
        return None

    bbox_list = first_list(item, "bbox", "label", "labels")
    kp_list = first_list(item, "kp", "keypoints", "keypoint")

    if not bbox_list or not kp_list:
        print(f"  skipping {name}: missing bbox or keypoints")
        return None

    # Pick the first puzzle_panel rectangle.
    bbox = None
    for b in bbox_list:
        labels = b.get("rectanglelabels") or b.get("labels") or []
        if not labels or "puzzle_panel" in labels:
            bbox = b
            break
    if bbox is None:
        print(f"  skipping {name}: no puzzle_panel rectangle")
        return None

    # x/y/width/height come in percent (0-100) of original image size.
    x = float(bbox["x"]) / 100.0
    y = float(bbox["y"]) / 100.0
    w = float(bbox["width"]) / 100.0
    h = float(bbox["height"]) / 100.0
    cx = x + w / 2.0
    cy = y + h / 2.0

    # Map keypoints by label.
    kp_by_label: dict[str, tuple[float, float]] = {}
    for k in kp_list:
        labels = k.get("keypointlabels") or k.get("labels") or []
        if not labels:
            continue
        lbl = labels[0]
        kx = float(k["x"]) / 100.0
        ky = float(k["y"]) / 100.0
        kp_by_label[lbl] = (kx, ky)

    missing = [lbl for lbl in KP_ORDER if lbl not in kp_by_label]
    if missing:
        print(f"  skipping {name}: missing keypoints {missing}")
        return None

    parts = [str(CLASS_ID), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for lbl in KP_ORDER:
        kx, ky = kp_by_label[lbl]
        parts.extend([f"{kx:.6f}", f"{ky:.6f}", "2"])
    return name, " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path,
                        help="Path to the Label Studio JSON-MIN file.")
    parser.add_argument("--out", type=Path, default=RAW,
                        help="Directory to write .txt label files into "
                             "(defaults to panel_detector/raw).")
    parser.add_argument("--require-image", action="store_true",
                        help="Only write a label if the matching image "
                             "exists in --out.")
    args = parser.parse_args()

    data = json.loads(args.export.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("Expected a JSON array at the top level.")
        return

    args.out.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for item in data:
        result = convert_item(item)
        if result is None:
            skipped += 1
            continue
        name, line = result
        img_path = args.out / name
        if args.require_image and not img_path.exists():
            print(f"  skipping {name}: image not found in {args.out}")
            skipped += 1
            continue
        label_path = args.out / (Path(name).stem + ".txt")
        label_path.write_text(line + "\n", encoding="utf-8")
        written += 1

    print(f"Wrote {written} label file(s) to {args.out}")
    if skipped:
        print(f"Skipped {skipped} item(s).")


if __name__ == "__main__":
    main()
