"""Prepare a Label Studio project for the panel-detector retrain.

Generates:
  panel_detector/labelstudio/labeling_config.xml   (rectangle-only config)
  panel_detector/labelstudio/tasks.json            (259 tasks, 40 pre-annotated)

The 40 existing YOLO bbox labels under
`project-1-at-2026-04-27-21-27-9c5892cc/labels/` are matched to the raw
frames by filename suffix and converted to Label Studio annotations.

Tasks reference frames via Local Files Storage URIs of the form
    /data/local-files/?d=panel_detector/raw/<file>.png
which requires `start_label_studio.ps1` (writes the env vars) to launch
the server.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "panel_detector" / "raw"
EXISTING = ROOT / "project-1-at-2026-04-27-21-27-9c5892cc"
OUT = ROOT / "panel_detector" / "labelstudio"
OUT.mkdir(parents=True, exist_ok=True)

LABELING_CONFIG = """\
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"
         rotateControl="false"/>
  <RectangleLabels name="label" toName="image">
    <Label value="puzzle_panel" background="#00ffaa"/>
  </RectangleLabels>
</View>
"""

# label-studio Local Files prefix; must agree with start_label_studio.ps1.
LS_PREFIX = "/data/local-files/?d=panel_detector/raw/"

# Existing labels are YOLO format: "<class> <cx> <cy> <w> <h>" (normalized).
# Filenames look like "<hash>-s01_0001.txt" -> map to "s01_0001.png".
SUFFIX_RE = re.compile(r"-(s01_\d{4})\.txt$")


def load_existing() -> dict[str, list[tuple[float, float, float, float]]]:
    """Returns {raw_stem -> [(cx, cy, w, h), ...]} from existing 40 labels."""
    out: dict[str, list[tuple[float, float, float, float]]] = {}
    for p in sorted((EXISTING / "labels").glob("*.txt")):
        m = SUFFIX_RE.search(p.name)
        if not m:
            continue
        stem = m.group(1)
        boxes = []
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = parts
            boxes.append((float(cx), float(cy), float(w), float(h)))
        if boxes:
            out[stem] = boxes
    return out


def main() -> None:
    (OUT / "labeling_config.xml").write_text(LABELING_CONFIG, encoding="utf-8")

    existing = load_existing()
    print(f"matched existing labels: {len(existing)} frames")

    # Raw frames excluding the *_panel.png derivatives.
    raw_frames = sorted(p for p in RAW.glob("s01_*.png")
                        if not p.stem.endswith("_panel"))
    print(f"raw frames: {len(raw_frames)}")

    tasks = []
    pre_count = 0
    for f in raw_frames:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  [skip unreadable] {f.name}")
            continue
        ih, iw = img.shape[:2]
        task = {"data": {"image": f"{LS_PREFIX}{f.name}"}}
        if f.stem in existing:
            results = []
            for cx, cy, w, h in existing[f.stem]:
                # YOLO normalized -> LS percent.
                x_pct = (cx - w / 2) * 100.0
                y_pct = (cy - h / 2) * 100.0
                results.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": iw,
                    "original_height": ih,
                    "image_rotation": 0,
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "width": w * 100.0,
                        "height": h * 100.0,
                        "rotation": 0,
                        "rectanglelabels": ["puzzle_panel"],
                    },
                })
            task["annotations"] = [{"result": results}]
            pre_count += 1
        tasks.append(task)

    out_tasks = OUT / "tasks.json"
    out_tasks.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    print(f"wrote {out_tasks}  ({len(tasks)} tasks, {pre_count} pre-annotated)")
    print(f"wrote {OUT/'labeling_config.xml'}")


if __name__ == "__main__":
    main()
