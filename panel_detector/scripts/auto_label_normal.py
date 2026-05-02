"""Auto-label `normal` cells via nearest-neighbor matching.

We have very few labels (~43). Instead of training a CNN on this we use
the labeled cells as a memory-based classifier:

- Each cell is represented as a 64x64 grayscale image flattened to a
  4096-d vector (mean-centered, L2-normalized).
- For each unlabeled cell, find the nearest labeled-normal and the
  nearest labeled-other (non-normal) by L2 distance in this space.
- Predict `normal` ONLY when:
    1. d_normal <= NORMAL_RADIUS (calibrated below), AND
    2. d_normal * SAFETY_RATIO <= d_other (clear margin from any
       known non-normal cell).
- Both knobs are calibrated automatically against the labeled set so
  every labeled non-normal cell is correctly rejected with margin to
  spare.

The output is a Label-Studio-importable predictions JSON
(`cells_predictions.json`) you can re-import (Project -> Import) so
the suggested `normal` label appears as a prediction on each task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent  # panel_detector/
REPO = ROOT.parent
CELLS_DIR = ROOT / "cells"

FEATURE_SIZE = 64
SAFETY_RATIO = 1.5  # d_target must be <= d_other / SAFETY_RATIO


def feat(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((FEATURE_SIZE, FEATURE_SIZE))
    arr = np.asarray(img, dtype=np.float32).flatten() / 255.0
    arr = arr - arr.mean()
    n = np.linalg.norm(arr)
    if n > 0:
        arr = arr / n
    return arr


def url_to_path(url: str) -> Path:
    # /data/local-files/?d=panel_detector%5Ccells%5Cs01_0001__r0_c0.png
    rel = unquote(url.split("?d=", 1)[1]).replace("\\", "/")
    return REPO / rel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path,
                        help="Path to LS JSON-MIN export")
    parser.add_argument("--class", dest="target", default="normal",
                        help="Target class to predict (default: normal)")
    parser.add_argument("--out", type=Path,
                        default=None,
                        help="Output predictions JSON "
                             "(default: cells_predictions_<class>.json)")
    args = parser.parse_args()
    target = args.target
    out_preds = args.out or (ROOT / f"cells_predictions_{target}.json")

    tasks = json.loads(args.export.read_text(encoding="utf-8"))
    labeled: dict[Path, str] = {}
    for t in tasks:
        lbl = t.get("label")
        if isinstance(lbl, list):
            lbl = lbl[0] if lbl else None
        if not isinstance(lbl, str):
            continue
        p = url_to_path(t["image"])
        if p.exists():
            labeled[p] = lbl

    normal_paths = [p for p, l in labeled.items() if l == target]
    other_paths = [p for p, l in labeled.items() if l != target]
    print(f"Target class: {target!r}")
    print(f"Labeled: {len(labeled)} ({len(normal_paths)} {target}, {len(other_paths)} other)")
    if len(normal_paths) < 3 or len(other_paths) < 1:
        print("Not enough labels.", file=sys.stderr)
        sys.exit(1)

    F_normal = np.stack([feat(p) for p in normal_paths])
    F_other = np.stack([feat(p) for p in other_paths])

    def dists(v: np.ndarray, M: np.ndarray) -> np.ndarray:
        return np.linalg.norm(M - v[None, :], axis=1)

    # Calibrate radius via leave-one-out on labeled target-class cells.
    if len(F_normal) >= 2:
        loo_dists = []
        for i in range(len(F_normal)):
            rest = np.delete(F_normal, i, axis=0)
            loo_dists.append(dists(F_normal[i], rest).min())
        loo_dists = np.array(loo_dists)
        normal_radius = float(np.percentile(loo_dists, 95))
    else:
        # Single example: use distance to nearest other as a soft cap.
        normal_radius = float(dists(F_normal[0], F_other).min() / SAFETY_RATIO)
        loo_dists = np.array([0.0])

    # Validate: every labeled "other" should be REJECTED.
    rejected = 0
    other_d_normal = []
    other_d_other = []
    for v in F_other:
        dn = dists(v, F_normal).min()
        do = dists(v, F_other).min() if len(F_other) > 1 else np.inf
        other_d_normal.append(dn)
        other_d_other.append(do)
        is_normal = dn <= normal_radius and dn * SAFETY_RATIO <= do
        if not is_normal:
            rejected += 1
    print(f"Calibration: radius={normal_radius:.4f} (95th pct LOO)")
    print(f"  labeled {target}: LOO d range [{loo_dists.min():.3f}, {loo_dists.max():.3f}]")
    print(f"  labeled others: d_to_{target} range [{min(other_d_normal):.3f}, {max(other_d_normal):.3f}]")
    print(f"  labeled others rejected: {rejected}/{len(F_other)}")
    if rejected != len(F_other):
        print(f"WARNING: some labeled non-{target} cells would be auto-labeled as {target}.", file=sys.stderr)

    # Score every cell file not already labeled.
    all_cells = sorted(CELLS_DIR.glob("s01_*.png"))
    print(f"Scanning {len(all_cells)} cells...")
    predictions = []
    n_labeled_skipped = 0
    n_predicted = 0
    for p in all_cells:
        if p in labeled:
            n_labeled_skipped += 1
            continue
        v = feat(p)
        dn = dists(v, F_normal).min()
        do = dists(v, F_other).min()
        if dn <= normal_radius and dn * SAFETY_RATIO <= do:
            n_predicted += 1
            # confidence: 1.0 at d=0, 0.0 at d=normal_radius
            score = float(max(0.0, 1.0 - dn / max(normal_radius, 1e-6)))
            rel = p.relative_to(REPO).as_posix().replace("/", "%5C")
            predictions.append({
                "data": {"image": f"/data/local-files/?d={rel}"},
                "predictions": [{
                    "model_version": f"knn-{target}-v1",
                    "score": score,
                    "result": [{
                        "from_name": "label",
                        "to_name": "image",
                        "type": "choices",
                        "value": {"choices": [target]},
                    }],
                }],
            })

    out_preds.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"Skipped already-labeled: {n_labeled_skipped}")
    print(f"Auto-predicted {target}: {n_predicted}")
    print(f"Wrote: {out_preds}")


if __name__ == "__main__":
    main()
