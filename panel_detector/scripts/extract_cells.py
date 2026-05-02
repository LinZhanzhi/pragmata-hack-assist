"""Slice every warped panel into individual cell images.

For each `<stem>_panel.png` in panel_detector/raw, infer the grid size
via dest_detector.infer_grid_from_dest and write one PNG per cell to
panel_detector/cells/<stem>__r<r>_c<c>.png.

Panels where the grid cannot be inferred (no green destination icon,
mid-animation frames, etc.) are skipped and logged.

A small CSV (panel_detector/cells/cells_meta.csv) records the grid
shape per panel so the labeler can verify nothing went wrong.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from dest_detector import infer_grid_from_dest  # noqa: E402

RAW_DIR = ROOT / "raw"
CELLS_DIR = ROOT / "cells"
META_CSV = CELLS_DIR / "cells_meta.csv"


def main() -> None:
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    panels = sorted(RAW_DIR.glob("*_panel.png"))
    if not panels:
        print(f"No *_panel.png found in {RAW_DIR}")
        return

    rows = []
    skipped: list[tuple[str, str]] = []
    n_cells_total = 0

    for panel in panels:
        stem = panel.stem.removesuffix("_panel")
        img = cv2.imread(str(panel))
        if img is None:
            skipped.append((panel.name, "imread failed"))
            continue

        info = infer_grid_from_dest(img)
        if info is None:
            skipped.append((panel.name, "no grid inferred"))
            continue

        H, W = img.shape[:2]
        nr, nc = info["n_rows"], info["n_cols"]
        # Use integer pixel boundaries from a linspace so rounding errors
        # don't drop a column on the right edge.
        xs = [round(c * W / nc) for c in range(nc + 1)]
        ys = [round(r * H / nr) for r in range(nr + 1)]

        for r in range(nr):
            for c in range(nc):
                cell = img[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
                out = CELLS_DIR / f"{stem}__r{r}_c{c}.png"
                cv2.imwrite(str(out), cell)
                n_cells_total += 1

        rows.append({
            "stem": stem,
            "n_rows": nr,
            "n_cols": nc,
            "dest_row": info["dest_row"],
            "dest_col": info["dest_col"],
        })

    with META_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "n_rows", "n_cols", "dest_row", "dest_col"])
        w.writeheader()
        w.writerows(rows)

    print(f"Panels processed: {len(rows)} / {len(panels)}")
    print(f"Cells written:    {n_cells_total} -> {CELLS_DIR}")
    if skipped:
        print(f"Skipped {len(skipped)} panel(s):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
