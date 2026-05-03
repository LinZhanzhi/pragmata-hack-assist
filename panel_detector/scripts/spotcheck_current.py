"""Run current_detector on every warped panel and dump debug overlays.

Outputs:
    panel_detector/runs/current_check/<stem>.png   (panel + box + cell)
    panel_detector/runs/current_check/_summary.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from current_detector import detect_orange_current, find_current_cell  # noqa: E402
from dest_detector import infer_grid_from_dest  # noqa: E402

RAW_DIR = ROOT / "raw"
OUT_DIR = ROOT / "runs" / "current_check"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = sorted(RAW_DIR.glob("*_panel.png"))
    rows: list[dict] = []
    n_with_dot = 0

    for p in panels:
        img = cv2.imread(str(p))
        if img is None:
            continue
        stem = p.stem.removesuffix("_panel")

        d = detect_orange_current(img)
        grid = infer_grid_from_dest(img)
        cell = None
        if d is not None and d["ready"] and grid is not None:
            cell = find_current_cell(img, grid["n_rows"], grid["n_cols"])
            n_with_dot += 1

        # draw
        vis = img.copy()
        if grid is not None:
            H, W = vis.shape[:2]
            cw = W / grid["n_cols"]
            ch = H / grid["n_rows"]
            for c in range(grid["n_cols"] + 1):
                x = int(round(c * cw))
                cv2.line(vis, (x, 0), (x, H), (60, 60, 60), 1)
            for r in range(grid["n_rows"] + 1):
                y = int(round(r * ch))
                cv2.line(vis, (0, y), (W, y), (60, 60, 60), 1)

        if d is not None:
            x, y, w, h = d["bbox"]
            color = (0, 255, 255) if d["ready"] else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        if cell is not None and grid is not None:
            r, c = cell
            cw = vis.shape[1] / grid["n_cols"]
            ch = vis.shape[0] / grid["n_rows"]
            x0, y0 = int(round(c * cw)), int(round(r * ch))
            x1, y1 = int(round((c + 1) * cw)), int(round((r + 1) * ch))
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(vis, f"r{r}c{c}", (x0 + 4, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(str(OUT_DIR / f"{stem}.png"), vis)
        rows.append({
            "stem": stem,
            "n_rows": grid["n_rows"] if grid else "",
            "n_cols": grid["n_cols"] if grid else "",
            "has_blob": d is not None,
            "ready": (d["ready"] if d else False),
            "area": (d["area"] if d else 0),
            "solidity": f'{d["solidity"]:.3f}' if d else "",
            "row": cell[0] if cell else "",
            "col": cell[1] if cell else "",
        })

    with (OUT_DIR / "_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"panels: {len(panels)}")
    print(f"with orange dot detected: {n_with_dot}")
    print(f"out -> {OUT_DIR}")


if __name__ == "__main__":
    main()
