"""
Slice a warped puzzle panel into per-cell crops for the cell classifier
labeling queue.

Pipeline (per panel image):
1. Infer grid dimension N and gridline positions via infer_grid.infer_grid.
2. For each cell (r, c), take the rect between gridlines (r,c)-(r+1,c+1).
3. Shrink the rect by --shrink fraction toward the cell center so the
   bright separator/border + intersection crosshair are dropped.
4. Resize the inner crop to --size x --size and save as
   <frame>_N<n>_r<row>_c<col>.png in the output queue folder.

Usage:
    # one panel
    python scripts/slice_cells.py path/to/foo_panel.png

    # batch a folder of *_panel.png
    python scripts/slice_cells.py path/to/folder

    # custom output dir, cell size, inner shrink, optional debug overlays
    python scripts/slice_cells.py panels/ --out cells/queue --size 64 --shrink 0.12 --debug
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from infer_grid import infer_grid, gridline_positions

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "dataset" / "cells" / "queue"


def slice_panel(
    panel_bgr: np.ndarray,
    size: int = 64,
    shrink: float = 0.12,
    force_n: int | None = None,
) -> tuple[int, list[tuple[int, int, np.ndarray]], dict]:
    """Infer the grid and return a list of (row, col, cell_crop) tuples.

    `shrink` is the fraction of cell width/height to trim from each side
    before resizing, so the bright gridlines are excluded.  e.g. 0.12
    means keep the central 76% of the cell.

    If `force_n` is given, the comb-fit at that N is used instead of
    the autocorrelation-voted N. Useful when the autodetect picks a
    harmonic (e.g. a 6x6 board with a merged 2x2 dead zone is sometimes
    misread as 3x3).
    """
    res = infer_grid(panel_bgr)
    n = force_n if force_n is not None else res["N"]
    if force_n is not None:
        res["N"] = force_n
        res["forced"] = True
    H, W = panel_bgr.shape[:2]
    if force_n is not None:
        # The warped panel spans the full canvas (0..W, 0..H) by
        # construction, so the most reliable fit at a forced N is just
        # an evenly-spaced grid. The comb-fit can drift toward a
        # harmonic offset when the autocorrelation voted a different N.
        xs = np.linspace(0, W, n + 1)
        ys = np.linspace(0, H, n + 1)
    else:
        xs = gridline_positions(res["col"], n)
        ys = gridline_positions(res["row"], n)
    res["xs"] = xs
    res["ys"] = ys

    cells: list[tuple[int, int, np.ndarray]] = []
    for r in range(n):
        y0_f, y1_f = ys[r], ys[r + 1]
        for c in range(n):
            x0_f, x1_f = xs[c], xs[c + 1]
            cw = x1_f - x0_f
            ch = y1_f - y0_f
            x0 = int(round(x0_f + shrink * cw))
            x1 = int(round(x1_f - shrink * cw))
            y0 = int(round(y0_f + shrink * ch))
            y1 = int(round(y1_f - shrink * ch))
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(W, x1); y1 = min(H, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = panel_bgr[y0:y1, x0:x1]
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
            cells.append((r, c, crop))
    return n, cells, res


def render_debug(panel_bgr: np.ndarray, res: dict, shrink: float, out_path: Path) -> None:
    n = res["N"]
    xs = res.get("xs")
    ys = res.get("ys")
    if xs is None or ys is None:
        xs = gridline_positions(res["col"], n)
        ys = gridline_positions(res["row"], n)
    overlay = panel_bgr.copy()
    H, W = overlay.shape[:2]
    for x in xs:
        cv2.line(overlay, (int(round(x)), 0), (int(round(x)), H - 1), (0, 255, 0), 1)
    for y in ys:
        cv2.line(overlay, (0, int(round(y))), (W - 1, int(round(y))), (0, 255, 0), 1)
    for r in range(n):
        for c in range(n):
            x0_f, x1_f = xs[c], xs[c + 1]
            y0_f, y1_f = ys[r], ys[r + 1]
            cw = x1_f - x0_f
            ch = y1_f - y0_f
            x0 = int(round(x0_f + shrink * cw))
            x1 = int(round(x1_f - shrink * cw))
            y0 = int(round(y0_f + shrink * ch))
            y1 = int(round(y1_f - shrink * ch))
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 255), 1)
    cv2.imwrite(str(out_path), overlay)


def frame_id_from_panel_path(p: Path) -> str:
    """Strip a trailing '_panel' from the stem if present."""
    stem = p.stem
    if stem.endswith("_panel"):
        stem = stem[: -len("_panel")]
    return stem


def process_one(img_path: Path, out_dir: Path, size: int, shrink: float, debug: bool,
                force_n: int | None = None) -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[skip] cannot read {img_path}")
        return {}
    n, cells, res = slice_panel(img, size=size, shrink=shrink, force_n=force_n)
    frame_id = frame_id_from_panel_path(img_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r, c, crop in cells:
        name = f"{frame_id}_N{n}_r{r}_c{c}.png"
        cv2.imwrite(str(out_dir / name), crop)
    if debug:
        dbg_dir = out_dir.parent / "debug_overlays"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        render_debug(img, res, shrink, dbg_dir / f"{frame_id}_slice.png")
    print(
        f"{img_path.name}: N={n} -> {len(cells)} cells "
        f"(agree={res['agree']}, col_margin={res['col']['margin']:.2f}, "
        f"row_margin={res['row']['margin']:.2f})"
    )
    return {"frame": frame_id, "n": n, "cells": len(cells), "agree": res["agree"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path,
                    help="A panel PNG, or a folder. If folder, all *_panel.png are processed.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Output folder for cell crops (default: {DEFAULT_OUT}).")
    ap.add_argument("--size", type=int, default=64,
                    help="Output cell size in pixels (square). Default 64.")
    ap.add_argument("--shrink", type=float, default=0.12,
                    help="Fraction trimmed from each side of every cell before resize. Default 0.12.")
    ap.add_argument("--debug", action="store_true",
                    help="Also write a per-panel overlay showing the inner cell rects.")
    ap.add_argument("--force-n", type=int, default=None,
                    help="Override the inferred N (e.g. 6 for a 6x6 board the autodetect missed).")
    args = ap.parse_args()

    if args.input.is_dir():
        files = sorted(args.input.rglob("*_panel.png"))
        if not files:
            print(f"No *_panel.png under {args.input}")
            return
    else:
        files = [args.input]

    total_cells = 0
    summary = []
    for f in files:
        info = process_one(f, args.out, args.size, args.shrink, args.debug,
                           force_n=args.force_n)
        if info:
            total_cells += info["cells"]
            summary.append(info)

    print()
    print(f"Wrote {total_cells} cell crops from {len(summary)} panels to {args.out}")
    disagree = [s for s in summary if not s["agree"]]
    if disagree:
        print(f"WARNING: {len(disagree)} panel(s) had col/row N disagreement; review them:")
        for s in disagree:
            print(f"  - {s['frame']} (chose N={s['n']})")


if __name__ == "__main__":
    main()
