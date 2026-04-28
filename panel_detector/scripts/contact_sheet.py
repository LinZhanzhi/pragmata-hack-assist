"""
Build per-panel contact sheets from cell crops.

Useful for fast visual labeling: open the panel's contact sheet,
identify each cell's class, then move the corresponding PNG from
queue/ into the right labeled/<class>/ folder.

Usage:
    # one frame
    python panel_detector/scripts/contact_sheet.py s01_0034

    # all frames in queue
    python panel_detector/scripts/contact_sheet.py --all

Output: <out>/<frame>_contact.png with the cells laid out in their
NxN positions, each tile labeled "rR cC" so you can match it to the
filename.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUEUE = ROOT / "dataset" / "cells" / "queue"
DEFAULT_OUT = ROOT / "dataset" / "cells" / "contact"

CELL_RE = re.compile(r"^(?P<frame>.+?)_N(?P<n>\d+)_r(?P<r>\d+)_c(?P<c>\d+)\.png$")


def list_frames(queue: Path) -> dict[str, dict]:
    """Group cell PNGs by frame. Returns frame -> {n, cells: {(r,c): path}}."""
    frames: dict[str, dict] = {}
    for p in queue.glob("*.png"):
        m = CELL_RE.match(p.name)
        if not m:
            continue
        f = m.group("frame")
        n = int(m.group("n"))
        r = int(m.group("r"))
        c = int(m.group("c"))
        frames.setdefault(f, {"n": n, "cells": {}})
        frames[f]["cells"][(r, c)] = p
    return frames


def make_sheet(frame: str, info: dict, tile_size: int = 96, gap: int = 6,
               border: int = 2, label_h: int = 18) -> np.ndarray:
    n = info["n"]
    cells = info["cells"]
    cell_box = tile_size + 2 * border
    row_h = cell_box + label_h
    sheet_w = n * cell_box + (n + 1) * gap
    sheet_h = n * row_h + (n + 1) * gap + 28  # extra for title
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    cv2.putText(sheet, f"{frame}  N={n}  ({len(cells)} cells)",
                (gap, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    y0_start = 28
    for r in range(n):
        for c in range(n):
            x0 = gap + c * (cell_box + gap)
            y0 = y0_start + gap + r * (row_h + gap)
            # black border
            sheet[y0:y0 + cell_box, x0:x0 + cell_box] = 0
            p = cells.get((r, c))
            if p is not None:
                img = cv2.imread(str(p))
                if img is not None:
                    img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
                    sheet[y0 + border:y0 + border + tile_size,
                          x0 + border:x0 + border + tile_size] = img
            # label below
            label = f"r{r} c{c}"
            cv2.putText(sheet, label,
                        (x0 + 2, y0 + cell_box + label_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1, cv2.LINE_AA)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame", nargs="?", help="Frame id, e.g. s01_0034. Omit with --all.")
    ap.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--all", action="store_true", help="Process every frame found in --queue.")
    ap.add_argument("--tile-size", type=int, default=96)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames = list_frames(args.queue)

    if args.all:
        targets = sorted(frames)
    else:
        if not args.frame:
            ap.error("Pass a frame id or --all")
        if args.frame not in frames:
            ap.error(f"No cells for frame {args.frame} under {args.queue}")
        targets = [args.frame]

    for f in targets:
        sheet = make_sheet(f, frames[f], tile_size=args.tile_size)
        out = args.out / f"{f}_contact.png"
        cv2.imwrite(str(out), sheet)
        print(f"{f}: N={frames[f]['n']}  ->  {out}")


if __name__ == "__main__":
    main()
