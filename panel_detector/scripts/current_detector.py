"""Current-location ('you are here') marker detector.

The marker is a set of 4 saturated-cyan dots placed at the midpoints of a
cell's 4 borders, sometimes plus an orange disc in the center. The cyan
ring is the universal signal — the orange disc is missing in some panels —
so we detect the cyan ring directly. For each cell we count cyan-ish pixels
in a thin border band and pick the cell with the strongest cyan signature.

Public API:
    detect_orange_current(bgr) -> dict | None      # legacy name (orange-disc detector)
    find_current_cell(bgr, n_rows, n_cols) -> (row, col) | None
"""

from __future__ import annotations

import cv2
import numpy as np


# Tuned visually on s01_0259_panel.png. Orange core is ~(255, 140, 50) RGB,
# i.e. BGR (50, 140, 255). The surrounding ring is more yellow but the
# saturated core alone gives a strong, isolated blob. We deliberately keep
# the threshold tight enough to reject the bright red "boost" icon (which
# has G well below R) and yellow grid text (which has B+G both high).
MIN_AREA = 30           # the dot core can be small; smaller than green dest icon
MIN_SOLIDITY = 0.30

# Minimum cyan-ring pixel count for a cell to be considered "current".
# Each of the 4 dots is on the order of ~30-100 px; require some margin.
MIN_CYAN_RING_PIXELS = 80


def _orange_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of saturated-orange pixels (current-location core)."""
    b, g, r = cv2.split(bgr)
    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)
    orange = (
        (r > 180)
        & ((r16 - b16) > 80)
        & ((r16 - g16) > 30)
        & ((g16 - b16) > 30)
        & (g > 60) & (g < 200)
    )
    m = (orange.astype(np.uint8)) * 255
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _cyan_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of saturated-cyan pixels (the 4-dot ring)."""
    b, g, r = cv2.split(bgr)
    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)
    cyan = (
        (b > 170) & (g > 140)
        & ((b16 - r16) > 50)
        & ((g16 - r16) > 30)
        & ((b16 - g16) > -25)   # reject pure green (dest icon: B << G)
    )
    return (cyan.astype(np.uint8)) * 255


def detect_orange_current(bgr: np.ndarray) -> dict | None:
    """Largest orange-disc blob in the panel (legacy detector).

    Kept for the visualization spotcheck script. The pathfinding pipeline
    uses :func:`find_current_cell` (cyan-ring based) instead.
    """
    m = _orange_mask(bgr)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, w, h, area = (int(stats[best, k]) for k in
                        (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                         cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT,
                         cv2.CC_STAT_AREA))
    solidity = area / max(w * h, 1)
    ready = (area >= MIN_AREA and solidity >= MIN_SOLIDITY)
    return dict(bbox=(x, y, w, h), area=area, solidity=solidity,
                center=(x + w / 2, y + h / 2), ready=ready)


def find_current_cell(
    bgr: np.ndarray,
    n_rows: int,
    n_cols: int,
    exclude: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Return (row, col) of the cell containing the 'you are here' marker.

    Strategy: the marker is 4 cyan dots on the cell's 4 border midpoints. We
    score each cell by how many cyan pixels lie within a thin band around its
    border, and pick the cell with the strongest score (above a minimum).
    The optional ``exclude`` cell (typically the destination) is skipped to
    avoid the dest icon's outer glow, which can pass the cyan threshold.
    """
    H, W = bgr.shape[:2]
    cm = _cyan_mask(bgr)

    cw = W / n_cols
    ch = H / n_rows
    # Band thickness ~12% of cell size (covers the cyan dot which sits on
    # the border and bleeds slightly inside).
    band = max(3, int(round(min(cw, ch) * 0.12)))

    # cv2.integral requires uint8/float. Convert 0/255 mask to 0/1 uint8.
    cm01 = (cm > 0).astype(np.uint8)
    integ = cv2.integral(cm01).astype(np.int64)  # (H+1, W+1)

    def rect_sum(x0, y0, x1, y1):
        x0 = max(0, min(W, int(x0)))
        x1 = max(0, min(W, int(x1)))
        y0 = max(0, min(H, int(y0)))
        y1 = max(0, min(H, int(y1)))
        if x1 <= x0 or y1 <= y0:
            return 0
        return int(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])

    best_score = -1
    best_cell = None
    for r in range(n_rows):
        for c in range(n_cols):
            if exclude is not None and (r, c) == exclude:
                continue
            x0 = c * cw
            y0 = r * ch
            x1 = (c + 1) * cw
            y1 = (r + 1) * ch
            # cyan in border band: full cell minus shrunken interior
            full = rect_sum(x0 - band, y0 - band, x1 + band, y1 + band)
            interior = rect_sum(x0 + band, y0 + band, x1 - band, y1 - band)
            ring = full - interior
            if ring > best_score:
                best_score = ring
                best_cell = (r, c)

    if best_cell is None or best_score < MIN_CYAN_RING_PIXELS:
        return None
    return best_cell
