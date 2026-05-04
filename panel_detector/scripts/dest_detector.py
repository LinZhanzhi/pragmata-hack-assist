"""Destination-cell detector for warped Cyberpunk hack panels.

Strategy (no ML): the destination cell is the only saturated-green object
in a warped panel. We isolate it with a fixed-color mask, take the largest
connected component, and reject anything that doesn't look like a fully-
loaded square icon.

Public API:
    detect_green_dest(bgr) -> dict | None
    infer_grid_from_dest(bgr) -> dict | None
"""

from __future__ import annotations
import cv2
import numpy as np

# --- Game-art constants -----------------------------------------------------
# In the warped panel, the destination icon is centered inside its cell with
# ~19% padding on each side, so icon_size / cell_size ≈ 0.62.
ICON_TO_CELL = 0.62

# --- "Fully-loaded" gate thresholds ----------------------------------------
# Tuned empirically on the s01 dataset (40 baseline + 219 new = 259 frames).
# Note: we deliberately do NOT bound the icon aspect ratio. Rectangular
# panels (e.g. 8x5) have cells with aspect ~1.6, and the icon inherits the
# cell aspect, so the icon bbox is naturally non-square. Mid-animation
# fragments are filtered instead by the grid-count bound (a partial icon
# yields a tiny bbox and therefore an implausibly large inferred grid).
MIN_AREA = 150           # px after morph close
MIN_SOLIDITY = 0.40      # area / (w*h). Filters thin/wispy mid-animation
                         # blobs and partly-rendered ring icons. Borderline
                         # cases should be retried on the next frame.
GRID_MIN = 3             # plausible R, C
GRID_MAX = 10


def _green_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of saturated-green pixels (destination icon color)."""
    b, g, r = cv2.split(bgr)
    gi = g.astype(np.int16)
    green = ((gi - r.astype(np.int16)) > 30) & \
            ((gi - b.astype(np.int16)) > 30) & \
            (g > 120)
    m = (green.astype(np.uint8)) * 255
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def detect_green_dest(bgr: np.ndarray) -> dict | None:
    """Find the destination-cell green icon in a warped panel.

    We enumerate green connected components and pick the largest one that
    passes the "fully-loaded" gate (area + solidity). Picking the
    overall-largest blob is wrong on big panels: scanline / glitch effects
    can connect dozens of small green pixels into a tall, low-solidity
    column whose area dwarfs the real icon. Falling back to the largest
    blob (with `ready=False`) is preserved so callers can still inspect
    why nothing passed.
    """
    m = _green_mask(bgr)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None

    def _stats(i):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        a = int(stats[i, cv2.CC_STAT_AREA])
        return x, y, w, h, a

    # First pass: only blobs that pass the gate.
    qualified = []
    for i in range(1, n):
        x, y, w, h, a = _stats(i)
        sol = a / max(w * h, 1)
        if a >= MIN_AREA and sol >= MIN_SOLIDITY:
            qualified.append((a, i, x, y, w, h, sol))

    if qualified:
        qualified.sort(reverse=True)
        a, _, x, y, w, h, sol = qualified[0]
        return dict(bbox=(x, y, w, h), area=a,
                    aspect=w / max(h, 1), solidity=sol, ready=True)

    # Fallback: report the largest blob with ready=False (legacy behavior).
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, w, h, a = _stats(best)
    sol = a / max(w * h, 1)
    return dict(bbox=(x, y, w, h), area=a,
                aspect=w / max(h, 1), solidity=sol, ready=False)


def infer_grid_from_dest(bgr: np.ndarray) -> dict | None:
    """Infer (n_rows, n_cols) and locate the destination cell.

    Returns None when no green blob exists, the icon fails the fully-loaded
    gate, or the inferred grid count is outside [GRID_MIN, GRID_MAX].
    """
    H, W = bgr.shape[:2]
    d = detect_green_dest(bgr)
    if d is None or not d['ready']:
        return None
    x, y, w, h = d['bbox']
    cw = w / ICON_TO_CELL
    ch = h / ICON_TO_CELL
    n_cols = int(round(W / cw))
    n_rows = int(round(H / ch))
    if not (GRID_MIN <= n_rows <= GRID_MAX and GRID_MIN <= n_cols <= GRID_MAX):
        return None
    # Snap a clean cell size from the round counts (more stable than raw cw/ch)
    cell_w = W / n_cols
    cell_h = H / n_rows
    icon_cx = x + w / 2
    icon_cy = y + h / 2
    dest_col = int(icon_cx // cell_w)
    dest_row = int(icon_cy // cell_h)
    dest_col = max(0, min(n_cols - 1, dest_col))
    dest_row = max(0, min(n_rows - 1, dest_row))
    return dict(
        n_rows=n_rows, n_cols=n_cols,
        dest_row=dest_row, dest_col=dest_col,
        cell_w=cell_w, cell_h=cell_h,
        icon_bbox=(x, y, w, h),
        icon_aspect=d['aspect'], icon_solidity=d['solidity'],
        icon_area=d['area'],
    )
