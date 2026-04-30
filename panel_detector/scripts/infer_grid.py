"""
Infer the puzzle grid dimension N from a warped panel image using
periodicity analysis on row/column brightness projections.

Pipeline:
1. Convert the warped panel to grayscale.
2. Build 1-D row and column profiles by summing brightness along each axis.
   The grid borders + the bright cross/dot at every gridline intersection
   produce strong periodic peaks in those profiles.
3. For each candidate N in {3..8}, score how well the profile is explained
   by a comb of N+1 evenly spaced gridlines.
4. Pick the N (and the offset/spacing) with the best score for rows and
   columns independently, then verify they agree.
5. Save a debug PNG so you can visually confirm the result.

Usage:
    python scripts/infer_grid.py path/to/panel.png
    python scripts/infer_grid.py path/to/panel.png --out debug.png
    python scripts/infer_grid.py path/to/folder      # batch all *_panel.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

CANDIDATE_NS = list(range(3, 11))  # 3..10 cells per axis (rows and cols inferred independently)


def build_profiles(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (col_profile, row_profile) as 1-D float32 arrays.

    col_profile[x] = brightness summed over column x (length = W).
    row_profile[y] = brightness summed over row y    (length = H).

    We use raw grayscale rather than a Sobel edge map because the
    gridlines are bright (white separators + white intersection
    crosshairs) on a dark panel background, so summing intensity
    already gives a strong periodic signal.
    """
    g = gray.astype(np.float32)
    col_profile = g.sum(axis=0)
    row_profile = g.sum(axis=1)
    return col_profile, row_profile


def detrend(p: np.ndarray, win_frac: float = 0.15) -> np.ndarray:
    """Remove a slow background trend so periodicity dominates."""
    n = len(p)
    win = max(3, int(n * win_frac) | 1)  # odd
    bg = cv2.GaussianBlur(p.reshape(1, -1), (win, 1), 0).ravel()
    return p - bg


def autocorr(p: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation of a 1-D signal (lag 0 .. len(p)-1)."""
    x = p - p.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] > 0:
        ac /= ac[0]
    return ac


def autocorr_score(profile: np.ndarray, n_cells: int) -> float:
    """Score N via autocorrelation of the detrended profile.

    A panel with n_cells columns has n_cells gridline-to-gridline
    repetitions of length T = L / n_cells, so the autocorrelation at
    lag T (and its harmonics 2T, 3T, ...) should be high.  We average
    autocorrelation values at the first few harmonics with a small
    tolerance to absorb sub-pixel period error.

    This metric does NOT depend on the number of teeth (unlike a mean
    over a fitted comb), so it is unbiased across candidate N.
    """
    p = detrend(profile)
    ac = autocorr(p)
    L = len(p)
    T = L / n_cells
    if T < 4:
        return -np.inf
    tol = max(2, int(round(T * 0.04)))
    # Look at harmonics k*T for k = 1 .. min(n_cells-1, 4).
    max_k = min(n_cells - 1, 4)
    if max_k < 1:
        return -np.inf
    vals = []
    for k in range(1, max_k + 1):
        lag = int(round(k * T))
        lo = max(1, lag - tol)
        hi = min(L - 1, lag + tol + 1)
        if lo >= hi:
            continue
        vals.append(float(ac[lo:hi].max()))
    if not vals:
        return -np.inf
    return float(np.mean(vals))


def comb_score(profile: np.ndarray, n_cells: int) -> tuple[float, float, float]:
    """Score how well `profile` matches a comb of (n_cells+1) gridlines.

    We slide a comb of n_cells+1 evenly spaced gridlines across the
    profile and pick the (offset, spacing) that maximizes the average
    profile value at the comb positions.

    Returns (best_score, best_offset_px, best_spacing_px).
    """
    L = len(profile)
    # Cell spacing must produce n_cells cells across the panel; allow a
    # small range around L/n_cells to absorb any inner padding.
    nominal = L / n_cells
    spacings = np.linspace(nominal * 0.85, nominal * 1.05, 21)
    best = (-np.inf, 0.0, nominal)
    for s in spacings:
        span = s * n_cells
        # The first gridline can start anywhere from 0 to (L - span).
        max_off = L - span
        if max_off < 0:
            continue
        offsets = np.linspace(0, max_off, 31)
        for off in offsets:
            xs = off + s * np.arange(n_cells + 1)
            xi = np.clip(np.round(xs).astype(int), 0, L - 1)
            score = float(profile[xi].mean())
            if score > best[0]:
                best = (score, float(off), float(s))
    return best


def infer_axis(profile: np.ndarray) -> dict:
    """Infer N for one axis. Returns a dict with diagnostics.

    Picking criterion: ``score = autocorr_score(N) * comb_score(N)``.

    Why the combination:
      * Autocorrelation alone is unbiased across N but ambiguous on
        harmonics: a true-N panel produces high autocorr at N as well
        as 2N, 3N, ... and at N/2, N/3, ... (when those are integers).
        Empirically that misfires on rectangular panels where one axis
        is squashed and the half-period peaks survive (e.g. cols=10
        instead of 5 on s01_0080).
      * Comb score (mean of detrended profile at fitted gridline
        positions) discriminates between true N and 2N: at 2N, half the
        comb teeth fall on cell interiors, dragging the mean down. So
        multiplying autocorr by comb suppresses the harmonic.
      * Comb alone is biased toward small N (fewer teeth = more chance
        to land on bright spots), so we multiply by autocorr instead of
        using comb on its own.
    """
    p_dt = detrend(profile)
    results = []
    for n in CANDIDATE_NS:
        ac_s = autocorr_score(profile, n)
        comb_s, off, spacing = comb_score(p_dt, n)
        # Use max(0, ...) so a negative autocorr can't flip the sign of
        # the product (negative * negative = positive false-positive).
        ac_clip = max(0.0, ac_s)
        score = ac_clip * comb_s
        results.append({
            "n": n,
            "score": score,
            "autocorr": ac_s,
            "comb_score": comb_s,
            "offset": off,
            "spacing": spacing,
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    second_score = results[1]["score"] if len(results) > 1 else -np.inf
    best["margin"] = best["score"] - second_score
    best["all"] = results
    return best


def infer_grid(panel_bgr: np.ndarray) -> dict:
    """Infer grid dimensions for a (possibly non-square) panel.

    Rows and columns are inferred independently from their respective
    1-D brightness profiles. The result exposes ``N_cols`` and
    ``N_rows`` separately. ``N`` is also set for backward compatibility:
    it equals ``N_rows`` when rows == cols, otherwise it is ``None``.
    ``agree`` indicates whether rows and cols matched.
    """
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    col_p, row_p = build_profiles(gray)
    col_res = infer_axis(col_p)
    row_res = infer_axis(row_p)
    n_col, n_row = col_res["n"], row_res["n"]
    agree = n_col == n_row
    return {
        "N_cols": n_col,
        "N_rows": n_row,
        "N": n_col if agree else None,
        "agree": agree,
        "col": col_res,
        "row": row_res,
        "col_profile": col_p,
        "row_profile": row_p,
        "gray": gray,
    }


def gridline_positions(axis_res: dict, n: int) -> np.ndarray:
    """Recover gridline pixel positions for the chosen N on one axis.

    If the axis voted for a different N, recompute at the requested N.
    """
    if axis_res["n"] == n:
        off, s = axis_res["offset"], axis_res["spacing"]
    else:
        # Re-fit at the agreed-upon N.
        # axis_res["all"] was sorted; find n there.
        cand = next(r for r in axis_res["all"] if r["n"] == n)
        off, s = cand["offset"], cand["spacing"]
    return off + s * np.arange(n + 1)


def render_debug(result: dict, panel_bgr: np.ndarray, out_path: Path) -> None:
    n_cols = result["N_cols"]
    n_rows = result["N_rows"]
    xs = gridline_positions(result["col"], n_cols)
    ys = gridline_positions(result["row"], n_rows)

    overlay = panel_bgr.copy()
    h, w = overlay.shape[:2]
    for x in xs:
        cv2.line(overlay, (int(round(x)), 0), (int(round(x)), h - 1), (0, 255, 0), 2)
    for y in ys:
        cv2.line(overlay, (0, int(round(y))), (w - 1, int(round(y))), (0, 255, 0), 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Warped panel")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(
        f"Inferred grid: rows={n_rows}  cols={n_cols}  (agree={result['agree']})"
    )
    axes[0, 1].axis("off")

    col_p = detrend(result["col_profile"])
    row_p = detrend(result["row_profile"])

    axes[1, 0].plot(col_p, color="steelblue")
    for x in xs:
        axes[1, 0].axvline(x, color="green", alpha=0.5, linewidth=1)
    axes[1, 0].set_title("Column profile (detrended) + fitted gridlines")
    axes[1, 0].set_xlabel("x")

    axes[1, 1].plot(row_p, color="indianred")
    for y in ys:
        axes[1, 1].axvline(y, color="green", alpha=0.5, linewidth=1)
    axes[1, 1].set_title("Row profile (detrended) + fitted gridlines")
    axes[1, 1].set_xlabel("y")

    # Per-N score bar: helpful to see the margin.
    scores_col = [(r["n"], r["score"]) for r in sorted(result["col"]["all"], key=lambda r: r["n"])]
    scores_row = [(r["n"], r["score"]) for r in sorted(result["row"]["all"], key=lambda r: r["n"])]
    fig.suptitle(
        f"col autocorr scores: {[(n_, round(s_, 3)) for n_, s_ in scores_col]}   "
        f"row autocorr scores: {[(n_, round(s_, 3)) for n_, s_ in scores_row]}",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def process_one(img_path: Path, out_dir: Path) -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[skip] cannot read {img_path}")
        return {}
    res = infer_grid(img)
    out_path = out_dir / (img_path.stem + "_grid.png")
    render_debug(res, img, out_path)
    print(
        f"{img_path.name}: rows={res['N_rows']} cols={res['N_cols']}  "
        f"(col_margin={res['col']['margin']:.2f}, "
        f"row_margin={res['row']['margin']:.2f}, "
        f"agree={res['agree']})  -> {out_path.name}"
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path,
                    help="A panel PNG, or a folder. If folder, all *_panel.png inside are processed.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write debug PNGs (default: alongside the input).")
    args = ap.parse_args()

    if args.input.is_dir():
        files = sorted(args.input.rglob("*_panel.png"))
        if not files:
            print(f"No *_panel.png under {args.input}")
            return
        out_dir = args.out_dir or (args.input / "grid_check")
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            process_one(f, out_dir)
    else:
        out_dir = args.out_dir or args.input.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        process_one(args.input, out_dir)


if __name__ == "__main__":
    main()
