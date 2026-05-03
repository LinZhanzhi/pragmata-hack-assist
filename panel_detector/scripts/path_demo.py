"""End-to-end path-planning demo on a single warped panel.

Pipeline for one panel image (default: panel_detector/raw/s01_0259_panel.png):
    1. Infer grid (n_rows, n_cols) + destination cell from the green icon.
    2. Find current-location cell from the orange dot.
    3. Slice the panel into cells, classify each with v1_best.pt.
    4. Map labels -> {forbidden, reward, normal, start, dest}:
         forbidden: red_warning, grey_warning, void, unknown
         dest:      destination
         normal:    normal
         reward:    everything else
    5. Run reward-aware best-first search (port of solver.js).
    6. Draw the planned path on the warped panel and on a labeled
       schematic, save to panel_detector/runs/path_demo/.

Usage:
    python panel_detector/scripts/path_demo.py
    python panel_detector/scripts/path_demo.py --panel raw/s01_0073_panel.png
"""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from current_detector import find_current_cell  # noqa: E402
from dest_detector import infer_grid_from_dest  # noqa: E402
from train_cell_classifier import IMG_SIZE, SmallCNN  # noqa: E402

WEIGHTS = ROOT / "runs" / "cell_classifier" / "v1_best.pt"
OUT_DIR = ROOT / "runs" / "path_demo"

FORBIDDEN_CLASSES = {"red_warning", "grey_warning", "void", "unknown"}
NORMAL_CLASSES = {"normal"}
DEST_CLASSES = {"destination"}
# all other classes -> reward


# --- classifier --------------------------------------------------------------

def classify_cells(panel_bgr, n_rows, n_cols, device):
    H, W = panel_bgr.shape[:2]
    xs = [round(c * W / n_cols) for c in range(n_cols + 1)]
    ys = [round(r * H / n_rows) for r in range(n_rows + 1)]

    ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
    classes: list[str] = ckpt["classes"]
    model = SmallCNN(len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    crops = []
    for r in range(n_rows):
        for c in range(n_cols):
            cell = panel_bgr[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
            crops.append(tf(Image.fromarray(rgb)))
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1).cpu().numpy()

    labels = np.empty((n_rows, n_cols), dtype=object)
    confs = np.zeros((n_rows, n_cols), dtype=np.float32)
    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        top = int(probs[i].argmax())
        labels[r, c] = classes[top]
        confs[r, c] = float(probs[i][top])
    return labels, confs


# --- solver (port of solver.js) ---------------------------------------------

def solve(grid_kind, start, goal, alpha=5.0, beta=1.0, max_expansions=20000):
    """grid_kind[r][c] in {'normal', 'forbidden', 'reward', 'dest'}.

    Reward-aware best-first search. Same logic as solver.js: maximize
    score = alpha * rewards - beta * (g + h). Returns (path, rewards, expansions).
    The goal cell itself counts as not-a-reward.
    """
    R = len(grid_kind)
    C = len(grid_kind[0])
    is_blocked = lambda r, c: grid_kind[r][c] == "forbidden"
    is_reward = lambda r, c: grid_kind[r][c] == "reward"

    rewards_pos = [(r, c) for r in range(R) for c in range(C) if is_reward(r, c)]
    total_rewards = len(rewards_pos)

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def heuristic(pos, visited):
        best_d = float("inf")
        best_p = None
        for rp in rewards_pos:
            if rp in visited:
                continue
            d = manhattan(pos, rp)
            if d < best_d:
                best_d = d
                best_p = rp
        if best_p is None:
            return manhattan(pos, goal)
        return best_d + manhattan(best_p, goal)

    pq: list = []
    uid = 0
    start_visited = frozenset({start})
    r0 = 1 if is_reward(*start) else 0
    h0 = heuristic(start, start_visited)
    heapq.heappush(pq, (-(alpha * r0 - beta * h0), uid,
                        start, 0, r0, start_visited, None))
    uid += 1

    expansions = 0
    best_goal = None
    best_score = -float("inf")

    while pq and expansions < max_expansions:
        _, _, pos, g, rew, visited, parent = heapq.heappop(pq)
        expansions += 1

        if pos == goal:
            score = alpha * rew - beta * g
            if score > best_score:
                best_score = score
                best_goal = (pos, g, rew, parent)
            continue

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            if is_blocked(nr, nc):
                continue
            np_ = (nr, nc)
            if np_ in visited:
                continue
            ng = g + 1
            nrew = rew + (1 if is_reward(nr, nc) else 0)

            remaining = total_rewards - nrew
            optim = alpha * (nrew + remaining) - beta * (ng + manhattan(np_, goal))
            if optim <= best_score:
                continue

            nvis = visited | {np_}
            nh = heuristic(np_, nvis)
            heapq.heappush(pq,
                (-(alpha * nrew - beta * nh), uid,
                 np_, ng, nrew, nvis, (pos, parent)))
            uid += 1

    if best_goal is None:
        return None, 0, expansions

    # rebuild path
    pos, g, rew, parent = best_goal
    path = [pos]
    while parent is not None:
        p_pos, parent = parent
        path.append(p_pos)
    path.reverse()
    return path, rew, expansions


# --- drawing -----------------------------------------------------------------

LABEL_COLORS = {
    "normal":       (40, 40, 40),
    "destination":  (0, 200, 0),
    "void":         (80, 0, 80),
    "open":         (180, 180, 180),
    "red_warning":  (0, 0, 200),
    "grey_warning": (140, 140, 140),
    "unknown":      (60, 60, 60),
}
REWARD_DEFAULT = (0, 200, 200)


def cell_color(label):
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    return REWARD_DEFAULT


def draw_overlay(panel_bgr, n_rows, n_cols, labels, path, start, goal):
    H, W = panel_bgr.shape[:2]
    cw = W / n_cols
    ch = H / n_rows
    vis = panel_bgr.copy()
    overlay = panel_bgr.copy()

    # tint cells by class
    for r in range(n_rows):
        for c in range(n_cols):
            x0, y0 = int(round(c * cw)), int(round(r * ch))
            x1, y1 = int(round((c + 1) * cw)), int(round((r + 1) * ch))
            color = cell_color(labels[r, c])
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    vis = cv2.addWeighted(overlay, 0.30, vis, 0.70, 0)

    # gridlines
    for c in range(n_cols + 1):
        x = int(round(c * cw))
        cv2.line(vis, (x, 0), (x, H), (200, 200, 200), 1)
    for r in range(n_rows + 1):
        y = int(round(r * ch))
        cv2.line(vis, (0, y), (W, y), (200, 200, 200), 1)

    # path polyline through cell centers
    if path:
        def center(rc):
            r, c = rc
            return (int(round((c + 0.5) * cw)), int(round((r + 0.5) * ch)))

        pts = [center(p) for p in path]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(vis, a, b, (0, 0, 255), 4, cv2.LINE_AA)
        for p in pts:
            cv2.circle(vis, p, 5, (0, 0, 255), -1, cv2.LINE_AA)

    # start (cyan ring) and goal (green ring)
    sx = int(round((start[1] + 0.5) * cw))
    sy = int(round((start[0] + 0.5) * ch))
    cv2.circle(vis, (sx, sy), int(min(cw, ch) * 0.35), (255, 200, 0), 3)
    cv2.putText(vis, "S", (sx - 8, sy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    gx = int(round((goal[1] + 0.5) * cw))
    gy = int(round((goal[0] + 0.5) * ch))
    cv2.circle(vis, (gx, gy), int(min(cw, ch) * 0.35), (0, 255, 0), 3)
    cv2.putText(vis, "G", (gx - 8, gy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return vis


def draw_schematic(n_rows, n_cols, labels, path, start, goal, cell_px=80):
    H = n_rows * cell_px
    W = n_cols * cell_px
    img = np.full((H, W, 3), 25, dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            x0, y0 = c * cell_px, r * cell_px
            x1, y1 = x0 + cell_px, y0 + cell_px
            cv2.rectangle(img, (x0, y0), (x1, y1), cell_color(labels[r, c]), -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
            cv2.putText(img, labels[r, c][:6], (x0 + 4, y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
    if path:
        pts = [(int((c + 0.5) * cell_px), int((r + 0.5) * cell_px)) for r, c in path]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (0, 0, 255), 4, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 6, (0, 0, 255), -1, cv2.LINE_AA)
    sx, sy = int((start[1] + 0.5) * cell_px), int((start[0] + 0.5) * cell_px)
    gx, gy = int((goal[1] + 0.5) * cell_px), int((goal[0] + 0.5) * cell_px)
    cv2.circle(img, (sx, sy), int(cell_px * 0.35), (255, 200, 0), 3)
    cv2.circle(img, (gx, gy), int(cell_px * 0.35), (0, 255, 0), 3)
    return img


# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(ROOT / "raw" / "s01_0259_panel.png"))
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()

    panel_path = Path(args.panel)
    img = cv2.imread(str(panel_path))
    if img is None:
        raise SystemExit(f"could not read {panel_path}")

    grid = infer_grid_from_dest(img)
    if grid is None:
        raise SystemExit("could not infer grid (no green destination icon)")
    n_rows, n_cols = grid["n_rows"], grid["n_cols"]
    goal = (grid["dest_row"], grid["dest_col"])
    print(f"grid: {n_rows}x{n_cols}, dest: {goal}")

    start = find_current_cell(img, n_rows, n_cols, exclude=goal)
    if start is None:
        raise SystemExit("could not find orange current-location dot")
    print(f"start: {start}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels, confs = classify_cells(img, n_rows, n_cols, device)

    # build kind grid
    kind = [["normal"] * n_cols for _ in range(n_rows)]
    for r in range(n_rows):
        for c in range(n_cols):
            lbl = labels[r, c]
            if (r, c) == goal:
                kind[r][c] = "dest"
            elif (r, c) == start:
                kind[r][c] = "normal"  # start cell traversable
            elif lbl in FORBIDDEN_CLASSES:
                kind[r][c] = "forbidden"
            elif lbl in NORMAL_CLASSES:
                kind[r][c] = "normal"
            elif lbl in DEST_CLASSES:
                # classifier says dest but it isn't the actual dest -> treat as normal
                kind[r][c] = "normal"
            else:
                kind[r][c] = "reward"

    # solver treats 'dest' the same as 'normal' for traversal; make goal walkable
    kind[goal[0]][goal[1]] = "normal"

    print("label grid:")
    for r in range(n_rows):
        print("  " + " ".join(f"{labels[r,c][:4]:>4}" for c in range(n_cols)))
    print("kind grid:")
    for r in range(n_rows):
        print("  " + " ".join(f"{kind[r][c][:4]:>4}" for c in range(n_cols)))

    path, rewards, expansions = solve(kind, start, goal,
                                      alpha=args.alpha, beta=args.beta)
    if path is None:
        print(f"NO PATH after {expansions} expansions")
    else:
        print(f"path length: {len(path) - 1} steps, rewards collected: {rewards}, "
              f"expansions: {expansions}")
        print(f"path: {path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = panel_path.stem.removesuffix("_panel")

    overlay = draw_overlay(img, n_rows, n_cols, labels, path, start, goal)
    cv2.imwrite(str(OUT_DIR / f"{stem}_overlay.png"), overlay)

    schem = draw_schematic(n_rows, n_cols, labels, path, start, goal)
    cv2.imwrite(str(OUT_DIR / f"{stem}_schematic.png"), schem)

    print(f"-> {OUT_DIR / (stem + '_overlay.png')}")
    print(f"-> {OUT_DIR / (stem + '_schematic.png')}")


if __name__ == "__main__":
    main()
