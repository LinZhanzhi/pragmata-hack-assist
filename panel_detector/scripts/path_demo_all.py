"""Batch-run path_demo over every *_panel.png that has an orange dot.

Output goes to panel_detector/runs/path_demo/ (overlay only).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from current_detector import find_current_cell  # noqa: E402
from dest_detector import infer_grid_from_dest  # noqa: E402
from path_demo import (  # noqa: E402
    DEST_CLASSES,
    FORBIDDEN_CLASSES,
    NORMAL_CLASSES,
    draw_overlay,
    solve,
)
from train_cell_classifier import IMG_SIZE, SmallCNN  # noqa: E402

WEIGHTS = ROOT / "runs" / "cell_classifier" / "v1_best.pt"
RAW_DIR = ROOT / "raw"
OUT_DIR = ROOT / "runs" / "path_demo"
ALPHA, BETA = 5.0, 1.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    panels = sorted(RAW_DIR.glob("*_panel.png"))
    print(f"scanning {len(panels)} panels")

    n_ok = n_skip = n_fail = 0
    for panel_path in panels:
        try:
            img = cv2.imread(str(panel_path))
            if img is None:
                n_fail += 1
                continue
            grid = infer_grid_from_dest(img)
            if grid is None:
                n_skip += 1
                continue
            n_rows, n_cols = grid["n_rows"], grid["n_cols"]
            goal = (grid["dest_row"], grid["dest_col"])

            start = find_current_cell(img, n_rows, n_cols, exclude=goal)
            if start is None:
                n_skip += 1
                continue

            H, W = img.shape[:2]
            xs = [round(c * W / n_cols) for c in range(n_cols + 1)]
            ys = [round(r * H / n_rows) for r in range(n_rows + 1)]
            crops = []
            for r in range(n_rows):
                for c in range(n_cols):
                    cell = img[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
                    rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
                    crops.append(tf(Image.fromarray(rgb)))
            batch = torch.stack(crops).to(device)
            with torch.no_grad():
                probs = F.softmax(model(batch), dim=1).cpu().numpy()

            import numpy as np
            labels = np.empty((n_rows, n_cols), dtype=object)
            for i in range(n_rows * n_cols):
                r, c = divmod(i, n_cols)
                labels[r, c] = classes[int(probs[i].argmax())]

            kind = [["normal"] * n_cols for _ in range(n_rows)]
            for r in range(n_rows):
                for c in range(n_cols):
                    lbl = labels[r, c]
                    if (r, c) == goal or (r, c) == start:
                        kind[r][c] = "normal"
                    elif lbl in FORBIDDEN_CLASSES:
                        kind[r][c] = "forbidden"
                    elif lbl in NORMAL_CLASSES or lbl in DEST_CLASSES:
                        kind[r][c] = "normal"
                    else:
                        kind[r][c] = "reward"

            path, rewards, expansions = solve(
                kind, start, goal, alpha=ALPHA, beta=BETA, max_expansions=20000)
            overlay = draw_overlay(img, n_rows, n_cols, labels, path, start, goal)

            stem = panel_path.stem.removesuffix("_panel")
            out_path = OUT_DIR / f"{stem}_overlay.png"
            cv2.imwrite(str(out_path), overlay)
            n_ok += 1
            steps = (len(path) - 1) if path else -1
            print(f"  {stem}: {n_rows}x{n_cols} start={start} goal={goal} "
                  f"steps={steps} rewards={rewards} exp={expansions}")
        except Exception:
            n_fail += 1
            print(f"  {panel_path.name}: FAILED")
            traceback.print_exc()

    print(f"done. ok={n_ok} skipped={n_skip} failed={n_fail}")
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
