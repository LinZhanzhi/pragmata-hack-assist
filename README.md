# pathSolver — Pragmata Hack-Panel Real-Time Assist

Real-time vision + planning + input-injection assist for the Pragmata
hacking mini-game. While the puzzle panel is on screen, hold a hot-key
combo and the app:

1. captures the game window,
2. detects the panel and warps it to a clean rectangle,
3. classifies every cell (start, destination, reward, normal, hazard, …),
4. plans a reward-aware optimal path from your current cell to the goal,
5. drives the in-game cursor along that path **in closed loop** by
   tracking the cyan-ring + orange dot every frame and nudging the OS
   mouse one screen axis at a time.

The walker is sensitivity-independent: it does not pre-compute screen
coordinates. It looks at where the cursor actually is in the panel each
tick and corrects until it arrives.

---

## Demo flow

```
[ pick game window ]
        │
        ▼
hold RMB + MOUSE 4, tap T   ──►   capture → YOLO panel → warp → classify
                                   → reward-aware best-first path
                                   → closed-loop walker (axis-only mouse)
                                   → goal reached
ESC, or release RMB/M4, aborts the walk at any time.
```

---

## Tech stack

### Vision

- **YOLOv8n-pose** (Ultralytics) — single-class panel detector with 4
  keypoints (TL, TR, BR, BL). Trained on hand-labelled screenshots; the
  dataset and split scripts live under `panel_detector/`.
- **Corner reconstruction** — at inference time we use the YOLO bbox
  plus a fixed pair of side-slope priors (`TOP_SLOPE = -0.0907`,
  `BOT_SLOPE = +0.0409`, `MAX_SIDE = 800`) to recover the four panel
  corners robustly. See `panel_detector/scripts/predict.py`.
- **Perspective warp** — `cv2.getPerspectiveTransform` /
  `cv2.warpPerspective` from the 4 corners to a canonical
  (panel_w × panel_h) rectangle. The 3×3 matrix `M_inv`
  (panel → client window) is preserved so we can map detections back
  into screen space at any time.
- **Cell classifier** — `SmallCNN` (3 conv blocks + 1 FC), 64×64 RGB
  input, 20-way softmax. Trained with `panel_detector/scripts/train_cell_classifier.py`,
  validation accuracy ≈ 0.983. Weights:
  `panel_detector/runs/cell_classifier/v1_best.pt`.
- **Grid inference** — `dest_detector.infer_grid_from_dest()` finds
  the green destination icon in the warped panel and infers
  `(n_rows, n_cols, start, goal)` from the panel's known geometry.
- **Cyan-ring detector** — `current_detector.find_current_cell()` builds
  an integral image of a saturated-cyan mask and scores every cell by
  the cyan count in a thin border band; the cell with the strongest
  ring is the cursor cell. Robust against missing/jittering orange dot.
- **Orange-dot localizer** — `current_detector.find_orange_centroid_in_cell()`
  runs a tight orange mask only inside the cyan-ring cell's bbox so it
  cannot confuse the cursor with red warning icons elsewhere on the
  panel. Gives sub-cell precision for the walker.

### Planning

- **Reward-aware best-first search** — `path_demo.solve()`. Heap-ordered
  by `α · rewards_collected − β · steps_taken`
  (`ALPHA = 5.0`, `BETA = 1.0`). 4-direction moves, no revisits,
  branch-and-bound prune on an admissible upper bound, expansion budget
  `20 000`. Forbidden cells (`red_warning`, `grey_warning`, `void`,
  `unknown`) are blocked.

### Real-time control

- **Window enumeration / capture** — `pywin32`
  (`win32gui` + `win32ui` + `PrintWindow` flag `0x02` with fallback
  `0x00`). Captures the client area of any visible window without
  stealing focus.
- **Hot-key trigger** — `pynput` global mouse + keyboard listeners.
  Trigger combo: **hold Right Mouse + hold Mouse 4 (XButton1) + tap T**.
  ESC aborts an in-progress walk.
- **Mouse injection** — `ctypes` `SendInput` with a `MOUSEINPUT` struct.
  We use **relative** `MOUSEEVENTF_MOVE` deltas (no `ABSOLUTE` flag), so
  the in-game sensitivity converts our OS-pixel nudges into in-panel
  motion. This is the only way the walker stays correct across
  different mouse + in-game sensitivity settings.
- **Closed-loop walker** — every iteration:
  1. capture client → warp to panel,
  2. find cyan-ring cell,
  3. find orange centroid inside that cell (sub-cell position),
  4. if cursor cell is on the planned path → target = next path cell;
     otherwise → target = nearest path cell (recovers from drift /
     overshoot before resuming),
  5. compute panel-space delta to target center, pick the dominant
     panel axis, map that single axis through `M_inv` to a screen axis,
  6. send a fixed-magnitude relative mouse move along that **single
     screen axis only** — never diagonal.
- **Tunables** — `--step <px>` (default 75) and `--tick <s>` (default 0,
  i.e. no artificial sleep; the loop is paced only by capture +
  detection cost, ~30–50 ms/iter).

### UI

- **Window picker** — small `tkinter` `Listbox` of visible windows
  (≥ 100 × 100), with refresh; double-click or "Start" returns the
  HWND.

---

## Repository layout

```
pathSolver/
  assist_app.py                     # the live assist (entry point)
  panel_detector/
    data.yaml                       # YOLO dataset descriptor
    raw/                            # original screenshots
    dataset/{images,labels}/{train,val,test}/
    runs/
      panel_detector/weights/best.pt   # YOLO weights (panel + 4 keypoints)
      cell_classifier/v1_best.pt       # cell classifier weights
    scripts/
      predict.py                    # YOLO inference + corner reconstruction
      train.py                      # YOLOv8n-pose trainer
      train_cell_classifier.py      # SmallCNN trainer
      slice_cells.py                # slice warped panel into cell crops
      dest_detector.py              # green destination icon -> grid
      current_detector.py           # cyan ring + orange dot detectors
      path_demo.py                  # solve() + draw_overlay()
      path_demo_all.py              # batch overlay generator
      ...                           # labelling helpers, splitters, etc.
  doc/                              # design notes
```

---

## Installation

Requires Windows 10/11, Python 3.11+ (developed on 3.13.11), and an
NVIDIA GPU with a recent CUDA build of PyTorch (developed on RTX 5060
Ti, CUDA 12.8, PyTorch 2.11). The classifier is small enough to run on
CPU but the YOLO model is much faster on GPU.

```powershell
# 1. clone
git clone https://github.com/LinZhanzhi/pathSolver.git
cd pathSolver

# 2. create + activate a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. install
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python pillow numpy pywin32 pynput
```

Weights are committed under `panel_detector/runs/` so you can use the
app immediately without retraining.

---

## User guide

### 1. Launch the assist

```powershell
python assist_app.py
```

A small window picker appears. Click your game's window
(e.g. `PRAGMATA`), then press **Start**.

CLI options:

| flag       | default | meaning                                            |
| ---------- | ------- | -------------------------------------------------- |
| `--list`   | —       | Print HWND + title of every visible window, exit.  |
| `--hwnd N` | —       | Skip the picker and bind directly to that HWND.    |
| `--step N` | `75`    | OS-pixel magnitude per mouse nudge.                |
| `--tick S` | `0.0`   | Sleep (seconds) between ticks; `0` = fastest.      |

Console will print:

```
[ready] target window: 'PRAGMATA' (hwnd=...)
[walker] step=75 px, tick=0.000 s
Hold RIGHT MOUSE + MOUSE 4, then tap T to trigger.
ESC or releasing RMB/M4 aborts the active walk. Ctrl+C to quit.
```

### 2. Use it in-game

1. Open a hack panel in Pragmata (offensive or combust mode — see
   *Known limitations* below).
2. **Wait for any animations to fully settle** (panel slide-in, dot
   pulse, glitch effects, …). The classifier and detectors expect a
   stable, "clean" frame at the moment you trigger.
3. Hold **Right Mouse Button**, hold **Mouse Button 4 (XButton1)**, then
   **tap T**.
4. The app captures, plans, and starts walking the cursor. You should
   see lines like:

   ```
   [trigger] plan: 7x10 start=(3,0) goal=(3,9) path_len=14 rewards=5 (412 ms)
   [walk] now at (3,1) (on-path, target (3,2))
   [walk] now at (3,2) (on-path, target (3,3))
   ...
   [walk] reached goal in 1.86s
   ```

5. To abort early: press **ESC**, or simply **release** RMB or Mouse 4.

Each trigger also writes `panel_detector/runs/live/<timestamp>_capture.png`
and `<timestamp>_overlay.png` for debugging.

### 3. Tuning

- **Step too small (slow / stuck)** — increase with `--step 100` or
  `--step 150`.
- **Step too large (overshooting cells)** — decrease, e.g. `--step 50`.
- **Tick** — leave at `0` unless your CPU is so loud that an explicit
  small sleep helps. The loop is naturally paced by capture+detect.

The walker recovers from overshoot automatically: if it lands on a cell
that is *not* on the plan, it routes back to the nearest path cell and
continues.

---

## Known limitations

These are the only two issues we know about. Otherwise the assist is
reliable and consistently picks a good path with a healthy number of
reward cells.

1. **Limited hacking-mode coverage.** Only the **offensive mode** and
   **combust mode** panels are represented in the cell-classifier
   training set. Other hacking modes (defensive, scan, etc.) may
   contain cell types the classifier has never seen, so classification
   accuracy on those panels is not guaranteed. To extend support: add
   labelled cells from the missing modes and retrain
   `panel_detector/scripts/train_cell_classifier.py`.

2. **Trigger only on a stable frame.** The user is responsible for
   making sure no disturbing animations are in progress when **T** is
   tapped. Panel intro slides, glitch overlays, damage flashes, etc.
   can throw off panel detection or cell classification. Wait until the
   panel is fully visible and steady, *then* trigger.

---

## Development notes

- Per-process safety: `num_workers = 0` for all DataLoaders (Windows +
  CUDA + PyTorch 2.11 has been the smoothest at 0).
- The cell classifier checkpoint is `{"state_dict", "classes"}`. Class
  list is alphabetical and 20-way; do **not** reorder it.
- Panel YOLO weights and cell classifier weights are committed; the
  raw screenshots and labels are kept in `panel_detector/raw/` and
  `panel_detector/dataset/` respectively for reproducibility.
- Useful spotcheck scripts:
  - `panel_detector/scripts/spotcheck_current.py` — visualizes cyan ring
    and orange dot detections.
  - `panel_detector/scripts/path_demo_all.py` — batch-renders path
    overlays over every panel screenshot in the dataset.

---

## License

Personal hobby project. Use at your own risk; do not use in
multiplayer, online, or competitive contexts. The Pragmata
trademarks/IP belong to their respective owners; this project is not
affiliated with or endorsed by them.
