# Puzzle Panel Detector

Stage-1 of the path-solver vision pipeline: detect the puzzle UI panel in a
full game screenshot and locate its **4 corners** (top-left, top-right,
bottom-right, bottom-left). The corners are then used to perspective-warp
the trapezoidal panel into a clean rectangle for downstream grid analysis.

## Pipeline goal

```
full screenshot
  --> [YOLO pose: bbox + 4 keypoints]
  --> 4 corners of panel
  --> cv2.warpPerspective
  --> rectangular canonical panel image
  --> (later stages: grid size, cell slicing, cell classifier)
```

Single class: `puzzle_panel`. Four keypoints per panel: `TL, TR, BR, BL`.

## Folder structure

```
panel_detector/
  raw/                     # original screenshots straight from capture (1920x1080 PNG)
    metadata.csv           # session, resolution, grid size, notes
  dataset/
    images/{train,val,test}/
    labels/{train,val,test}/   # YOLO pose label files
  data.yaml                # YOLO dataset descriptor (1 class + 4 keypoints)
  scripts/
    split_dataset.py       # split raw -> train/val/test by session id
    train.py               # train YOLOv8n-pose
    predict.py             # detect corners and save a perspective-warped crop
  runs/                    # YOLO output (auto-created on training)
  models/                  # final exported weights
  README.md                # this file
```

## YOLO pose label format

Each image gets a `.txt` label file. One line per object:

```
<class_id> <x_c> <y_c> <w> <h> <kp1_x> <kp1_y> <kp1_v> <kp2_x> <kp2_y> <kp2_v> ...
```

All coordinates normalized to [0, 1]. Visibility flag: `2` = labeled & visible,
`1` = labeled & occluded, `0` = not labeled.

For us: bbox = the axis-aligned box that fully contains the trapezoid; the
4 keypoints are the actual panel corners in this exact order:

1. TL — top-left corner
2. TR — top-right corner
3. BR — bottom-right corner
4. BL — bottom-left corner

So one labeled object looks like:

```
0 0.742 0.518 0.312 0.441 0.602 0.310 2 0.880 0.305 2 0.895 0.760 2 0.598 0.755 2
```

Tools that support keypoint annotation: **CVAT**, **Label Studio**,
**Roboflow Annotate** (export as "YOLOv8 pose").

## Annotation rules

- Click the 4 corners of the trapezoid in the order TL, TR, BR, BL.
- The bbox should tightly enclose the four corners.
- Be consistent: TL = the corner that is top *and* left in the rendered
  image, even though the panel is tilted. Do not relabel based on perceived
  3D geometry.
- Skip mid-transition / partially hidden frames for v1.
- Cover all grid sizes 3x3 ... 8x8 evenly.

## Workflow

1. Capture screenshots into `raw/` named `<session>_<frame>.png` at 1920x1080.
2. Annotate with 4 keypoints + bbox per panel; export YOLO pose format.
3. Place images + labels per session, then run `scripts/split_dataset.py`.
4. Install deps: `pip install -r requirements.txt`.
5. Train: `python scripts/train.py`.
6. Predict: `python scripts/predict.py path/to/screenshot.png` -- saves the
   warped panel beside the input as `<name>_panel.png`.

## Target dataset size

200-500 frames, split by **capture session** (not random) so val/test
sessions are unseen. See `scripts/split_dataset.py` for session assignment.
