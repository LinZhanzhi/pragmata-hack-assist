# Puzzle Panel Detector

Stage-1 of the path-solver vision pipeline: detect the puzzle UI panel in a full game screenshot.

## Pipeline goal

```
full screenshot --> [panel detector] --> bounding box --> crop --> (later stages: grid size, cells)
```

Single class: `puzzle_panel`.

## Folder structure

```
panel_detector/
  raw/                     # original screenshots straight from capture (any size)
    metadata.csv           # session, resolution, grid size, notes
  dataset/
    images/
      train/               # training images
      val/                 # validation images
      test/                # held-out test images
    labels/
      train/               # YOLO .txt label files (one per image)
      val/
      test/
  data.yaml                # YOLO dataset descriptor
  scripts/
    split_dataset.py       # split raw -> train/val/test by session
    train.py               # train YOLO panel detector
    predict.py             # run trained model on a screenshot, save crop
  runs/                    # YOLO output (auto-created on training)
  models/                  # final exported weights
  README.md                # this file
```

## Workflow

1. Capture screenshots into `raw/`. Keep filenames `<session>_<frame>.png`.
2. Annotate each image with one `puzzle_panel` box using LabelImg / Roboflow / Label Studio. Export YOLO format.
3. Place images + labels per session, then run `scripts/split_dataset.py` to fill `dataset/{images,labels}/{train,val,test}`.
4. Train: `python scripts/train.py`.
5. Predict on a new screenshot: `python scripts/predict.py path/to/img.png`.

## Annotation rules

- One tight axis-aligned box around the full puzzle panel.
- Class id `0` = `puzzle_panel`.
- Skip mid-transition / partially hidden frames for v1.
- Cover all grid sizes 3x3 ... 8x8 evenly.

## Target dataset size

200-500 frames, split by **capture session** (not random) so val/test sessions are unseen.
