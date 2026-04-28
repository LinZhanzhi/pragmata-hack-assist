# Panel Detection — Status & Plan

> Updated April 28, 2026. The bulk of this document below is historical
> design notes; this top section is the live plan.

## Where we are now

Stages 1–3 of the original plan are working end-to-end on the 40
labeled training frames:

1. **Panel localization** — YOLOv8n-pose detector trained on
   `panel_detector/dataset` finds the `puzzle_panel` bbox with conf
   ≈ 0.98 on every test frame.
2. **Corner reconstruction** — Instead of trusting the (noisy) keypoint
   head, [predict.py](../panel_detector/scripts/predict.py) reconstructs
   the four puzzle corners from the bbox using two empirically-fitted,
   constant edge slopes (`TOP=-0.0907`, `BOT=+0.0409` in 1920×1080 px
   space). Mean reconstruction error ≈ 5 px / corner across all 40
   labeled frames. Result is then warped to a canonical 800×800 panel.
3. **Grid dimension inference** —
   [infer_grid.py](../panel_detector/scripts/infer_grid.py) infers
   `N ∈ {3..8}` from the warped panel using autocorrelation of grayscale
   row/column projections (the bright gridline crosshairs produce
   periodic peaks). Cols and rows vote independently and must agree.
   Verified 8/8 on the verify set.
4. **Cell slicing** —
   [slice_cells.py](../panel_detector/scripts/slice_cells.py) refits
   gridline offset+spacing at the chosen N, shrinks each cell rect by
   12% to drop the bright separator/crosshair, and dumps
   `<frame>_N<n>_r<row>_c<col>.png` (64×64) into a labeling queue.
   191 clean crops produced from the 8 verify panels.

## What's left: Stage 4 — per-cell classification

This is where new training data + a new model are needed.

### 4.1 Class taxonomy (proposed)

Based on the cell crops generated so far:

| Class           | Description                                              |
|-----------------|----------------------------------------------------------|
| `empty`         | Walkable cell (just the inner crosshair, no icon).       |
| `forbidden`     | Grey/red warning triangle — must not enter.              |
| `start`         | Green power button with the active/highlighted ring.     |
| `destination`   | Green power button without the active ring.              |
| `reward_open`   | Blue `OPEN` arrows — teleport / shortcut tile.           |
| `reward_freeze` | Yellow `FREEZE` lightning — special reward tile.         |
| `cursor`        | Bright sparkle currently always at `r0_c0` — possibly a  |
|                 | fixed UI marker; keep as its own class until we confirm. |
| `unknown`       | Transition / mid-animation frames; keep as a sink class  |
|                 | so the classifier can defer instead of misfiring.        |

Notes:

- `start` vs `destination` distinction needs to be confirmed — they may
  actually be the same icon and the role is just positional. If so,
  collapse into a single `power` class and let the solver decide which
  endpoint is which (start is the cell the player currently occupies).
- We may discover more classes (e.g. coin reward, enemy tile) once we
  see more diverse boards. Plan for adding classes without retraining
  from scratch — e.g. start with a small backbone + replaceable head.

### 4.2 Dataset

Folder layout (matches `dataset/cell_classification/` from the original
plan):

```text
panel_detector/dataset/cells/
  queue/                      # unlabeled output of slice_cells.py
  train/
    empty/
    forbidden/
    start/
    destination/
    reward_open/
    reward_freeze/
    cursor/
    unknown/
  val/
    <same class subfolders>
  test/
    <same class subfolders>
```

Labeling workflow:

1. Run `slice_cells.py` over every captured frame → fills `queue/`.
2. Manually move each PNG into the matching class folder. The filename
   already encodes `<frame>_N<n>_r<row>_c<col>` so we always know which
   board it came from for split-by-session.
3. Periodically dedupe near-identical crops (same icon, similar
   background) — for v1 prefer diversity over volume.

Target sizes for the first usable classifier:

- ≥ 100 labeled examples per class for `empty`, `forbidden`,
  `power` (or `start`/`destination`).
- ≥ 30 examples per class for the rare ones (`reward_open`,
  `reward_freeze`, `cursor`).
- Cover all observed board sizes (3×3 .. 8×8). Smaller cells at 8×8
  matter most because icons get tiny.

### 4.3 Train/val/test split

Split **by capture session**, never randomly across cells from the same
panel. Cells from one panel are highly correlated (same lighting,
same icon style) and randomly splitting them inflates val accuracy. A
sensible default:

- train: sessions 1–6
- val:   session 7
- test:  session 8

When we have only one capture session, hold out whole *frames* (not
individual cells) for val/test.

### 4.4 Model

Start tiny:

- Input: 64×64 RGB.
- Backbone: MobileNetV3-Small or a 4-conv CNN (~50 K params).
- Head: linear → `num_classes` softmax.
- Loss: cross-entropy with class weights to handle imbalance.
- Optimizer: AdamW, cosine LR, 20–50 epochs.
- Augmentations:
  - `±5°` rotation (small, the warp is already aligned).
  - brightness/contrast jitter.
  - slight scale jitter (0.9–1.1×) to mimic 3×3 vs 8×8 cell sizes.
  - random horizontal flip **only if** flipping doesn't change the
    class (probably fine for icons here, but verify on
    `reward_open` arrows).

Metrics to watch:

- Per-class precision/recall (not just overall accuracy — `empty`
  dominates the dataset).
- Confusion matrix — especially `start` vs `destination`,
  `reward_open` vs `reward_freeze`.
- Calibration: we'll use the predicted probability for the
  temporal-vote / `unknown` deferral logic.

### 4.5 New scripts to add

Once we have ≥ ~50 labeled cells/class, add:

- `scripts/train_cell_classifier.py` — load
  `dataset/cells/{train,val}`, train a small CNN, save `best.pt`.
- `scripts/predict_cells.py` — given a panel image (or a folder of
  panels), run the full pipeline (warp → infer N → slice → classify)
  and output an `N×N` label grid as JSON, plus a debug overlay PNG with
  each cell labeled.
- `scripts/eval_cells.py` — confusion matrix + per-class P/R on the
  test split, with a CLI flag to print misclassified cells for review.

### 4.6 Inference pipeline (target end-to-end)

```text
frame
  -> YOLO detect (best bbox)
  -> corners_from_bbox (fixed slopes)
  -> warpPerspective -> 800x800 panel
  -> infer_grid -> N + gridline positions
  -> slice cells (12% shrink, 64x64)
  -> cell classifier -> per-cell label + confidence
  -> NxN label grid -> solver
```

Plus a temporal-vote layer: run on 2–3 consecutive frames and majority-
vote per cell before handing the grid to the solver. Cells with
ambiguous votes or `unknown` predictions cause the system to wait one
more frame.

### 4.7 Immediate next steps (in order)

1. **Capture more frames** across diverse boards (priority: 3×3, 7×7,
   8×8 — least represented today). The current 40-frame set is
   panel-detector training data, not cell-classifier data.
2. **Run `slice_cells.py`** over the full set to populate `queue/`.
3. **Manually sort** the queue into class folders; aim for the targets
   in §4.2.
4. **Implement `train_cell_classifier.py`** and train a v1 model.
5. **Implement `predict_cells.py`** and validate end-to-end on held-out
   frames.
6. Iterate: add new classes, capture hard cases (motion blur,
   transitions), retrain.

---

# Historical design notes

The remainder of this document is the original brainstorming and
remains for context.

---

Yes — with those constraints, I’d update the recommendation quite a bit.

Because the puzzle UI stays as a **clean rectangular screen-space element with no perspective distortion**, but its **position and overall scale vary**, this is a very favorable CV problem. You probably do **not** need a heavy end-to-end model first; a two-stage pipeline should work very well: **detect/localize the puzzle panel, then normalize it and classify cells**. [github](https://github.com/MulongXie/UIED)

## What this means

Your problem is now roughly:

1. Find the puzzle panel on the right half of the screen.
2. Crop that rectangle.
3. Infer the grid dimension, such as 3×3 or 8×8.
4. Resize / normalize each cell crop to a fixed size.
5. Classify each cell as start, destination, reward, forbidden, or empty.

That structure is important because once the panel is cropped, the “3×3 cells are bigger, 8×8 cells are smaller” issue becomes much easier: after grid extraction, every cell can just be resized to the same classifier input size. [docs.unity3d](https://docs.unity3d.com/Packages/com.unity.ugui@1.0/manual/HOWTO-UIMultiResolution.html)

## Best architecture

I’d recommend this exact stack:

### Stage 1: puzzle panel localization
Use one of these:

- **Option A: detector for the whole puzzle panel**
  Train a lightweight detector to output one bounding box for the puzzle UI. [arxiv](https://arxiv.org/html/2408.03507v1)
- **Option B: CV-based rectangular panel search**
  Since the UI is neat, rectangular, and screen-aligned, you may be able to find it with contour/edge/region heuristics plus right-half prior. [dl.acm](https://dl.acm.org/doi/fullHtml/10.1145/3411764.3445784)

My practical advice:
Start with **CV heuristics first**, but if they are even a little flaky, switch quickly to a **small detector that finds only the puzzle panel**. Detecting one large rectangle is much easier than detecting many tiny symbols. [github](https://github.com/MulongXie/UIED)

### Stage 2: panel normalization
After localization:

- Crop the panel.
- Resize it to a canonical size.
- Optionally sharpen / threshold / normalize colors.
- Estimate grid lines or cell boundaries.

Because there is no distortion, you likely only need translation + uniform scaling, not homography correction. [arxiv](https://arxiv.org/html/2603.19773v1)

### Stage 3: grid size recognition
Determine whether the board is 3×3, 4×4, …, 8×8.

You can do this in two good ways:

- **Method 1: direct grid-line analysis**
  Detect repeated vertical and horizontal separators.
- **Method 2: small classifier on the whole cropped panel**
  Predict grid dimension as one of \(\{3,4,5,6,7,8\}\).

I’d try grid-line analysis first because it is interpretable and probably stable for a neat UI.

### Stage 4: per-cell classification
Once the board is sliced into \(N \times N\) cells:

- Resize each cell to fixed input size, e.g. 48×48 or 64×64.
- Run a tiny classifier for cell type.

This is still the key ML part, and it is much easier than detecting all cells directly. [linkedin](https://www.linkedin.com/learning/ai-in-connected-products-aiot/build-a-computer-vision-model-for-image-classification)

## What model to use

Given your updated description, my preferred model choices are:

| Subproblem | Best first choice | Backup choice |
|---|---|---|
| Find puzzle panel | CV rectangle search | YOLOv8n / nano detector for one panel box. [github](https://github.com/MulongXie/UIED) |
| Detect grid dimension | Grid-line / separator analysis | Small classifier on whole panel crop |
| Recognize tile type per cell | Tiny CNN / MobileNet-style classifier | Template matching with multi-scale support. [docs.adaptive-vision](https://docs.adaptive-vision.com/5.5/studio/machine_vision_guide/TemplateMatching.html) |

## Should you use training data from game images?

Yes, definitely for the **cell classifier**, and maybe for the **panel detector** if heuristic localization is not reliable. [betterprogramming](https://betterprogramming.pub/how-to-train-yolov5-for-recognizing-custom-game-objects-in-real-time-9d78369928a8)

Real game images matter because your panel scale changes and the cell graphics will be rendered at different sizes depending on board dimension. Training on real crops from 3×3 through 8×8 boards will help the model learn those appearance shifts. [arxiv](https://arxiv.org/abs/1608.01745)

## How I would train it

## Panel detector dataset

Only if needed.

- Label full frames with one bounding box around the puzzle UI.
- Collect examples across:
  - different board dimensions,
  - slight UI position shifts,
  - different scenes / combat lighting,
  - partial animation states.

This is a very small annotation burden because it is just one box per image. [betterprogramming](https://betterprogramming.pub/how-to-train-yolov5-for-recognizing-custom-game-objects-in-real-time-9d78369928a8)

Model:
- YOLOv8n-class lightweight detector is reasonable here because the target is one medium-large UI object, not a tiny free-form object. [arxiv](https://arxiv.org/html/2408.03507v1)

## Cell classifier dataset

This is the main one.

For every panel crop:

1. Record the true grid size.
2. Slice the board into cells.
3. Label each cell:
   - `start`
   - `destination`
   - `reward`
   - `forbidden`
   - `empty`
   - optional `unknown` / `transition`

Important: include examples from **all board sizes**, because the appearance of each tile at 3×3 vs 8×8 can differ in effective resolution.

A simple and strong training setup is:

- Input: 64×64 RGB crop.
- Model: MobileNetV3-small or similar tiny CNN.
- Loss: multiclass cross-entropy.
- Metric: per-class precision/recall, not just overall accuracy.

## Why this is better than one big detector

A one-shot detector that tries to detect every tile in the full frame would have to handle:

- panel localization,
- varying board size,
- many small objects,
- class recognition,

all at once. Small object detection across scales is a known difficulty, especially when symbols get tiny. Splitting the problem into localization then cell classification is usually more data-efficient and easier to debug. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2603.19773)

## Data collection advice

Collect data intentionally across board dimensions:

- 3×3
- 4×4
- 5×5
- 6×6
- 7×7
- 8×8

And balance tile classes across them. If most of your data is 5×5 or 6×6, the model may underperform on 8×8 where symbols are smallest.

Also save hard cases:

- motion blur,
- transition frames,
- glow effects,
- partial overlays,
- low contrast scenes.

Then either exclude transition frames from training or label them as `unknown` so the system can wait one more frame instead of making a bad decision.

## Inference strategy

For responsiveness and robustness, I’d use:

1. Read current frame.
2. Detect or crop puzzle panel.
3. Infer grid size.
4. Slice cells.
5. Classify cells.
6. Repeat for 2–3 consecutive frames.
7. Use majority vote / confidence averaging before sending to solver.

That temporal smoothing is cheap and often removes single-frame mistakes without much latency penalty.

## My concrete suggestion

If I were building this now, I would do:

- **First prototype**
  - Hardcode “search right half of screen”
  - Use CV heuristics to find the rectangular puzzle panel
  - Use line/separator analysis to estimate \(N\)
  - Slice cells
  - Use template matching as a baseline classifier

- **Second prototype**
  - Replace template matching with a tiny cell classifier trained on real game cell crops
  - Keep the rest of the pipeline deterministic

- **Third prototype only if needed**
  - Add a lightweight detector for panel localization

So yes, I agree with using computer vision and training on game images, but I would train **a small cell classifier first**, not an end-to-end big model. The varying board dimension changes cell scale, but because the panel is clean and undistorted, that is still very manageable with “localize → normalize → slice → classify.” [voxel51](https://voxel51.com/blog/how-to-detect-small-objects)

If you want, I can next design the exact labeling format and folder structure for your dataset, including what files to save from each captured game frame.
Yes — the next concrete step is to prepare a dataset of puzzle screenshots and annotate the **puzzle region** first. For your first pipeline, I would train only a **single-class detector for the puzzle panel**, then build the rest deterministically from the crop. [docs.ultralytics](https://docs.ultralytics.com/datasets/detect/)

That gives you the fastest usable sensor pipeline: **frame → puzzle box → crop → infer grid → slice cells**. It keeps annotation cheap because each training image only needs one tight box around the whole puzzle UI. [roboflow](https://roboflow.com/how-to-label/yolov8-obb)

## What to annotate first

For the first model, annotate only:

- **Class name:** `puzzle_panel`
- **Annotation type:** one axis-aligned bounding box per image
- **Target:** the full visible puzzle UI rectangle, tightly around the panel edges. [docs.ultralytics](https://docs.ultralytics.com/guides/data-collection-and-annotation/)

Since you said the UI is neat and not rotated, you do **not** need polygons or oriented boxes at this stage. Plain bounding boxes are enough. [roboflow](https://roboflow.com/how-to-label/yolov8-obb)

## How to collect images

Use real gameplay captures and sample frames that cover the variation you expect:

- Different grid sizes: 3×3 through 8×8.
- Different puzzle positions in the right half.
- Different combat scenes / backgrounds.
- Slight animation states.
- Different resolutions or graphics settings if you plan to support them. [docs.ultralytics](https://docs.ultralytics.com/guides/data-collection-and-annotation/)

Do **not** save every frame from a video. Adjacent frames are highly redundant, so prefer diverse frames from different sessions or clips. [lightly](https://www.lightly.ai/blog/video-segmentation)

A good starter target is:

- **200–500 full-frame images** for panel detection.
- Make sure all board sizes appear.
- Include some “hard” samples where contrast is worse or the scene is visually busy.

## How to annotate

Use any standard bounding-box annotation tool such as LabelImg, Roboflow Annotate, or Label Studio; these tools support YOLO-compatible export formats. [labelstud](https://labelstud.io/blog/object-detection-with-yolov8/)

Labeling rules:

- Draw the box **tightly** around the full puzzle panel. [roboflow](https://roboflow.com/how-to-label/yolov8-obb)
- Be consistent about edges — don’t include lots of extra padding in some images and none in others.
- If the panel is partially hidden or mid-transition, either:
  - exclude that frame from v1, or
  - label it only if you truly want the detector to handle partial states.
- If the puzzle is absent, either:
  - leave the image unlabeled and keep it as a negative example, or
  - maintain a separate set of no-puzzle frames for testing false positives.

For your first version, I’d mostly use **clear fully visible puzzle frames** and a smaller set of negative frames.

## Suggested folder structure

A clean project dataset layout:

```text
dataset/
  detection_panel/
    raw/
      session_01_0001.png
      session_01_0042.png
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
    data.yaml
```

If you later add per-cell classification, keep that as a separate dataset:

```text
dataset/
  cell_classification/
    train/
      reward/
      forbidden/
      start/
      destination/
      empty/
    val/
    test/
```

This separation is helpful because detection and classification are different tasks. [github](https://github.com/ultralytics/ultralytics/issues/13754)

## Train/val/test split

Do **not** randomly split near-duplicate frames from the same short clip into all splits, because that inflates validation performance. [innovatiana](https://www.innovatiana.com/en/post/video-classification)

Instead:

- Split by **capture session** or **video chunk**.
- Example:
  - train: sessions 1–6
  - val: session 7
  - test: session 8

This makes evaluation much more honest.

## YOLO label format

If you use YOLO-style detection, each image gets a `.txt` label file with:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to \([0,1]\). [scribd](https://www.scribd.com/document/808135422/Tra-Ning)

Since you only have one class:

- `puzzle_panel = 0`

Example:

```text
0 0.742 0.518 0.312 0.441
```

And your `data.yaml` would look like:

```yaml
path: /path/to/dataset/detection_panel
train: images/train
val: images/val
test: images/test

names:
  0: puzzle_panel
```

This structure matches standard Ultralytics YOLO dataset organization. [docs.ultralytics](https://docs.ultralytics.com/datasets/detect/)

## What model to use first

For the first pipeline, use a lightweight detector such as a **nano** detection model. A small YOLO-family model is a practical starting point for custom detection and works well with standard dataset tooling. [labelstud](https://labelstud.io/blog/object-detection-with-yolov8/)

Because you are detecting one medium/large rectangular UI component, this is much easier than small-object detection. You do not need a large model first.

Good first choice:

- **YOLO nano / small class model**
- Input size around 640
- One class only: `puzzle_panel`

## How to train the first pipeline

### Step 1: train panel detector
Train the detector only to return the puzzle panel bounding box. [docs.ultralytics](https://docs.ultralytics.com/datasets/detect/)

### Step 2: crop the panel
At inference time:

- Run detector on full screenshot.
- Keep the best `puzzle_panel` box.
- Crop that region.
- Optionally add a tiny margin around the box.

### Step 3: normalize the crop
Because the panel is not distorted:

- Resize crop to standard size.
- Optionally normalize brightness / contrast.
- Run edge or line analysis to infer board size.

### Step 4: infer grid dimension
Use deterministic CV here first:

- Detect repeated separators / lines.
- Count rows and columns.
- Infer whether board is 3×3 to 8×8.

You do **not** need to train this part first unless the separators are hard to recover.

### Step 5: slice cells
Once \(N\) is known:

- Divide the normalized crop evenly into \(N \times N\) cells.
- Export each cell crop.

### Step 6: build the second dataset
Now save those cell crops into a review queue and label them for:
- start
- destination
- reward
- forbidden
- empty

That becomes your next model’s dataset.

## Practical first milestone

A very good v1 milestone is:

- Input: full screenshot
- Output:
  - puzzle panel box
  - cropped panel image
  - estimated grid size
  - visual overlay showing the sliced grid

If you can do this reliably, the project becomes much easier because all later work happens inside normalized cell crops.

## Annotation tips that will save time

- Keep naming simple and sequential.
- Store source session ID in filenames.
- Save a small metadata CSV with session, resolution, puzzle size, and notes.
- Start with high-quality obvious frames before adding edge cases.
- Review annotations after every 50–100 images for consistency. [docs.ultralytics](https://docs.ultralytics.com/guides/data-collection-and-annotation/)

## My recommendation

I suggest this exact order:

1. Capture 300–500 diverse puzzle screenshots.
2. Annotate one `puzzle_panel` box per image.
3. Train a lightweight panel detector.
4. Build crop + grid-size inference.
5. Save auto-sliced cell crops.
6. Start labeling cells for the second-stage classifier.

That is the lowest-risk way to get a functioning first sensor pipeline quickly. [docs.ultralytics](https://docs.ultralytics.com/datasets/detect/)

If you want, I can next give you:
- a concrete `data.yaml`,
- a recommended directory structure,
- and a short Python training script for the panel detector.
