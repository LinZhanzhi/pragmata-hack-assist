# Label Studio re-annotation project (panel detector v2)

## Goal
Re-train the panel detector on all 259 raw frames so we stop getting
false-positive detections on HUD/standby/menu screens. We're using a
**bbox-only** label config (no keypoints) because `predict.py` reconstructs
the 4 corners from the bbox via fixed slopes — the previous keypoint head
was unreliable anyway.

## One-time setup

1. Generate `tasks.json` and `labeling_config.xml`:
   ```pwsh
   python panel_detector/scripts/prepare_labelstudio.py
   ```
   This pre-fills 40 of the 259 tasks with the existing bounding-box
   labels (keypoints dropped per request).

2. Start the server (sets the Local Files env vars first):
   ```pwsh
   pwsh panel_detector/scripts/start_label_studio.ps1
   ```
   First launch will prompt you to create an admin account — pick anything.

## Project setup in the UI (one time, ~30 seconds)

Open http://localhost:8080 and:

1. **Create Project** → name it `panel-detector-v2`.
2. **Labeling Setup** → click *Code* and paste the contents of
   `panel_detector/labelstudio/labeling_config.xml`. Save.
3. **Settings → Cloud Storage → Add Source Storage → Local files**:
   - Storage Title: `raw`
   - Absolute local path: `<repo>/panel_detector/raw`
   - File Filter Regex: `^s01_\d{4}\.png$`  (excludes `*_panel.png`)
   - Click *Add Storage*, then *Sync Storage*.
4. **Import → Upload Files** → choose `panel_detector/labelstudio/tasks.json`.
   This creates 259 tasks; 40 of them will already show a green bbox you
   can accept or edit.

## Labeling guidance

- **Real panels** — drag a tight bounding box around the puzzle panel.
  The 40 pre-annotated frames should already be correct; just submit
  them as-is unless you spot a bad box.
- **No-panel frames** (HUD only, standby screens, REFramework menu, etc.):
  submit with **no boxes**. Empty annotations are valid negatives and
  will teach the model to stop firing on those scenes.

## Export → train

When done:

1. **Export** → format `YOLO` → download zip.
2. Replace the `dataset/images` and `dataset/labels` contents.
3. Run `python panel_detector/scripts/split_dataset.py`.
4. Retrain with `python panel_detector/scripts/train.py`.
