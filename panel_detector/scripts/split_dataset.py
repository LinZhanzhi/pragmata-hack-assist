"""
Build the YOLO dataset directory from a Label Studio YOLO export.

Reads images from `panel_detector/raw/s01_NNNN.png` and labels from
`<export>/labels/s01_NNNN.txt`. Splits 80/10/10 with a fixed seed,
stratified so the small set of negative samples (frames where the panel
is absent) is spread proportionally across train/val/test.

Output layout (cleared on each run):
    panel_detector/dataset/images/{train,val,test}/<frame>.png
    panel_detector/dataset/labels/{train,val,test}/<frame>.txt

Run:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --export project-1-at-2026-05-01-17-41-374fc67b
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
RAW = ROOT / "raw"
DATASET = ROOT / "dataset"

SEED = 42
SPLITS = (("train", 0.80), ("val", 0.10), ("test", 0.10))
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def find_export(arg: str | None) -> Path:
    """Locate the LS YOLO export folder.  If --export is omitted, pick the
    most recent `project-*-*` folder at the repo root."""
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = REPO / p
        return p
    candidates = sorted(REPO.glob("project-*-at-*-*"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        raise SystemExit("No LS export folder found at repo root "
                         "(looking for project-*-at-*-*).")
    return candidates[0]


def stratified_split(positives: list[Path], negatives: list[Path]):
    """Returns {split: [image_paths, ...]} with negatives spread
    proportionally across splits."""
    rng = random.Random(SEED)
    out: dict[str, list[Path]] = {name: [] for name, _ in SPLITS}
    for bucket in (positives, negatives):
        items = list(bucket)
        rng.shuffle(items)
        n = len(items)
        cuts = []
        running = 0
        for name, frac in SPLITS[:-1]:
            count = int(round(frac * n))
            cuts.append((name, running, running + count))
            running += count
        cuts.append((SPLITS[-1][0], running, n))
        for name, lo, hi in cuts:
            out[name].extend(items[lo:hi])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", help="LS YOLO export folder (default: newest "
                                     "project-*-at-*-* at repo root)")
    args = ap.parse_args()

    export = find_export(args.export)
    labels_dir = export / "labels"
    if not labels_dir.is_dir():
        raise SystemExit(f"Missing labels dir: {labels_dir}")

    # Reset the dataset tree so stale files from previous runs can't leak
    # across splits.
    for kind in ("images", "labels"):
        for split, _ in SPLITS:
            d = DATASET / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    # Collect (image_path, label_path) pairs by matching label stems.
    positives: list[tuple[Path, Path]] = []
    negatives: list[tuple[Path, Path]] = []
    missing_image: list[str] = []
    for lbl in sorted(labels_dir.glob("*.txt")):
        stem = lbl.stem
        img = None
        for ext in IMAGE_EXTS:
            cand = RAW / f"{stem}{ext}"
            if cand.exists():
                img = cand
                break
        if img is None:
            missing_image.append(stem)
            continue
        if lbl.stat().st_size == 0:
            negatives.append((img, lbl))
        else:
            positives.append((img, lbl))

    if missing_image:
        head = missing_image[:5]
        tail = "..." if len(missing_image) > 5 else ""
        print(f"WARN: {len(missing_image)} label(s) without matching image "
              f"(skipped): {head}{tail}")

    print(f"export    : {export}")
    print(f"positives : {len(positives)}")
    print(f"negatives : {len(negatives)}")

    splits = stratified_split(
        [pair[0] for pair in positives],
        [pair[0] for pair in negatives],
    )
    label_by_stem = {img.stem: lbl for img, lbl in positives + negatives}

    counts = {name: {"pos": 0, "neg": 0} for name, _ in SPLITS}
    for split_name, _ in SPLITS:
        for img in splits[split_name]:
            lbl = label_by_stem[img.stem]
            shutil.copy2(img, DATASET / "images" / split_name / img.name)
            shutil.copy2(lbl, DATASET / "labels" / split_name / lbl.name)
            if lbl.stat().st_size == 0:
                counts[split_name]["neg"] += 1
            else:
                counts[split_name]["pos"] += 1

    print("\nsplit counts (pos / neg / total):")
    for name, _ in SPLITS:
        c = counts[name]
        print(f"  {name:5s}  {c['pos']:3d} / {c['neg']:2d} / "
              f"{c['pos']+c['neg']:3d}")


if __name__ == "__main__":
    main()
