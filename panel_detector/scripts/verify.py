"""Quick verification of the trained panel detector.

Runs best.pt against the test split and the labeled negatives,
printing detection counts and confidences.
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "runs" / "panel_detector" / "weights" / "best.pt"

NEG_STEMS = [
    "s01_0141", "s01_0143", "s01_0148", "s01_0183", "s01_0184",
    "s01_0189", "s01_0190", "s01_0191", "s01_0248",
]


def main() -> None:
    model = YOLO(str(WEIGHTS))

    test_imgs = sorted((ROOT / "dataset/images/test").glob("*.png"))
    print(f"=== TEST SPLIT ({len(test_imgs)} images) ===")
    results = model.predict(source=[str(p) for p in test_imgs], conf=0.25, verbose=False)
    fp = fn = tp = tn = 0
    for p, r in zip(test_imgs, results):
        n = len(r.boxes)
        confs = [round(c.item(), 3) for c in r.boxes.conf] if n else []
        lbl = ROOT / "dataset/labels/test" / f"{p.stem}.txt"
        is_neg = lbl.exists() and lbl.stat().st_size == 0
        tag = "[NEG]" if is_neg else "[POS]"
        if is_neg:
            if n == 0:
                tn += 1
            else:
                fp += 1
        else:
            if n >= 1:
                tp += 1
            else:
                fn += 1
        print(f"  {tag} {p.name}: {n} det, conf={confs}")
    print(f"  -> TP={tp} TN={tn} FP={fp} FN={fn}")

    neg_imgs = [ROOT / "raw" / f"{s}.png" for s in NEG_STEMS]
    neg_imgs = [p for p in neg_imgs if p.exists()]
    print()
    print(f"=== ALL LABELED NEGATIVES FROM RAW ({len(neg_imgs)} images) ===")
    results = model.predict(source=[str(p) for p in neg_imgs], conf=0.25, verbose=False)
    fp = 0
    for p, r in zip(neg_imgs, results):
        n = len(r.boxes)
        confs = [round(c.item(), 3) for c in r.boxes.conf] if n else []
        if n > 0:
            fp += 1
        print(f"  {p.name}: {n} det, conf={confs}")
    print(f"  -> false positives on negatives: {fp}/{len(neg_imgs)}")


if __name__ == "__main__":
    main()
