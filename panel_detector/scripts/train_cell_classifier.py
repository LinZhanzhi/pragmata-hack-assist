"""Train a multi-class cell classifier (pure PyTorch) and emit LS predictions.

Uses a small CNN trained from scratch on cells_cls/{train,val}/<class>/.
Avoids ultralytics' subprocess spawn issues on Python 3.13.

Pipeline:
1. Read JSON-MIN export, dedupe by image, build cells_cls/{train,val}/.
   Classes with >=5 samples: 80/20 stratified split.
   Classes with <5 samples: all in train (val gets 1 sample copy).
2. Train SmallCNN on 64x64 RGB cells, 30 epochs, GPU, AdamW.
3. Predict all 5304 cells.
4. Sanity check: report disagreements vs ground-truth on the labeled set.
5. Emit LS prediction JSON for unlabeled cells in two confidence tiers.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO = Path(__file__).resolve().parents[2]
CELLS_DIR = REPO / "panel_detector" / "cells"
CLS_DIR = REPO / "panel_detector" / "cells_cls"
RUNS_DIR = REPO / "panel_detector" / "runs" / "cell_classifier"
PRED_OUT = REPO / "cells_predictions_cls.json"
PRED_OUT_REVIEW = REPO / "cells_predictions_cls_review.json"
DISAGREE_REPORT = REPO / "panel_detector" / "cls_disagreements.json"

CONF_HIGH = 0.95
CONF_REVIEW = 0.80
SEED = 42
IMG_SIZE = 64


def url_to_cell(url: str) -> str:
    qs = url.split("?d=", 1)[-1]
    decoded = urllib.parse.unquote(qs).replace("\\", "/")
    return Path(decoded).name


def cell_to_url(name: str) -> str:
    encoded = urllib.parse.quote(f"panel_detector\\cells\\{name}", safe="")
    return f"/data/local-files/?d={encoded}"


def load_labels(export_path: Path) -> dict[str, str]:
    rows = json.loads(export_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for r in rows:
        lbl = r.get("label")
        if not lbl:
            continue
        out[url_to_cell(r["image"])] = lbl
    return out


def build_dataset(labels: dict[str, str]) -> tuple[dict[str, int], dict[str, int]]:
    if CLS_DIR.exists():
        shutil.rmtree(CLS_DIR)
    by_class: dict[str, list[str]] = defaultdict(list)
    for name, lbl in labels.items():
        by_class[lbl].append(name)

    rng = random.Random(SEED)
    train_counts, val_counts = {}, {}
    for lbl, files in by_class.items():
        rng.shuffle(files)
        if len(files) >= 5:
            n_val = max(1, int(round(len(files) * 0.2)))
            val_files, train_files = files[:n_val], files[n_val:]
        else:
            val_files, train_files = [], files
        (CLS_DIR / "train" / lbl).mkdir(parents=True, exist_ok=True)
        for f in train_files:
            shutil.copy2(CELLS_DIR / f, CLS_DIR / "train" / lbl / f)
        (CLS_DIR / "val" / lbl).mkdir(parents=True, exist_ok=True)
        for f in val_files:
            shutil.copy2(CELLS_DIR / f, CLS_DIR / "val" / lbl / f)
        if not val_files:
            shutil.copy2(CELLS_DIR / train_files[0],
                         CLS_DIR / "val" / lbl / train_files[0])
            val_counts[lbl] = 1
        else:
            val_counts[lbl] = len(val_files)
        train_counts[lbl] = len(train_files)
    return train_counts, val_counts


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class CellDataset(Dataset):
    def __init__(self, root: Path, classes: list[str], augment: bool):
        self.samples = []
        self.classes = classes
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for p in (root / c).iterdir():
                if p.suffix.lower() == ".png":
                    self.samples.append((p, self.cls_to_idx[c]))
        norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
        if augment:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                norm,
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                norm,
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.tf(Image.open(p).convert("RGB")), y


def train_model(epochs: int, device: str) -> tuple[Path, list[str]]:
    classes = sorted(p.name for p in (CLS_DIR / "train").iterdir() if p.is_dir())
    train_ds = CellDataset(CLS_DIR / "train", classes, augment=True)
    val_ds = CellDataset(CLS_DIR / "val", classes, augment=False)

    counts = Counter(y for _, y in train_ds.samples)
    weights = torch.tensor(
        [1.0 / max(counts[i], 1) for i in range(len(classes))],
        dtype=torch.float32, device=device,
    )
    weights = weights / weights.sum() * len(classes)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = SmallCNN(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = RUNS_DIR / "v1_best.pt"
    best_val = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_n, tr_correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
            tr_n += x.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
        sched.step()

        model.eval()
        val_n, val_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_n += x.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
        tr_acc = tr_correct / max(tr_n, 1)
        val_acc = val_correct / max(val_n, 1)
        print(f"  ep {ep:3d}/{epochs}  loss {tr_loss/max(tr_n,1):.4f}  "
              f"train_acc {tr_acc:.4f}  val_acc {val_acc:.4f}", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, best_path)
    print(f"best val_acc: {best_val:.4f}")
    return best_path, classes


def predict_all(weights: Path, device: str):
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    classes: list[str] = ckpt["classes"]
    model = SmallCNN(len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        norm,
    ])

    files = sorted(p for p in CELLS_DIR.glob("*.png"))
    out = []
    BATCH = 256
    with torch.no_grad():
        for i in range(0, len(files), BATCH):
            chunk = files[i:i + BATCH]
            imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(device)
            probs = F.softmax(model(imgs), dim=1).cpu().numpy()
            for path, p in zip(chunk, probs):
                top = int(p.argmax())
                out.append((path.name, classes[top], float(p[top]), p))
    return classes, out


def write_ls_predictions(
    preds, labeled, out_path: Path, min_conf: float, max_conf,
) -> int:
    payload = []
    for name, top, conf, _ in preds:
        if name in labeled:
            continue
        if conf < min_conf:
            continue
        if max_conf is not None and conf >= max_conf:
            continue
        payload.append({
            "data": {"image": cell_to_url(name)},
            "predictions": [{
                "model_version": "cls-v1",
                "score": round(conf, 4),
                "result": [{
                    "from_name": "label",
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": [top]},
                }],
            }],
        })
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    export = Path(args.export).resolve()
    if not export.exists():
        sys.exit(f"export not found: {export}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    labels = load_labels(export)
    print(f"labeled cells: {len(labels)}")
    print(f"class dist: {Counter(labels.values()).most_common()}")

    weights_path = RUNS_DIR / "v1_best.pt"
    if not args.skip_train:
        tc, vc = build_dataset(labels)
        print("train:", tc)
        print("val  :", vc)
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        weights_path, _ = train_model(args.epochs, device)
    print(f"weights: {weights_path}", flush=True)

    print("predicting all cells...", flush=True)
    classes, preds = predict_all(weights_path, device)
    print(f"predicted {len(preds)} cells")

    correct = 0; n_eval = 0
    disagreements = []
    by_class_correct = Counter(); by_class_total = Counter()
    for name, top, conf, _ in preds:
        if name not in labels:
            continue
        gt = labels[name]
        n_eval += 1
        by_class_total[gt] += 1
        if gt == top:
            correct += 1
            by_class_correct[gt] += 1
        else:
            disagreements.append({"cell": name, "gt": gt, "pred": top, "conf": round(conf, 4)})
    print(f"labeled-set acc: {correct}/{n_eval} = {correct/max(n_eval,1):.4f}")
    print("per-class accuracy:")
    for c in classes:
        if by_class_total[c]:
            print(f"  {c:15s} {by_class_correct[c]}/{by_class_total[c]} "
                  f"= {by_class_correct[c]/by_class_total[c]:.3f}")
    DISAGREE_REPORT.write_text(json.dumps(disagreements, indent=2), encoding="utf-8")
    print(f"disagreements: {len(disagreements)} -> {DISAGREE_REPORT}")

    unl = [(n, t, c) for n, t, c, _ in preds if n not in labels]
    print(f"unlabeled cells: {len(unl)}")
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
    hist = Counter()
    for _, _, c in unl:
        for lo, hi in zip(bins[:-1], bins[1:]):
            if lo <= c < hi:
                hist[f"{lo:.2f}-{hi:.2f}"] += 1
                break
    print("conf histogram (unlabeled):")
    for k in sorted(hist):
        print(f"  {k}: {hist[k]}")

    high_by_class = Counter()
    for n, t, c in unl:
        if c >= CONF_HIGH:
            high_by_class[t] += 1
    print(f"unlabeled w/ conf >= {CONF_HIGH} by predicted class:")
    for c, n in high_by_class.most_common():
        print(f"  {c:15s} {n}")

    n_high = write_ls_predictions(preds, labels, PRED_OUT, CONF_HIGH, None)
    print(f"high-confidence (>= {CONF_HIGH}): {n_high} -> {PRED_OUT}")
    n_rev = write_ls_predictions(preds, labels, PRED_OUT_REVIEW, CONF_REVIEW, CONF_HIGH)
    print(f"review tier ({CONF_REVIEW}-{CONF_HIGH}): {n_rev} -> {PRED_OUT_REVIEW}")


if __name__ == "__main__":
    main()
