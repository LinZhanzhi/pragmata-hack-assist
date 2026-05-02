"""Diagnose Label Studio task duplication.

Counts annotated vs unlabeled tasks per data-key style.
"""

import json
import sqlite3
from collections import Counter

DB = r"D:\pathSolver\panel_detector\labelstudio\.lsdata\label_studio.sqlite3"

con = sqlite3.connect(DB)
cur = con.cursor()

rows = cur.execute("select id, project_id, data from task where project_id=3").fetchall()
buckets: dict[str, list[int]] = {"$undefined$": [], "image": [], "other": []}
for tid, pid, data in rows:
    d = json.loads(data)
    if "$undefined$" in d:
        buckets["$undefined$"].append(tid)
    elif "image" in d:
        buckets["image"].append(tid)
    else:
        buckets["other"].append(tid)

print("data-key   | n_tasks | n_annotated | n_predicted")
for k, ids in buckets.items():
    n = len(ids)
    if n == 0:
        continue
    placeholders = ",".join("?" * len(ids))
    n_ann = cur.execute(
        f"select count(distinct task_id) from task_completion where task_id in ({placeholders})",
        ids,
    ).fetchone()[0]
    n_pred = cur.execute(
        f"select count(distinct task_id) from prediction where task_id in ({placeholders})",
        ids,
    ).fetchone()[0]
    print(f"{k:10s} | {n:7d} | {n_ann:11d} | {n_pred:11d}")


def img_of(data: str) -> str:
    d = json.loads(data)
    return d.get("image") or d.get("$undefined$") or ""


undef_imgs = set(img_of(data) for tid, pid, data in rows if "$undefined$" in json.loads(data))
image_imgs = set(img_of(data) for tid, pid, data in rows if "image" in json.loads(data))
print()
print(f"distinct image URLs in $undefined$ bucket: {len(undef_imgs)}")
print(f"distinct image URLs in image bucket      : {len(image_imgs)}")
print(f"present in BOTH buckets                  : {len(undef_imgs & image_imgs)}")
print(f"only in $undefined$ (untouched originals): {len(undef_imgs - image_imgs)}")
print(f"only in image bucket  (orphan imports)   : {len(image_imgs - undef_imgs)}")
