"""Fix Label Studio task duplication caused by mixing $undefined$ and image keys.

Problem:
- The original 5304 tasks were synced from Local Files Storage with data
  key `$undefined$` (sync ran before the labeling config defined the
  variable name).
- Subsequent prediction-JSON imports used data key `image`, so LS could
  not match them to existing tasks and instead created duplicates.
- We end up with 5304 originals + 2079 duplicates = 7383 tasks. Each
  duplicate carries the user-accepted annotation; the original is
  un-annotated.

Fix:
1. For every `image`-keyed task, find the matching `$undefined$` original
   (same image URL).
2. Move annotations and predictions from the duplicate to the original.
3. Delete the duplicate task.
4. Rewrite all remaining `$undefined$` keys to `image` so future imports
   line up.

Pre-condition: the LS server is stopped (DB not locked).
The DB has already been backed up.
"""

import json
import sqlite3
from collections import defaultdict

DB = r"D:\pathSolver\panel_detector\labelstudio\.lsdata\label_studio.sqlite3"

con = sqlite3.connect(DB)
con.execute("PRAGMA foreign_keys = ON;")
cur = con.cursor()


def img_of(data: str) -> str:
    d = json.loads(data)
    return d.get("image") or d.get("$undefined$") or ""


# Index every project-3 task by its image URL, separated by data-key style.
undef_by_img: dict[str, int] = {}
image_by_img: dict[str, list[int]] = defaultdict(list)
for tid, data in cur.execute("select id, data from task where project_id=3"):
    d = json.loads(data)
    if "$undefined$" in d:
        undef_by_img[d["$undefined$"]] = tid
    elif "image" in d:
        image_by_img[d["image"]].append(tid)

print(f"undef tasks: {len(undef_by_img)}")
print(f"image tasks: {sum(len(v) for v in image_by_img.values())}")

# Build merge plan: for every image-keyed dup, find the matching original.
moves = []  # (dup_id, orig_id)
unmatched: list[int] = []
for url, dup_ids in image_by_img.items():
    orig = undef_by_img.get(url)
    if orig is None:
        unmatched.extend(dup_ids)
        continue
    for dup in dup_ids:
        moves.append((dup, orig))

print(f"will merge: {len(moves)} dup tasks -> originals")
print(f"unmatched dup tasks (no original): {len(unmatched)}")

if unmatched:
    print("First few unmatched:")
    for tid in unmatched[:5]:
        d = cur.execute("select data from task where id=?", (tid,)).fetchone()
        print(" ", tid, d)
    raise SystemExit("Aborting: unmatched duplicates exist; investigate first.")


# Re-point everything that references the dup task to the original.
# Tables to update (FKs into task.id):
#   task_completion.task_id          (annotations)
#   prediction.task_id
#   tasks_annotationdraft.task_id
#   tasks_failedprediction.task_id
#   tasks_tasklock.task_id
#   io_storages_localfilesimportstoragelink.task_id
#   ml_backend* prediction-job tables -- skip; auto-cleared on cascade
TASK_FK_TABLES = [
    "task_completion",
    "prediction",
    "prediction_meta",      # only relevant if FK points to prediction; safe no-op otherwise
    "tasks_annotationdraft",
    "tasks_failedprediction",
    "tasks_tasklock",
    "fsm_taskstate",
    "data_export_export",   # only has a many-to-many; ignore if column missing
]

# We do the simple ones explicitly; let CASCADE handle the rest on delete.
SAFE_REPOINT_TABLES = [
    ("task_completion", "task_id"),
    ("prediction", "task_id"),
    ("tasks_annotationdraft", "task_id"),
    ("tasks_failedprediction", "task_id"),
    ("fsm_taskstate", "task_id"),
]


def column_exists(table: str, col: str) -> bool:
    rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == col for r in rows)


repoint_counts: dict[str, int] = {}
for table, col in SAFE_REPOINT_TABLES:
    if not column_exists(table, col):
        continue
    n = 0
    for dup, orig in moves:
        n += cur.execute(
            f"update {table} set {col}=? where {col}=?", (orig, dup)
        ).rowcount
    repoint_counts[table] = n
    print(f"  repointed {n} rows in {table}")

# Delete tasklocks on dup ids (no need to move; they'll re-acquire if needed).
if column_exists("tasks_tasklock", "task_id"):
    n = 0
    for dup, _ in moves:
        n += cur.execute("delete from tasks_tasklock where task_id=?", (dup,)).rowcount
    print(f"  deleted {n} task locks on dup tasks")

# Delete the storage link rows pointing at dup tasks (LS will recreate
# them on next sync if needed).
if column_exists("io_storages_localfilesimportstoragelink", "task_id"):
    n = 0
    for dup, _ in moves:
        n += cur.execute(
            "delete from io_storages_localfilesimportstoragelink where task_id=?",
            (dup,),
        ).rowcount
    print(f"  deleted {n} storage links on dup tasks")

# Now delete the duplicate tasks themselves.
n_del = 0
for dup, _ in moves:
    n_del += cur.execute("delete from task where id=?", (dup,)).rowcount
print(f"deleted {n_del} duplicate tasks")

# Finally normalize: rewrite $undefined$ -> image on remaining tasks.
n_norm = 0
for tid, data in cur.execute("select id, data from task where project_id=3").fetchall():
    d = json.loads(data)
    if "$undefined$" in d and "image" not in d:
        d = {"image": d["$undefined$"]}
        cur.execute("update task set data=? where id=?", (json.dumps(d), tid))
        n_norm += 1
print(f"normalized {n_norm} task data keys ($undefined$ -> image)")

con.commit()

# Final sanity counts.
print()
print("=== POST-FIX ===")
print("total tasks:", cur.execute("select count(*) from task where project_id=3").fetchone()[0])
print("annotated tasks:", cur.execute(
    "select count(distinct task_id) from task_completion where task_id in "
    "(select id from task where project_id=3)").fetchone()[0])
con.close()
