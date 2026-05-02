"""Generic LS task deduplicator.

Merges duplicate tasks (same data.image) into the one with a storage link
(the 'original'). Moves annotations/predictions, deletes the duplicate.
Run with LS server stopped.
"""

import json
import sqlite3
from collections import defaultdict

DB = r"D:\pathSolver\panel_detector\labelstudio\.lsdata\label_studio.sqlite3"
PROJECT_ID = 3

con = sqlite3.connect(DB)
con.execute("PRAGMA foreign_keys = ON;")
cur = con.cursor()

# Group tasks by image URL.
by_url: dict[str, list[int]] = defaultdict(list)
for tid, data in cur.execute("select id, data from task where project_id=?", (PROJECT_ID,)):
    d = json.loads(data)
    img = d.get("image") or d.get("$undefined$")
    if img:
        by_url[img].append(tid)

groups = {u: ids for u, ids in by_url.items() if len(ids) > 1}
print(f"total tasks: {sum(len(v) for v in by_url.values())}")
print(f"groups with duplicates: {len(groups)}")

# Identify originals (the one with a storage link). Fall back to lowest id.
storage_owners: dict[int, int] = {}
for row in cur.execute(
    "select task_id, id from io_storages_localfilesimportstoragelink"
):
    storage_owners[row[0]] = row[1]

moves: list[tuple[int, int]] = []  # (dup_id, orig_id)
no_storage_groups = 0
for url, ids in groups.items():
    originals = [i for i in ids if i in storage_owners]
    if originals:
        orig = originals[0]
    else:
        orig = min(ids)
        no_storage_groups += 1
    for i in ids:
        if i != orig:
            moves.append((i, orig))

print(f"groups w/o storage link: {no_storage_groups}")
print(f"will merge {len(moves)} dup tasks")

REPOINT = [
    ("task_completion", "task_id"),
    ("prediction", "task_id"),
    ("tasks_annotationdraft", "task_id"),
    ("tasks_failedprediction", "task_id"),
    ("fsm_taskstate", "task_id"),
]


def col_exists(t, c):
    return any(r[1] == c for r in cur.execute(f"PRAGMA table_info({t})"))


for table, col in REPOINT:
    if not col_exists(table, col):
        continue
    n = 0
    for dup, orig in moves:
        n += cur.execute(
            f"update {table} set {col}=? where {col}=?", (orig, dup)
        ).rowcount
    print(f"  repointed {n} rows in {table}")

if col_exists("tasks_tasklock", "task_id"):
    n = 0
    for dup, _ in moves:
        n += cur.execute("delete from tasks_tasklock where task_id=?", (dup,)).rowcount
    print(f"  deleted {n} task locks")

if col_exists("io_storages_localfilesimportstoragelink", "task_id"):
    n = 0
    for dup, _ in moves:
        n += cur.execute(
            "delete from io_storages_localfilesimportstoragelink where task_id=?",
            (dup,),
        ).rowcount
    print(f"  deleted {n} storage links on dup tasks")

n_del = 0
for dup, _ in moves:
    n_del += cur.execute("delete from task where id=?", (dup,)).rowcount
print(f"deleted {n_del} duplicate tasks")

# Refresh denormalized counters on remaining tasks.
cur.execute("""
update task set
  total_annotations = (select count(*) from task_completion tc
                       where tc.task_id=task.id and (tc.was_cancelled=0 or tc.was_cancelled is null)),
  cancelled_annotations = (select count(*) from task_completion tc
                           where tc.task_id=task.id and tc.was_cancelled=1),
  total_predictions = (select count(*) from prediction p where p.task_id=task.id)
where project_id=?
""", (PROJECT_ID,))
cur.execute("update task set is_labeled=(total_annotations >= overlap) where project_id=?", (PROJECT_ID,))

con.commit()

print()
print("=== POST-FIX ===")
print("total tasks:", cur.execute("select count(*) from task where project_id=?", (PROJECT_ID,)).fetchone()[0])
print("annotated  :", cur.execute("select count(*) from task where project_id=? and total_annotations>0", (PROJECT_ID,)).fetchone()[0])
con.close()
