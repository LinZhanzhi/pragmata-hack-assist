"""Rewrite Label Studio Local Files Storage paths after moving the repo
between machines.

The actual `path` column lives on the mixin table
`io_storages_localfilesmixin` shared by both import and export storages.
We rewrite any non-existent path that contains 'panel_detector' to the
equivalent folder under the current repo root.

Usage:
    python panel_detector/scripts/fix_ls_storage_paths.py            # dry run
    python panel_detector/scripts/fix_ls_storage_paths.py --apply    # write
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "labelstudio" / ".lsdata" / "label_studio.sqlite3"
REPO = Path(__file__).resolve().parents[2]
TABLE = "io_storages_localfilesmixin"


def remap(old):
    if not old:
        return None
    parts = Path(old).parts
    if "panel_detector" in parts:
        idx = parts.index("panel_detector")
        return str(REPO.joinpath(*parts[idx:]))
    return str(REPO / "panel_detector" / "raw")


def main(apply):
    if not DB.exists():
        sys.exit(f"DB not found: {DB}")
    con = sqlite3.connect(DB)
    rows = list(con.execute(f"SELECT id, path, regex_filter FROM {TABLE}"))
    if not rows:
        print("No storage rows.")
        return
    changed = 0
    for rid, path, rx in rows:
        ok = bool(path) and Path(path).exists()
        new_path = path if ok else remap(path)
        marker = "OK" if ok else ("FIX" if new_path != path else "??")
        print(f"  id={rid} [{marker}]  regex={rx!r}")
        print(f"    current: {path}   exists={ok}")
        if not ok:
            print(f"    new    : {new_path}   exists={Path(new_path).exists() if new_path else False}")
            if apply and new_path:
                con.execute(f"UPDATE {TABLE} SET path=? WHERE id=?", (new_path, rid))
                changed += 1
    if apply:
        con.commit()
        print(f"\nUpdated {changed} row(s).")
    else:
        print("\n(dry-run; pass --apply to write changes)")
    con.close()


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
