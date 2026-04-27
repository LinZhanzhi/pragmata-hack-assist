"""
Rename raw screenshots to the canonical `s<session>_<frame>.png` form.

Files already matching that pattern are left alone.  Any other image is
renamed to s<session>_NNNN.png where NNNN is a zero-padded frame index
assigned in sorted-by-current-filename order (so timestamped captures
keep chronological order).

Usage:
    python scripts/rename_raw.py              # session=01, .png only
    python scripts/rename_raw.py --session 02
    python scripts/rename_raw.py --dry-run

Run from the repo root or anywhere; paths are resolved relative to this
script's location.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"

CANONICAL = re.compile(r"^s\d{2}_\d{4}\.(png|jpg|jpeg)$", re.IGNORECASE)
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="01",
                        help="Two-digit session id, e.g. '01'.")
    parser.add_argument("--start", type=int, default=1,
                        help="Starting frame index for new names.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not RAW.exists():
        print(f"raw directory not found: {RAW}")
        return

    if not re.fullmatch(r"\d{2}", args.session):
        print(f"--session must be two digits, got: {args.session}")
        return

    all_files = sorted(p for p in RAW.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    to_rename = [p for p in all_files if not CANONICAL.match(p.name)]

    if not to_rename:
        print("Nothing to rename. All image files already match s##_####.<ext>.")
        return

    # Reserve frame numbers already used by canonical files in this session
    # so we don't collide with them.
    used: set[int] = set()
    prefix = f"s{args.session}_"
    for p in all_files:
        if p.name.startswith(prefix) and CANONICAL.match(p.name):
            try:
                used.add(int(p.stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                pass

    next_idx = args.start
    plan: list[tuple[Path, Path]] = []
    for src in to_rename:
        while next_idx in used:
            next_idx += 1
        used.add(next_idx)
        dst = RAW / f"s{args.session}_{next_idx:04d}{src.suffix.lower()}"
        plan.append((src, dst))
        next_idx += 1

    print(f"Renaming {len(plan)} file(s) in {RAW}:")
    for src, dst in plan:
        print(f"  {src.name}  ->  {dst.name}")
        if not args.dry_run:
            src.rename(dst)

    if args.dry_run:
        print("\n(dry run -- no files were renamed)")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
