"""clean_duplicates.py -- hash-group the raw coin image tree, collapse exact duplicates, and
drop images whose bytes were filed under more than one emperor label.

check_split_leakage.py found the failure mode; this fixes it at the source, before any split
ever sees the data. Two things can be true of a hash group (2+ files with identical bytes):

  * duplicate      -- same bytes, SAME label, different filename/folder-instance (a re-upload,
                       a resize-and-resave, a copy). Keep exactly one canonical copy -- the
                       lexicographically smallest relative path, so re-running is deterministic
                       and picks the same survivor every time -- drop the rest.
  * label_conflict -- same bytes, DIFFERENT labels. We don't know which label is right, so we
                       don't guess: every copy in the group is dropped and logged for a human.

Read-only by default: writes a clean file list + a drop log, touches nothing in data_dir. Pass
--quarantine-dir to actually MOVE dropped files out of the training tree (a reversible move, not
a delete) so discover_dataset's glob naturally stops seeing them.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from coin_clf.data import folder_label
from coin_clf.hashing import file_hash


def hash_groups(data_dir: Path) -> dict[str, list[Path]]:
    filepaths = sorted(data_dir.glob("*/side_a/*.jpg"))
    groups: dict[str, list[Path]] = defaultdict(list)
    for i, p in enumerate(filepaths):
        groups[file_hash(p)].append(p)
        if (i + 1) % 5000 == 0:
            print(f"  hashed {i + 1}/{len(filepaths)}", file=sys.stderr)
    return groups


def plan_cleanup(data_dir: Path, groups: dict[str, list[Path]]) -> tuple[list[Path], list[dict]]:
    """Decide what survives. Returns (keep, drops); drops is log-ready rows."""
    keep: list[Path] = []
    drops: list[dict] = []

    for h, paths in groups.items():
        if len(paths) == 1:
            keep.append(paths[0])
            continue

        labels = {folder_label(p.parent.parent.name) for p in paths}

        if len(labels) == 1:
            canonical = min(paths, key=lambda p: str(p.relative_to(data_dir)))
            keep.append(canonical)
            for p in paths:
                if p != canonical:
                    drops.append({
                        "path": str(p.relative_to(data_dir)),
                        "hash": h,
                        "reason": "duplicate",
                        "kept": str(canonical.relative_to(data_dir)),
                        "labels": next(iter(labels)),
                    })
        else:
            for p in paths:
                drops.append({
                    "path": str(p.relative_to(data_dir)),
                    "hash": h,
                    "reason": "label_conflict",
                    "kept": "",
                    "labels": "|".join(sorted(labels)),
                })

    keep.sort(key=lambda p: str(p.relative_to(data_dir)))
    return keep, drops


def write_outputs(
    data_dir: Path, keep: list[Path], drops: list[dict], clean_list_path: Path, drop_log_path: Path
) -> None:
    clean_list_path.parent.mkdir(parents=True, exist_ok=True)
    clean_list_path.write_text("\n".join(str(p.relative_to(data_dir)) for p in keep) + "\n")

    drop_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(drop_log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "hash", "reason", "kept", "labels"])
        writer.writeheader()
        writer.writerows(drops)


def protected_relpaths(manifest_path: Path, active_train_list: Path) -> set[str]:
    """Every relpath a live partition depends on: the manifest's three pools plus the active
    training list. Quarantining any of these would silently shrink a split that other code
    believes is fixed -- which is exactly the class of bug this whole cleanup exists to end.
    """
    protected: set[str] = set()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for pool in manifest.get("splits", {}).values():
            protected.update(pool)
    if active_train_list.exists():
        protected.update(l for l in active_train_list.read_text().splitlines() if l.strip())
    return protected


def quarantine(data_dir: Path, drops: list[dict], quarantine_dir: Path,
               protected: set[str] | None = None) -> None:
    """MOVE dropped files out of the training tree. Reversible: the relative layout is preserved
    under quarantine_dir, so `mv` back restores the tree exactly.

    Refuses to move anything a live partition references (see protected_relpaths).
    """
    if protected:
        collisions = [row["path"] for row in drops if row["path"] in protected]
        if collisions:
            raise RuntimeError(
                f"refusing to quarantine {len(collisions)} file(s) that a live split still "
                f"references (e.g. {collisions[0]}). The clean list and the manifest disagree -- "
                f"re-carve before quarantining."
            )
    for row in drops:
        src = data_dir / row["path"]
        dst = quarantine_dir / row["path"]
        if not src.exists():
            continue  # already quarantined by an earlier run; move is idempotent
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--clean-list", default="data/clean_files.txt")
    p.add_argument("--drop-log", default="data/drop_log.csv")
    p.add_argument("--quarantine-dir", default=None,
                    help="if set, MOVE dropped files here instead of leaving them in place")
    p.add_argument("--manifest", default="data/splits_manifest.json",
                    help="live splits manifest whose files must never be quarantined")
    p.add_argument("--active-train-list", default="data/active_train.txt",
                    help="live active training list whose files must never be quarantined")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Hashing {data_dir} ...")
    groups = hash_groups(data_dir)
    total_files = sum(len(v) for v in groups.values())

    keep, drops = plan_cleanup(data_dir, groups)
    dup_drops = [d for d in drops if d["reason"] == "duplicate"]
    conflict_drops = [d for d in drops if d["reason"] == "label_conflict"]

    write_outputs(data_dir, keep, drops, Path(args.clean_list), Path(args.drop_log))

    print(f"{total_files} files -> {len(keep)} kept, {len(drops)} dropped "
          f"({len(dup_drops)} duplicate, {len(conflict_drops)} label_conflict)")
    print(f"clean list -> {args.clean_list}")
    print(f"drop log   -> {args.drop_log}")

    if args.quarantine_dir:
        protected = protected_relpaths(Path(args.manifest), Path(args.active_train_list))
        print(f"{len(protected)} relpath(s) protected by live splits -- verifying no overlap "
              f"with the {len(drops)} drops...")
        quarantine(data_dir, drops, Path(args.quarantine_dir), protected=protected)
        print(f"moved {len(drops)} dropped files -> {args.quarantine_dir}")
        remaining = len(sorted(data_dir.glob("*/side_a/*.jpg")))
        print(f"training tree now holds {remaining} image(s) (expected {len(keep)})")
    else:
        print("dry-run: nothing moved. Pass --quarantine-dir to actually pull dropped files "
              "out of the training tree.")


if __name__ == "__main__":
    main()
