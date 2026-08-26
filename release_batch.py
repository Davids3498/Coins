"""release_batch.py -- pop the next future-pool batch and fold it into training.

The retraining DAG's first task, once per simulated batch release: advance a persisted cursor
over splits.py's FuturePool, append the batch's files to data/active_train.txt (bootstrapping
that list from the manifest's clean train split on first use), and write the batch's own
(path,label) rows to disk for validate_batch to score before anything trains on it.

Deliberately a root-level script, not inside src/coin_clf: it needs splits.py's DatasetSplits /
FuturePool, and src/coin_clf must never import a root-level script (see
coin_clf.validate_batch's module docstring on the dependency direction) -- so this lives next to
splits.py, train.py, evaluate.py, promote.py, the same way check_split_leakage.py already does.

The future-pool running out is treated as a hard error, not a "bad batch": there's no more
simulated data left to arrive, so the run should fail loudly rather than let validate/train/
promote quietly proceed on nothing.

Usage:
    python release_batch.py --data-dir data/FOR_TRAINNING --manifest data/splits_manifest.json \
        --active-train-list data/active_train.txt --cursor-file data/future_pool_cursor.json \
        --batch-size 200 --batch-out /path/to/run/batch.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from splits import DatasetSplits


def _load_cursor(cursor_file: Path) -> int:
    if not cursor_file.exists():
        return 0
    return json.loads(cursor_file.read_text())["next_batch"]


def _save_cursor(cursor_file: Path, next_batch: int) -> None:
    cursor_file.parent.mkdir(parents=True, exist_ok=True)
    cursor_file.write_text(json.dumps({"next_batch": next_batch}))


def release_next_batch(
    *,
    data_dir: str,
    manifest_path: str,
    active_train_list: str,
    cursor_file: str,
    batch_size: int,
) -> tuple[int, list[tuple[Path, str]]]:
    """Pop the next not-yet-released future-pool batch, fold its files into active_train_list,
    advance the cursor, and return (batch_num, [(path, label), ...]) for the caller to write out
    and validate.

    batch_size must stay the SAME across every call sharing one cursor_file -- the cursor counts
    batch NUMBERS, so changing batch_size mid-stream would silently shift what "batch 7" means.
    """
    splits = DatasetSplits.load(manifest_path, data_dir=data_dir)
    pool = splits.future_pool(batch_size)
    
    cursor_path = Path(cursor_file)
    batch_num = _load_cursor(cursor_path)
    if batch_num >= len(pool):
        raise RuntimeError(
            f"future-pool exhausted: {len(pool)} batch(es) of size {batch_size} already "
            f"released (next would be {batch_num}) -- no more simulated data to release from "
            f"{manifest_path}"
        )
    batch = pool[batch_num]

    image_dir = Path(data_dir)
    active_path = Path(active_train_list)
    if not active_path.exists():
        active_path.parent.mkdir(parents=True, exist_ok=True)
        seed_relpaths = [str(splits.filepaths[i].relative_to(image_dir)) for i in splits.train_idx]
        active_path.write_text("".join(f"{rel}\n" for rel in seed_relpaths))
        print(f"bootstrapped {active_train_list} from the manifest's clean train split "
              f"({len(seed_relpaths)} files)")

    batch_relpaths = [str(p.relative_to(image_dir)) for p in batch.filepaths]
    with active_path.open("a") as f:
        for rel in batch_relpaths:
            f.write(f"{rel}\n")

    _save_cursor(cursor_path, batch_num + 1)

    labels = [splits.idx_to_label[int(lab)] for lab in batch.labels.tolist()]
    rows = list(zip(batch.filepaths, labels))
    return batch_num, rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--manifest", default="data/splits_manifest.json")
    p.add_argument("--active-train-list", default="data/active_train.txt")
    p.add_argument("--cursor-file", default="data/future_pool_cursor.json")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument(
        "--batch-out", required=True,
        help="where to write this run's (path,label) batch -- absolute paths, one "
             "'path,label' row per line, ready for coin_clf.validate_batch to read",
    )
    args = p.parse_args()

    batch_num, rows = release_next_batch(
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        active_train_list=args.active_train_list,
        cursor_file=args.cursor_file,
        batch_size=args.batch_size,
    )

    out_path = Path(args.batch_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for path, label in rows:
            f.write(f"{Path(path).resolve()},{label}\n")

    print(f"released batch {batch_num} ({len(rows)} images) -> {args.batch_out}")
    print(f"active_train_list now includes this batch -> {args.active_train_list}")


if __name__ == "__main__":
    main()
