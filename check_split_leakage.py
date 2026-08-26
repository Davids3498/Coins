"""check_split_leakage.py -- verify train / frozen-holdout / future-pool share NO images.

Two independent checks, because splits.py's own index bookkeeping can't catch the second one:
  1. Index-level: no file index appears in more than one split (DatasetSplits already enforces
     this at construction, but this re-checks independently, e.g. against a hand-edited manifest).
  2. Content-level: no two files in DIFFERENT splits are byte-identical. The same physical coin
     photographed/uploaded twice under different filenames would sail past check #1 (different
     indices) yet still leak -- the model would effectively be tested on an image it trained on.

Usage:
    python check_split_leakage.py --data-dir data/FOR_TRAINNING
    python check_split_leakage.py --data-dir data/FOR_TRAINNING --manifest data/splits_manifest.json
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from coin_clf.hashing import file_hash
from splits import DatasetSplits, carve


def check(splits: DatasetSplits) -> bool:
    ok = True
    by_split = {"train": splits.train_idx, "holdout": splits.holdout_idx, "future_pool": splits.future_idx}

    # 1. index-level: same file index claimed by two splits
    owner: dict[int, str] = {}
    for name, idx in by_split.items():
        for i in idx.tolist():
            if i in owner:
                ok = False
                print(f"INDEX LEAK: {splits.filepaths[i]} is in both {owner[i]!r} and {name!r}")
            owner[i] = name

    # 2. content-level: identical bytes on opposite sides of a split
    total = sum(len(idx) for idx in by_split.values())
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    hash_to_paths: dict[str, list[Path]] = defaultdict(list)
    done = 0
    for name, idx in by_split.items():
        for i in idx.tolist():
            path = splits.filepaths[i]
            h = file_hash(path)
            hash_to_splits[h].add(name)
            hash_to_paths[h].append(path)
            done += 1
            if done % 5000 == 0:
                print(f"  hashed {done}/{total}", file=sys.stderr)

    for h, split_names in hash_to_splits.items():
        if len(split_names) > 1:
            ok = False
            print(f"CONTENT LEAK: identical bytes appear in {sorted(split_names)}:")
            for p in hash_to_paths[h]:
                print(f"    {p}")

    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--manifest", help="load a saved manifest instead of re-carving from scratch")
    p.add_argument("--clean-list", default=None,
                    help="clean_duplicates.py survivor list to filter through before carving "
                         "(default: data/clean_files.txt; ignored if --manifest is given)")
    p.add_argument("--allow-raw-tree", action="store_true",
                    help="carve the raw, uncleaned tree -- expected to report leakage")
    args = p.parse_args()

    if args.manifest:
        splits = DatasetSplits.load(args.manifest, data_dir=args.data_dir)
    else:
        splits = carve(args.data_dir, clean_list=args.clean_list,
                       allow_raw_tree=args.allow_raw_tree)
    sizes = splits.sizes
    print(f"Checking {sum(sizes.values())} images: "
          f"train={sizes['train']} holdout={sizes['holdout']} future_pool={sizes['future_pool']}")

    if check(splits):
        print("OK -- no leakage between splits.")
    else:
        print("LEAKAGE FOUND -- see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
