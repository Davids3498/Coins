"""splits.py -- the ONE place the full coin dataset gets carved into three disjoint pools.

    train           (~60%) -- everything an initial/current model is allowed to train on.
    frozen-holdout  (~20%) -- THE holdout. The manifest this module writes is the single
                              definition of it in the codebase: coin_clf.data.build_manifest_holdout
                              reads it back, and evaluate.py / promote.py / train.py / export.py
                              all resolve through that one function. The two rival definitions
                              that used to exist (frozen_split's independent 80/20 draw and
                              build_test_dataset) are gone -- carving is now the only way a
                              holdout comes into existence.
    future-pool     (~20%) -- withheld entirely. Never trained on, never evaluated against.
                              Stands in for data that "arrives" after the model is live; the
                              retraining DAG pulls it in fixed-size batches (`FuturePool`, via
                              `DatasetSplits.future_pool()`) to grow the training set and trigger
                              a train -> evaluate -> promote cycle.

Carved ONCE: `carve()` is a deterministic function of (data_dir, seed) via stratified
train_test_split, but a DAG re-deriving the partition on every trigger is one accident away from
silently seeing a DIFFERENT partition (a relabeled folder, an image added/removed, a sklearn
version bump). `save`/`load` pin one carve to a JSON manifest of relative file paths -- not raw
indices, which are only meaningful for the exact glob() that produced them -- so every later run
(DAG tasks, notebooks, ad-hoc scripts) reads the SAME three pools instead of recomputing them.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from coin_clf.data import discover_dataset, split_dataset

DEFAULT_HOLDOUT_SIZE = 0.2
DEFAULT_FUTURE_SIZE = 0.2
DEFAULT_SEED = 42
MANIFEST_VERSION = 1


@dataclass(eq=False)
class DatasetSplits:
    """Three disjoint index sets over a single (filepaths, all_labs) pair from discover_dataset.

    future_idx is intentionally left in the shuffled order carve() produced (NOT sorted like
    train_idx/holdout_idx): that order IS the simulated arrival order, so consecutive future-pool
    batches mix classes the way organic incoming data would, instead of one batch per emperor.
    """

    filepaths: list[Path]
    all_labs: torch.Tensor
    label_encoder: dict[str, int]
    idx_to_label: dict[int, str]
    train_idx: np.ndarray
    holdout_idx: np.ndarray
    future_idx: np.ndarray

    def __post_init__(self) -> None:
        train, hold, fut = set(self.train_idx.tolist()), set(self.holdout_idx.tolist()), set(self.future_idx.tolist())
        if train & hold or train & fut or hold & fut:
            raise ValueError("train / holdout / future-pool overlap -- the carve is broken")

    def __repr__(self) -> str:
        return (f"DatasetSplits(num_classes={self.num_classes}, train={len(self.train_idx)}, "
                f"holdout={len(self.holdout_idx)}, future_pool={len(self.future_idx)})")

    @property
    def num_classes(self) -> int:
        return len(self.label_encoder)

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train_idx),
            "holdout": len(self.holdout_idx),
            "future_pool": len(self.future_idx),
        }

    def future_pool(self, batch_size: int) -> "FuturePool":
        return FuturePool(self, batch_size)

    def save(self, path: str | Path, data_dir: str | Path) -> Path:
        """Pin this carve to disk as relative file paths. Refuses to overwrite -- the whole
        point of a manifest is that a partition, once carved, does not move.
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} exists -- splits are carved once; refusing to overwrite")
        data_dir = Path(data_dir)
        manifest = {
            "version": MANIFEST_VERSION,
            "label_encoder": self.label_encoder,
            "splits": {
                "train": [str(self.filepaths[i].relative_to(data_dir)) for i in self.train_idx],
                "holdout": [str(self.filepaths[i].relative_to(data_dir)) for i in self.holdout_idx],
                "future_pool": [str(self.filepaths[i].relative_to(data_dir)) for i in self.future_idx],
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path, data_dir: str | Path,
             *, allow_raw_tree: bool = False) -> "DatasetSplits":
        """Re-derive indices for the CURRENT discover_dataset(data_dir) from a saved manifest.

        Guards against silent drift: if the label space no longer matches (folder added,
        removed, renamed, or GORDIAN_MERGES changed) or a manifest file has disappeared from
        data_dir, this raises rather than quietly returning a partition that no longer means
        what it did when it was carved.
        """
        manifest = json.loads(Path(path).read_text())
        if manifest["version"] != MANIFEST_VERSION:
            raise ValueError(f"manifest version {manifest['version']} != supported {MANIFEST_VERSION}")

        filepaths, all_labs, label_encoder, idx_to_label, _ = discover_dataset(
            data_dir, allow_raw_tree=allow_raw_tree
        )
        if label_encoder != manifest["label_encoder"]:
            raise ValueError(
                "label_encoder in the manifest no longer matches discover_dataset(data_dir) -- "
                "the label space has drifted since this split was carved. Re-carve before "
                "trusting this manifest."
            )

        data_dir = Path(data_dir)
        path_to_idx = {str(p.relative_to(data_dir)): i for i, p in enumerate(filepaths)}

        def _resolve(names: list[str]) -> np.ndarray:
            missing = [n for n in names if n not in path_to_idx]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} file(s) from the manifest are gone from {data_dir} "
                    f"(e.g. {missing[0]}) -- the dataset changed since this split was carved"
                )
            return np.array([path_to_idx[n] for n in names], dtype=int)

        return cls(
            filepaths=filepaths,
            all_labs=all_labs,
            label_encoder=label_encoder,
            idx_to_label=idx_to_label,
            train_idx=_resolve(manifest["splits"]["train"]),
            holdout_idx=_resolve(manifest["splits"]["holdout"]),
            future_idx=_resolve(manifest["splits"]["future_pool"]),
        )


def carve(
    data_dir: str | Path,
    *,
    holdout_size: float = DEFAULT_HOLDOUT_SIZE,
    future_size: float = DEFAULT_FUTURE_SIZE,
    seed: int = DEFAULT_SEED,
    clean_list: str | Path | None = None,
    allow_raw_tree: bool = False,
) -> DatasetSplits:
    """Deterministically carve discover_dataset(data_dir) into train / holdout / future-pool.

    Stage 1 calls coin_clf.data.split_dataset and keeps ONLY its test_idx. Stage 2 stratified-
    splits everything NOT in the holdout again, with future_size treated as a fraction of the
    FULL dataset (not of the 1-holdout_size remainder) -- e.g. holdout_size=0.2, future_size=0.2
    leaves train at 60% of the total, not 60% of the leftover 80%.

    CLEAN BY DEFAULT: discover_dataset filters to data/clean_files.txt unless allow_raw_tree=True,
    dropping duplicates and cross-label conflicts BEFORE any of the three pools are carved, so bad
    data can't end up split across train/holdout/future-pool in the first place. Carving the raw
    tree produced a partition with ~6,900 train/holdout byte-collisions; that is why it is no
    longer reachable by accident.
    """
    if not 0 < holdout_size < 1:
        raise ValueError(f"holdout_size must be in (0, 1), got {holdout_size}")
    if not 0 < future_size < 1:
        raise ValueError(f"future_size must be in (0, 1), got {future_size}")
    if holdout_size + future_size >= 1:
        raise ValueError(
            f"holdout_size + future_size must leave room for a train split, "
            f"got {holdout_size} + {future_size} >= 1"
        )

    filepaths, all_labs, label_encoder, idx_to_label, _ = discover_dataset(
        data_dir, clean_list=clean_list, allow_raw_tree=allow_raw_tree
    )
    labels_np = all_labs.numpy()

    _, _, holdout_idx = split_dataset(all_labs, test_size=holdout_size, random_state=seed)

    remaining_idx = np.setdiff1d(np.arange(len(all_labs)), holdout_idx, assume_unique=True)
    future_frac_of_remaining = future_size / (1.0 - holdout_size)
    train_idx, future_idx = train_test_split(
        remaining_idx,
        test_size=future_frac_of_remaining,
        shuffle=True,
        stratify=labels_np[remaining_idx],
        random_state=seed,
    )

    return DatasetSplits(
        filepaths=filepaths,
        all_labs=all_labs,
        label_encoder=label_encoder,
        idx_to_label=idx_to_label,
        train_idx=np.sort(train_idx),
        holdout_idx=np.sort(holdout_idx),
        future_idx=future_idx,  # keep the shuffled "arrival" order -- do not sort
    )


@dataclass(eq=False)
class Batch:
    """One future-pool batch, ready for a DAG task to append to the training set."""

    batch_num: int
    indices: np.ndarray
    filepaths: list[Path]
    labels: torch.Tensor

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str:
        return f"Batch(batch_num={self.batch_num}, size={len(self)})"


class FuturePool(Sequence):
    """Fixed-size batches over the future-pool, in carve()'s frozen arrival order.

    Indexable (`pool[i]`) as well as iterable, on purpose: a DAG task is usually parameterized by
    a run/batch number and needs idempotent random access ("give me batch 7 again"), not just
    "call next() until state runs out."
    """

    def __init__(self, splits: DatasetSplits, batch_size: int):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.splits = splits
        self.batch_size = batch_size

    def __len__(self) -> int:
        return -(-len(self.splits.future_idx) // self.batch_size)  # ceil division

    def __getitem__(self, batch_num: int) -> Batch:
        if batch_num < 0:
            batch_num += len(self)
        if not 0 <= batch_num < len(self):
            raise IndexError(f"batch {batch_num} out of range (0..{len(self) - 1})")
        start = batch_num * self.batch_size
        end = min(start + self.batch_size, len(self.splits.future_idx))
        idx = self.splits.future_idx[start:end]
        return Batch(
            batch_num=batch_num,
            indices=idx,
            filepaths=[self.splits.filepaths[i] for i in idx],
            labels=self.splits.all_labs[idx],
        )

    def ingested_through(self, batch_num: int) -> np.ndarray:
        """Indices from future-pool batches 0..batch_num inclusive -- what a DAG has folded
        into the training set by round `batch_num`, for a retrain that accumulates batches
        rather than training on each one in isolation.
        """
        if batch_num < 0:
            batch_num += len(self)
        if not 0 <= batch_num < len(self):
            raise IndexError(f"batch {batch_num} out of range (0..{len(self) - 1})")
        end = min((batch_num + 1) * self.batch_size, len(self.splits.future_idx))
        return self.splits.future_idx[:end]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Carve the coin dataset into train / frozen-holdout / future-pool, once."
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="data/splits_manifest.json", help="manifest path to write")
    p.add_argument("--holdout-size", type=float, default=DEFAULT_HOLDOUT_SIZE)
    p.add_argument("--future-size", type=float, default=DEFAULT_FUTURE_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--clean-list", default=None,
        help="clean_duplicates.py survivor list (default: data/clean_files.txt; missing is a "
             "hard error, not a fallback to raw data)",
    )
    p.add_argument(
        "--allow-raw-tree", action="store_true",
        help="carve the RAW, uncleaned tree -- duplicates and cross-label conflicts included. "
             "This produces a partition with thousands of train/holdout byte-collisions and must "
             "never be used for a model you intend to score or promote.",
    )
    args = p.parse_args()

    if args.allow_raw_tree:
        print("WARNING: --allow-raw-tree -- carving the raw, uncleaned tree. The resulting "
              "partition is NOT leakage-free.")

    splits = carve(
        args.data_dir,
        holdout_size=args.holdout_size,
        future_size=args.future_size,
        seed=args.seed,
        clean_list=args.clean_list,
        allow_raw_tree=args.allow_raw_tree,
    )
    out = splits.save(args.out, data_dir=args.data_dir)

    sizes = splits.sizes
    total = sum(sizes.values())
    print(
        f"{total} images -> train={sizes['train']} ({sizes['train'] / total:.1%})  "
        f"holdout={sizes['holdout']} ({sizes['holdout'] / total:.1%})  "
        f"future_pool={sizes['future_pool']} ({sizes['future_pool'] / total:.1%})"
    )
    print(f"manifest written -> {out}")


if __name__ == "__main__":
    main()
