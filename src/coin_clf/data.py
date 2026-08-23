import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler

from coin_clf.hashing import file_hash
from coin_clf.transforms import val_transform

GORDIAN_MERGES = {"GORDIAN II": "GORDIAN I"}


def folder_label(name: str) -> str:
    return re.sub(r"^\d+_", "", name)


def discover_dataset(data_dir: Path, clean_list: Path | str | None = None):
    """Glob the labeled image tree, encode labels, and apply the GORDIAN II -> GORDIAN I merge.

    clean_list: optional path to a newline-separated file of paths relative to data_dir, as
    written by clean_duplicates.py. When given, only files on the list are kept -- exact
    duplicates and cross-label conflicts it found are dropped before label encoding and
    splitting even start. None (default) keeps every *.jpg the glob finds, so existing callers
    are unaffected unless they opt in.

    Returns (filepaths, all_labs, label_encoder, idx_to_label, num_classes).
    """
    image_dir = Path(data_dir)
    filepaths = sorted(image_dir.glob("*/side_a/*.jpg"))  # same order as embeddings
    if clean_list is not None:
        allowed = set(Path(clean_list).read_text().splitlines())
        filepaths = [p for p in filepaths if str(p.relative_to(image_dir)) in allowed]
    unique_labels = sorted({folder_label(p.parent.parent.name) for p in filepaths})
    label_encoder = {name: i for i, name in enumerate(unique_labels)}
    raw_labels = [folder_label(p.parent.parent.name) for p in filepaths]
    labels_int = [label_encoder[l] for l in raw_labels]

    kept_names = [n for n in unique_labels if n not in GORDIAN_MERGES]
    new_label_encoder = {name: i for i, name in enumerate(kept_names)}
    old_to_new = {label_encoder[name]: new_label_encoder[name] for name in kept_names}
    for src, tgt in GORDIAN_MERGES.items():
        old_to_new[label_encoder[src]] = new_label_encoder[tgt]

    all_labs = torch.tensor([old_to_new[l] for l in labels_int], dtype=torch.long)
    label_encoder = new_label_encoder
    idx_to_label = {v: k for k, v in label_encoder.items()}
    num_classes = len(label_encoder)

    return filepaths, all_labs, label_encoder, idx_to_label, num_classes


def split_dataset(all_labs: torch.Tensor, test_size: float = 0.2, random_state: int = 42):
    """Reproduces the frozen train/val/test split: two stratified train_test_split calls,
    the same test_size and random_state both times (carve test, then carve val from the rest).
    """
    indices = np.arange(len(all_labs))
    labels_np = all_labs.numpy()
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, shuffle=True, stratify=labels_np, random_state=random_state)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=test_size, shuffle=True,
        stratify=labels_np[train_idx], random_state=random_state)
    return train_idx, val_idx, test_idx


class CoinDistilDataset(Dataset):
    """Returns (image, soft_label, hard_label) for distillation training."""

    def __init__(self, filepaths, all_labs, teacher_soft, indices, transform):
        self.filepaths = filepaths
        self.all_labs = all_labs
        self.teacher_soft = teacher_soft
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = Image.open(self.filepaths[idx]).convert("RGB")
        img_tensor = self.transform(img)
        soft = self.teacher_soft[idx]  # (num_classes,) float
        hard = self.all_labs[idx]      # int
        return img_tensor, soft, hard


class CoinEvalDataset(Dataset):
    """Returns (image, label) — no teacher soft labels, for eval/serving-parity scoring."""

    def __init__(self, filepaths, all_labs, indices, transform):
        self.filepaths = filepaths
        self.all_labs = all_labs
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = Image.open(self.filepaths[idx]).convert("RGB")
        return self.transform(img), self.all_labs[idx]


def frozen_split(
    data_dir: Path, test_size: float = 0.2, random_state: int = 42, clean_list: Path | str | None = None
):
    """The ONE frozen train/val/test boundary -- computed on the raw tree so it never moves,
    then (optionally) cleaned. train.py and build_test_dataset both call this so an
    already-trained model and the holdout that scores it are always talking about the same
    boundary, cleaned the same way; two separate reimplementations is how they'd quietly drift.

    Splits FIRST, on the raw tree, THEN filters -- deliberately not the other way around.
    Filtering before splitting (like splits.carve() does) shrinks the universe train_test_split
    sees, and since its stratified split consumes one shared RNG stream class by class, dropping
    even one image from an early class reshuffles every class split after it -- the whole
    boundary moves. That's fine for splits.carve(), which is carving a brand-new partition
    nothing has trained on yet. It's NOT fine here: the boundary has to stay fixed so an
    already-trained model and any newly-trained one are still comparable on the same holdout, and
    so a freshly-trained model can never end up training on what is supposed to be frozen holdout.

    clean_list, when given, filters each side after the split:
      * train_idx / val_idx -- dropped if not a clean_list survivor (duplicate or label_conflict
        that clean_duplicates.py removed from the corpus entirely).
      * test_idx -- the same membership filter, PLUS dropped if its content hash matches ANY
        (unfiltered) train/val file, even if clean_duplicates.py's filename tie-break happened to
        keep THIS copy as the corpus-wide canonical. Because that hash check runs against the
        full, unfiltered train/val set, shrinking train/val afterward (the membership filter
        above) can only preserve this disjointness, never break it -- so train/val don't need
        their own hash check on top.

    Returns (filepaths, all_labs, label_encoder, idx_to_label, num_classes, train_idx, val_idx, test_idx).
    """
    filepaths, all_labs, label_encoder, idx_to_label, num_classes = discover_dataset(data_dir)
    train_idx, val_idx, test_idx = split_dataset(all_labs, test_size=test_size, random_state=random_state)

    if clean_list is not None:
        image_dir = Path(data_dir)
        allowed = set(Path(clean_list).read_text().splitlines())

        def _by_membership(idx):
            return np.array(
                [i for i in idx if str(filepaths[i].relative_to(image_dir)) in allowed],
                dtype=idx.dtype,
            )

        train_val_hashes = {file_hash(filepaths[i]) for i in np.concatenate([train_idx, val_idx])}
        test_idx = _by_membership(test_idx)
        test_idx = np.array(
            [i for i in test_idx if file_hash(filepaths[i]) not in train_val_hashes],
            dtype=test_idx.dtype,
        )
        train_idx = _by_membership(train_idx)
        val_idx = _by_membership(val_idx)

    return filepaths, all_labs, label_encoder, idx_to_label, num_classes, train_idx, val_idx, test_idx


def build_test_dataset(
    data_dir: Path, test_size: float = 0.2, random_state: int = 42, clean_list: Path | str | None = None
) -> Dataset:
    """The frozen test split, transformed the way serving transforms images. The one seam
    evaluate.py needs to score any registered version without reproducing the split logic.
    See frozen_split for what clean_list actually does.
    """
    filepaths, all_labs, *_, test_idx = frozen_split(data_dir, test_size, random_state, clean_list)
    return CoinEvalDataset(filepaths, all_labs, test_idx, val_transform)


def class_balanced_weights(labels: torch.Tensor, num_classes: int, beta: float = 0.9999) -> torch.Tensor:
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes
    return weights


def weighted_sampler(labels: torch.Tensor, weights: torch.Tensor, num_samples: int | None = None) -> WeightedRandomSampler:
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples=num_samples or len(labels), replacement=True)
