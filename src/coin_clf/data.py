import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler

GORDIAN_MERGES = {"GORDIAN II": "GORDIAN I"}


def folder_label(name: str) -> str:
    return re.sub(r"^\d+_", "", name)


def discover_dataset(data_dir: Path):
    """Glob the labeled image tree, encode labels, and apply the GORDIAN II -> GORDIAN I merge.

    Returns (filepaths, all_labs, label_encoder, idx_to_label, num_classes).
    """
    image_dir = Path(data_dir)
    filepaths = sorted(image_dir.glob("*/side_a/*.jpg"))  # same order as embeddings
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


def class_balanced_weights(labels: torch.Tensor, num_classes: int, beta: float = 0.9999) -> torch.Tensor:
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes
    return weights


def weighted_sampler(labels: torch.Tensor, weights: torch.Tensor, num_samples: int | None = None) -> WeightedRandomSampler:
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples=num_samples or len(labels), replacement=True)
