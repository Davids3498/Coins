"""
Fixed infrastructure: data loading, model definitions, and training utilities.
Do NOT modify this file — autoresearch agents modify train.py only.
"""

import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Paths & device
# ---------------------------------------------------------------------------
DATA_DIR = Path('/home/david/coin/FOR_TRAINNING')
EMBEDDINGS_PATH = DATA_DIR / 'cradio_v4_embeddings.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Label encoder (derived from folder structure)
# ---------------------------------------------------------------------------
def _folder_label(name: str) -> str:
    return re.sub(r'^\d+_', '', name)

_unique_labels = sorted({
    _folder_label(p.name)
    for p in DATA_DIR.iterdir()
    if p.is_dir() and (p / 'side_a').exists()
})
_base_encoder = {name: i for i, name in enumerate(_unique_labels)}

# Merge tiny/hard-to-distinguish classes
MERGES = {'GORDIAN II': 'GORDIAN I'}

_kept_names = [n for n in _unique_labels if n not in MERGES]
label_encoder = {name: i for i, name in enumerate(_kept_names)}
idx_to_label = {v: k for k, v in label_encoder.items()}
num_classes = len(label_encoder)   # 51
FEAT_DIM = 2560                    # C-RADIO v4-H embedding dimension

# ---------------------------------------------------------------------------
# Load pre-extracted embeddings and apply merges
# ---------------------------------------------------------------------------
def _load_embeddings():
    saved = torch.load(EMBEDDINGS_PATH, map_location='cpu')
    feats, labs = saved['features'], saved['labels']

    remap = {_base_encoder[src]: _base_encoder[tgt] for src, tgt in MERGES.items()}
    mapped = labs.clone()
    for src_idx, tgt_idx in remap.items():
        mapped[labs == src_idx] = tgt_idx

    old_to_new = {_base_encoder[name]: label_encoder[name] for name in _kept_names}
    labs = torch.tensor([old_to_new[int(x)] for x in mapped], dtype=torch.long)
    return feats, labs

_all_feats, _all_labs = _load_embeddings()

# ---------------------------------------------------------------------------
# Stratified splits (fixed seeds — never change these)
# ---------------------------------------------------------------------------
_train_feats, _test_feats, _train_labs, _test_labs = train_test_split(
    _all_feats, _all_labs, test_size=0.2, shuffle=True,
    stratify=_all_labs, random_state=42,
)
_train_feats, _val_feats, _train_labs, _val_labs = train_test_split(
    _train_feats, _train_labs, test_size=0.2, shuffle=True,
    stratify=_train_labs, random_state=42,
)

# Expose read-only tensors for kNN or custom loaders
train_feats, val_feats, test_feats = _train_feats, _val_feats, _test_feats
train_labs,  val_labs,  test_labs  = _train_labs,  _val_labs,  _test_labs

# ---------------------------------------------------------------------------
# Class-balanced sample weights (Cui et al. 2019 effective number)
# ---------------------------------------------------------------------------
_class_counts = torch.bincount(_train_labs, minlength=num_classes).float()
_beta = 0.9999
_eff = 1.0 - torch.pow(_beta, _class_counts)
cb_weights = (1.0 - _beta) / _eff
cb_weights = cb_weights / cb_weights.sum() * num_classes
cb_weights = cb_weights.to(device)

# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def make_loaders(batch_size: int = 512):
    sample_w = cb_weights.cpu()[_train_labs]
    sampler = WeightedRandomSampler(sample_w, len(_train_labs), replacement=True)
    train_loader = DataLoader(
        TensorDataset(_train_feats, _train_labs),
        batch_size=batch_size, sampler=sampler,
    )
    val_loader = DataLoader(
        TensorDataset(_val_feats, _val_labs),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(_test_feats, _test_labs),
        batch_size=batch_size, shuffle=False,
    )
    return train_loader, val_loader, test_loader

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, n_cls: int, scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_cls, in_dim) * 0.01)
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(scale))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return torch.exp(self.log_scale) * (x @ w.T)


class CoinHead(nn.Module):
    """MLP projection head + cosine classifier over frozen C-RADIO features."""

    def __init__(
        self,
        in_dim: int = FEAT_DIM,
        n_cls: int = num_classes,
        hidden: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls = CosineClassifier(hidden, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.body(x))

# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def smoothed_ce(logits, y_a, y_b, lam, weight=None, smoothing=0.1):
    log_p = F.log_softmax(logits, dim=-1)

    def _one(y):
        nll = -log_p.gather(1, y.unsqueeze(1)).squeeze(1)
        smooth = -log_p.mean(dim=-1)
        per_ex = (1 - smoothing) * nll + smoothing * smooth
        if weight is not None:
            per_ex = per_ex * weight[y]
        return per_ex.mean()

    return lam * _one(y_a) + (1 - lam) * _one(y_b)


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def _lr(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for feats, targets in loader:
        feats, targets = feats.to(device), targets.to(device)
        correct += (model(feats).argmax(-1) == targets).sum().item()
        total += targets.size(0)
    return correct / total
