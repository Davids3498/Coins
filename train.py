"""
Autoresearch training script — agents modify this file freely.
The only contract: print  `val_acc: X.XXXX`  as the last line of stdout.
Higher val_acc is better (baseline: 0.8747).
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from prepare import (
    device, num_classes, FEAT_DIM, cb_weights,
    train_feats, val_feats, train_labs, val_labs,
    make_loaders, CoinHead, mixup_batch, smoothed_ce, make_scheduler, accuracy,
)

# ===========================================================================
# Hyperparameters — edit freely
# ===========================================================================
BATCH_SIZE    = 512
EPOCHS        = 60
LR            = 3e-4
WEIGHT_DECAY  = 1e-2
MIXUP_ALPHA   = 0.2
LABEL_SMOOTH  = 0.1
HIDDEN        = 1024
DROPOUT       = 0.3
N_MODELS      = 5      # ensemble size
WARMUP_EPOCHS = 2
# ===========================================================================

train_loader, val_loader, _ = make_loaders(BATCH_SIZE)


def train_one(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CoinHead(FEAT_DIM, num_classes, hidden=HIDDEN, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps   = EPOCHS * len(train_loader)
    warmup_steps  = WARMUP_EPOCHS * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps)

    best_val_acc, best_state = 0.0, None

    for _ in range(EPOCHS):
        model.train()
        for feats, targets in train_loader:
            feats, targets = feats.to(device), targets.to(device)
            mixed, y_a, y_b, lam = mixup_batch(feats, targets, MIXUP_ALPHA)
            optimizer.zero_grad()
            loss = smoothed_ce(model(mixed), y_a, y_b, lam, cb_weights, LABEL_SMOOTH)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_acc = accuracy(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model.eval(), best_val_acc


# Train ensemble
ensemble = []
for k in range(N_MODELS):
    m, vacc = train_one(seed=1000 + k)
    print(f'head {k + 1}/{N_MODELS}  val_acc={vacc:.4f}')
    ensemble.append(m)


# Ensemble evaluation on val
@torch.no_grad()
def ensemble_acc(loader):
    correct = total = 0
    for feats, targets in loader:
        feats, targets = feats.to(device), targets.to(device)
        probs = sum(F.softmax(m(feats), dim=-1) for m in ensemble) / len(ensemble)
        correct += (probs.argmax(-1) == targets).sum().item()
        total += targets.size(0)
    return correct / total


val_acc = ensemble_acc(val_loader)
print(f'val_acc: {val_acc:.4f}')
