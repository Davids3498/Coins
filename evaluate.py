"""Score a registered model version on the frozen coin holdout and log the metric to its run.

Two jobs:
  * Standalone pipeline step (roadmap §8.5) — score any version, record the number.
  * The scoring primitive the promotion gate re-uses. `promote.py` imports `evaluate_version`
    so champion and challenger are always scored on the SAME frozen split at decision time —
    the gate never trusts a stale or missing logged number.

Backfilling: running this against @champion once logs v1's holdout accuracy onto its seed
run, closing the "seed_champion logged no test_acc" gap. The gate doesn't need this (it
re-scores live), but it makes the registry honest for humans reading the runs.
"""
from __future__ import annotations

import argparse
from typing import Callable

import mlflow
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader, Dataset

MODEL_NAME = "coin-classifier"
TRACKING_URI = "http://127.0.0.1:5000"
# Same name train.py logs, so a backfilled v1 lines up with trained challengers. Re-running
# against a run that already has it appends another point (cosmetic — the gate reads none of
# these). Point it at a distinct name if you'd rather keep eval numbers separate from training.
EVAL_METRIC = "test_acc"


# --- the ONE seam to wire to your repo -------------------------------------
# data.py already owns the frozen split (train_test_split, test_size=0.2, random_state=42,
# stratified, GORDIAN II -> GORDIAN I merge). Point this at whatever it exports so there is
# exactly ONE definition of the holdout — reproducing the split here would reintroduce the
# train/serve skew the refactor just killed. Must return a Dataset yielding
# (image_tensor, label_idx) built with coin_clf.transforms.val_transform (serving's transform).
def load_holdout(data_dir: str) -> Dataset:
    from coin_clf.data import build_test_dataset

    return build_test_dataset(data_dir)
# ---------------------------------------------------------------------------


def _resolve(client: MlflowClient, version: str | None, alias: str | None) -> str:
    if (version is None) == (alias is None):
        raise ValueError("pass exactly one of version / alias")
    if version is not None:
        return version
    return client.get_model_version_by_alias(MODEL_NAME, alias).version


@torch.no_grad()
def score(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval().to(device)
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    if total == 0:
        raise RuntimeError("holdout is empty — check data_dir / load_holdout")
    return correct / total


def evaluate_version(
    *,
    version: str | None = None,
    alias: str | None = None,
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    device: torch.device | None = None,
    holdout_loader: Callable[[str], Dataset] = load_holdout,
) -> tuple[str, float]:
    """Load a version from the registry, score it on the frozen holdout, return (version, acc).

    Pure scoring — logs NOTHING, so the gate can call it on both models without writing to
    runs. The CLI wrapper below handles logging.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    resolved = _resolve(client, version, alias)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{resolved}")
    dataset = holdout_loader(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    acc = score(model, loader, device)
    return resolved, acc


def log_eval_metric(version: str, acc: float) -> None:
    """Record the frozen-holdout accuracy on the version's originating run. Record-keeping only."""
    client = MlflowClient()
    run_id = client.get_model_version(MODEL_NAME, version).run_id
    if not run_id:
        print(f"v{version} has no source run; skipping metric log")
        return
    client.log_metric(run_id, EVAL_METRIC, acc)


def main() -> None:
    p = argparse.ArgumentParser(description="Score a coin-classifier version on the frozen holdout.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--version")
    g.add_argument("--alias")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-log", action="store_true", help="score only; do not log to the run")
    args = p.parse_args()

    version, acc = evaluate_version(
        version=args.version,
        alias=args.alias,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"coin-classifier v{version}  {EVAL_METRIC}={acc:.4f}")
    if not args.no_log:
        log_eval_metric(version, acc)
        print(f"logged {EVAL_METRIC}={acc:.4f} to v{version}'s run")


if __name__ == "__main__":
    main()
