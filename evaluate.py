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
import os
from typing import Callable

import mlflow
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader, Dataset

MODEL_NAME = "coin-classifier"
# Env-overridable, matching app/main.py and drift_report.py: one way to point every
# component at a tracking server, and a default that keeps the local setup working.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
# Same name train.py logs, so a backfilled v1 lines up with trained challengers. Re-running
# against a run that already has it appends another point (cosmetic — the gate reads none of
# these). Point it at a distinct name if you'd rather keep eval numbers separate from training.
EVAL_METRIC = "test_acc"


# --- the ONE seam to wire to your repo -------------------------------------
# coin_clf.data.build_manifest_holdout is the single definition of the holdout in this codebase.
# It resolves splits.py's carve manifest (data/splits_manifest.json by default) and RAISES if
# that manifest is absent — there is deliberately no fallback, because every fallback that ever
# existed here turned out to be a different partition than the one models were trained against.
#
# There used to be a clean_list parameter selecting between frozen_split's independent 80/20 draw
# and a clean-filtered variant of it. Those were two more holdouts (12,427 and 11,382 images,
# sharing only ~2,400 images with this one) and they are gone. Do not reintroduce a parameter
# here that can change which images come back.
def load_holdout(data_dir: str, manifest: str | None = None) -> Dataset:
    from coin_clf.data import build_manifest_holdout

    return build_manifest_holdout(data_dir, manifest)
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
    manifest: str | None = None,
    holdout_loader: Callable[[str, str | None], Dataset] = load_holdout,
) -> tuple[str, float]:
    """Load a version from the registry, score it on the frozen holdout, return (version, acc).

    Pure scoring — logs NOTHING, so the gate can call it on both models without writing to
    runs. The CLI wrapper below handles logging.

    manifest=None resolves to data/splits_manifest.json and raises if it is missing. A bare
    `evaluate_version(...)` call therefore scores on the SAME 11,559-image holdout the DAG and
    the promotion gate use — it used to quietly score on a 12,427-image raw-tree draw that
    overlapped the training set by 5,689 images.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    resolved = _resolve(client, version, alias)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{resolved}")
    dataset = holdout_loader(data_dir, manifest)
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
    p.add_argument(
        "--manifest", default=None,
        help="splits.py manifest pinning the frozen holdout (default: data/splits_manifest.json). "
             "This is the ONLY holdout definition; a missing manifest is a hard error, not a "
             "fallback.",
    )
    args = p.parse_args()

    version, acc = evaluate_version(
        version=args.version,
        alias=args.alias,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        manifest=args.manifest,
    )
    print(f"coin-classifier v{version}  {EVAL_METRIC}={acc:.4f}")
    if not args.no_log:
        log_eval_metric(version, acc)
        print(f"logged {EVAL_METRIC}={acc:.4f} to v{version}'s run")


if __name__ == "__main__":
    main()
