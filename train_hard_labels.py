"""Headless twin of notebooks/coin_mobilenet_hard_labels.ipynb -- the training script the
retrain_coin_clf DAG shells out to.

WHY THIS EXISTS ALONGSIDE train.py. train.py is the general-purpose CLI: its defaults (60
epochs, batch 128) are a reasonable starting point, not a result anyone has earned. This script
is one specific, already-validated recipe -- the notebook's 120-epoch / batch-256 run that
produced coin-classifier v6 (val 0.9059, holdout 0.9030) -- turned into something a scheduler can
trigger unattended. The DAG needs a run whose hyperparameters are a fact about a past result, not
a default someone may retune tomorrow.

WHAT IT DOES NOT DEFINE. No split, no transforms, no loaders, no scoring, no LR schedule. Every
one of those is imported from train.py, exactly as the notebook imports them, for the reason
stated in the notebook's preamble: a second split definition is what produced three rival
holdouts, and a differently-computed accuracy is not comparable to @champion's. The only thing
that lives here is the epoch loop, the notebook's config, and the CLI/artifact plumbing an
unattended run needs.

WHAT IT ADDS OVER THE NOTEBOOK: argparse (so the DAG can override epochs and point at a run's
own paths), --version-out (the DAG reads the registered challenger version back out of it for
evaluate/promote), and a non-interactive matplotlib backend. The holdout-fingerprint assertion
that guards the notebook's cell 3 is kept as a hard precondition here too -- a scheduled run that
silently scores against a moved holdout is worse than one that fails loudly, because nothing
downstream (evaluate.py, promote.py) would notice.

Registers a challenger. It does NOT touch the @champion alias -- promote.py does, as its own
deliberate DAG task.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

import matplotlib

# Airflow workers have no display, and this must be set before pyplot is imported anywhere --
# including by `import train` below, which imports pyplot at module scope.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.pytorch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

from checkpoint import save_checkpoint  # noqa: E402
from coin_clf.labels import save_labels  # noqa: E402
from coin_clf.model import build_model  # noqa: E402

# --- the canonical seams. Nothing below is reimplemented here. ---
from train import (  # noqa: E402
    class_display_order,
    confusion_matrix,
    evaluate_hard,
    make_scheduler,
    prepare_data,
)

# Repo-relative, not machine-absolute -- same anchor pattern as train.py and
# verify_data_integrity.py. Defined here rather than imported from train.py so this file
# keeps working if that import list ever changes.
REPO_ROOT = Path(__file__).resolve().parent

EXPERIMENT_NAME = "coin-classifier"
MODEL_NAME = "coin-classifier"

# Content-hash fingerprint of the canonical 11,559-image holdout, as proven by
# verify_data_integrity.py and asserted by the notebook this script is derived from. The
# retraining loop grows data/active_train.txt and never touches the manifest's holdout, so this
# value must survive every DAG run; if it changes, the holdout moved and no number produced here
# is comparable to @champion's.
CANONICAL_HOLDOUT_FP = "c47d3f321a0e13c2"

# Below this, the cosine schedule never leaves warmup + early plateau, so the run trains on a
# truncated LR curve rather than a shorter version of the same one. Smoke runs are legitimate;
# silently promoting one is not, hence the printed warning.
SMOKE_RUN_EPOCHS = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "FOR_TRAINNING"))
    p.add_argument("--weights-dir", default=str(REPO_ROOT / "weights"))
    p.add_argument("--labels-out", default=None, help="default: <weights-dir>/coin_labels.json")
    p.add_argument("--num-classes", type=int, default=51)
    p.add_argument("--input-size", type=int, default=224)

    # --- the notebook's config: the numbers that earned v6's 0.9030 holdout. ---
    # Batch size is the one deliberate departure -- the notebook ran 256, this defaults to 128 to
    # match train.py and stay within a smaller GPU. It changes steps/epoch, and therefore the
    # cosine curve's resolution, so a run at 128 is a close relative of v6's rather than a rerun
    # of it; pass --batch-size 256 to reproduce v6 exactly.
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--class-balance-beta", type=float, default=0.9999)
    p.add_argument("--split-seed", type=int, default=42)

    p.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    p.add_argument("--run-name", default="mobilenetv3-hard-labels")
    p.add_argument(
        "--active-train-list", default=None,
        help="the retraining loop's growing training universe (default: data/active_train.txt). "
             "Missing is a hard error -- release_batch.py bootstraps it, or pass "
             "--train-from-manifest to train on the manifest's own train split.",
    )
    p.add_argument(
        "--train-from-manifest", action="store_true",
        help="train on the manifest's train split instead of the active training list -- the "
             "clean starting state, before any future-pool batch has been released",
    )
    p.add_argument(
        "--manifest", default=None,
        help="splits.py manifest pinning the frozen holdout (default: data/splits_manifest.json). "
             "The only holdout definition; a missing manifest is a hard error.",
    )
    p.add_argument(
        "--expect-holdout-fingerprint", default=CANONICAL_HOLDOUT_FP,
        help="abort before training if the resolved holdout's content-hash fingerprint differs "
             f"(default: the canonical {CANONICAL_HOLDOUT_FP}). Pass 'any' to skip the check -- "
             "do not do that for a run you intend to promote.",
    )
    p.add_argument(
        "--version-out", default=None,
        help="if given, write the newly registered challenger's version string to this path",
    )
    args = p.parse_args()
    if args.labels_out is None:
        args.labels_out = str(Path(args.weights_dir) / "coin_labels.json")
    if args.train_from_manifest and args.active_train_list is not None:
        p.error("pass either --active-train-list or --train-from-manifest, not both")
    return args


def check_holdout_fingerprint(actual: str, expected: str) -> None:
    """The notebook's cell-3 assertion. Runs before the model exists, so a moved holdout costs
    seconds instead of two hours of GPU time and a challenger nobody can compare to anything.
    """
    if expected.lower() == "any":
        print(f"holdout fingerprint {actual} -- CHECK SKIPPED (--expect-holdout-fingerprint any). "
              "This run's accuracy may not be comparable to @champion's.")
        return
    if actual != expected:
        raise SystemExit(
            f"holdout fingerprint {actual} != expected {expected}. This run would be scored on a "
            "different set than evaluate.py / promote.py, making its accuracy incomparable to "
            "@champion's. Refusing to train.\n"
            "Investigate with: PYTHONPATH=src python3 verify_data_integrity.py"
        )
    print(f"holdout is the expected set ({actual}) -- comparable to @champion")


def log_training_curves(history: list[dict], best_val_acc: float) -> None:
    """The notebook's curve cell. Per-epoch metrics are already in MLflow; this is the at-a-glance
    artifact for a run nobody watched live.
    """
    hist = pd.DataFrame(history).set_index("epoch")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    hist["train_loss"].plot(ax=ax1, title="train loss")
    hist["val_acc"].plot(ax=ax2, title=f"val accuracy (best {best_val_acc:.4f})")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    mlflow.log_figure(fig, "training_curves.png")
    plt.close(fig)


def log_confusion_matrix(cm: np.ndarray, data_dir: str, idx_to_label: dict, num_classes: int,
                         test_acc: float) -> np.ndarray:
    """Log the per-class heatmap and return per-class recall (the CM's row accuracies)."""
    row_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    mlflow.log_metrics({f"class_acc/{idx_to_label[i]}": float(row_acc[i])
                        for i in range(num_classes)})

    class_order, display_names = class_display_order(data_dir, idx_to_label, num_classes)
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(pd.DataFrame(cm[np.ix_(class_order, class_order)],
                             index=display_names, columns=display_names),
                annot=True, fmt="d", cmap="Blues", annot_kws={"size": 7}, ax=ax)
    ax.set_title(f"Confusion Matrix — MobileNetV3 hard labels (frozen holdout)  acc={test_acc:.4f}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)
    return row_acc


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data = prepare_data(
        args.data_dir,
        active_train_list=args.active_train_list,
        manifest=args.manifest,
        from_manifest_train=args.train_from_manifest,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        class_balance_beta=args.class_balance_beta,
    )

    print()
    print(data.summary())
    print(f"train source  : {data.train_source}")
    print(f"classes       : {data.num_classes}")
    print(f"batches/epoch : {len(data.train_loader)}")

    assert data.num_classes == args.num_classes, (
        f"Discovered {data.num_classes} classes in {args.data_dir}, expected {args.num_classes} — "
        "this changes the split/label space, investigate before continuing."
    )
    check_holdout_fingerprint(data.holdout_fingerprint, args.expect_holdout_fingerprint)

    model = build_model(data.num_classes, pretrained=True).to(device)
    print(f"MobileNetV3-Large: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    cb_weights_dev = data.cb_weights.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(data.train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps)
    print(f"{args.epochs} epochs x {steps_per_epoch} batches = {total_steps} steps "
          f"({args.warmup_epochs} warmup epochs, then cosine decay to 0)")
    if args.epochs < SMOKE_RUN_EPOCHS:
        print(f"NOTE: {args.epochs} epochs is a smoke run. The cosine schedule is defined over "
              "epochs x batches, so this trains on a truncated LR curve that never decays -- the "
              "resulting accuracy is not a shorter version of the full-length number, it is a "
              "different and much worse one. Expect promote.py to HOLD.")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=args.run_name) as run:
        print("run_id:", run.info.run_id)
        mlflow.log_params({
            "arch": "mobilenet_v3_large",
            "pretrained": True,
            "num_classes": data.num_classes,
            "input_size": args.input_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "scheduler": "linear_warmup_cosine_decay",
            "warmup_epochs": args.warmup_epochs,
            "grad_clip": args.grad_clip,
            "label_smoothing": args.label_smoothing,
            "class_balance_beta": args.class_balance_beta,
            "sampler": "class_balanced_weighted_random",
            "objective": "cross_entropy_hard_labels",
            "split_seed": args.split_seed,
            "train_source": data.train_source,
            "n_train": len(data.train_idx),
            "n_val": len(data.val_idx),
            "n_holdout": len(data.test_idx),
            # Provenance: which holdout this run's number was actually earned against.
            "holdout_fingerprint": data.holdout_fingerprint,
            "source": "train_hard_labels.py (from notebooks/coin_mobilenet_hard_labels.ipynb)",
        })

        best_val_acc, best_state = 0.0, None
        history = []

        for epoch in range(1, args.epochs + 1):
            t0 = datetime.now()
            model.train()
            running_loss = 0.0

            for imgs, labels in data.train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(imgs), labels, weight=cb_weights_dev,
                                       label_smoothing=args.label_smoothing)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

            val_acc = evaluate_hard(model, data.val_loader, device)
            train_loss = running_loss / len(data.train_loader)
            lr_now = optimizer.param_groups[0]["lr"]
            improved = val_acc > best_val_acc
            if improved:
                best_val_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

            mlflow.log_metrics({"train_loss": train_loss, "val_acc": val_acc, "lr": lr_now},
                               step=epoch)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc,
                            "lr": lr_now})
            # Every epoch, unlike train.py's throttled print: nobody is watching a DAG run live,
            # so the task log is the only record of the curve's shape.
            print(f"ep {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  val {val_acc:.4f}  "
                  f"lr {lr_now:.2e}  [{datetime.now() - t0}]" + ("  <--" if improved else ""),
                  flush=True)

        print(f"\nBest val acc: {best_val_acc:.4f}")
        log_training_curves(history, best_val_acc)

        # --- score the best-val weights on the canonical holdout ------------------------------
        model.load_state_dict(best_state)
        model.eval()
        ckpt_path = save_checkpoint(best_state, args.weights_dir, run.info.run_id)
        print(f"weights -> {ckpt_path}")

        test_acc = evaluate_hard(model, data.test_loader, device)
        print(f"Holdout acc: {test_acc:.4f}  on {len(data.test_idx)} images "
              f"(fingerprint {data.holdout_fingerprint})")
        mlflow.log_metrics({"best_val_acc": best_val_acc, "test_acc": test_acc})

        cm = confusion_matrix(model, data.test_loader, device, data.num_classes)
        row_acc = log_confusion_matrix(cm, args.data_dir, data.idx_to_label, data.num_classes,
                                       test_acc)
        worst = sorted(range(data.num_classes), key=lambda i: row_acc[i])[:10]
        print("\nWeakest 10 classes:")
        for i in worst:
            print(f"  {row_acc[i]:.3f}  {data.idx_to_label[i]}  (n={cm.sum(axis=1)[i]})")

        save_labels(data.idx_to_label, args.labels_out)
        print(f"\nSaved label mapping -> {args.labels_out}")

        # --- register as a challenger ---------------------------------------------------------
        model.eval()
        model.to("cpu")
        example_input = torch.randn(1, 3, args.input_size, args.input_size)
        with torch.no_grad():
            example_output = model(example_input)
        signature = infer_signature(example_input.numpy(), example_output.numpy())

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",  # if this errors on an older client: change to artifact_path="model"
            signature=signature,
            input_example=example_input.numpy(),
            serialization_format="pickle",  # avoids pt2 (needs torch>=2.4), matches v1
        )

    mv = mlflow.register_model(model_info.model_uri, MODEL_NAME)
    print(f"Registered {MODEL_NAME} v{mv.version} — challenger only, @champion unchanged")
    if args.version_out:
        Path(args.version_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.version_out).write_text(mv.version)
        print(f"Wrote challenger version -> {args.version_out}")


if __name__ == "__main__":
    main()
