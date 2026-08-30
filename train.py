"""Supervised training: MobileNetV3-Large on the verified-clean coin corpus.

Trains on ONE boundary -- coin_clf.data.active_split -- and scores on ONE holdout, the
splits.py manifest's. There is no raw-tree path, no second split definition, and no way to
reach either by omitting a flag.

Registers a new model version in the MLflow registry as a challenger; it does NOT set or move
the @champion alias. Promotion is a separate, deliberate step (promote.py).

NO DISTILLATION. This used to distill from the v6 teacher ensemble via cached soft labels. The
teacher was trained on contaminated data, so everything distilled from it inherited the
contamination -- teacher, soft labels and the compression-gap metric are all gone, and the
student now learns from hard labels alone.
"""
from __future__ import annotations

import argparse
import copy
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from mlflow.models import infer_signature
from torch.utils.data import DataLoader

from coin_clf.data import (
    CoinImageDataset,
    active_split,
    class_balanced_weights,
    weighted_sampler,
)
from coin_clf.hashing import fingerprint, hash_many
from coin_clf.labels import save_labels
from coin_clf.model import build_model
from coin_clf.transforms import train_transform, val_transform
from checkpoint import save_checkpoint

# Repo-relative, not machine-absolute: this file is run by the Airflow DAG, by hand, and (for
# --help / arg parsing) by CI, each from a different working directory. Same anchor pattern as
# verify_data_integrity.py and coin_clf.data.
REPO_ROOT = Path(__file__).resolve().parent

EXPERIMENT_NAME = "coin-classifier"
MODEL_NAME = "coin-classifier"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "FOR_TRAINNING"))
    p.add_argument("--weights-dir", default=str(REPO_ROOT / "weights"))
    p.add_argument("--labels-out", default=None,
                    help="default: <weights-dir>/coin_labels.json")
    p.add_argument("--num-classes", type=int, default=51)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--class-balance-beta", type=float, default=0.9999)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
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
        "--version-out", default=None,
        help="if given, write the newly registered challenger's version string to this path",
    )
    args = p.parse_args()
    if args.labels_out is None:
        args.labels_out = str(Path(args.weights_dir) / "coin_labels.json")
    if args.train_from_manifest and args.active_train_list is not None:
        p.error("pass either --active-train-list or --train-from-manifest, not both")
    return args


@dataclass
class TrainingData:
    """Everything the training loop needs, built from the ONE canonical data path.

    Exists so the training notebook imports this instead of rebuilding its own datasets,
    sampler and loaders. The notebook and the pipeline drifting apart is what produced two
    different ideas of the holdout last time; there is now nothing left for them to disagree
    about, because they call the same function.
    """

    filepaths: list
    all_labs: torch.Tensor
    label_encoder: dict
    idx_to_label: dict
    num_classes: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    cb_weights: torch.Tensor
    train_source: str
    holdout_fingerprint: str

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train_idx), "val": len(self.val_idx),
                "holdout": len(self.test_idx)}

    def summary(self) -> str:
        s = self.sizes
        return (f"train={s['train']}  val={s['val']}  holdout={s['holdout']}  "
                f"holdout_fingerprint={self.holdout_fingerprint}")


def prepare_data(
    data_dir,
    *,
    active_train_list=None,
    manifest=None,
    from_manifest_train: bool = False,
    split_seed: int = 42,
    batch_size: int = 128,
    class_balance_beta: float = 0.9999,
    num_workers: int = 4,
) -> TrainingData:
    """Resolve the canonical split and build the loaders. The single seam both train.py's CLI
    and notebooks/coin_mobilenet_hard_labels.ipynb go through.

    The holdout fingerprint it returns is the content-hash fingerprint of the holdout images --
    print it at the top of any run to make it visible, on the face of the run, which holdout was
    actually scored.
    """
    filepaths, all_labs, label_encoder, idx_to_label, num_classes, train_idx, val_idx, test_idx = active_split(
        data_dir,
        active_train_list=active_train_list,
        manifest_path=manifest,
        random_state=split_seed,
        from_manifest_train=from_manifest_train,
    )

    train_ds = CoinImageDataset(filepaths, all_labs, train_idx, train_transform)
    val_ds = CoinImageDataset(filepaths, all_labs, val_idx, val_transform)
    test_ds = CoinImageDataset(filepaths, all_labs, test_idx, val_transform)

    train_labs_idx = all_labs[train_idx]
    cb_weights = class_balanced_weights(train_labs_idx, num_classes, beta=class_balance_beta)
    train_sampler = weighted_sampler(train_labs_idx, cb_weights, num_samples=len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    holdout_fp = fingerprint(hash_many([filepaths[i] for i in test_idx]).values())
    train_source = "manifest train split" if from_manifest_train else str(
        active_train_list or "data/active_train.txt"
    )

    return TrainingData(
        filepaths=filepaths, all_labs=all_labs, label_encoder=label_encoder,
        idx_to_label=idx_to_label, num_classes=num_classes,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        cb_weights=cb_weights, train_source=train_source, holdout_fingerprint=holdout_fp,
    )


def make_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_hard(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += (model(imgs).argmax(-1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def confusion_matrix(model, loader, device, num_classes: int) -> np.ndarray:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(-1)
        for t, p in zip(labels.cpu(), preds.cpu()):
            cm[t.item(), p.item()] += 1
    return cm


def class_display_order(data_dir: Path, idx_to_label: dict, num_classes: int):
    name_to_prefix = {
        re.sub(r"^\d+_", "", p.name): int(re.match(r"^(\d+)_", p.name).group(1))
        for p in Path(data_dir).iterdir()
        if p.is_dir() and (p / "side_a").exists()
    }
    class_order = sorted(range(num_classes), key=lambda i: name_to_prefix.get(idx_to_label[i], 99))
    display_names = [f"{name_to_prefix.get(idx_to_label[i], 99):02d}_{idx_to_label[i]}" for i in class_order]
    return class_order, display_names


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
    num_classes, idx_to_label = data.num_classes, data.idx_to_label
    train_idx, val_idx, test_idx = data.train_idx, data.val_idx, data.test_idx
    train_loader, val_loader, test_loader = data.train_loader, data.val_loader, data.test_loader
    cb_weights = data.cb_weights

    print(f"training on {data.train_source}  "
          f"holdout={args.manifest or 'data/splits_manifest.json'}")
    assert num_classes == args.num_classes, (
        f"Discovered {num_classes} classes in {args.data_dir}, expected {args.num_classes} — "
        "this changes the split/label space, investigate before continuing."
    )
    print(f"{len(data.filepaths)} images | {num_classes} classes (after GORDIAN merge)")
    print(data.summary())
    print(f"Loaders ready. Batches per epoch: {len(train_loader)}")

    model = build_model(num_classes, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV3-Large: {total_params / 1e6:.1f}M params")

    cb_weights_dev = cb_weights.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="mobilenetv3-supervised") as run:
        mlflow.log_params({
            "arch": "mobilenet_v3_large",
            "pretrained": True,
            "num_classes": num_classes,
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
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_holdout": len(test_idx),
            # Provenance: which holdout this number was actually earned against.
            "holdout_fingerprint": data.holdout_fingerprint,
        })

        best_val_acc, best_state = 0.0, None
        for epoch in range(1, args.epochs + 1):
            t0 = datetime.now()
            model.train()
            running_loss = 0.0

            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels, weight=cb_weights_dev,
                                       label_smoothing=args.label_smoothing)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

            val_acc = evaluate_hard(model, val_loader, device)
            train_loss = running_loss / len(train_loader)
            lr_now = optimizer.param_groups[0]["lr"]
            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

            mlflow.log_metrics({"train_loss": train_loss, "val_acc": val_acc, "lr": lr_now}, step=epoch)
            if epoch <= 5 or epoch % 5 == 0 or improved:
                print(f"ep {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  val {val_acc:.4f}  "
                      f"lr {lr_now:.2e}  [{datetime.now() - t0}]" + (" <--" if improved else ""))

        model.load_state_dict(best_state)
        model.eval()
        ckpt_path = save_checkpoint(best_state, args.weights_dir, run.info.run_id)
        print(f"Saved student weights -> {ckpt_path}")
        test_acc = evaluate_hard(model, test_loader, device)
        print(f"Best val acc: {best_val_acc:.4f}  Holdout acc: {test_acc:.4f}")
        mlflow.log_metrics({
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
        })

        cm = confusion_matrix(model, test_loader, device, num_classes)
        row_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
        mlflow.log_metrics({f"class_acc/{idx_to_label[i]}": float(row_acc[i]) for i in range(num_classes)})

        class_order, display_names = class_display_order(args.data_dir, idx_to_label, num_classes)
        cm_ordered = cm[np.ix_(class_order, class_order)]
        fig, ax = plt.subplots(figsize=(22, 18))
        sns.heatmap(pd.DataFrame(cm_ordered, index=display_names, columns=display_names),
                    annot=True, fmt="d", cmap="Blues", annot_kws={"size": 7}, ax=ax)
        ax.set_title(f"Confusion Matrix — MobileNetV3 (frozen holdout)  acc={test_acc:.4f}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
        fig.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        save_labels(idx_to_label, args.labels_out)
        print(f"Saved label mapping -> {args.labels_out}")

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
            serialization_format="pickle",   # avoids pt2 (needs torch>=2.4), matches v1
        )

    mv = mlflow.register_model(model_info.model_uri, MODEL_NAME)
    print(f"Registered {MODEL_NAME} v{mv.version} — challenger only, @champion unchanged")
    if args.version_out:
        Path(args.version_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.version_out).write_text(mv.version)
        print(f"Wrote challenger version -> {args.version_out}")


if __name__ == "__main__":
    main()
