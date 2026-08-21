"""Knowledge-distillation training: v6 teacher -> MobileNetV3-Large student.

Converted from notebooks/emp_model_knowledge_distilation.ipynb. Registers a new
model version in the MLflow registry as a challenger — it does NOT set or move
the @champion alias. Promotion is a separate, deliberate step.
"""
from __future__ import annotations

import argparse
import copy
import math
import re
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
    CoinDistilDataset,
    class_balanced_weights,
    discover_dataset,
    split_dataset,
    weighted_sampler,
)
from coin_clf.labels import save_labels
from coin_clf.model import build_model
from coin_clf.teacher import CoinHeadV6, SubHead
from coin_clf.transforms import train_transform, val_transform
from checkpoint import save_checkpoint

EXPERIMENT_NAME = "coin-classifier"
MODEL_NAME = "coin-classifier"

# Per-cluster sub-classifier redistribution weights, tuned in v6.
SUB_WEIGHTS = {"QAC": 0.7, "VAG": 0.9, "SDP": 0.8, "TETRARCHY": 0.2, "SEVERAN_HARD": 1.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/home/david/coin/FOR_TRAINNING")
    p.add_argument("--weights-dir", default="/home/david/coin/weights")
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
    p.add_argument("--distill-temp", type=float, default=4.0)
    p.add_argument("--distill-alpha", type=float, default=0.7)
    p.add_argument("--class-balance-beta", type=float, default=0.9999)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--teacher-test-acc", type=float, default=0.9271,
                    help="v6 full-pipeline test accuracy, used for the compression-gap metric")
    p.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = p.parse_args()
    if args.labels_out is None:
        args.labels_out = str(Path(args.weights_dir) / "coin_labels.json")
    return args


def precompute_teacher_soft_labels(data_dir: Path, weights_dir: Path, num_classes: int, device) -> Path:
    """Runs the full v6 pipeline (ensemble + sub-classifiers) once and caches soft labels to disk."""
    soft_labels_path = Path(data_dir) / "teacher_soft_labels.pt"
    if soft_labels_path.exists():
        print("Teacher soft labels already on disk — skipping generation.")
        return soft_labels_path

    v41_data = torch.load(Path(data_dir) / "cradio_v41_embeddings.pt")
    crops_data = torch.load(Path(data_dir) / "cradio_v5_crops.pt")

    feats_full = v41_data["features"].to(device)
    feats_portrait = crops_data["features_portrait"].to(device)
    feats_legend = crops_data["features_legend"].to(device)
    print(f"Embeddings loaded: {feats_full.shape[0]} images")

    ensemble_states = torch.load(Path(weights_dir) / "emp_model_v6_ensemble.pth", map_location=device)
    teacher_ensemble = []
    for state in ensemble_states:
        m = CoinHeadV6(num_classes).to(device)
        m.load_state_dict(state)
        m.eval()
        teacher_ensemble.append(m)
    print(f"Loaded {len(teacher_ensemble)}-head main ensemble.")

    sub_save = torch.load(Path(weights_dir) / "emp_model_v6_sub.pth", map_location=device)
    sub_classifiers = {}
    for name, info in sub_save.items():
        cluster_idxs = info["idxs"]
        sub_ens = []
        for state in info["states"]:
            m = SubHead(3840, len(cluster_idxs)).to(device)
            m.load_state_dict(state)
            m.eval()
            sub_ens.append(m)
        sub_classifiers[name] = {"ensemble": sub_ens, "idxs": cluster_idxs}
    print(f"Loaded sub-classifiers: {list(sub_classifiers.keys())}")

    BATCH = 512
    N = feats_full.shape[0]
    all_soft = torch.zeros(N, num_classes)

    with torch.no_grad():
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            xf = feats_full[start:end]
            xp = feats_portrait[start:end]
            xl = feats_legend[start:end]

            probs = None
            for m in teacher_ensemble:
                p = F.softmax(m(xf, xp, xl), dim=-1)
                probs = p if probs is None else probs + p
            probs /= len(teacher_ensemble)

            for cluster_name, info in sub_classifiers.items():
                sw = SUB_WEIGHTS.get(cluster_name, 0.0)
                if sw == 0.0:
                    continue
                cluster_idxs = info["idxs"]
                sub_p = None
                for m in info["ensemble"]:
                    p = F.softmax(m(xf), dim=-1)
                    sub_p = p if sub_p is None else sub_p + p
                sub_p /= len(info["ensemble"])

                mass = probs[:, cluster_idxs].sum(dim=-1, keepdim=True)
                redistributed = mass * sub_p
                blended = (1 - sw) * probs[:, cluster_idxs] + sw * redistributed
                for local_i, global_i in enumerate(cluster_idxs):
                    probs[:, global_i] = blended[:, local_i]
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

            all_soft[start:end] = probs.cpu()
            if (start // BATCH) % 10 == 0:
                print(f"  {end}/{N}")

    torch.save(all_soft, soft_labels_path)
    print(f"Saved teacher soft labels -> {soft_labels_path}  shape={all_soft.shape}")

    del teacher_ensemble, sub_classifiers, feats_full, feats_portrait, feats_legend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return soft_labels_path


def make_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def distillation_loss(student_logits, teacher_probs, hard_labels, temp, alpha, cb_weights):
    student_log_soft = F.log_softmax(student_logits / temp, dim=-1)
    soft_loss = F.kl_div(student_log_soft, teacher_probs, reduction="batchmean") * (temp ** 2)
    hard_loss = F.cross_entropy(student_logits, hard_labels, weight=cb_weights)
    return alpha * soft_loss + (1 - alpha) * hard_loss


@torch.no_grad()
def evaluate_hard(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for imgs, _, hard_labels in loader:
        imgs, hard_labels = imgs.to(device), hard_labels.to(device)
        correct += (model(imgs).argmax(-1) == hard_labels).sum().item()
        total += hard_labels.size(0)
    return correct / total


@torch.no_grad()
def confusion_matrix(model, loader, device, num_classes: int) -> np.ndarray:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for imgs, _, hard_labels in loader:
        imgs, hard_labels = imgs.to(device), hard_labels.to(device)
        preds = model(imgs).argmax(-1)
        for t, p in zip(hard_labels.cpu(), preds.cpu()):
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

    filepaths, all_labs, label_encoder, idx_to_label, num_classes = discover_dataset(args.data_dir)
    assert num_classes == args.num_classes, (
        f"Discovered {num_classes} classes in {args.data_dir}, expected {args.num_classes} — "
        "this changes the frozen split/label space, investigate before continuing."
    )
    print(f"{len(filepaths)} images | {num_classes} classes (after GORDIAN merge)")

    soft_labels_path = precompute_teacher_soft_labels(args.data_dir, args.weights_dir, num_classes, device)
    teacher_soft = torch.load(soft_labels_path)
    print(f"Teacher soft labels: {teacher_soft.shape}")

    train_idx, val_idx, test_idx = split_dataset(all_labs, test_size=0.2, random_state=args.split_seed)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    train_ds = CoinDistilDataset(filepaths, all_labs, teacher_soft, train_idx, train_transform)
    val_ds = CoinDistilDataset(filepaths, all_labs, teacher_soft, val_idx, val_transform)
    test_ds = CoinDistilDataset(filepaths, all_labs, teacher_soft, test_idx, val_transform)

    train_labs_idx = all_labs[train_idx]
    cb_weights = class_balanced_weights(train_labs_idx, num_classes, beta=args.class_balance_beta)
    train_sampler = weighted_sampler(train_labs_idx, cb_weights, num_samples=len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
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

    with mlflow.start_run(run_name="distill-mobilenetv3") as run:
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
            "distill_temp": args.distill_temp,
            "distill_alpha": args.distill_alpha,
            "class_balance_beta": args.class_balance_beta,
            "sampler": "class_balanced_weighted_random",
            "teacher": "v6_ensemble+subheads",
            "split_seed": args.split_seed,
        })

        best_val_acc, best_state = 0.0, None
        for epoch in range(1, args.epochs + 1):
            t0 = datetime.now()
            model.train()
            running_loss = 0.0

            for imgs, soft_labels, hard_labels in train_loader:
                imgs = imgs.to(device)
                soft_labels = soft_labels.to(device)
                hard_labels = hard_labels.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = distillation_loss(logits, soft_labels, hard_labels,
                                          args.distill_temp, args.distill_alpha, cb_weights_dev)
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
        compression_gap_pp = (args.teacher_test_acc - test_acc) * 100
        print(f"Best val acc: {best_val_acc:.4f}  Test acc: {test_acc:.4f}  "
              f"Compression gap: {compression_gap_pp:.2f}pp")
        mlflow.log_metrics({
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "compression_gap_pp": compression_gap_pp,
        })

        cm = confusion_matrix(model, test_loader, device, num_classes)
        row_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
        mlflow.log_metrics({f"class_acc/{idx_to_label[i]}": float(row_acc[i]) for i in range(num_classes)})

        class_order, display_names = class_display_order(args.data_dir, idx_to_label, num_classes)
        cm_ordered = cm[np.ix_(class_order, class_order)]
        fig, ax = plt.subplots(figsize=(22, 18))
        sns.heatmap(pd.DataFrame(cm_ordered, index=display_names, columns=display_names),
                    annot=True, fmt="d", cmap="Blues", annot_kws={"size": 7}, ax=ax)
        ax.set_title(f"Confusion Matrix — Distillation Student (Test Set)  acc={test_acc:.4f}")
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
            serialization_format="pickle",   # <-- add this; avoids pt2 (needs torch>=2.4), matches v1
        )

    mv = mlflow.register_model(model_info.model_uri, MODEL_NAME)
    print(f"Registered {MODEL_NAME} v{mv.version} — challenger only, @champion unchanged")


if __name__ == "__main__":
    main()
