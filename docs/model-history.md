# Model history

[← back to the README](../README.md)

What was tried before any of the pipeline existed, as a ranking of approaches. Read the caveat above the table before quoting any number in it — these accuracies are not comparable to the registry's.

Before any of the above existed, the project was a series of notebooks iterating
on feature extraction, head architecture and ensembling. These numbers are the
record of what was tried. They are **not comparable to the registry's
accuracies**: they predate the duplicate/label-conflict cleanup and the
single-holdout carve, and the ad-hoc splits they were scored on leaked — one such
raw-tree 80/20 draw overlapped the training set by 5,689 images (see `export.py`).
Read the table as a ranking of approaches, not as accuracy anyone should quote.

| Notebook | Approach | Test acc (pre-cleanup) |
|---|---|---|
| `emp_model_v2.ipynb` | Frozen C-RADIO v4-H + linear head | 84.99% |
| `emp_model_v3.ipynb` | + MLP/cosine head, mixup, class-balanced sampling, 5-head ensemble | 87.51% |
| `emp_model_v3_TTA.ipynb` | + TTA feature extraction, 10-head ensemble | 88.82% |
| `emp_model_v4.ipynb` | + ArcFace, patch tokens, hierarchical classifier (frozen backbone) | 88.69% |
| `emp_model_v4.1.ipynb` | + fine-tuned C-RADIO backbone | 92.05% |
| `emp_model_v5.ipynb` | Multi-stream (portrait/legend crops + DINOv2) — regressed | 91.73% |
| `emp_model_v6.ipynb` | + per-stream projection, hard-pair sub-classifiers | **92.71% (best)** |
| `emp_model_v7.ipynb` | Frozen DINOv2-only baseline (phase 1, no fine-tuning follow-up) | 85.78% |
| `emp_model_knowledge_distilation.ipynb` | MobileNetV3-Large distilled from the v6 ensemble | 89.61% |
| `emp_model_mobilenet_baseline.ipynb` | Same MobileNetV3, plain CE (no distillation) — comparison | 89.02% |

**The distillation path is retired.** The v6 teacher trained on contaminated data,
so the soft labels and everything distilled from them inherited it — `train.py` no
longer imports `coin_clf.teacher`, and `verify_data_integrity.py` (section C5)
checks that it cannot come back. The live recipe is
`notebooks/coin_mobilenet_hard_labels.ipynb` and its headless twin
`train_hard_labels.py`: MobileNetV3-Large, class-balanced cross-entropy with label
smoothing, 120 epochs, warmup + cosine decay, scored on the canonical
11,559-image holdout.
