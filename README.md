# Roman Coin Classifier

Classifies photos of Roman Imperial coins by emperor (51 classes, GORDIAN II
merged into GORDIAN I). Built as a series of notebooks that iterate on feature
extraction, head architecture, and ensembling, then compressed into a small
deployable model via knowledge distillation.

## Model history

| Notebook | Approach | Test acc |
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

Full write-up of what worked and what didn't is in `docs/improvement_or_not.md`
(kept locally, not pushed).

v6 has the best raw accuracy but is an ensemble + kNN + sub-classifier
pipeline that's impractical to serve. The **distilled MobileNetV3-Large**
(`emp_model_distil_student.pth`) trades ~3pp of accuracy for a single
forward pass over a 5.4M-param model, and is what the serving app below
loads.

## Serving

`app/main.py` is a FastAPI app that loads `emp_model_distil_student.pth` and
exposes:

- `GET /health` — status, device, class count, checkpoint name
- `POST /predict?topk=N` — multipart image upload → top-k `{label, probability}`

Preprocessing matches training: `Resize(256) -> CenterCrop(224) -> ImageNet normalize`.
Labels are read from `app/coin_labels.json` (index → emperor name).

### Run with Docker

```bash
docker build -f serve.Dockerfile -t coin-classifier .
docker run -d -p 8000:8000 --name coin-classifier coin-classifier
```

```bash
curl http://localhost:8000/health

curl -X POST "http://localhost:8000/predict?topk=3" \
  -F "file=@/path/to/coin.jpg;type=image/jpeg"
```

The image is CPU-only (`python:3.10-slim` + CPU torch/torchvision wheels) for
portability — inference on this model size is fast enough without a GPU. For
GPU serving, swap the base image for an `nvidia/cuda` runtime and install the
`cu121` torch wheels in `serve.Dockerfile`.

### Run locally without Docker

Requires Python 3.10 with `torch`, `torchvision`, and the packages in
`requirements-serve.txt`:

```bash
pip install -r requirements-serve.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Layout

Code sits in two layers, deliberately. `src/coin_clf/` is the shared library — the only
installed package, and everything the serving image and every script imports. The pipeline
CLIs stay at the repo root: they need `splits.py`'s `DatasetSplits`/`FuturePool`, and
`coin_clf` must never import a root script, so they live one level above the package rather
than inside it. The DAGs shell out to them by name (`PYTHONPATH=src python3 train_hard_labels.py`),
which is also how the Makefile runs them, so a run reproduced by hand is the run Airflow makes.

```
src/coin_clf/  the shared library — model, transforms, data/split access, labels, hashing,
               image metadata, batch validation, prediction log
app/           FastAPI serving app (main.py, coin_labels.json)
dags/          Airflow DAGs — retrain_coin_clf (release → validate → train → evaluate →
               promote) and monitor_coin_clf (drift → conditional retrain trigger)
*.py at root   the pipeline CLIs — see "Root scripts" below
tests/         the whole pytest suite
scripts/       one-off utilities (seed_champion.py)
notebooks/     archived model-development notebooks (emp_model_v2..v7, distillation, gradcam,
               predict, ...) — history, not maintained code; excluded from ruff and pytest
makefile       control panel: mlflow, airflow, build, serve, train, retrain, drift, health
```

Generated or too large to track — gitignored, except where noted:

```
data/          the dataset itself is gitignored and DVC-tracked (FOR_TRAINNING/, archives/,
               quarantine/), but the small manifests that DEFINE the splits are tracked:
               splits_manifest.json, clean_files.txt, active_train.txt, future_pool_cursor.json
weights/       all .pth / .onnx checkpoints
mlflow/        local MLflow tracking store (mlflow.db)
outputs/       generated artifacts — monitoring/ (prediction log, drift reports, verdicts),
               dag_runs/, gradcam_out/, misclassified/
docs/          working notes (improvement_or_not.md, notebook_summary.md, ...) are local-only;
               a few references are tracked (folders.txt, ProjectBook.docx.pdf)
misc/          loose non-project images
```

Every notebook uses **absolute paths** (`/home/david/coin/FOR_TRAINNING`,
`/home/david/coin/weights/emp_model_*.pth`) rather than paths relative to
the notebook's own location — this matters because VS Code's Jupyter
extension sets a notebook's `cwd` to wherever the `.ipynb` file lives, so a
bare relative filename would silently break the moment the notebook moved
into `notebooks/`. All checkpoint load/save cells and `predict.ipynb`'s
`prediction_result.png` output were updated to absolute paths as part of
this move. Three compatibility symlinks at the repo root keep those paths
working: `FOR_TRAINNING` → `data/FOR_TRAINNING`, `misclassified` →
`outputs/misclassified`, and `prediction_result.png` →
`outputs/prediction_result.png`. The first is not decoration —
`verify_data_integrity.py` (check B1c) asserts the alias discovers the same
file set as `data/FOR_TRAINNING`, so a stale symlink is caught rather than
silently trained through.

## Root scripts

The retraining and monitoring pipeline, as individually runnable CLIs. Grouped by what they do:

**Dataset boundaries** — the code all three leakage incidents came out of.

- `splits.py` — the ONE place the dataset is carved into train / frozen-holdout / future-pool. The manifest it writes is the single definition of the holdout in the codebase.
- `verify_data_integrity.py` — read-only proof, from the bytes on disk, that the partitions are what the code thinks they are. Writes nothing.
- `check_split_leakage.py` — index-level and content-level checks that the three splits share no image.
- `clean_duplicates.py` — hash-groups the raw tree, collapses exact duplicates, drops bytes filed under more than one emperor.

**Train → evaluate → promote** — what the `retrain_coin_clf` DAG shells out to, in order.

- `release_batch.py` — pops the next future-pool batch and folds it into `data/active_train.txt`.
- `train_hard_labels.py` — the headless twin of the notebook recipe that earned v6; the DAG's training step.
- `train.py` — the general-purpose training CLI (`make train`).
- `evaluate.py` — scores a registered version on the frozen holdout; `promote.py` re-uses `evaluate_version` so both sides of the gate are scored on the same split.
- `promote.py` — the promotion gate: moves `@champion` only if the challenger clears the margin.
- `checkpoint.py` — safe checkpoint saving for `train.py`; never silently clobbers a run's weights.

**Production feedback loop** — what `monitor_coin_clf` runs.

- `replay_traffic.py` — replays future-pool images through the live `/predict` endpoint as stand-in production traffic.
- `drift_report.py` — scores a window of the prediction log against the champion's reference distribution; emits an Evidently HTML report plus a machine-readable verdict.
- `monitor_state.py` — the monitoring state machine and its persisted state (cursor + circuit breaker). Stdlib-only, so the Airflow venv can import it.

**Elsewhere**

- `export.py` — ONNX export + int8 quantization, verified against the same frozen holdout.
- `scripts/seed_champion.py` — registers a checkpoint as the `@champion` model in the MLflow registry.

## Tests

```bash
pip install -r requirements-dev.txt
pip install -e . --no-deps    # `make install` — editable coin_clf, leaves your CUDA torch alone
python -m pytest -q -m "not needs_data and not needs_gpu"
```

Every test lives under `tests/` and runs with no dataset, no GPU, no MLflow server and no AWS
credentials — the same conditions the CI runner has. `.github/workflows/ci.yml` runs that
command on every PR alongside `ruff check .` and `mypy src/coin_clf`, and separately builds the
serving image and imports it with `--network none` to prove the app is importable without
infrastructure. Anything that would need the real 57k-image tree or CUDA gets the `needs_data` /
`needs_gpu` marker instead of being deleted; nothing carries either one today.

## Data & weights

The training dataset (`data/FOR_TRAINNING/`) and all `.pth`/`.onnx`
checkpoints (`weights/`) are gitignored — too large for the repo. The dataset
is version-controlled with DVC instead (`data/FOR_TRAINNING.dvc`); what git
tracks is source, notebooks, this README, and the small JSON/TXT manifests
under `data/` that define the split boundaries.
