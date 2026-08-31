# Roman Coin Classifier

Classifies photos of Roman Imperial coins by emperor (51 classes, GORDIAN II
merged into GORDIAN I). It began as a series of notebooks that iterated on
feature extraction, head architecture and ensembling; what is in the repo now is
the pipeline that grew around one small deployable model:

* a dataset carved **once** into train / frozen-holdout / future-pool, with the
  carve provable from the bytes on disk (`verify_data_integrity.py`),
* an MLflow registry with a champion/challenger **promotion gate** that re-scores
  both models on that one holdout before moving `@champion`,
* a FastAPI server that loads `@champion` from the registry at startup and logs
  every prediction to SQLite,
* an Evidently **drift check** over that log, and
* two Airflow DAGs that close the loop: drift → retrain → evaluate → promote.

The serving model is a MobileNetV3-Large trained on hard labels — 4.27M
parameters once the 1000-class head is replaced with a 51-class one, small enough
that CPU inference is fine for single-image requests.
`@champion` at the time of writing is registry **v10**, holdout accuracy
**0.9201**, produced by a `retrain_coin_clf` run (120 epochs, batch 128) rather
than by hand. It is resolved by alias at runtime, never baked into the image —
`make champion` prints the current one. The pipeline has also *declined* a
promotion: v11 scored 0.9150 on the same holdout and the gate held it, which is
the behaviour the `--margin 0.005` exists for.

## Model history

These are the **notebook-era** numbers, kept because they are the record of what
was tried. They are **not comparable to the registry's accuracies**: they predate
the duplicate/label-conflict cleanup (`clean_duplicates.py`, 4,342 files
quarantined) and the single-holdout carve (`splits.py`), and the ad-hoc splits
they were scored on leaked — one such raw-tree 80/20 draw overlapped the training
set by 5,689 images (see `export.py`'s docstring), and the v6 teacher itself
trained on contaminated data. Read the table as a ranking of approaches, not as
accuracy anyone should quote today.

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

Full write-up of what worked and what didn't is in `docs/improvement_or_not.md`
(kept locally, not pushed).

**The distillation path is retired.** The v6 teacher was trained on contaminated
data, so the soft labels and everything distilled from them inherited it —
`train.py` no longer imports `coin_clf.teacher` at all, and
`verify_data_integrity.py` (section C5) checks that it can't come back.
`coin_clf/teacher.py` survives only so the frozen v6 checkpoints can still be
loaded for inspection. The live recipe is
`notebooks/coin_mobilenet_hard_labels.ipynb` and its headless twin
`train_hard_labels.py`: MobileNetV3-Large, class-balanced cross-entropy with
label smoothing, 120 epochs, warmup + cosine decay, scored on the canonical
11,559-image holdout.

## Dataset and its boundaries

Every number below is re-derived from file bytes by `verify_data_integrity.py`;
the full output of the last run is tracked at `docs/verify_data_integrity_run.txt`.

```
62,134  original images
 4,342  quarantined by clean_duplicates.py (byte-duplicates + cross-label conflicts)
57,792  clean corpus (data/clean_files.txt) — hash-unique, all decode end-to-end

carved once by splits.py, seed 42, into data/splits_manifest.json:
  34,674  train        (~60%)
  11,559  frozen holdout (~20%)  fingerprint c47d3f321a0e13c2 — THE holdout
  11,559  future pool  (~20%)    withheld; stands in for data arriving after launch
```

The future pool is released in batches of 5,000 (`BATCH_SIZE` in
`dags/retrain_coin_clf.py`); `data/future_pool_cursor.json` counts batch
*numbers*, so the size must stay fixed for as long as one cursor is in use.
Two batches are released, which is why `data/active_train.txt` currently holds
44,674 files (34,674 manifest train + 10,000 released) and 1,559 future-pool
images remain unreleased.

Three rules the code enforces rather than documents:

* **Clean by default.** Every entry point in `coin_clf.data` resolves to the
  clean list; a missing clean list or manifest raises `RawTreeError` instead of
  falling back to the raw tree. Reaching unfiltered data requires typing
  `allow_raw_tree=True` at the call site.
* **One holdout.** `build_manifest_holdout` is the only definition in live code.
  The two that used to compete with it (`frozen_split`, `build_test_dataset`) are
  deleted, and there is a test plus an integrity check asserting they stay deleted.
* **Disjointness by content hash, not filename.** `active_split` re-hashes both
  sides and refuses to return if any training image is byte-identical to a
  holdout image.

## Serving

`app/main.py` is a FastAPI app. It loads the model from the MLflow registry —
`models:/coin-classifier@champion`, the full `nn.Module`, not a state dict — in a
FastAPI **lifespan** (startup), not at import. That is what lets the module be
imported with no registry reachable, which is how the tests inject a fake bundle
through `app.dependency_overrides` and how CI's `--network none` import smoke test
works.

- `GET /health` — `status`, `device`, `num_classes`, `model_name`, `model_version`
- `POST /predict?topk=N` — multipart image upload → `{model_version, predictions: [{label, probability}]}`
  - optional `X-Traffic-Source` header: a monitoring tag (the replay script sets
    it to separate a normal from a deliberately skewed batch). It changes nothing
    about the prediction, only how the logged row is grouped later.

Preprocessing is `coin_clf.transforms.val_transform`, the same object training
uses: `Resize(256) -> CenterCrop(224) -> ImageNet normalize`. Labels come from
`app/coin_labels.json` (index → emperor name), overridable with `LABELS_PATH`.

Every request writes one row — version, label, confidence, width/height/mode,
latency, source — to the SQLite prediction log (`coin_clf.prediction_log`). That
log is the only thing observing this model in production and the sole input to the
drift check. It is strictly an observer: `log()` never raises, `/predict` wraps it
anyway, and it is the last thing the endpoint does. Image metadata is read from
the upload **before** `.convert("RGB")`, or every row would report `mode="RGB"`
forever and the mode drift signal would be permanently dead.

Configuration (all environment variables, with these defaults):
`MLFLOW_TRACKING_URI=http://127.0.0.1:5000`, `MODEL_NAME=coin-classifier`,
`MODEL_ALIAS=champion`, `LABELS_PATH=app/coin_labels.json`,
`PREDICTION_LOG_PATH=<repo>/outputs/monitoring/predictions.db`.

### Run it

The container needs the tracking server (for the alias) and its artifact store
(for the weights), so `make serve` runs it on the host network with `~/.aws`
mounted read-only and the prediction-log directory bind-mounted — without that
mount the log dies with the `--rm` container and the host-side drift check has
nothing to read.

```bash
make mlflow          # terminal 1: tracking server, SQLite backend + S3 artifacts
make build           # docker build -f serve.Dockerfile -t coin-classifier:latest .
make serve           # terminal 2: the container, wired to both
make health          # curl :8000/health
```

```bash
curl -X POST "http://localhost:8000/predict?topk=3" \
  -F "file=@/path/to/coin.jpg;type=image/jpeg"
```

The equivalent raw `docker run`, if you'd rather not use the Makefile:

```bash
docker run --rm --network host \
  -v ~/.aws:/root/.aws:ro \
  -v "$PWD/outputs/monitoring:/srv/outputs/monitoring" \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e PREDICTION_LOG_PATH=/srv/outputs/monitoring/predictions.db \
  coin-classifier:latest
```

The image is CPU-only (`python:3.10-slim` + CPU torch/torchvision wheels) for
portability — inference on this model size is fast enough without a GPU. For GPU
serving, swap the base image for an `nvidia/cuda` runtime and install the `cu121`
torch wheels in `serve.Dockerfile`. `.dockerignore` is an allowlist, not a
denylist: it excludes everything and re-includes exactly the four paths the
Dockerfile copies. The denylist it replaced had fallen behind the repo and was
streaming a 50 GB context (the DVC cache, `weights/`, the Airflow venv) to build
a ~1 GB image; under an allowlist a new large directory is ignored by default and
a new `COPY` that needs something has to say so.

### Run locally without Docker

Requires Python 3.10 with `torch`, `torchvision`, and the packages in
`requirements-serve.txt`, plus a reachable tracking server:

```bash
pip install -r requirements-serve.txt
pip install -e . --no-deps
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Retraining pipeline

`dags/retrain_coin_clf.py` — `schedule=None`, one run per simulated batch
arrival, `max_active_runs=1`, `retries=0` (every task has a side effect that is
not safe to replay: appending to the active-train list, advancing the cursor,
registering a version, moving an alias).

```
release_batch → validate → gate_on_validation → train → evaluate → promote
```

Every step shells out to an existing CLI with `PYTHONPATH=src /usr/bin/python3`,
which is exactly what the Makefile does — a run reproduced by hand is the run
Airflow makes, not a near-miss. Airflow's own process never imports torch.

* **release_batch** — pops the next 5,000-image future-pool batch, appends it to
  `data/active_train.txt`, writes the batch's `(path,label)` rows for validation.
  A drained pool is a hard error, not a "bad batch".
* **validate** — `python -m coin_clf.validate_batch`: readable (a real decode, not
  a header parse), RGB mode, minimum dimensions, known label, class balance
  (no class over 50% of a batch), no intra-batch duplicates, no leakage against
  the reference set — by SHA-256 over bytes, the same hash `clean_duplicates.py`
  uses. It reports a verdict; it never raises for a bad image.
* **gate_on_validation** — short-circuits train/evaluate/promote on an invalid
  batch. The DAG run stays **green**: a rejected batch is a normal outcome
  recorded in `report.json`, not a pipeline failure.
* **train** — `train_hard_labels.py`. Reads `epochs` / `batch_size` from
  `dag_run.conf` directly rather than through `params`, because that override
  depends on an `airflow.cfg` setting; a triggered retrain silently getting the
  3-epoch smoke default would train on a truncated cosine curve and then look
  like a model failure at the gate. It aborts if the holdout fingerprint has
  moved. Registers a challenger; it does not touch `@champion`.
* **evaluate** — scores the challenger on the frozen holdout, logs the metric.
* **promote** — moves `@champion` only if the challenger beats it by
  `--margin 0.005`. Re-scores **both** models fresh (never trusts a logged
  number), is idempotent, bootstraps when there is no champion, and holds
  fail-safe if the champion cannot be scored. A HOLD is logged, not raised.

## Monitoring loop

`dags/monitor_coin_clf.py` — the closed loop.

```
check_serving_version → read_state → drift_check → gate_on_drift → trigger_retrain
```

* **check_serving_version** fails fast if the container is serving something other
  than `@champion`; a blind monitor must look broken rather than green.
* **drift_check** runs `drift_report.py` over the prediction log since the state
  cursor. The reference is **the champion's own predictions on the holdout**, not
  the training labels — comparing predictions to labels would bake the model's
  ~8% error rate into the baseline, and labels carry no confidence at all. It is
  cached per champion version, and production rows are filtered to that version,
  so a v7-vs-v6 comparison can't manufacture drift. Three signals —
  `predicted_class`, `confidence`, and `image_metadata` (width/height/mode) —
  scored with Jensen-Shannon at 0.35 on the categorical columns and normalized
  Wasserstein at 0.25 on the numeric ones. Those thresholds were *measured*
  against a simulated null at n=500 rather than defaulted (the obvious 0.1 would
  fire on every normal batch); the confidence one is marked provisional in the
  source because it has no measured null behind it yet. Output: an Evidently HTML
  report plus a JSON verdict.
* **gate_on_drift** triggers only on `status == "ok" AND drift_detected`.
  `insufficient_data` (under 500 rows), `no_data` and `no_reference` all mean
  "couldn't tell", which is not "no drift" and is certainly not grounds for a
  120-epoch run — that is why the verdict carries a status field and not a bare
  boolean. It also skips (rather than queues) when a retrain is already in flight.
* **trigger_retrain** passes the real recipe: `epochs=120, batch_size=256`, plus
  the reason, so the retrain run permanently records why it exists.

Runaway protection lives in `monitor_state.py`, and it is two mechanisms because
they cover different failures: a **consuming cursor** (advances only when a
verdict was actually rendered, so the same rows can never re-fire) and a
**circuit breaker** (two consecutive triggered retrains that end without a
promotion means retraining is not the fix — drift keeps being reported, but the
trigger stops).

There is no real traffic to this model, so `replay_traffic.py` simulates it over
HTTP against the running container: `normal` mode sends untouched file bytes from
the *unreleased* future pool (unseen by the champion, and the calibration run for
the provisional confidence threshold), `skewed` mode restricts classes, downsizes
and grayscales. Requests go over HTTP rather than through in-process inference on
purpose — the point is to exercise the real serving path, decode and log write
included. It is read-only with respect to pipeline state: it reads the cursor,
never advances it, and never appends to `active_train.txt`.

## Layout

Code sits in two layers, deliberately. `src/coin_clf/` is the shared library — the only
installed package, and everything the serving image and every script imports. The pipeline
CLIs stay at the repo root: they need `splits.py`'s `DatasetSplits`/`FuturePool`, and
`coin_clf` must never import a root script, so they live one level above the package rather
than inside it. The DAGs shell out to them by name (`PYTHONPATH=src python3 train_hard_labels.py`),
which is also how the Makefile runs them, so a run reproduced by hand is the run Airflow makes.

```
src/coin_clf/  the shared library — model, transforms, data/split access, labels, hashing,
               image metadata, batch validation, prediction log, v6 teacher (load-only)
app/           FastAPI serving app (main.py, coin_labels.json)
dags/          Airflow DAGs — retrain_coin_clf (release → validate → train → evaluate →
               promote) and monitor_coin_clf (drift → conditional retrain trigger)
*.py at root   the pipeline CLIs — see "Root scripts" below
tests/         the whole pytest suite, plus fixtures/make_tree.py (a ~60-image stand-in
               for the real tree, so verify_data_integrity.py can be exercised in CI)
scripts/       one-off utilities (seed_champion.py)
notebooks/     archived model-development notebooks (emp_model_v2..v7, distillation, the live
               hard-labels recipe, gradcam, predict, ...) — history, not maintained code;
               excluded from ruff and pytest
docs/          tracked: folders.txt, ProjectBook.docx.pdf, verify_data_integrity_run.txt
               (the full integrity run over the real tree)
.github/       CI workflow — the merge gate
makefile       control panel: mlflow, airflow, build, serve, train, retrain, drift, health
.airflow_env.sh  AIRFLOW_HOME / DAGs folder / port + the airflow venv, sourced by `make airflow`
```

Generated or too large to track — gitignored, except where noted:

```
data/          the dataset itself is gitignored and DVC-tracked (FOR_TRAINNING/, archives/,
               quarantine/), but the small files that DEFINE the splits are tracked:
               splits_manifest.json, clean_files.txt, active_train.txt,
               future_pool_cursor.json, drop_log.csv
weights/       all .pth / .onnx checkpoints
mlflow/        local MLflow tracking store (mlflow.db)
outputs/       generated artifacts — monitoring/ (prediction log, drift reports, verdicts,
               monitor_state.json), dag_runs/, gradcam_out/, misclassified/
docs/          working notes (improvement_or_not.md, notebook_summary.md, ...) are local-only
misc/          loose non-project images
.venv-airflow/ the Airflow interpreter — deliberately separate from the torch/mlflow one
```

Every notebook that is part of this repo's history uses **absolute paths**
(`/home/david/coin/FOR_TRAINNING`, `/home/david/coin/weights/emp_model_*.pth`)
rather than paths relative to the notebook's own location — this matters because
VS Code's Jupyter extension sets a notebook's `cwd` to wherever the `.ipynb` file
lives, so a bare relative filename would silently break the moment the notebook
moved into `notebooks/`. All checkpoint load/save cells and `predict.ipynb`'s
`prediction_result.png` output were updated to absolute paths as part of that
move. (Two exceptions: `CoinClassifierDontEdit.ipynb` and `emp_model.ipynb` are
the original Colab notebooks and still carry `/content/drive` paths; and
`coin_mobilenet_hard_labels.ipynb`, the current recipe, resolves the repo root
from `cwd` on purpose so it imports the same `train.py` seams the CLI does.)

Three compatibility symlinks at the repo root keep the absolute paths working:
`FOR_TRAINNING` → `data/FOR_TRAINNING`, `misclassified` → `outputs/misclassified`,
and `prediction_result.png` → `outputs/prediction_result.png`. The first is not
decoration — `verify_data_integrity.py` (check B1c) asserts the alias discovers
the same file set as `data/FOR_TRAINNING`, so a stale symlink is caught rather
than silently trained through.

## Root scripts

The retraining and monitoring pipeline, as individually runnable CLIs. Grouped by what they do:

**Dataset boundaries** — the code all three leakage incidents came out of.

- `splits.py` — the ONE place the dataset is carved into train / frozen-holdout / future-pool. The manifest it writes is the single definition of the holdout in the codebase.
- `verify_data_integrity.py` — read-only proof, from the bytes on disk, that the partitions are what the code thinks they are. Writes nothing. Three sections: partition integrity, every data-loading entry point actually called and fingerprinted, and unreachability of the deleted raw-tree/rival-holdout paths.
- `check_split_leakage.py` — index-level and content-level checks that the three splits share no image.
- `clean_duplicates.py` — hash-groups the raw tree, collapses exact duplicates, drops bytes filed under more than one emperor.

**Train → evaluate → promote** — what the `retrain_coin_clf` DAG shells out to, in order.

- `release_batch.py` — pops the next future-pool batch and folds it into `data/active_train.txt`.
- `train_hard_labels.py` — the headless twin of `coin_mobilenet_hard_labels.ipynb`, the recipe that earned registry v6; the DAG's training step, and what produced the current champion. Defaults: 120 epochs, batch 128 (a monitor-triggered run passes 256).
- `train.py` — the general-purpose training CLI (`make train`). Defaults: 60 epochs, batch 128.
- `evaluate.py` — scores a registered version on the frozen holdout; `promote.py` re-uses `evaluate_version` so both sides of the gate are scored on the same split.
- `promote.py` — the promotion gate: moves `@champion` only if the challenger clears the margin.
- `checkpoint.py` — safe checkpoint saving for `train.py`; writes a run-id-tied filename and refuses to overwrite, so a local `.pth` always maps back to a known run.

**Production feedback loop** — what `monitor_coin_clf` runs.

- `replay_traffic.py` — replays future-pool images through the live `/predict` endpoint as stand-in production traffic (`--mode normal|skewed`).
- `drift_report.py` — scores a window of the prediction log against the champion's reference distribution; emits an Evidently HTML report plus a machine-readable verdict.
- `monitor_state.py` — the monitoring state machine and its persisted state (cursor + circuit breaker), plus the `--check-serving` preflight. Stdlib-only at module scope, so the Airflow venv can import it.

**Elsewhere**

- `export.py` — ONNX export + int8 quantization, verified against the same frozen holdout.
- `scripts/seed_champion.py` — registers a checkpoint as the `@champion` model in the MLflow registry. This is how v1 got there (from `emp_model_mobilenet_baseline.pth`); every version since was registered by a training run.

## Make targets

| Target | What it does |
|---|---|
| `make mlflow` | tracking server — SQLite backend, S3 artifact root, `--no-serve-artifacts` |
| `make airflow` | `airflow standalone` on port 8090 (not 8080 — see `.airflow_env.sh`) |
| `make build` / `make serve` | build the serving image / run it wired to MLflow + the log mount |
| `make health` | pretty-print `/health` |
| `make champion` | print the version currently aliased `@champion` |
| `make train` / `make retrain` | `train.py` / `train_hard_labels.py`, `ARGS="--epochs 1"` for a smoke run |
| `make install` | `pip install -e . --no-deps` — editable `coin_clf`, leaves your CUDA torch alone |
| `make monitoring-install` | evidently, for the offline drift check (deliberately not in the serving image) |
| `make drift` | run the drift check → HTML report + verdict in `outputs/monitoring` |

## Tests

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
pip install -e . --no-deps    # `make install`
python -m pytest -q -m "not needs_data and not needs_gpu"
```

Install order matters: `pyproject.toml` declares torch/torchvision as
dependencies, so a plain `pip install -e .` re-resolves them from PyPI and drags
in ~2.5 GB of CUDA wheels no test can use. `--no-deps` keeps the CPU wheels.

159 tests, ~35s, all under `tests/`, with no dataset, no GPU, no MLflow server and
no AWS credentials — the same conditions the CI runner has.
`.github/workflows/ci.yml` runs that command on every PR alongside `ruff check .`
and `mypy src/coin_clf` (scoped to the shared library; the root scripts still
report 11 errors, widening it is a follow-up), asserts no `nvidia-*` wheel slipped
into the install, and in a second job builds the serving image and imports it with
`--network none` — the mechanical check that the registry load stays inside the
lifespan and the app stays importable without infrastructure.

Anything that would need the real 57,792-image tree or CUDA gets the `needs_data`
/ `needs_gpu` marker instead of being deleted; nothing carries either one today
(`--strict-markers` makes a typo'd marker an error rather than a silent no-op).
`verify_data_integrity.py` is the one check that cannot run in CI — it re-hashes
the whole tree — so `tests/test_verify_data_integrity.py` runs it against
`tests/fixtures/make_tree.py`'s ~60-image stand-in, injecting one specific fault
per test and asserting the check that should notice actually names it.

## Data & weights

The training dataset (`data/FOR_TRAINNING/`) and all `.pth`/`.onnx` checkpoints
(`weights/`) are gitignored — too large for the repo. The dataset is
version-controlled with DVC instead: `data/FOR_TRAINNING.dvc` is the tracked
pointer, and the remote is `s3://davids-mlops-artifacts-8412/dvc` — the same
bucket MLflow writes model artifacts to. What git tracks is source, notebooks,
this README, and the small JSON/TXT/CSV files under `data/` that define the split
boundaries and record what the cleanup dropped.
