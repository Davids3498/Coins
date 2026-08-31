# Development

[← back to the README](../README.md)

Working on this repo: how the code is laid out and why, what every root script does, the make targets, how to run the tests, and where the data and weights actually live.

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
dags/          Airflow DAGs — retrain_coin_clf and monitor_coin_clf
*.py at root   the pipeline CLIs — see "Root scripts" below
tests/         the whole pytest suite, plus fixtures/make_tree.py (a ~60-image stand-in
               for the real tree, so verify_data_integrity.py can be exercised in CI)
scripts/       one-off utilities — seed_champion.py, export_champion.py (freezes the
               registry's @champion into build/champion/ for the cloud image)
infra/         Terraform for the Fargate deploy, split by concern (network, ecr, ecs, iam,
               outputs, variables) + destroy.sh, which tears down and then verifies
notebooks/     archived model-development notebooks — history, not maintained code;
               excluded from ruff and pytest. They use absolute paths, because VS Code
               sets a notebook's cwd to its own directory
docs/          reference material + the tracked integrity run; working notes are local-only
.github/       CI workflow — the merge gate, plus the manual-only deploy job
serve.Dockerfile       the serving image; resolves @champion at startup, bakes in nothing
serve-cloud.Dockerfile FROM the above + the exported champion; MODEL_SOURCE=local, for Fargate
makefile       control panel: mlflow, airflow, build, serve, train, retrain, drift, health
.airflow_env.sh  AIRFLOW_HOME / DAGs folder / port + the airflow venv, sourced by `make airflow`
```

Generated or too large to track — gitignored, except where noted:

```
data/          the dataset is gitignored and DVC-tracked (FOR_TRAINNING/, archives/,
               quarantine/), but the small files that DEFINE the splits are tracked:
               splits_manifest.json, clean_files.txt, active_train.txt,
               future_pool_cursor.json, drop_log.csv
weights/       all .pth / .onnx checkpoints
mlflow/        local MLflow tracking store (mlflow.db)
build/         champion/ — the exported model the cloud image bakes in (~24 MB), plus
               anything setuptools leaves behind
infra/*.tfstate  Terraform state and .terraform/ — local and gitignored, see [cloud-deploy.md](cloud-deploy.md)
outputs/       generated artifacts — monitoring/ (prediction log, drift reports, verdicts,
               monitor_state.json), dag_runs/, gradcam_out/, misclassified/
misc/          loose non-project images
.venv-airflow/ the Airflow interpreter — deliberately separate from the torch/mlflow one
```

Three compatibility symlinks at the repo root (`FOR_TRAINNING`, `misclassified`,
`prediction_result.png`) keep the notebooks' absolute paths working. The first is
not decoration: `verify_data_integrity.py` (check B1c) asserts the alias discovers
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
- `train_hard_labels.py` — the headless twin of `coin_mobilenet_hard_labels.ipynb`; the DAG's training step, and what produced the current champion. Defaults: 120 epochs, batch 128 (a monitor-triggered run passes 256).
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
- `scripts/seed_champion.py` — registers a checkpoint as the `@champion` model in the MLflow registry. This is how v1 got there; every version since was registered by a training run.

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

## Tests and CI

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
pip install -e . --no-deps    # `make install`
python -m pytest -q -m "not needs_data and not needs_gpu"
```

Install order matters: `pyproject.toml` declares torch/torchvision as
dependencies, so a plain `pip install -e .` re-resolves them from PyPI and drags
in ~2.5 GB of CUDA wheels no test can use. `--no-deps` keeps the CPU wheels.

**159 tests, ~35s**, all under `tests/`, with no dataset, no GPU, no MLflow server
and no AWS credentials — the same conditions the CI runner has.
`.github/workflows/ci.yml` runs that command on every PR alongside `ruff check .`
and `mypy src/coin_clf` (scoped to the shared library; the root scripts still
report 11 errors, widening it is a follow-up), asserts no `nvidia-*` wheel slipped
into the install, and in a second job builds the serving image and imports it with
`--network none` — the mechanical check that the registry load stays inside the
lifespan and the app stays importable without infrastructure.

The workflow has a third job, `deploy`, which is **manual only** — `workflow_dispatch`
plus an `if:` on the event, plus a `confirm` input the operator has to type. It
builds and pushes the cloud image and forces a new ECS deployment, authenticating
with OIDC rather than stored keys. Nothing on `push` or `pull_request` can reach
it, because it starts a task that bills. See [cloud-deploy.md](cloud-deploy.md) — and
note it fails at the export step until a tracking server exists that a hosted
runner can reach.

Anything that would need the real 57,792-image tree or CUDA gets the `needs_data`
/ `needs_gpu` marker instead of being deleted; nothing carries either one today
(`--strict-markers` makes a typo'd marker an error rather than a silent no-op).
`verify_data_integrity.py` is the one check that cannot run in CI — it re-hashes
the whole tree — so `tests/test_verify_data_integrity.py` runs it against a
~60-image stand-in, injecting one specific fault per test and asserting the check
that should notice actually names it.

## Data & weights

The training dataset (`data/FOR_TRAINNING/`) and all `.pth`/`.onnx` checkpoints
(`weights/`) are gitignored — too large for the repo. The dataset is
version-controlled with DVC instead: `data/FOR_TRAINNING.dvc` is the tracked
pointer, and the remote is `s3://davids-mlops-artifacts-8412/dvc` — the same
bucket MLflow writes model artifacts to. What git tracks is source, notebooks,
this README, and the small JSON/TXT/CSV files under `data/` that define the split
boundaries and record what the cleanup dropped.
