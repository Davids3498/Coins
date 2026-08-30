"""dags/retrain_coin_clf.py -- Day 5: orchestrate release -> validate -> train -> evaluate ->
promote, one Airflow run per simulated future-pool batch arrival.

IMPORT-LIGHT ON PURPOSE: the scheduler re-parses every DAG file on a timer, so nothing here may
import torch / pandas / coin_clf at module scope, or every parse pays a multi-second tax. Only
stdlib + airflow are imported above the task definitions. All heavy work happens inside
BashOperator subprocesses that invoke PYTHON (a torch/pandas/mlflow-capable interpreter) --
Airflow's own process never needs any of that, which is also why BashOperator was chosen over an
in-process PythonOperator for every ML step.

Decisions this DAG encodes:
  * HOLD/REJECT (both the promotion gate's and the validation gate's) = log-and-keep: the DAG
    run itself stays GREEN, @champion is left alone, and the verdict lives in task logs /
    report.json / XCom -- never in the run's success/failure state. gate_on_validation is the
    validation twin of promote.py's own hold-without-raising behaviour.
  * Every task shells out to an existing CLI -- release_batch.py, `-m coin_clf.validate_batch`,
    train_hard_labels.py, evaluate.py, promote.py -- no in-process training. The train task runs
    train_hard_labels.py rather than train.py: it is the headless form of the notebook recipe
    that actually earned coin-classifier v6, so its defaults are a past result rather than a
    tunable starting point, and it refuses to start if the holdout fingerprint has moved (it
    imports train.py's prepare_data / evaluate_hard / make_scheduler, so the split, loaders and
    scoring are still the single canonical ones). release_batch.py and train_hard_labels.py /
    evaluate.py / promote.py stay root-level scripts (matching the Makefile's own
    `PYTHONPATH=src $(PY) train.py ...` pattern) because they need splits.py's DatasetSplits /
    FuturePool, and splits.py is itself a root-level script -- src/coin_clf must never import a
    root script (see coin_clf.validate_batch's module docstring on the dependency direction), so
    those three can't be package modules without inverting it. `-m coin_clf.validate_batch` is
    used for the one task that's genuinely inside the package already.
  * schedule=None: triggered manually, once per simulated batch arrival -- there's no wall-clock
    cadence to simulate here, just "a batch showed up."
  * max_active_runs=1: two runs both moving @champion (or both appending to active_train.txt /
    advancing the future-pool cursor) concurrently is corruption, not a race worth tolerating.
  * retries=0 everywhere: every task has an external side effect that isn't safe to replay --
    release_batch appends to active_train.txt and advances a cursor file (a retry double-
    releases the same batch), train registers a brand-new MLflow model version (a retry mid-
    registration double-registers), promote moves the @champion alias (a retry after a crash
    mid-move risks exactly the corruption max_active_runs=1 also guards against). None of these
    are pure functions of their inputs the way a stateless API call would be, so Airflow's retry
    machinery would do more harm than good here.
"""
from __future__ import annotations

import json
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator

# --- module constants ---------------------------------------------------------------------
PYTHON = "/usr/bin/python3"  # the python3.10 interpreter with torch/pandas/mlflow installed --
# deliberately NOT the bare `python3` on PATH (miniconda's base env, no torch). Airflow's own
# venv/process never imports either interpreter's packages; this is purely what BashOperator
# shells out to.
# Overridable so the DAG file is importable (and parseable by CI / a second Airflow host) off
# this one machine. The default keeps the deployed scheduler working with no env change.
PROJECT_ROOT = os.environ.get("COIN_PROJECT_ROOT", "/home/david/coin")

# One future-pool batch per DAG run. Must stay constant for as long as one future_pool_cursor.json
# is in use -- the cursor counts batch NUMBERS, so changing this mid-stream silently redefines
# what "batch 7" means. See release_batch.py.
BATCH_SIZE = 5000

# {{ run_id }} can contain ':' and '+' (e.g. "manual__2024-01-01T00:00:00+00:00") -- sanitized
# the same way in both the Jinja-templated bash_commands below and _run_dir() (used by the one
# Python task), so every task in a run agrees on the same directory without passing it through
# XCom.
_RUN_DIR_JINJA = "{{ run_id | replace(':', '-') | replace('+', '-') }}"
RUN_DIR = f"{PROJECT_ROOT}/outputs/dag_runs/{_RUN_DIR_JINJA}"


def _run_dir(run_id: str) -> str:
    safe_run_id = run_id.replace(":", "-").replace("+", "-")
    return os.path.join(PROJECT_ROOT, "outputs", "dag_runs", safe_run_id)


def _check_batch_valid(**context) -> bool:
    """gate_on_validation's callable. Reads validate's report.json and short-circuits the rest
    of the run (train/evaluate/promote) when the batch is invalid. Never raises on a bad batch
    -- that's a normal outcome (the batch stays quarantined in report.json / task logs), not a
    DAG failure. A missing/malformed report.json IS still a hard failure: that means validate
    itself broke, not that the batch was bad.
    """
    report_path = os.path.join(_run_dir(context["run_id"]), "report.json")
    with open(report_path) as f:
        report = json.load(f)

    is_valid = report["is_valid"]
    print(f"batch report: is_valid={is_valid}  n_images={report['n_images']}")
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}: {check['detail']}")

    if not is_valid:
        print(
            "INVALID BATCH -- short-circuiting train/evaluate/promote. @champion is untouched. "
            "Note: release_batch already appended this batch's files to data/active_train.txt "
            "unconditionally (task order is release -> validate); a failing report here means "
            "this run won't train/evaluate/promote on it, not that active_train.txt was rolled "
            "back. See report.json's offending_rows for what tripped the gate."
        )
    return is_valid


with DAG(
    dag_id="retrain_coin_clf",
    description="release -> validate -> train -> evaluate -> promote, one run per simulated batch",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0},
    params={
        # Smoke-test defaults; a real retrain run overrides via trigger config with the recipe
        # train_hard_labels.py's own defaults encode (epochs=120, batch_size=256 -- the config
        # that earned coin-classifier v6). Kept low here so a pipeline-shape test costs minutes,
        # not hours; train_hard_labels.py prints a loud warning when epochs are this short,
        # because a truncated cosine schedule is a different run, not a shorter one.
        "epochs": 3,
        "batch_size": 256,
    },
    tags=["coin-clf", "retraining"],
) as dag:

    release_batch = BashOperator(
        task_id="release_batch",
        bash_command=(
            f"cd {PROJECT_ROOT} && mkdir -p {RUN_DIR} && "
            f"PYTHONPATH=src {PYTHON} release_batch.py "
            f"--data-dir data/FOR_TRAINNING "
            f"--manifest data/splits_manifest.json "
            f"--active-train-list data/active_train.txt "
            f"--cursor-file data/future_pool_cursor.json "
            f"--batch-size {BATCH_SIZE} "
            f"--batch-out {RUN_DIR}/batch.txt"
        ),
    )

    validate = BashOperator(
        task_id="validate",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"PYTHONPATH=src {PYTHON} -m coin_clf.validate_batch "
            f"--batch-file {RUN_DIR}/batch.txt "
            f"--data-dir data/FOR_TRAINNING "
            f"--manifest data/splits_manifest.json "
            f"--active-train-list data/active_train.txt "
            f"--report-out {RUN_DIR}/report.json"
        ),
    )

    gate_on_validation = ShortCircuitOperator(
        task_id="gate_on_validation",
        python_callable=_check_batch_valid,
    )

    train = BashOperator(
        task_id="train",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"PYTHONPATH=src {PYTHON} train_hard_labels.py "
            f"--data-dir data/FOR_TRAINNING "
            f"--active-train-list data/active_train.txt "
            f"--manifest data/splits_manifest.json "
            # Reads dag_run.conf DIRECTLY rather than relying on params being overridden by it.
            # That override is gated on the [core] dag_run_conf_overrides_params setting, which
            # lives in airflow.cfg -- outside this file. It is currently True, but if it were
            # ever flipped, an automated trigger asking for 120 epochs would silently get the
            # 3-epoch smoke-test default, produce a challenger on a truncated cosine schedule,
            # and the promotion gate's HOLD would look like a model problem instead of the
            # config bug it is. A guarantee this expensive to get wrong does not belong in a
            # global setting. `or {}` because dag_run.conf is None for a scheduled run.
            f"--epochs {{{{ (dag_run.conf or {{}}).get('epochs', params.epochs) }}}} "
            f"--batch-size {{{{ (dag_run.conf or {{}}).get('batch_size', params.batch_size) }}}} "
            f"--run-name \"retrain-{_RUN_DIR_JINJA}\" "
            f"--version-out {RUN_DIR}/challenger_version.txt "
            f"&& cat {RUN_DIR}/challenger_version.txt"
        ),
        do_xcom_push=True,  # last stdout line (the `cat`) -> XCom, read by evaluate/promote below
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"PYTHONPATH=src {PYTHON} evaluate.py "
            f"--version \"{{{{ ti.xcom_pull(task_ids='train') }}}}\" "
            f"--data-dir data/FOR_TRAINNING "
            f"--manifest data/splits_manifest.json"
        ),
    )

    promote = BashOperator(
        task_id="promote",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"PYTHONPATH=src {PYTHON} promote.py "
            f"--challenger \"{{{{ ti.xcom_pull(task_ids='train') }}}}\" "
            f"--data-dir data/FOR_TRAINNING "
            f"--manifest data/splits_manifest.json "
            f"--margin 0.005"
        ),
    )

    release_batch >> validate >> gate_on_validation >> train >> evaluate >> promote
