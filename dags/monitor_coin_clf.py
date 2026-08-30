"""dags/monitor_coin_clf.py -- the closed loop: observe production, and retrain when the model's
input distribution has actually moved.

Turns the manual retrain button into a feedback loop. app/main.py logs every /predict into the
SQLite prediction log; drift_report.py scores a window of that log against the champion's own
predictions on the frozen holdout; this DAG decides whether the verdict is worth 120 epochs.

IMPORT-LIGHT, same rule as retrain_coin_clf.py: the scheduler re-parses this file on a timer, so
nothing torch/pandas/mlflow-sized may be imported at module scope. The only non-stdlib,
non-airflow import is monitor_state, which is deliberately stdlib-only -- the airflow venv cannot
import coin_clf (it is installed into the /usr/bin/python3 interpreter, not this one), so the
decision logic lives in a root-level module both this DAG and the test suite can reach. Every
heavy step shells out to PYTHON.

WHAT FIRES A RETRAIN, exactly: status == "ok" AND drift_detected. Nothing else. insufficient_data,
no_data and no_reference all mean "could not tell", which is not "no drift" and is certainly not
grounds for an expensive retrain -- that distinction is why drift_report.py carries a status field
instead of a bare boolean.

TRIGGERED RUNS GET THE REAL RECIPE: conf={"epochs": 120, "batch_size": 256}. The retrain DAG's
own params default to 3 epochs for pipeline smoke tests; a triggered retrain silently inheriting
that would train a challenger on a truncated cosine schedule and then look like a model failure
when the promotion gate HELD it. retrain_coin_clf.py's train task reads these out of dag_run.conf
explicitly so the guarantee does not depend on an airflow.cfg setting.

SKIP, DO NOT QUEUE, when a retrain is already in flight. TriggerDagRunOperator creates a QUEUED
DagRun; with the retrain DAG's max_active_runs=1 those stack up and execute serially, so N
monitoring cycles would mean N full retrains, each releasing another 5,000-image future-pool
batch. TriggerDagRunOperator's own skip_when_already_exists does NOT cover this -- it only
catches a duplicate run_id (DagRunAlreadyExists) -- so the gate queries DagRun.find explicitly.

Runaway protection (a consuming cursor plus a circuit breaker) lives in monitor_state.py; see its
docstring for why one mechanism is not enough.

FUTURE-POOL BUDGET: each triggered retrain releases BATCH_SIZE (5,000) images and the pool is
finite. When it runs dry release_batch.py fails hard, on purpose -- a loop that quietly retrained
on nothing would be worse than one that stops.
"""
from __future__ import annotations

import json
import os
import sys

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# --- module constants ---------------------------------------------------------------------
PYTHON = "/usr/bin/python3"   # the interpreter with torch/pandas/mlflow -- see retrain_coin_clf.py
# Overridable so the DAG file is importable (and parseable by CI / a second Airflow host) off
# this one machine. The default keeps the deployed scheduler working with no env change.
PROJECT_ROOT = os.environ.get("COIN_PROJECT_ROOT", "/home/david/coin")

# Airflow puts the DAGS folder on sys.path, not the repo root, so monitor_state (a root-level
# module, stdlib-only by design) needs this to be importable here.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from monitor_state import (  # noqa: E402  (must follow the sys.path insert above)
    DEFAULT_COOLDOWN_HOURS,
    DEFAULT_MAX_CONSECUTIVE_HOLDS,
    apply_decision,
    decide,
    load_state,
    resolve_pending,
    save_state,
    utc_now_iso,
)

RETRAIN_DAG_ID = "retrain_coin_clf"

# The recipe that earned coin-classifier v6 -- train_hard_labels.py's own defaults. NOT the
# retrain DAG's 3-epoch smoke-test params. See the module docstring.
RETRAIN_EPOCHS = 120
RETRAIN_BATCH_SIZE = 256

STATE_FILE = f"{PROJECT_ROOT}/outputs/monitoring/monitor_state.json"
PREDICTIONS_DB = f"{PROJECT_ROOT}/outputs/monitoring/predictions.db"
HEALTH_URL = "http://localhost:8000/health"

_RUN_DIR_JINJA = "{{ run_id | replace(':', '-') | replace('+', '-') }}"
RUN_DIR = f"{PROJECT_ROOT}/outputs/dag_runs/{_RUN_DIR_JINJA}"


def _run_dir(run_id: str) -> str:
    safe_run_id = run_id.replace(":", "-").replace("+", "-")
    return os.path.join(PROJECT_ROOT, "outputs", "dag_runs", safe_run_id)


def _read_state(**context) -> str:
    """Load monitor state, bootstrapping on first run, and hand the drift check its window.

    Returns cursor_ts (pushed to XCom) -- the --since the drift check runs with. On a first run
    there is no state file and the cursor initialises to now, so the first window is empty and
    the run reports insufficient_data instead of triggering a retrain on all of history.
    """
    state = load_state(STATE_FILE)
    save_state(STATE_FILE, state)   # persist the bootstrap so the next run continues from here
    print(f"monitor state: cursor_ts={state['cursor_ts']} "
          f"last_trigger_ts={state['last_trigger_ts']} "
          f"consecutive_holds={state['consecutive_holds']} "
          f"pending_trigger={state['pending_trigger']}")
    return state["cursor_ts"]


def _retrain_is_active() -> bool:
    """True if a retrain run is RUNNING or QUEUED. See the module docstring on skip-don't-queue."""
    from airflow.models import DagRun
    from airflow.utils.state import DagRunState

    active = []
    for state in (DagRunState.RUNNING, DagRunState.QUEUED):
        active.extend(DagRun.find(dag_id=RETRAIN_DAG_ID, state=state))
    if active:
        print(f"{RETRAIN_DAG_ID} already active: {[r.run_id for r in active]}")
    return bool(active)


def _gate_on_drift(**context) -> bool:
    """Read the verdict, resolve the previous trigger's outcome, decide, persist.

    Never raises on a drift outcome -- every verdict, including 'could not tell', is a normal
    result here, the same log-and-keep contract retrain_coin_clf.py's gate_on_validation follows.
    A missing/malformed verdict file IS a hard failure: that means the drift check broke, not
    that the traffic was fine.
    """
    ti = context["ti"]
    now = utc_now_iso()
    verdict_path = os.path.join(_run_dir(context["run_id"]), "drift_verdict.json")
    with open(verdict_path) as f:
        verdict = json.load(f)

    champion = (verdict.get("reference") or {}).get("model_version")
    sample = verdict.get("sample") or {}
    print(f"drift verdict: status={verdict['status']} drift_detected={verdict['drift_detected']} "
          f"signals={verdict.get('drifted_signals')} n_production={sample.get('n_production')} "
          f"champion=v{champion}")

    state = load_state(STATE_FILE)
    retrain_active = _retrain_is_active()

    # Score the PREVIOUS triggered retrain before deciding anything new: if it finished without
    # moving @champion, the gate HELD, and that is what the circuit breaker counts.
    state, outcome = resolve_pending(state, champion, retrain_active)
    if outcome:
        print(f"previous triggered retrain outcome: {outcome.upper()} "
              f"(consecutive_holds now {state['consecutive_holds']})")

    decision = decide(verdict, state, now, retrain_active,
                      cooldown_hours=DEFAULT_COOLDOWN_HOURS,
                      max_consecutive_holds=DEFAULT_MAX_CONSECUTIVE_HOLDS)
    print(f"decision: {decision.action} -- {decision.detail}")

    # window_end is the moment the drift check looked, not the newest row: consuming up to "now"
    # means rows that arrived mid-check are not skipped over.
    state = apply_decision(state, decision, now, window_end=now, champion=champion,
                           retrain_run_id=context["run_id"])
    save_state(STATE_FILE, state)
    print(f"monitor state saved: cursor_ts={state['cursor_ts']} "
          f"consecutive_holds={state['consecutive_holds']}")

    # Read by the trigger's conf, so the retrain run records WHY it exists.
    ti.xcom_push(key="trigger_reason", value=decision.detail)
    return decision.trigger


with DAG(
    dag_id="monitor_coin_clf",
    description="drift check over production predictions -> triggers retrain_coin_clf on real drift",
    # Manual, like retrain_coin_clf: there is no continuous traffic to this model (it arrives
    # only when replay_traffic.py runs), so a timer would mostly log insufficient_data. @daily is
    # the sensible production default. The interval is NOT correctness-critical either way: the
    # cursor holds on insufficient_data, so thin windows accumulate rather than being discarded,
    # and running too often only costs a no-op run.
    schedule=None,
    catchup=False,
    # Two monitoring runs interleaving on monitor_state.json would corrupt the cursor and the
    # breaker count -- the very state that prevents runaway retraining.
    max_active_runs=1,
    # A retry after a partial gate run could double-trigger a 120-epoch retrain.
    default_args={"retries": 0},
    tags=["coin-clf", "monitoring", "drift"],
) as dag:

    # Fail-fast, BEFORE the drift check possibly spends minutes rebuilding a reference it cannot
    # use. A container serving a different version than @champion makes monitoring blind, and a
    # blind monitor must look broken rather than green. See monitor_state.check_serving_version.
    check_serving_version = BashOperator(
        task_id="check_serving_version",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"PYTHONPATH=src {PYTHON} monitor_state.py --check-serving "
            f"--health-url {HEALTH_URL}"
        ),
    )

    read_state = PythonOperator(
        task_id="read_state",
        python_callable=_read_state,
    )

    drift_check = BashOperator(
        task_id="drift_check",
        bash_command=(
            f"cd {PROJECT_ROOT} && mkdir -p {RUN_DIR} && "
            f"PYTHONPATH=src {PYTHON} drift_report.py "
            f"--data-dir data/FOR_TRAINNING "
            f"--manifest data/splits_manifest.json "
            f"--db {PREDICTIONS_DB} "
            f"--since \"{{{{ ti.xcom_pull(task_ids='read_state') }}}}\" "
            f"--verdict-out {RUN_DIR}/drift_verdict.json "
            f"--html-out {RUN_DIR}/drift_report.html"
        ),
    )

    gate_on_drift = ShortCircuitOperator(
        task_id="gate_on_drift",
        python_callable=_gate_on_drift,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id=RETRAIN_DAG_ID,
        # THE REAL RECIPE, never the 3-epoch smoke default. trigger_reason rides along so the
        # retrain run's conf permanently records which drift signal caused it -- visible in the
        # Airflow UI long after the task logs have rotated.
        conf={
            "epochs": RETRAIN_EPOCHS,
            "batch_size": RETRAIN_BATCH_SIZE,
            "triggered_by": "monitor_coin_clf",
            "monitor_run_id": "{{ run_id }}",
            "trigger_reason": "{{ ti.xcom_pull(task_ids='gate_on_drift', key='trigger_reason') }}",
        },
        # Do NOT wait: a 120-epoch retrain runs for hours and holding a monitoring slot open for
        # it would block the next drift check behind max_active_runs=1.
        wait_for_completion=False,
    )

    check_serving_version >> read_state >> drift_check >> gate_on_drift >> trigger_retrain
