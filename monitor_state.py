"""monitor_state.py -- the monitoring loop's decision logic and persisted state.

Two things live here, deliberately separated:

  * THE STATE MACHINE (top of the file): pure functions over plain dicts, stdlib only. No
    airflow, no mlflow, no requests at module scope. dags/monitor_coin_clf.py imports these
    directly (the airflow venv cannot import coin_clf, and the test environment cannot import
    airflow -- keeping the decision logic in an airflow-free module is the only way both the DAG
    and the tests can reach the same code).
  * THE SERVING PREFLIGHT (--check-serving, bottom): a CLI the monitoring DAG shells out to with
    the torch/mlflow interpreter, imports lazy so the module stays cheap for the scheduler.

WHY A STATE FILE EXISTS AT ALL -- the runaway-retrain problem.
An automated loop that retrains every cycle on the same unfixed drift is worse than no
automation: each run costs 120 epochs and consumes a 5,000-image future-pool batch. Two
independent mechanisms prevent it, because they cover different failures.

  1. A CONSUMING CURSOR stops re-firing on the SAME rows. The drift check runs with
     --since cursor_ts, and the cursor advances if and only if status == "ok" -- that is, only
     when a verdict was actually rendered. Drift detected -> triggered -> those rows are behind
     the cursor and can never fire again. The classic failure ("retrain ran, gate HELD, next
     cycle fires on the identical data") cannot happen. And because the cursor HOLDS on
     insufficient_data, thin traffic accumulates instead of being discarded, so a too-frequent
     schedule is harmless -- the interval controls detection latency, never correctness.

  2. A CIRCUIT BREAKER stops re-firing on CONTINUOUSLY ARRIVING drift, which the cursor cannot
     help with: every window legitimately contains fresh drifted rows. The monitor tracks
     whether its triggered retrains accomplish anything by comparing @champion before and after.
     Two consecutive triggered retrains that end without a promotion means retraining is not the
     fix -- the data or the recipe is -- and the breaker opens. Drift keeps being REPORTED; it
     just stops firing retrains until a human clears the state file.

A COOLDOWN sits in front of the breaker as a cheaper first guard.

Deliberate tradeoff: drift detected while a retrain is in flight, during cooldown, or with the
breaker open is logged, and its rows ARE consumed. It must reappear in fresh traffic to trigger.
The alternative -- holding the rows back -- would produce a delayed storm the moment the
cooldown expired, which is the same disease with a longer incubation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_COOLDOWN_HOURS = 6.0
DEFAULT_MAX_CONSECUTIVE_HOLDS = 2

# The only statuses that mean "a verdict was rendered". Anything else is "couldn't tell", which
# must never advance the cursor and must never trigger. See drift_report.build_verdict.
ASSESSED = ("ok",)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


# --- state I/O ----------------------------------------------------------------------------------

def initial_state(now: str | None = None) -> dict:
    """A fresh monitor starts its cursor at NOW, not at the beginning of time.

    A brand-new monitor must not open by triggering a retrain on months of accumulated history.
    Starting at now makes the first window empty, so the first run reports insufficient_data and
    does nothing -- the safe bootstrap falls out of the ordinary cursor rule rather than needing
    a special case.
    """
    return {
        "cursor_ts": now or utc_now_iso(),
        "last_trigger_ts": None,
        "pending_trigger": None,
        "consecutive_holds": 0,
    }


def load_state(path: str | Path, now: str | None = None) -> dict:
    path = Path(path)
    if not path.exists():
        return initial_state(now)
    state = json.loads(path.read_text())
    for key, value in initial_state(now).items():
        state.setdefault(key, value)   # tolerate a state file written by an older version
    return state


def save_state(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


# --- resolving the outcome of a previous trigger -------------------------------------------------

def resolve_pending(state: dict, champion_now: str | None, pending_run_active: bool) -> tuple[dict, str | None]:
    """Score the previous triggered retrain, once it is no longer running.

    The champion version is the evidence: it is resolved every run anyway, and it is the only
    thing that distinguishes "the retrain produced a better model" from "the gate HELD". A HOLD
    leaves @champion untouched, which is exactly the signal the breaker counts.

    Returns (state, outcome) where outcome is "promoted" | "held" | None (nothing to resolve).
    """
    pending = state.get("pending_trigger")
    if not pending or pending_run_active:
        return state, None

    state = dict(state)
    if champion_now is not None and str(champion_now) != str(pending.get("champion_before")):
        state["consecutive_holds"] = 0
        outcome = "promoted"
    else:
        state["consecutive_holds"] = int(state.get("consecutive_holds", 0)) + 1
        outcome = "held"
    state["pending_trigger"] = None
    return state, outcome


# --- the decision ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """What the monitoring run should do, and whether the window it just looked at is spent."""

    trigger: bool
    advance_cursor: bool
    action: str      # trigger | no_drift | skip_in_flight | skip_cooldown | skip_breaker | not_assessed
    detail: str


def cooldown_remaining(state: dict, now: str, cooldown_hours: float) -> timedelta:
    last = state.get("last_trigger_ts")
    if not last or cooldown_hours <= 0:
        return timedelta(0)
    elapsed = _parse(now) - _parse(last)
    remaining = timedelta(hours=cooldown_hours) - elapsed
    return remaining if remaining > timedelta(0) else timedelta(0)


def decide(
    verdict: dict,
    state: dict,
    now: str,
    retrain_active: bool,
    cooldown_hours: float = DEFAULT_COOLDOWN_HOURS,
    max_consecutive_holds: int = DEFAULT_MAX_CONSECUTIVE_HOLDS,
) -> Decision:
    """Map a drift verdict plus monitor state onto an action. Pure; no I/O, no clock."""
    status = verdict.get("status")
    if status not in ASSESSED:
        return Decision(
            False, False, "not_assessed",
            f"status={status!r} -- no verdict was rendered, so nothing was assessed and the "
            "cursor holds. This is 'could not tell', never 'no drift'.",
        )

    if not verdict.get("drift_detected"):
        return Decision(False, True, "no_drift", "no drift; window assessed and consumed")

    reason = verdict.get("trigger_reason") or "drift detected"

    if retrain_active:
        return Decision(False, True, "skip_in_flight",
                        f"drift detected but a retrain is already running or queued -- skipping "
                        f"rather than queueing a second one behind max_active_runs=1. {reason}")

    remaining = cooldown_remaining(state, now, cooldown_hours)
    if remaining > timedelta(0):
        return Decision(False, True, "skip_cooldown",
                        f"drift detected but the {cooldown_hours}h cooldown has "
                        f"{remaining.total_seconds() / 3600:.1f}h left. {reason}")

    holds = int(state.get("consecutive_holds", 0))
    if holds >= max_consecutive_holds:
        return Decision(False, True, "skip_breaker",
                        f"CIRCUIT BREAKER OPEN: {holds} consecutive triggered retrain(s) ended "
                        f"without a promotion, so retraining is not fixing this drift. No further "
                        f"retrains will be triggered until consecutive_holds is reset in the state "
                        f"file. Drift is still real: {reason}")

    return Decision(True, True, "trigger", reason)


def apply_decision(state: dict, decision: Decision, now: str, window_end: str,
                   champion: str | None = None, retrain_run_id: str | None = None) -> dict:
    """Fold a decision back into the state: advance the cursor, record the trigger."""
    state = dict(state)
    if decision.advance_cursor:
        state["cursor_ts"] = window_end
    if decision.trigger:
        state["last_trigger_ts"] = now
        state["pending_trigger"] = {"run_id": retrain_run_id, "champion_before": champion}
    return state


# --- serving preflight (CLI) -----------------------------------------------------------------------

def check_serving_version(health_url: str, tracking_uri: str, model_name: str,
                          model_alias: str) -> tuple[bool, str]:
    """Is the container serving the same model version the drift reference is built for?

    The serving app loads @champion ONCE, in its FastAPI lifespan. After a promotion moves the
    alias, the container keeps serving and LOGGING the old version until it is restarted, while
    drift_report.py builds its reference for the new one. Every production row then fails the
    version filter, and monitoring reports insufficient_data forever.

    That fails safe -- no spurious trigger, no runaway loop -- but it is silently blind, and
    silence is indistinguishable from "no drift". So this is a hard failure: monitoring that
    cannot see production should look broken, not green.

    KNOWN LIMITATION, deliberately not fixed here: there is no reload endpoint and nothing
    restarts serving automatically after a promotion. Closing that gap is a change to the
    serving path and is being decided separately.
    """
    import mlflow          # lazy: keeps this module stdlib-only for the airflow scheduler
    import requests

    response = requests.get(health_url, timeout=10)
    response.raise_for_status()
    serving_version = str(response.json()["model_version"])

    mlflow.set_tracking_uri(tracking_uri)
    champion = str(mlflow.MlflowClient().get_model_version_by_alias(model_name, model_alias).version)

    if serving_version == champion:
        return True, f"serving v{serving_version} == @champion v{champion}"
    return False, (
        f"VERSION SKEW: the container is serving v{serving_version} but @champion is v{champion}. "
        f"Every production row would be filtered out of the drift check and the verdict would be "
        f"insufficient_data forever -- monitoring is blind, not quiet. Restart the serving "
        f"container (`make serve`) so it loads v{champion}."
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="monitoring preflight: serving/champion version check")
    p.add_argument("--check-serving", action="store_true", required=True)
    p.add_argument("--health-url", default="http://localhost:8000/health")
    p.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    p.add_argument("--model-name", default="coin-classifier")
    p.add_argument("--model-alias", default="champion")
    args = p.parse_args()

    ok, message = check_serving_version(args.health_url, args.tracking_uri,
                                        args.model_name, args.model_alias)
    print(message)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
