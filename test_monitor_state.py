"""Tests for the monitoring loop's decision logic, and for the epochs guarantee.

The decision logic lives in monitor_state.py rather than in the DAG file precisely so it can be
tested here: this environment has no airflow (it lives in .venv-airflow, a different interpreter),
so dags/monitor_coin_clf.py cannot be imported. monitor_state is stdlib-only and imports cleanly
in both.

The most important tests are the ones about NOT triggering. A false trigger costs a 120-epoch
retrain and a 5,000-image future-pool batch, so every "could not tell" path and every runaway
guard has an explicit test.

The last section guards the epochs requirement by rendering the retrain DAG's ACTUAL train
template with jinja2 (present in both interpreters) rather than asserting on source text -- so it
catches a broken Jinja expression, not just a reverted string.
"""
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from monitor_state import (
    DEFAULT_COOLDOWN_HOURS,
    DEFAULT_MAX_CONSECUTIVE_HOLDS,
    apply_decision,
    cooldown_remaining,
    decide,
    initial_state,
    load_state,
    resolve_pending,
    save_state,
)

NOW = "2026-08-25T12:00:00+00:00"


def ago(hours):
    return (datetime.fromisoformat(NOW) - timedelta(hours=hours)).isoformat()


def verdict(status="ok", drift=False, reason=None, champion="10"):
    return {
        "status": status,
        "drift_detected": drift,
        "drifted_signals": ["predicted_class"] if drift else [],
        "trigger_reason": reason or ("predicted_class:predicted_label (jensenshannon 0.61 > 0.35)"
                                     if drift else None),
        "sample": {"n_production": 600},
        "reference": {"model_version": champion},
    }


def state(**overrides):
    s = initial_state(NOW)
    s.update(overrides)
    return s


def decide_now(v, s, retrain_active=False, **kw):
    return decide(v, s, NOW, retrain_active, **kw)


# --- the only path that triggers ---------------------------------------------------------------

def test_drift_with_everything_clear_triggers():
    d = decide_now(verdict(drift=True), state())
    assert d.trigger is True
    assert d.action == "trigger"
    assert "jensenshannon" in d.detail          # the reason is carried, for the retrain's conf


def test_no_drift_does_not_trigger_but_consumes_the_window():
    d = decide_now(verdict(drift=False), state())
    assert d.trigger is False
    assert d.advance_cursor is True


# --- "could not tell" is never drift -------------------------------------------------------------

@pytest.mark.parametrize("status", ["insufficient_data", "no_data", "no_reference"])
def test_unassessed_statuses_never_trigger_and_never_advance_the_cursor(status):
    """The cursor must HOLD here or thin traffic would be discarded and could never accumulate to
    min_samples -- a low-traffic model would become permanently unmonitorable.
    """
    d = decide_now(verdict(status=status, drift=False), state())
    assert d.trigger is False
    assert d.advance_cursor is False
    assert d.action == "not_assessed"


def test_drift_true_under_a_non_ok_status_still_never_triggers():
    """Defence in depth: even if a verdict somehow carried drift_detected=True with a non-ok
    status, the gate must not fire.
    """
    d = decide_now(verdict(status="insufficient_data", drift=True), state())
    assert d.trigger is False


# --- runaway guard 1: in-flight retrain ------------------------------------------------------------

def test_in_flight_retrain_skips_instead_of_queueing():
    d = decide_now(verdict(drift=True), state(), retrain_active=True)
    assert d.trigger is False
    assert d.action == "skip_in_flight"
    assert d.advance_cursor is True


# --- runaway guard 2: cooldown -----------------------------------------------------------------------

def test_recent_trigger_is_within_cooldown():
    d = decide_now(verdict(drift=True), state(last_trigger_ts=ago(1)))
    assert d.trigger is False
    assert d.action == "skip_cooldown"


def test_cooldown_expires():
    d = decide_now(verdict(drift=True), state(last_trigger_ts=ago(DEFAULT_COOLDOWN_HOURS + 1)))
    assert d.trigger is True


def test_cooldown_remaining_is_zero_when_never_triggered():
    assert cooldown_remaining(state(), NOW, DEFAULT_COOLDOWN_HOURS) == timedelta(0)


# --- runaway guard 3: circuit breaker ------------------------------------------------------------------

def test_breaker_opens_after_max_consecutive_holds():
    """The scenario that matters: drift keeps arriving, retrains keep running, the gate keeps
    HOLDing. Continuing to fire 120-epoch runs at that point is worse than not automating.
    """
    d = decide_now(verdict(drift=True), state(consecutive_holds=DEFAULT_MAX_CONSECUTIVE_HOLDS))
    assert d.trigger is False
    assert d.action == "skip_breaker"
    assert "CIRCUIT BREAKER OPEN" in d.detail
    assert "jensenshannon" in d.detail          # drift is still REPORTED, just not actioned


def test_breaker_stays_closed_below_the_limit():
    d = decide_now(verdict(drift=True), state(consecutive_holds=DEFAULT_MAX_CONSECUTIVE_HOLDS - 1))
    assert d.trigger is True


# --- scoring the previous trigger --------------------------------------------------------------------

def test_hold_increments_the_breaker_count():
    s = state(pending_trigger={"run_id": "r1", "champion_before": "10"}, consecutive_holds=0)
    s, outcome = resolve_pending(s, champion_now="10", pending_run_active=False)
    assert outcome == "held"
    assert s["consecutive_holds"] == 1
    assert s["pending_trigger"] is None


def test_promotion_resets_the_breaker_count():
    s = state(pending_trigger={"run_id": "r1", "champion_before": "10"}, consecutive_holds=1)
    s, outcome = resolve_pending(s, champion_now="11", pending_run_active=False)
    assert outcome == "promoted"
    assert s["consecutive_holds"] == 0


def test_still_running_retrain_is_not_scored_yet():
    s = state(pending_trigger={"run_id": "r1", "champion_before": "10"}, consecutive_holds=0)
    s2, outcome = resolve_pending(s, champion_now="10", pending_run_active=True)
    assert outcome is None
    assert s2["consecutive_holds"] == 0
    assert s2["pending_trigger"] is not None      # still pending; scored on a later run


def test_nothing_pending_is_a_no_op():
    s, outcome = resolve_pending(state(), champion_now="10", pending_run_active=False)
    assert outcome is None


# --- the full HOLD scenario, end to end -----------------------------------------------------------------

def test_after_a_hold_the_same_data_cannot_fire_again():
    """The exact loop this design exists to prevent: drift -> trigger -> retrain -> gate HOLDs ->
    next monitoring run. The rows were consumed at trigger time, so the next window is empty and
    the verdict is insufficient_data, not a second trigger on identical data.
    """
    s = state()
    d1 = decide_now(verdict(drift=True), s)
    assert d1.trigger is True
    s = apply_decision(s, d1, NOW, window_end=NOW, champion="10", retrain_run_id="monitor_1")
    assert s["cursor_ts"] == NOW                       # the drifted rows are now behind us

    s, outcome = resolve_pending(s, champion_now="10", pending_run_active=False)
    assert outcome == "held"

    # next run: empty window -> the drift check reports insufficient_data
    d2 = decide_now(verdict(status="insufficient_data"), s)
    assert d2.trigger is False
    assert d2.advance_cursor is False


# --- state persistence ---------------------------------------------------------------------------------

def test_bootstrap_cursor_starts_at_now_not_the_epoch(tmp_path):
    """A brand-new monitor must not open by retraining on all of history."""
    s = load_state(tmp_path / "monitor_state.json", now=NOW)
    assert s["cursor_ts"] == NOW
    assert s["consecutive_holds"] == 0


def test_state_round_trips(tmp_path):
    path = tmp_path / "monitor_state.json"
    save_state(path, state(consecutive_holds=2, last_trigger_ts=ago(3)))
    assert load_state(path)["consecutive_holds"] == 2


def test_older_state_files_gain_new_keys(tmp_path):
    path = tmp_path / "monitor_state.json"
    path.write_text(json.dumps({"cursor_ts": NOW}))
    s = load_state(path)
    assert s["consecutive_holds"] == 0 and s["pending_trigger"] is None


def test_apply_decision_records_the_trigger():
    s = apply_decision(state(), decide_now(verdict(drift=True), state()), NOW,
                       window_end=NOW, champion="10", retrain_run_id="monitor_1")
    assert s["last_trigger_ts"] == NOW
    assert s["pending_trigger"] == {"run_id": "monitor_1", "champion_before": "10"}


def test_unassessed_decision_leaves_the_cursor_alone():
    s = state(cursor_ts="2026-01-01T00:00:00+00:00")
    out = apply_decision(s, decide_now(verdict(status="no_data"), s), NOW, window_end=NOW)
    assert out["cursor_ts"] == "2026-01-01T00:00:00+00:00"


# --- the epochs guarantee -------------------------------------------------------------------------------

DAG_PATH = Path(__file__).resolve().parent / "dags" / "retrain_coin_clf.py"


def render_train_flag(flag, conf, params):
    """Render the retrain DAG's real --<flag> Jinja expression against a given conf/params.

    Extracted from the DAG source and rendered with jinja2 rather than compared as text: this
    fails if the expression is reverted to `params.epochs` AND if the expression itself breaks.
    airflow cannot be imported here, but its templating engine can.
    """
    import jinja2

    # The DAG builds bash_command with f-strings, so in the SOURCE the Jinja braces are doubled
    # ({{{{ ... }}}}). Undo that escaping to recover the template the operator actually receives.
    # Must match the TEMPLATED occurrence: --batch-size appears twice in that DAG with unrelated
    # meanings -- the future-pool batch (a literal 5000) in release_batch, and the training batch
    # size in train. Requiring the escaped Jinja opener picks the train one.
    marker = f"--{flag} " + "{{{{"
    line = next((l for l in DAG_PATH.read_text().splitlines() if marker in l), None)
    assert line, f"no templated --{flag} found in {DAG_PATH}"
    template = line.replace("{{", "{").replace("}}", "}")

    match = re.search(r"(\{\{.*\}\})", template)
    assert match, f"--{flag} is no longer a Jinja expression in {DAG_PATH}: {line.strip()}"
    return jinja2.Template(match.group(1)).render(dag_run=type("R", (), {"conf": conf}), params=params)


def test_triggered_retrain_renders_120_epochs():
    """THE critical guarantee: a monitor-triggered retrain must not silently smoke-test."""
    rendered = render_train_flag("epochs", conf={"epochs": 120, "batch_size": 256},
                                 params={"epochs": 3, "batch_size": 256})
    assert rendered == "120"


def test_triggered_retrain_renders_the_real_batch_size():
    rendered = render_train_flag("batch-size", conf={"epochs": 120, "batch_size": 256},
                                 params={"epochs": 3, "batch_size": 256})
    assert rendered == "256"


def test_manual_run_without_conf_still_gets_the_smoke_default():
    """The 3-epoch default must survive for manual pipeline tests -- conf-reading must not break
    a scheduled/manual run, where dag_run.conf is None.
    """
    assert render_train_flag("epochs", conf=None, params={"epochs": 3}) == "3"


def test_epochs_does_not_depend_on_params_being_overridden_by_conf():
    """Simulates dag_run_conf_overrides_params=False: params still hold the smoke default while
    conf asks for 120. Reading conf directly must still yield 120.
    """
    assert render_train_flag("epochs", conf={"epochs": 120}, params={"epochs": 3}) == "120"
