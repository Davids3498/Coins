"""Tests for drift_report.py -- the drift check's verdict logic and its Evidently wiring.

No MLflow, no model, no registry. The expensive half of the script (scoring the champion over
the holdout to build a reference) is deliberately separated from the half that decides anything,
so everything decision-bearing is testable from plain DataFrames:

    compute_drift(reference_frame, current_frame, spec) -> per-column scores   [real Evidently]
    build_verdict(status, column_results, ...)          -> the DAG's input     [pure]
    load_production(db, since, source, model_version)   -> the selection rules [real SQLite]

The drift tests use real Evidently rather than stubbed scores: the point of a threshold is what
it does to an actual statistic, and a stub would only prove the comparison operator works.
"""
import json

import numpy as np
import pandas as pd
import pytest

from coin_clf.prediction_log import PredictionLog, PredictionRecord, utc_now_iso
from drift_report import (
    DEFAULT_CLASS_METHOD,
    DEFAULT_CLASS_THRESHOLD,
    DEFAULT_NUMERIC_METHOD,
    DEFAULT_NUMERIC_THRESHOLD,
    build_verdict,
    column_spec,
    compute_drift,
    load_production,
    reference_paths,
)

CLASSES = [f"EMPEROR_{i:02d}" for i in range(51)]  # same cardinality as the real label space


@pytest.fixture
def spec():
    return column_spec(DEFAULT_CLASS_METHOD, DEFAULT_CLASS_THRESHOLD,
                       DEFAULT_NUMERIC_METHOD, DEFAULT_NUMERIC_THRESHOLD)


def make_frame(n, seed=0, classes=CLASSES, conf=(8, 2), size=(469, 434), mode="RGB"):
    """A production-shaped frame. Defaults approximate the real holdout; every argument is a
    dial for one signal, so a test can move exactly one thing.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "predicted_label": rng.choice(classes, n),
        "confidence": rng.beta(conf[0], conf[1], n),
        "width": rng.normal(size[0], size[1], n).clip(32),
        "height": rng.normal(size[0], size[1], n).clip(32),
        "mode": [mode] * n,
    })


def write_rows(log, n, model_version="6", source=None, ts=None, label="NERO"):
    for _ in range(n):
        log.log(PredictionRecord(
            ts=ts or utc_now_iso(), model_version=model_version, predicted_label=label,
            confidence=0.9, width=224, height=224, mode="RGB", latency_ms=10.0, source=source,
        ))


# --- the demo's core claim: normal traffic is quiet, skewed traffic is not ---------------------

def test_same_distribution_does_not_drift(spec, tmp_path):
    """Normal traffic at the minimum sample size must not fire. This is the false-positive test:
    a fire here means the thresholds are miscalibrated and the closed loop would retrain on noise.
    """
    results = compute_drift(make_frame(11559, seed=1), make_frame(500, seed=2), spec,
                            html_out=tmp_path / "r.html")
    assert not any(r["drifted"] for r in results.values()), results


def test_class_restriction_drifts(spec, tmp_path):
    """The skew the replay script produces: traffic restricted to a handful of classes."""
    results = compute_drift(make_frame(11559, seed=1),
                            make_frame(500, seed=2, classes=CLASSES[:5]), spec,
                            html_out=tmp_path / "r.html")
    assert results["predicted_label"]["drifted"] is True
    assert results["predicted_label"]["score"] > DEFAULT_CLASS_THRESHOLD


def test_downsizing_drifts_metadata_not_class(spec, tmp_path):
    """Degraded image size must fire metadata WITHOUT fabricating class drift -- the signals have
    to be independently attributable or the trigger reason is meaningless.
    """
    results = compute_drift(make_frame(11559, seed=1),
                            make_frame(500, seed=2, size=(96, 4)), spec,
                            html_out=tmp_path / "r.html")
    assert results["width"]["drifted"] is True
    assert results["height"]["drifted"] is True
    assert results["predicted_label"]["drifted"] is False


def test_grayscale_drifts_mode(spec, tmp_path):
    results = compute_drift(make_frame(11559, seed=1), make_frame(500, seed=2, mode="L"), spec,
                            html_out=tmp_path / "r.html")
    assert results["mode"]["drifted"] is True


def test_confidence_collapse_drifts(spec, tmp_path):
    results = compute_drift(make_frame(11559, seed=1), make_frame(500, seed=2, conf=(2, 8)), spec,
                            html_out=tmp_path / "r.html")
    assert results["confidence"]["drifted"] is True


def test_html_report_is_written(spec, tmp_path):
    out = tmp_path / "nested" / "report.html"
    compute_drift(make_frame(2000, seed=1), make_frame(500, seed=2), spec, html_out=out)
    assert out.exists() and out.stat().st_size > 0


# --- verdict assembly ---------------------------------------------------------------------------

def drifted(score, threshold=0.35, method="jensenshannon"):
    return {"score": score, "method": method, "threshold": threshold, "drifted": score > threshold}


def all_quiet():
    return {c: drifted(0.01) for c in ("predicted_label", "confidence", "width", "height", "mode")}


def test_verdict_groups_columns_into_three_signals():
    v = build_verdict("ok", all_quiet(), {}, {})
    assert set(v["signals"]) == {"predicted_class", "confidence", "image_metadata"}
    assert set(v["signals"]["image_metadata"]["columns"]) == {"width", "height", "mode"}
    assert v["drift_detected"] is False
    assert v["trigger_reason"] is None


def test_one_metadata_column_fires_the_whole_signal():
    cols = all_quiet()
    cols["mode"] = drifted(0.83)
    v = build_verdict("ok", cols, {}, {})
    assert v["drift_detected"] is True
    assert v["drifted_signals"] == ["image_metadata"]
    assert v["signals"]["image_metadata"]["drifted"] is True
    assert v["signals"]["predicted_class"]["drifted"] is False


def test_trigger_reason_names_signal_column_method_and_threshold():
    cols = all_quiet()
    cols["predicted_label"] = drifted(0.879)
    reason = build_verdict("ok", cols, {}, {})["trigger_reason"]
    for fragment in ("predicted_class", "predicted_label", "jensenshannon", "0.879", "0.35"):
        assert fragment in reason


def test_insufficient_data_is_never_drift():
    """A thin sample must not read as drift to Piece 4's trigger, even if the scores are extreme."""
    cols = all_quiet()
    cols["predicted_label"] = drifted(0.99)
    v = build_verdict("insufficient_data", cols, {"n_production": 12}, {})
    assert v["drift_detected"] is False
    assert v["drifted_signals"] == []
    assert v["trigger_reason"] is None


@pytest.mark.parametrize("status", ["no_data", "no_reference", "insufficient_data"])
def test_non_ok_statuses_never_report_drift(status):
    assert build_verdict(status, None, {}, {})["drift_detected"] is False


def test_verdict_is_json_serializable():
    """The DAG reads this off disk -- numpy scalars leaking in would break json.dump."""
    v = build_verdict("ok", all_quiet(), {"n_production": 500}, {"split": "holdout"})
    assert json.loads(json.dumps(v))["status"] == "ok"


# --- production selection -------------------------------------------------------------------------

def test_source_tag_isolates_one_replay(tmp_path):
    log = PredictionLog(tmp_path / "p.db")
    write_rows(log, 5, source="normal-replay")
    write_rows(log, 3, source="skewed-replay")
    write_rows(log, 2, source=None)

    frame, _ = load_production(tmp_path / "p.db", None, "skewed-replay", "6")
    assert len(frame) == 3


def test_untagged_rows_are_selected_when_no_source_given(tmp_path):
    log = PredictionLog(tmp_path / "p.db")
    write_rows(log, 4, source=None)
    frame, _ = load_production(tmp_path / "p.db", None, None, "6")
    assert len(frame) == 4


def test_other_model_versions_are_excluded_and_counted(tmp_path):
    """The stale-reference guard. Rows predicted by a different champion must not be compared
    against this champion's reference -- that comparison is the infinite-retrain-loop bug.
    """
    log = PredictionLog(tmp_path / "p.db")
    write_rows(log, 6, model_version="6")
    write_rows(log, 4, model_version="7")

    frame, info = load_production(tmp_path / "p.db", None, None, "6")
    assert len(frame) == 6
    assert info["excluded_other_versions"] == 4
    assert info["model_versions_seen"] == ["6", "7"]


def test_since_filters_the_window(tmp_path):
    log = PredictionLog(tmp_path / "p.db")
    write_rows(log, 3, ts="2020-01-01T00:00:00+00:00")
    cutoff = utc_now_iso()
    write_rows(log, 2, ts=cutoff)

    frame, _ = load_production(tmp_path / "p.db", cutoff, None, "6")
    assert len(frame) == 2


def test_missing_database_raises_for_the_caller_to_report(tmp_path):
    """load_production surfaces the missing file; main() turns it into a no_data verdict rather
    than letting it escape as a task failure.
    """
    with pytest.raises(FileNotFoundError):
        load_production(tmp_path / "nope.db", None, None, "6")


# --- cache keying ------------------------------------------------------------------------------

def test_reference_cache_path_is_keyed_on_version_and_split(tmp_path):
    v6, _ = reference_paths(tmp_path, "6", "holdout")
    v7, _ = reference_paths(tmp_path, "7", "holdout")
    train, _ = reference_paths(tmp_path, "6", "train")
    assert v6 != v7 != train and v6 != train
    assert "v6" in v6.name and "holdout" in v6.name
