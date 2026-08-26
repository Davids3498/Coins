"""drift_report.py -- compare recent production predictions against a cached reference
distribution and emit an Evidently HTML report plus a machine-readable verdict.

Nothing observed the champion in production before this. app/main.py logs one row per /predict
into the SQLite prediction log (coin_clf.prediction_log); this script is what turns that log into
a drift signal a DAG task can branch on.

THE REFERENCE IS THE CHAMPION'S OWN PREDICTIONS, NOT THE TRAINING LABELS.
Production gives us model OUTPUT (a predicted class, a confidence). Ground-truth labels are a
different quantity, and comparing the two would be wrong in two ways. First, the champion is
~92% accurate, so on a perfectly in-distribution batch its predicted-class histogram already
differs from the label histogram by ~8% of probability mass -- a permanent non-zero baseline
that is model error, not drift, and that silently moves every time @champion is promoted.
Second, labels carry no confidence at all, so the confidence signal could not exist. Scoring the
champion over a reference set makes both sides the same quantity: "what this model outputs given
an image." Model error then appears identically on both sides and cancels.

THE REFERENCE SET IS THE HOLDOUT, NOT THE TRAIN SPLIT.
The champion trained on the train split for 120 epochs, so its confidence there is memorization-
inflated toward 1.0. Production traffic is unseen images, so a train-based reference would report
large confidence drift on day one against perfectly normal traffic -- a false positive wired
straight into the retraining trigger, where it costs a 120-epoch run. The holdout is unseen by
the champion, carved from the same clean corpus by splits.py, and is like-for-like with the
future-pool images the replay script sends. --reference-split train is the documented override
and is expected to inflate the confidence signal (it also runs active_split's content-hash
disjointness guard, so it is minutes slower).

THE CACHE IS KEYED ON CHAMPION VERSION, AND PRODUCTION ROWS ARE FILTERED TO IT.
This is the guard against the worst failure this pipeline can have. Compare v7's production
predictions against a v6 reference and you get guaranteed spurious drift -> a triggered retrain
-> a promoted v8 -> spurious drift again: an infinite retrain loop that looks like a working
system. So the current @champion version is resolved on every run, a cache built for any other
version is rebuilt rather than used, and production rows from a different model_version are
excluded and counted (if that leaves too few rows the verdict is insufficient_data, which is the
safe direction).

REPORT, DON'T RAISE. Mirrors coin_clf.validate_batch: every drift outcome, a missing prediction
log, and an unreachable registry are all NORMAL results written to the verdict file with exit 0.
Only a malformed CALL (a bad threshold, an unparseable --since) is a hard error. The verdict's
`status` field is what separates "no drift" from "couldn't tell" -- a DAG task must trigger only
on status == "ok" AND drift_detected, never on the boolean alone.

torch and mlflow are imported LAZILY, inside the reference builder. A warm-cache run -- the DAG's
normal path -- never loads them, following the same import-light principle dags/retrain_coin_clf.py
documents for the scheduler.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from coin_clf.image_meta import metadata_from_path
from coin_clf.prediction_log import PredictionLog

REPO_ROOT = Path(__file__).resolve().parent

MODEL_NAME = os.environ.get("MODEL_NAME", "coin-classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

DEFAULT_DB = REPO_ROOT / "outputs" / "monitoring" / "predictions.db"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "monitoring"
DEFAULT_CACHE_DIR = DEFAULT_OUT_DIR / "reference"

# The columns compared. Reference and production frames are subset to exactly these, in this
# order, so the two sides cannot silently disagree about what is being compared.
CATEGORICAL_COLUMNS = ("predicted_label", "mode")
NUMERICAL_COLUMNS = ("confidence", "width", "height")
COMPARED_COLUMNS = ("predicted_label", "confidence", "width", "height", "mode")

# Three signals, each one or more columns. A signal fires if ANY of its columns fires; the
# verdict names the specific column so a triggered retrain traces back to a cause.
SIGNALS = {
    "predicted_class": ("predicted_label",),
    "confidence": ("confidence",),
    "image_metadata": ("width", "height", "mode"),
}

# --- thresholds ------------------------------------------------------------------------------
# MEASURED, not defaulted. The null distribution (what a NO-DRIFT sample of this size actually
# scores against this reference) was simulated against the real 51-class distribution and real
# holdout image dimensions:
#
#   n=500 vs ref=11559   jensenshannon on class: median 0.138, p95 0.164   (skewed batch: 0.88)
#                        normed-wasserstein on width: median 0.051, p95 0.100  (degraded: 0.87)
#
# So the obvious 0.1 default would fire on EVERY normal batch for class drift and ~5% of the time
# for width. These thresholds sit at ~2.5x the empirical null p95 at DEFAULT_MIN_SAMPLES, which
# leaves a wide band on both sides: normal traffic scores well under, real skew scores 3-5x over.
#
# Magnitude tests, not p-values: with a reference of 11,559 rows a p-value answers "is there ANY
# difference", which at that sample size is always yes (KS's critical D here is ~0.063, so
# same-distribution samples fire routinely). A magnitude answers "is it big enough to care",
# which is what should gate an expensive retrain. --class-method / --numeric-method can still
# select ks/psi/chisquare if you want the other behaviour.
#
# jensenshannon for categorical rather than PSI or chi-square: it is bounded [0,1] so scores are
# comparable across signals, it is symmetric, and it stays stable when a class is absent from one
# side -- exactly the skewed-batch case, where most of the 51 reference classes have zero
# production mass. PSI needs smoothing there and chi-square's expected counts collapse.
#
# NOTE ON SCALE: evidently's "wasserstein" is normalized by the REFERENCE STANDARD DEVIATION and
# is NOT bounded by 1 (a large shift scores >1). The null figures above are on that same scale.
DEFAULT_CLASS_METHOD = "jensenshannon"
DEFAULT_CLASS_THRESHOLD = 0.35
DEFAULT_NUMERIC_METHOD = "wasserstein"
DEFAULT_NUMERIC_THRESHOLD = 0.25

# CONFIDENCE'S THRESHOLD IS PROVISIONAL. Unlike the class and metadata thresholds it has no
# measured null distribution behind it -- observing one requires an inference pass, which is what
# the reference builder below does. It is set by analogy with width (both numeric, both
# std-normalized). The traffic simulator's NORMAL run is its calibration: if normal traffic scores
# above this, the threshold is wrong and should be moved deliberately, with the number written
# down here -- not nudged to make a demo pass.
CONFIDENCE_THRESHOLD_IS_PROVISIONAL = True

# Below this many production rows the verdict is insufficient_data, never drift. At n=200 the
# no-drift class-distribution score is already 0.227 (median) -- close enough to a meaningful
# threshold that a thin sample would be indistinguishable from real drift, and a false trigger
# here costs a 120-epoch retrain.
DEFAULT_MIN_SAMPLES = 500


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --- reference ---------------------------------------------------------------------------------

def resolve_champion_version(tracking_uri: str, model_name: str, model_alias: str) -> str:
    """The version currently behind the alias. Resolved on EVERY run -- see the module docstring
    on the stale-reference retrain loop this prevents.
    """
    import mlflow  # lazy: a warm-cache run still needs this, but nothing torch-sized

    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.MlflowClient().get_model_version_by_alias(model_name, model_alias).version


def _load_split_dataset(data_dir: str, manifest: str | None, split: str):
    """The reference image set, through the canonical data path -- never an ad-hoc glob.

    holdout: build_manifest_holdout, the single definition of the holdout in this codebase.
    train:   active_split(from_manifest_train=True), the canonical clean-start training boundary.
             Slower on purpose: it runs the content-hash disjointness guard. That cost belongs to
             an override that is expected to be rare.
    """
    from coin_clf.data import CoinImageDataset, active_split, build_manifest_holdout
    from coin_clf.transforms import val_transform

    if split == "holdout":
        return build_manifest_holdout(data_dir, manifest)
    filepaths, all_labs, _, _, _, train_idx, _, _ = active_split(
        data_dir, manifest_path=manifest, from_manifest_train=True
    )
    return CoinImageDataset(filepaths, all_labs, train_idx, val_transform)


def build_reference(
    data_dir: str,
    manifest: str | None,
    split: str,
    model_version: str,
    tracking_uri: str,
    model_name: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Score the champion over the reference split and record exactly the production columns.

    The expensive half of this script (an inference pass plus a metadata read over ~11.5k images),
    which is why the result is cached and keyed on model_version.
    """
    import mlflow  # noqa: F401  (set_tracking_uri happens in resolve_champion_version)
    import mlflow.pytorch
    import torch
    from torch.utils.data import DataLoader

    mlflow.set_tracking_uri(tracking_uri)

    # idx_to_label comes from the manifest's own encoder. Safe because _load_split_dataset's
    # canonical loaders re-validate that encoder against the discovered tree and RAISE if the
    # label space has drifted -- so if these disagreed we would never reach this line.
    manifest_path = Path(manifest) if manifest else (REPO_ROOT / "data" / "splits_manifest.json")
    encoder = json.loads(manifest_path.read_text())["label_encoder"]
    idx_to_label = {int(v): k for k, v in encoder.items()}

    dataset = _load_split_dataset(data_dir, manifest, split)
    print(f"reference: scoring coin-classifier v{model_version} over {len(dataset)} "
          f"{split} image(s)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mlflow.pytorch.load_model(f"models:/{model_name}/{model_version}")
    model.to(device).eval()

    labels, confidences = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for images, _ in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            top_p, top_i = probs.max(dim=1)
            labels.extend(idx_to_label[int(i)] for i in top_i.tolist())
            confidences.extend(float(p) for p in top_p.tolist())

    # Metadata through the SAME function app/main.py logs with, read from the file as stored --
    # so "width" means the same thing on both sides of the comparison. shuffle=False above keeps
    # this aligned with the prediction order.
    print(f"reference: reading image metadata for {len(dataset.indices)} file(s)...")
    widths, heights, modes = [], [], []
    for i in dataset.indices:
        meta = metadata_from_path(dataset.filepaths[i])
        widths.append(meta.width if meta else None)
        heights.append(meta.height if meta else None)
        modes.append(meta.mode if meta else None)

    return pd.DataFrame({
        "predicted_label": labels,
        "confidence": confidences,
        "width": widths,
        "height": heights,
        "mode": modes,
    })


def reference_paths(cache_dir: Path, model_version: str, split: str) -> tuple[Path, Path]:
    stem = f"reference_v{model_version}_{split}"
    return cache_dir / f"{stem}.csv", cache_dir / f"{stem}.meta.json"


def load_or_build_reference(
    cache_dir: Path, model_version: str, split: str, refresh: bool, **build_kwargs
) -> tuple[pd.DataFrame, dict]:
    """Cache read-through. The version is in the FILENAME, so a cache built for another champion
    is not a stale hit to detect -- it is simply a different file, and this one gets built.
    """
    csv_path, meta_path = reference_paths(cache_dir, model_version, split)
    if csv_path.exists() and meta_path.exists() and not refresh:
        frame = pd.read_csv(csv_path)
        meta = json.loads(meta_path.read_text())
        print(f"reference: cache hit -> {csv_path} ({len(frame)} rows)")
        return frame, meta

    frame = build_reference(model_version=model_version, split=split, **build_kwargs)
    meta = {
        "model_version": model_version,
        "split": split,
        "n_rows": len(frame),
        "generated_at": utc_now_iso(),
        "data_dir": str(build_kwargs.get("data_dir")),
        "manifest": str(build_kwargs.get("manifest")),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"reference: cached -> {csv_path} ({len(frame)} rows)")
    return frame, meta


# --- production ---------------------------------------------------------------------------------

def load_production(
    db_path: Path, since: str | None, source: str | None, model_version: str
) -> tuple[pd.DataFrame, dict]:
    """Recent predictions, filtered to the window, the traffic tag, and the reference's version.

    Returns (frame, selection_info). The version filter is not optional -- see the module
    docstring. Rows from any other model_version are excluded and counted so the exclusion is
    visible in the verdict rather than silently shrinking the sample.
    """
    rows = PredictionLog(db_path).read_records(since=since)
    frame = pd.DataFrame(rows)
    info = {"n_matched_window": len(frame), "excluded_other_versions": 0,
            "model_versions_seen": []}
    if frame.empty:
        return frame, info

    if source is not None:
        frame = frame[frame["source"] == source]
        info["n_matched_window"] = len(frame)
    if frame.empty:
        return frame, info

    seen = sorted(str(v) for v in frame["model_version"].dropna().unique())
    info["model_versions_seen"] = seen
    keep = frame["model_version"].astype(str) == str(model_version)
    info["excluded_other_versions"] = int((~keep).sum())
    return frame[keep], info


# --- drift ---------------------------------------------------------------------------------------

def column_spec(class_method: str, class_threshold: float,
                numeric_method: str, numeric_threshold: float) -> dict:
    """{column: (method, threshold)} -- the single place a column's test is decided, so the HTML
    report and the verdict cannot be computed under different settings.
    """
    spec = {c: (class_method, class_threshold) for c in CATEGORICAL_COLUMNS}
    spec.update({c: (numeric_method, numeric_threshold) for c in NUMERICAL_COLUMNS})
    return spec


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, spec: dict,
                  html_out: Path | None = None) -> dict:
    """Run one Evidently report over all compared columns; return {column: result}.

    The verdict is read back OUT of Evidently rather than recomputed in scipy on the side. One
    computation, one number: if the HTML and the verdict came from two implementations they could
    disagree, and a report saying "no drift" beside a verdict saying "drift" is worse than either.

    The metric -> result mapping is by metric id (ValueDrift.get_metric_id, a hash of the metric's
    config) into snapshot.metric_results, NOT by position in snapshot.dict()["metrics"].
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import ValueDrift

    columns = list(COMPARED_COLUMNS)
    definition = DataDefinition(
        numerical_columns=[c for c in columns if c in NUMERICAL_COLUMNS],
        categorical_columns=[c for c in columns if c in CATEGORICAL_COLUMNS],
    )
    ref_ds = Dataset.from_pandas(reference[columns], data_definition=definition)
    cur_ds = Dataset.from_pandas(current[columns], data_definition=definition)

    metrics = [ValueDrift(column=c, method=spec[c][0], threshold=spec[c][1]) for c in columns]
    snapshot = Report(metrics).run(cur_ds, ref_ds)

    if html_out is not None:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_out))

    results = {}
    for metric, column in zip(metrics, columns):
        method, threshold = spec[column]
        score = float(snapshot.metric_results[metric.get_metric_id()].value)
        results[column] = {
            "score": round(score, 6),
            "method": method,
            "threshold": threshold,
            "drifted": bool(score > threshold),
        }
    return results


def build_verdict(status: str, column_results: dict | None, sample: dict, reference: dict,
                  report_html: str | None = None, error: str | None = None) -> dict:
    """Assemble the machine-readable verdict a DAG task branches on.

    drift_detected is False for every non-"ok" status -- "couldn't tell" must never read as
    "drift" to a trigger that costs a 120-epoch retrain. That is why status is a separate field
    and not encoded as a third boolean state.
    """
    signals, drifted_signals, reasons = {}, [], []
    if column_results:
        for signal, cols in SIGNALS.items():
            per_column = {c: column_results[c] for c in cols if c in column_results}
            fired = sorted(c for c, r in per_column.items() if r["drifted"])
            signals[signal] = {"drifted": bool(fired), "columns": per_column}
            if fired:
                drifted_signals.append(signal)
                for c in fired:
                    r = per_column[c]
                    reasons.append(f"{signal}:{c} ({r['method']} {r['score']:.4f} > {r['threshold']})")

    drift_detected = status == "ok" and bool(drifted_signals)
    verdict = {
        "status": status,
        "drift_detected": drift_detected,
        "drifted_signals": drifted_signals if drift_detected else [],
        "trigger_reason": "; ".join(reasons) if drift_detected else None,
        "signals": signals,
        "sample": sample,
        "reference": reference,
        "report_html": report_html,
        "generated_at": utc_now_iso(),
    }
    if error:
        verdict["error"] = error
    return verdict


# --- CLI -----------------------------------------------------------------------------------------

def _print_summary(verdict: dict) -> None:
    print(f"\nstatus={verdict['status']}  drift_detected={verdict['drift_detected']}")
    for signal, payload in verdict["signals"].items():
        flag = "DRIFT" if payload["drifted"] else "  ok "
        for column, r in payload["columns"].items():
            print(f"  [{flag}] {signal}:{column:16} {r['method']:14} "
                  f"{r['score']:.4f} (threshold {r['threshold']})")
    if verdict.get("trigger_reason"):
        print(f"  reason: {verdict['trigger_reason']}")
    if verdict.get("error"):
        print(f"  error: {verdict['error']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                   help="image tree root, for building the reference (unused on a cache hit)")
    p.add_argument("--manifest", default=None,
                   help="splits.py manifest (default: data/splits_manifest.json)")
    p.add_argument("--db", default=str(DEFAULT_DB), help="prediction log written by /predict")
    p.add_argument("--since", default=None,
                   help="ISO-8601 UTC lower bound on the prediction timestamp")
    p.add_argument("--source", default=None,
                   help="X-Traffic-Source tag to isolate one replay run")
    p.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                   help="below this many production rows the verdict is insufficient_data")
    p.add_argument("--reference-split", choices=("holdout", "train"), default="holdout",
                   help="holdout (default) is unseen by the champion; train is the documented "
                        "override and inflates the confidence signal (see module docstring)")
    p.add_argument("--refresh-reference", action="store_true",
                   help="rebuild the cached reference even if it exists for this version")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--class-method", default=DEFAULT_CLASS_METHOD)
    p.add_argument("--class-threshold", type=float, default=DEFAULT_CLASS_THRESHOLD)
    p.add_argument("--numeric-method", default=DEFAULT_NUMERIC_METHOD)
    p.add_argument("--numeric-threshold", type=float, default=DEFAULT_NUMERIC_THRESHOLD)
    p.add_argument("--html-out", default=None)
    p.add_argument("--verdict-out", default=str(DEFAULT_OUT_DIR / "drift_verdict.json"))
    p.add_argument("--tracking-uri", default=MLFLOW_TRACKING_URI)
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--model-alias", default=MODEL_ALIAS)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    # Malformed CALL -> hard error, exactly like validate_batch. Everything downstream of here is
    # an outcome to report.
    if args.class_threshold <= 0 or args.numeric_threshold <= 0:
        raise ValueError("thresholds must be positive")
    if args.min_samples < 1:
        raise ValueError("--min-samples must be at least 1")

    verdict_out = Path(args.verdict_out)
    html_out = Path(args.html_out) if args.html_out else (
        DEFAULT_OUT_DIR / f"drift_report_{_timestamp_slug()}.html")
    spec = column_spec(args.class_method, args.class_threshold,
                       args.numeric_method, args.numeric_threshold)
    sample = {"n_production": 0, "n_reference": 0, "min_samples": args.min_samples,
              "since": args.since, "source": args.source, "model_version": None,
              "excluded_other_versions": 0, "model_versions_seen": []}

    def finish(verdict: dict) -> None:
        verdict_out.parent.mkdir(parents=True, exist_ok=True)
        verdict_out.write_text(json.dumps(verdict, indent=2))
        _print_summary(verdict)
        print(f"\nwrote verdict -> {verdict_out}")

    # 1. champion version + reference. An unreachable registry or a failed build is an
    #    environmental outcome the DAG should see as "no_reference", not a task crash.
    try:
        model_version = resolve_champion_version(args.tracking_uri, args.model_name, args.model_alias)
        sample["model_version"] = model_version
        reference, ref_meta = load_or_build_reference(
            Path(args.cache_dir), model_version, args.reference_split, args.refresh_reference,
            data_dir=args.data_dir, manifest=args.manifest, tracking_uri=args.tracking_uri,
            model_name=args.model_name, batch_size=args.batch_size, num_workers=args.num_workers,
        )
    except Exception as exc:
        finish(build_verdict("no_reference", None, sample, {},
                             error=f"{type(exc).__name__}: {exc}"))
        return

    sample["n_reference"] = len(reference)
    ref_info = {"split": args.reference_split, "model_version": model_version,
                "cache_path": str(reference_paths(Path(args.cache_dir), model_version,
                                                  args.reference_split)[0]),
                "generated_at": ref_meta.get("generated_at")}

    # 2. production selection
    try:
        current, sel = load_production(Path(args.db), args.since, args.source, model_version)
    except (FileNotFoundError, sqlite3.Error) as exc:
        finish(build_verdict("no_data", None, sample, ref_info,
                             error=f"{type(exc).__name__}: {exc}"))
        return

    sample.update({"n_production": len(current),
                   "excluded_other_versions": sel["excluded_other_versions"],
                   "model_versions_seen": sel["model_versions_seen"]})

    if len(current) < args.min_samples:
        finish(build_verdict("insufficient_data", None, sample, ref_info))
        return

    # 3. drift
    results = compute_drift(reference, current, spec, html_out=html_out)
    verdict = build_verdict("ok", results, sample, ref_info, report_html=str(html_out))

    conf = results.get("confidence")
    if CONFIDENCE_THRESHOLD_IS_PROVISIONAL and conf and conf["drifted"] and args.source is None:
        print("\nNOTE: the confidence threshold is provisional (no measured null distribution). "
              "If this run is NORMAL traffic, that is a calibration finding -- move the threshold "
              "deliberately, do not tune it per-run.")
    finish(verdict)


if __name__ == "__main__":
    main()
