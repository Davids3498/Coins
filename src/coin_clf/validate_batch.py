"""validate_batch.py -- the quality gate every incoming future-pool batch passes before it's
allowed into training. Front gate of the retraining pipeline: a DAG task calls validate_batch()
on a freshly pulled batch and only proceeds to fold it into the training set if the report says
so; a failing report tells it what to log or quarantine instead.

Mirrors promote.py's gate pattern on purpose: validate_batch() REPORTS a structured verdict
(BatchValidationReport), it never raises for a bad IMAGE -- only a malformed call (an empty batch,
a bad max_class_share) is a hard error, exactly like decide() rejecting a negative margin.

`batch` is a sequence of (path, label) pairs rather than splits.Batch itself: splits.Batch carries
GORDIAN-merge-encoded label INDICES (int), but known_classes here is the human-readable class
vocabulary (coin_labels.json's values), and this module lives in src/coin_clf -- it must not
import splits.py, a root-level script, without inverting the package's dependency direction. A DAG
wiring a FuturePool batch through this gate zips batch.filepaths with string labels (decoded via
whatever idx_to_label mapping it already has) before calling validate_batch().

Duplicate-hash detection reuses coin_clf.hashing.file_hash -- the same SHA-256-over-bytes logic
clean_duplicates.py uses to hash the raw tree -- so a batch image and a reference image collide
under validate_batch() if and only if clean_duplicates.py would also call them duplicates.
Width/height/mode come from coin_clf.image_meta for the same reason: the serving layer's
prediction log reads image metadata too, and the gate's notion of an image's mode and the
monitoring layer's must be one notion, not two that happen to agree today.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from coin_clf.hashing import file_hash
from coin_clf.image_meta import metadata_from_path

DEFAULT_MIN_DIM = 32
DEFAULT_MAX_CLASS_SHARE = 0.5  # one class filling more than half a batch is "wildly over-represented"


@dataclass(frozen=True)
class CheckResult:
    """One row of the gate's verdict: did this check pass, and which images tripped it."""

    name: str
    passed: bool
    detail: str
    offending_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class BatchValidationReport:
    """The gate's verdict for one batch. `is_valid` is what the DAG branches on; `checks` and
    `offending_rows` are what it logs or quarantines against when `is_valid` is False.
    """

    is_valid: bool
    checks: tuple[CheckResult, ...]
    frame: pd.DataFrame

    def __repr__(self) -> str:
        passed = sum(c.passed for c in self.checks)
        return (f"BatchValidationReport(is_valid={self.is_valid}, "
                f"checks_passed={passed}/{len(self.checks)}, n_images={len(self.frame)})")

    @property
    def failed_checks(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)

    @property
    def offending_rows(self) -> list[dict]:
        """Flat, CSV-ready rows (path + which check failed + why) for logging or quarantining."""
        return [
            {"path": str(path), "check": check.name, "detail": check.detail}
            for check in self.failed_checks
            for path in check.offending_paths
        ]


def _build_frame(batch: Iterable[tuple[str | Path, str]]) -> pd.DataFrame:
    rows = []
    for path, label in batch:
        path = Path(path)
        meta = metadata_from_path(path)  # None == unreadable/corrupt
        rows.append({
            "path": path,
            "label": label,
            "width": meta.width if meta else None,
            "height": meta.height if meta else None,
            "mode": meta.mode if meta else None,
            "hash": file_hash(path),
            "readable": meta is not None,
        })
    return pd.DataFrame(rows, columns=["path", "label", "width", "height", "mode", "hash", "readable"])


def _check_readable(frame: pd.DataFrame) -> CheckResult:
    bad = frame.loc[~frame["readable"], "path"]
    return CheckResult("readable", bad.empty, f"{len(bad)} unreadable/corrupt image(s)", tuple(bad))


def _check_rgb_mode(frame: pd.DataFrame) -> CheckResult:
    checked = frame[frame["readable"]]
    bad = checked.loc[checked["mode"] != "RGB", "path"]
    return CheckResult("rgb_mode", bad.empty, f"{len(bad)} non-RGB image(s) (grayscale/RGBA/other)", tuple(bad))


def _check_min_dimensions(frame: pd.DataFrame, min_dim: int) -> CheckResult:
    checked = frame[frame["readable"]]
    bad = checked.loc[(checked["width"] < min_dim) | (checked["height"] < min_dim), "path"]
    return CheckResult("min_dimensions", bad.empty, f"{len(bad)} image(s) smaller than {min_dim}px", tuple(bad))


def _check_known_label(frame: pd.DataFrame, known_classes: Iterable[str]) -> CheckResult:
    known = set(known_classes)
    bad = frame.loc[~frame["label"].isin(known), "path"]
    return CheckResult("known_label", bad.empty, f"{len(bad)} image(s) with an unrecognized label", tuple(bad))


def _check_class_balance(frame: pd.DataFrame, max_class_share: float) -> CheckResult:
    """Flags any label filling more than max_class_share of the batch. A single-class batch is
    the extreme case of this (share == 1.0), so it's caught by the same rule, not a second one.
    """
    shares = frame["label"].value_counts(normalize=True)
    offending_labels = shares[shares > max_class_share].index
    bad = frame.loc[frame["label"].isin(offending_labels), "path"]
    worst = f"{shares.max():.0%}" if len(shares) else "0%"
    detail = f"{len(offending_labels)} class(es) exceed {max_class_share:.0%} of the batch (worst: {worst})"
    return CheckResult("class_balance", bad.empty, detail, tuple(bad))


def _check_no_intra_batch_duplicates(frame: pd.DataFrame) -> CheckResult:
    bad = frame.loc[frame["hash"].duplicated(keep=False), "path"]
    return CheckResult("no_intra_batch_duplicates", bad.empty,
                        f"{len(bad)} image(s) byte-identical to another image in this batch", tuple(bad))


def _check_no_reference_leakage(frame: pd.DataFrame, known_hashes: Iterable[str]) -> CheckResult:
    known = set(known_hashes)
    bad = frame.loc[frame["hash"].isin(known), "path"]
    return CheckResult("no_reference_set_leakage", bad.empty,
                        f"{len(bad)} image(s) byte-identical to an already-ingested train/holdout image",
                        tuple(bad))


def validate_batch(
    batch: Iterable[tuple[str | Path, str]],
    known_classes: Iterable[str],
    min_dim: int = DEFAULT_MIN_DIM,
    known_hashes: Iterable[str] | None = None,
    max_class_share: float = DEFAULT_MAX_CLASS_SHARE,
) -> BatchValidationReport:
    """Run the full quality gate over one future-pool batch.

    batch: (path, label) pairs -- see the module docstring for why not splits.Batch directly.
    known_classes: the accepted label vocabulary (e.g. coin_labels.json's values).
    known_hashes: precomputed hashes of the train/holdout reference set, for the leakage guard.
        Pass None to skip that check (still checks for duplicates WITHIN the batch); passing it
        precomputed keeps this fast -- callers should hash the ~57k reference images once and
        reuse the set, not re-hash them on every validate_batch() call.
    max_class_share: a label filling more than this fraction of the batch fails class_balance.

    Never raises for a bad IMAGE -- that's what the report is for. Only rejects a malformed CALL.
    """
    batch = list(batch)
    if not batch:
        raise ValueError("batch is empty -- nothing to validate")
    if not 0 < max_class_share <= 1:
        raise ValueError(f"max_class_share must be in (0, 1], got {max_class_share}")

    frame = _build_frame(batch)
    checks = [
        _check_readable(frame),
        _check_rgb_mode(frame),
        _check_min_dimensions(frame, min_dim),
        _check_known_label(frame, known_classes),
        _check_class_balance(frame, max_class_share),
        _check_no_intra_batch_duplicates(frame),
    ]
    if known_hashes is not None:
        checks.append(_check_no_reference_leakage(frame, known_hashes))

    return BatchValidationReport(
        is_valid=all(c.passed for c in checks),
        checks=tuple(checks),
        frame=frame,
    )


def _read_batch_file(path: Path) -> list[tuple[str, str]]:
    """Parse release_batch.py's '<absolute path>,<label>' rows, one image per line."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        img_path, label = line.rsplit(",", 1)
        rows.append((img_path, label))
    return rows


def main() -> None:
    """CLI wrapper for the retraining DAG's validate task: score one released batch and write a
    JSON report. Always exits 0 -- validate_batch() never raises for a bad batch, only for a
    malformed call, so a bad batch here is a normal outcome the DAG quarantines (via
    gate_on_validation reading is_valid out of the report), not a task failure.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-file", required=True,
                    help="'path,label' rows, as written by release_batch.py")
    p.add_argument("--data-dir", required=True,
                    help="root the manifest's/active-train-list's relative paths resolve against")
    p.add_argument("--manifest", required=True,
                    help="splits.py manifest -- source of known_classes and the holdout half of "
                         "the leakage reference set")
    p.add_argument("--active-train-list", required=True,
                    help="source of the train half of the leakage reference set")
    p.add_argument("--report-out", required=True)
    p.add_argument("--min-dim", type=int, default=DEFAULT_MIN_DIM)
    p.add_argument("--max-class-share", type=float, default=DEFAULT_MAX_CLASS_SHARE)
    args = p.parse_args()

    batch = _read_batch_file(Path(args.batch_file))
    manifest = json.loads(Path(args.manifest).read_text())
    known_classes = list(manifest["label_encoder"].keys())

    data_dir = Path(args.data_dir).resolve()
    # release_batch.py already appended THIS batch's own files to active_train_list before
    # validate ever runs (its task order is release -> validate, not the other way round) -- so
    # the reference set has to exclude this batch's own relpaths, or every batch would trivially
    # "already be ingested" against itself. Anything else in active_train_list (the clean seed
    # plus any earlier, already-validated batches) is fair game for the leakage check. A batch
    # row that isn't under data_dir at all (never true for release_batch.py's own output, but
    # not a reason to crash the task) just has nothing to exclude -- it's obviously not one of
    # the pre-existing reference files either way.
    batch_relpaths = set()
    for path, _ in batch:
        try:
            batch_relpaths.add(str(Path(path).resolve().relative_to(data_dir)))
        except ValueError:
            continue
    reference_relpaths = (
        set(Path(args.active_train_list).read_text().splitlines()) | set(manifest["splits"]["holdout"])
    ) - batch_relpaths
    print(f"hashing {len(reference_relpaths)} reference image(s) (active train + holdout, "
          "excluding this batch) for the leakage check...")
    known_hashes = {file_hash(data_dir / rel) for rel in reference_relpaths}

    report = validate_batch(
        batch, known_classes, min_dim=args.min_dim, known_hashes=known_hashes,
        max_class_share=args.max_class_share,
    )

    print(repr(report))
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.name}: {check.detail}")

    out = {
        "is_valid": report.is_valid,
        "n_images": len(report.frame),
        "checks": [
            {"name": c.name, "passed": c.passed, "detail": c.detail, "n_offending": len(c.offending_paths)}
            for c in report.checks
        ],
        "offending_rows": report.offending_rows,
    }
    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote report -> {args.report_out}  is_valid={report.is_valid}")


if __name__ == "__main__":
    main()
