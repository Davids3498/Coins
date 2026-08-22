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
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError

from coin_clf.hashing import file_hash

DEFAULT_MIN_DIM = 32
DEFAULT_MAX_CLASS_SHARE = 0.5  # one class filling more than half a batch is "wildly over-represented"
_UNREADABLE_EXCEPTIONS = (OSError, SyntaxError, UnidentifiedImageError)


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
        width = height = mode = None
        readable = True
        try:
            with Image.open(path) as img:
                img.verify()  # cheap corruption check; invalidates `img` for further reads
            with Image.open(path) as img:  # re-open: verify() leaves the handle unusable
                width, height = img.size
                mode = img.mode
        except _UNREADABLE_EXCEPTIONS:
            readable = False
        rows.append({
            "path": path,
            "label": label,
            "width": width,
            "height": height,
            "mode": mode,
            "hash": file_hash(path),
            "readable": readable,
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
