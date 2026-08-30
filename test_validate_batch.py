"""Tests for coin_clf.validate_batch -- the retraining pipeline's front-gate quality check.

Real tiny images are written to tmp_path (not fakes) because the readable/mode/dimension checks
actually open and decode files with PIL -- an empty or fake file would trip the "corrupt" check
for every fixture, not just the one meant to.
"""
import random
import shutil

import pytest
from PIL import Image

from coin_clf.hashing import file_hash
from coin_clf.image_meta import decodes, metadata_from_path
from coin_clf.validate_batch import validate_batch

KNOWN_CLASSES = {"AUGUSTUS", "NERO", "TRAJAN", "HADRIAN", "VESPASIAN"}


def make_image(path, size=(64, 64), mode="RGB", color=(200, 50, 50)):
    Image.new(mode, size, color).save(path, format="JPEG")
    return path


def failed_names(report):
    return {c.name for c in report.failed_checks}


# --- clean batch --------------------------------------------------------------

def test_clean_batch_passes(tmp_path):
    batch = [
        (make_image(tmp_path / "a.jpg", color=(10, 10, 10)), "AUGUSTUS"),
        (make_image(tmp_path / "b.jpg", color=(20, 20, 20)), "NERO"),
        (make_image(tmp_path / "c.jpg", color=(30, 30, 30)), "TRAJAN"),
    ]
    report = validate_batch(batch, KNOWN_CLASSES)
    assert report.is_valid is True
    assert all(c.passed for c in report.checks)
    assert report.offending_rows == []


# --- one bad input per check ---------------------------------------------------

def test_corrupt_file_trips_only_readable_check(tmp_path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"this is not a valid jpeg")
    batch = [
        (make_image(tmp_path / "a.jpg", color=(1, 1, 1)), "AUGUSTUS"),
        (make_image(tmp_path / "b.jpg", color=(2, 2, 2)), "NERO"),
        (corrupt, "TRAJAN"),
    ]
    report = validate_batch(batch, KNOWN_CLASSES)
    assert report.is_valid is False
    assert failed_names(report) == {"readable"}
    readable = next(c for c in report.checks if c.name == "readable")
    assert corrupt in readable.offending_paths


def make_truncated_image(path, keep=0.5):
    """A JPEG whose HEADER survives but whose scan data is cut short.

    Noise, not a solid colour, and that matters: a flat 64x64 JPEG compresses to ~693 bytes, of
    which the header is most of it, so truncating one destroys the header and every reader
    rejects it -- including the weak one, which makes it useless for proving anything. Noise
    compresses to ~2.9 KB, so half the file is still well past the header. The assertion below
    pins that property rather than trusting it.
    """
    img = Image.new("RGB", (64, 64))
    rng = random.Random(20260831)
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(64 * 64)])
    img.save(path, format="JPEG")

    raw = path.read_bytes()
    path.write_bytes(raw[: int(len(raw) * keep)])

    assert metadata_from_path(path) is not None, (
        "truncated too far -- the header is gone, so this fixture no longer isolates the "
        "header-parse-vs-full-decode distinction it exists to test"
    )
    return path


def test_truncated_file_trips_only_readable_check(tmp_path):
    """The case the gate used to pass.

    A JPEG cut mid-scan keeps an intact header, so PIL's verify() accepts it and reports its real
    64x64 RGB metadata -- `readable` used to be derived from exactly that and called the file
    fine. It is not: a DataLoader raises OSError on it mid-epoch. Only a full decode sees it.

    Asserting == {"readable"} pins the other half too: on paper the file is still a valid RGB
    64x64 image, so no other check should fire on it.
    """
    truncated = make_truncated_image(tmp_path / "truncated.jpg")
    batch = [
        (make_image(tmp_path / "a.jpg", color=(1, 1, 1)), "AUGUSTUS"),
        (make_image(tmp_path / "b.jpg", color=(2, 2, 2)), "NERO"),
        (truncated, "TRAJAN"),
    ]
    report = validate_batch(batch, KNOWN_CLASSES)

    assert report.is_valid is False
    assert failed_names(report) == {"readable"}
    readable = next(c for c in report.checks if c.name == "readable")
    assert truncated in readable.offending_paths


def test_header_parse_and_full_decode_disagree_on_truncation(tmp_path):
    """Guards the distinction itself, not just the outcome.

    metadata_from_path stays cheap for the serving path, which decodes every upload anyway;
    decodes() is the strict one the gate uses. If these two ever agree on a truncated file,
    someone has changed one of them, and this failing is how they find out -- rather than the two
    predicates quietly collapsing into one and serving paying for a full decode per request.
    """
    truncated = make_truncated_image(tmp_path / "truncated.jpg")

    meta = metadata_from_path(truncated)
    assert meta is not None, "header parse should still succeed -- that is the whole problem"
    assert (meta.width, meta.height, meta.mode) == (64, 64, "RGB")
    assert decodes(truncated) is False


def test_grayscale_trips_only_rgb_mode_check(tmp_path):
    batch = [
        (make_image(tmp_path / "a.jpg", color=(1, 1, 1)), "AUGUSTUS"),
        (make_image(tmp_path / "b.jpg", color=(2, 2, 2)), "NERO"),
        (make_image(tmp_path / "gray.jpg", mode="L", color=128), "TRAJAN"),
    ]
    report = validate_batch(batch, KNOWN_CLASSES)
    assert report.is_valid is False
    assert failed_names(report) == {"rgb_mode"}
    rgb_mode = next(c for c in report.checks if c.name == "rgb_mode")
    assert (tmp_path / "gray.jpg") in rgb_mode.offending_paths


def test_undersized_trips_only_min_dimensions_check(tmp_path):
    batch = [
        (make_image(tmp_path / "a.jpg", color=(1, 1, 1)), "AUGUSTUS"),
        (make_image(tmp_path / "b.jpg", color=(2, 2, 2)), "NERO"),
        (make_image(tmp_path / "small.jpg", size=(10, 10), color=(3, 3, 3)), "TRAJAN"),
    ]
    report = validate_batch(batch, KNOWN_CLASSES, min_dim=32)
    assert report.is_valid is False
    assert failed_names(report) == {"min_dimensions"}
    min_dims = next(c for c in report.checks if c.name == "min_dimensions")
    assert (tmp_path / "small.jpg") in min_dims.offending_paths


def test_unknown_label_trips_only_known_label_check(tmp_path):
    batch = [
        (make_image(tmp_path / "a.jpg", color=(1, 1, 1)), "AUGUSTUS"),
        (make_image(tmp_path / "b.jpg", color=(2, 2, 2)), "NERO"),
        (make_image(tmp_path / "c.jpg", color=(3, 3, 3)), "NOT_A_REAL_EMPEROR"),
    ]
    report = validate_batch(batch, KNOWN_CLASSES)
    assert report.is_valid is False
    assert failed_names(report) == {"known_label"}
    known_label = next(c for c in report.checks if c.name == "known_label")
    assert (tmp_path / "c.jpg") in known_label.offending_paths


def test_duplicate_pair_trips_only_intra_batch_duplicates_check(tmp_path):
    original = make_image(tmp_path / "a.jpg", color=(9, 9, 9))
    copy = tmp_path / "a_copy.jpg"
    shutil.copy(original, copy)  # byte-identical, guaranteed (not just visually identical)
    batch = [
        (original, "AUGUSTUS"),
        (copy, "NERO"),
        (make_image(tmp_path / "c.jpg", color=(3, 3, 3)), "TRAJAN"),
    ]
    report = validate_batch(batch, KNOWN_CLASSES)
    assert report.is_valid is False
    assert failed_names(report) == {"no_intra_batch_duplicates"}
    dup_check = next(c for c in report.checks if c.name == "no_intra_batch_duplicates")
    assert set(dup_check.offending_paths) == {original, copy}


def test_over_represented_class_trips_only_class_balance_check(tmp_path):
    batch = [(make_image(tmp_path / f"aug_{i}.jpg", color=(i, i, i)), "AUGUSTUS") for i in range(4)]
    batch.append((make_image(tmp_path / "nero.jpg", color=(99, 99, 99)), "NERO"))
    report = validate_batch(batch, KNOWN_CLASSES, max_class_share=0.5)
    assert report.is_valid is False
    assert failed_names(report) == {"class_balance"}
    balance = next(c for c in report.checks if c.name == "class_balance")
    assert len(balance.offending_paths) == 4  # the over-represented AUGUSTUS images, not NERO


# --- leakage guard (known_hashes) ----------------------------------------------

def test_known_hashes_collision_is_caught(tmp_path):
    holdout_dir = tmp_path / "holdout"
    holdout_dir.mkdir()
    holdout_image = make_image(holdout_dir / "ref.jpg", color=(77, 77, 77))
    known_hashes = {file_hash(holdout_image)}

    leaked = tmp_path / "leaked.jpg"
    shutil.copy(holdout_image, leaked)  # a "new" batch image that is actually already in holdout

    batch = [
        (leaked, "AUGUSTUS"),
        (make_image(tmp_path / "b.jpg", color=(2, 2, 2)), "NERO"),
        (make_image(tmp_path / "c.jpg", color=(3, 3, 3)), "TRAJAN"),
    ]
    report = validate_batch(batch, KNOWN_CLASSES, known_hashes=known_hashes)
    assert report.is_valid is False
    assert failed_names(report) == {"no_reference_set_leakage"}
    leakage = next(c for c in report.checks if c.name == "no_reference_set_leakage")
    assert leaked in leakage.offending_paths

    # not a duplicate WITHIN the batch -- the leak is against the external reference set only.
    dup_check = next(c for c in report.checks if c.name == "no_intra_batch_duplicates")
    assert dup_check.passed is True


def test_no_leakage_check_run_when_known_hashes_not_passed(tmp_path):
    batch = [(make_image(tmp_path / "a.jpg", color=(1, 1, 1)), "AUGUSTUS")]
    report = validate_batch(batch, KNOWN_CLASSES)
    assert "no_reference_set_leakage" not in {c.name for c in report.checks}


# --- malformed calls -----------------------------------------------------------

def test_empty_batch_raises():
    with pytest.raises(ValueError):
        validate_batch([], KNOWN_CLASSES)


def test_bad_max_class_share_raises(tmp_path):
    batch = [(make_image(tmp_path / "a.jpg"), "AUGUSTUS")]
    with pytest.raises(ValueError):
        validate_batch(batch, KNOWN_CLASSES, max_class_share=0)
