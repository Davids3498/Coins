"""Tests for coin_clf.data -- the clean-by-default contract and the ONE training boundary.

What these lock down, all learned the hard way over three rounds of "fixed it":

1. CLEAN BY DEFAULT (F7). Every entry point resolves to the clean survivor list unless a caller
   passes allow_raw_tree=True out loud. A missing clean list or manifest RAISES; it never
   degrades to the raw tree. Safety being opt-in at each call site is why a new leak kept
   appearing behind the last one.

2. ONE HOLDOUT (F2). build_manifest_holdout is the only definition. The two that used to compete
   with it -- frozen_split()'s independent 80/20 draw and build_test_dataset() -- must stay
   deleted, so there is an explicit regression test that they cannot be imported.

3. HASH-BASED LEAKAGE GUARD (F6). active_split's disjointness check is by content hash. The old
   filename-set check could not see the case it existed to catch: the same coin re-uploaded under
   a different name sits in both lists with different relpaths and identical bytes.
"""
import json
from pathlib import Path

import pytest

from coin_clf.data import (
    RawTreeError,
    active_split,
    build_manifest_holdout,
    discover_dataset,
    resolve_clean_list,
    resolve_manifest,
)

PER_CLASS = 40
FAKE_CLASS_NAMES = ["GORDIAN I", "GORDIAN II", "EMPEROR2", "EMPEROR3", "EMPEROR4"]


def make_fake_dataset(tmp_path, class_names=FAKE_CLASS_NAMES, per_class=PER_CLASS):
    # Unique content per file (NOT .touch()) -- every file needs a distinct hash by default so
    # the duplicate tests below can introduce exactly one controlled collision.
    for c, name in enumerate(class_names):
        side_a = tmp_path / f"{c:02d}_{name}" / "side_a"
        side_a.mkdir(parents=True)
        for i in range(per_class):
            (side_a / f"img_{i}.jpg").write_bytes(f"{name}-{i}".encode())
    return tmp_path


def all_relpaths(data_dir):
    return sorted(str(p.relative_to(data_dir)) for p in Path(data_dir).glob("*/side_a/*.jpg"))


@pytest.fixture
def data_dir(tmp_path):
    return make_fake_dataset(tmp_path)


@pytest.fixture
def clean_list(data_dir):
    path = data_dir / "clean_files.txt"
    path.write_text("\n".join(all_relpaths(data_dir)) + "\n")
    return path


@pytest.fixture
def manifest(data_dir, clean_list):
    """A real carve, saved the way splits.py writes it -- not a hand-rolled dict."""
    from splits import carve

    splits = carve(data_dir, clean_list=clean_list, seed=42)
    path = data_dir / "splits_manifest.json"
    splits.save(path, data_dir=data_dir)
    return path


@pytest.fixture
def active_train_list(data_dir, manifest):
    """The starting state release_batch.py bootstraps: the manifest's train split, verbatim."""
    train = json.loads(manifest.read_text())["splits"]["train"]
    path = data_dir / "active_train.txt"
    path.write_text("\n".join(train) + "\n")
    return path


def relpaths_of(ds, data_dir):
    return {str(Path(ds.filepaths[i]).relative_to(data_dir)) for i in ds.indices}


# --- 1. clean by default (F7) --------------------------------------------------------------

def test_resolve_clean_list_raises_instead_of_falling_back_to_raw(tmp_path):
    with pytest.raises(RawTreeError, match="Refusing to fall back to the raw tree"):
        resolve_clean_list(tmp_path / "nope.txt")


def test_resolve_clean_list_allows_raw_tree_only_when_asked(tmp_path):
    assert resolve_clean_list(allow_raw_tree=True) is None


def test_clean_list_and_allow_raw_tree_are_mutually_exclusive(clean_list):
    with pytest.raises(ValueError, match="not both"):
        resolve_clean_list(clean_list, allow_raw_tree=True)


def test_resolve_manifest_raises_when_missing(tmp_path):
    with pytest.raises(RawTreeError, match="no fallback"):
        resolve_manifest(tmp_path / "nope.json")


def test_discover_dataset_filters_to_the_clean_list(data_dir):
    keep = all_relpaths(data_dir)[:50]
    path = data_dir / "clean_files.txt"
    path.write_text("\n".join(keep) + "\n")

    filepaths, *_ = discover_dataset(data_dir, clean_list=path)
    assert {str(p.relative_to(data_dir)) for p in filepaths} == set(keep)


def test_discover_dataset_raises_when_a_clean_list_file_vanished(data_dir, clean_list):
    # Someone quarantined too much: the list still names a file the tree no longer has. Every
    # split derived from here would silently shrink, so this must raise rather than shrink.
    victim = all_relpaths(data_dir)[0]
    (data_dir / victim).unlink()
    with pytest.raises(RawTreeError, match="missing from"):
        discover_dataset(data_dir, clean_list=clean_list)


def test_discover_dataset_raw_tree_requires_the_explicit_flag(data_dir):
    filepaths, *_ = discover_dataset(data_dir, allow_raw_tree=True)
    assert len(filepaths) == PER_CLASS * len(FAKE_CLASS_NAMES)


# --- 2. one holdout (F2) -------------------------------------------------------------------

def test_the_deleted_holdout_definitions_stay_deleted():
    # Regression guard. These were the 12,427-image and 11,382-image rivals to the manifest's
    # 11,559 -- reintroducing either is how the leak came back last time.
    import coin_clf.data as data_mod

    assert not hasattr(data_mod, "frozen_split")
    assert not hasattr(data_mod, "build_test_dataset")
    assert not hasattr(data_mod, "CoinDistilDataset")


def test_build_manifest_holdout_returns_exactly_the_manifest_holdout(data_dir, manifest, clean_list):
    expected = set(json.loads(manifest.read_text())["splits"]["holdout"])
    ds = build_manifest_holdout(data_dir, manifest, clean_list=clean_list)
    assert relpaths_of(ds, data_dir) == expected


def test_build_manifest_holdout_raises_if_a_holdout_file_is_missing(data_dir, manifest, clean_list):
    # A holdout that silently shrinks silently flatters whatever is scored on it.
    victim = json.loads(manifest.read_text())["splits"]["holdout"][0]
    (data_dir / victim).unlink()
    survivors = [r for r in all_relpaths(data_dir)]
    clean_list.write_text("\n".join(survivors) + "\n")

    with pytest.raises(RawTreeError, match="not found in the discovered dataset"):
        build_manifest_holdout(data_dir, manifest, clean_list=clean_list)


def test_active_split_holdout_is_identical_to_build_manifest_holdout(
    data_dir, manifest, active_train_list, clean_list
):
    ds = build_manifest_holdout(data_dir, manifest, clean_list=clean_list)
    filepaths, _, _, _, _, _, _, test_idx = active_split(
        data_dir, active_train_list=active_train_list, manifest_path=manifest, clean_list=clean_list, jobs=1
    )
    from_split = {str(Path(filepaths[i]).relative_to(data_dir)) for i in test_idx}
    assert from_split == relpaths_of(ds, data_dir)


# --- 3. the training boundary + hash guard (F6) ---------------------------------------------

def test_active_split_partitions_the_active_list_into_train_and_val(
    data_dir, manifest, active_train_list, clean_list
):
    active = set(l for l in active_train_list.read_text().splitlines() if l.strip())
    filepaths, _, _, _, _, train_idx, val_idx, _ = active_split(
        data_dir, active_train_list=active_train_list, manifest_path=manifest, clean_list=clean_list, jobs=1
    )
    train_rel = {str(Path(filepaths[i]).relative_to(data_dir)) for i in train_idx}
    val_rel = {str(Path(filepaths[i]).relative_to(data_dir)) for i in val_idx}

    assert train_rel | val_rel == active
    assert not (train_rel & val_rel)


def test_active_split_catches_a_byte_duplicate_under_a_different_filename(
    data_dir, manifest, active_train_list, clean_list
):
    # THE case the old filename-set check could not see: different relpath, identical bytes.
    m = json.loads(manifest.read_text())
    victim_holdout = m["splits"]["holdout"][0]
    twin_in_train = m["splits"]["train"][0]
    (data_dir / twin_in_train).write_bytes((data_dir / victim_holdout).read_bytes())

    with pytest.raises(ValueError, match="BYTE-IDENTICAL"):
        active_split(data_dir, active_train_list=active_train_list, manifest_path=manifest, clean_list=clean_list, jobs=1)


def test_active_split_accepts_a_clean_active_list(data_dir, manifest, active_train_list, clean_list):
    # The control for the test above: same fixture, no injected duplicate, must not raise.
    _, _, _, _, _, train_idx, val_idx, test_idx = active_split(
        data_dir, active_train_list=active_train_list, manifest_path=manifest, clean_list=clean_list, jobs=1
    )
    assert len(train_idx) and len(val_idx) and len(test_idx)


def test_active_split_from_manifest_train_needs_no_active_list(data_dir, manifest, clean_list):
    expected = set(json.loads(manifest.read_text())["splits"]["train"])
    filepaths, _, _, _, _, train_idx, val_idx, _ = active_split(
        data_dir, manifest_path=manifest, from_manifest_train=True, clean_list=clean_list, jobs=1
    )
    got = {str(Path(filepaths[i]).relative_to(data_dir)) for i in list(train_idx) + list(val_idx)}
    assert got == expected


def test_active_split_rejects_both_sources_at_once(data_dir, manifest, active_train_list, clean_list):
    with pytest.raises(ValueError, match="not both"):
        active_split(
            data_dir, active_train_list=active_train_list, manifest_path=manifest,
            from_manifest_train=True, clean_list=clean_list, jobs=1,
        )


def test_active_split_raises_when_the_active_list_is_missing(data_dir, manifest, clean_list):
    with pytest.raises(RawTreeError, match="not found"):
        active_split(
            data_dir, active_train_list=data_dir / "nope.txt", manifest_path=manifest, clean_list=clean_list, jobs=1
        )
