"""Tests for coin_clf.data.build_test_dataset's clean_list handling.

Two properties being tested, both learned the hard way:

1. clean_list must filter the ALREADY-COMPUTED test_idx, not shrink the universe before
   splitting. Recomputing the split on a smaller universe would move the train/test boundary
   (sklearn's stratified split consumes one shared RNG stream class by class, so dropping even
   one image from an early class reshuffles every class split after it) -- risking an
   already-trained model being scored on images that were on the TRAIN side of the boundary it
   was actually trained under.

2. clean_list MEMBERSHIP alone is not enough on top of that. clean_duplicates.py picks a
   canonical survivor per duplicate hash-group by filename, with no idea which split either copy
   landed in. If the copy it kept is the one sitting in test while its byte-identical twin sits
   in train, the file passes the membership check yet the model still trained on those exact
   pixels under the twin's name. build_test_dataset also has to drop by content hash.
"""
from pathlib import Path

import pytest

from coin_clf.data import build_test_dataset, discover_dataset, frozen_split, split_dataset
from coin_clf.hashing import file_hash

PER_CLASS = 40
FAKE_CLASS_NAMES = ["GORDIAN I", "GORDIAN II", "EMPEROR2", "EMPEROR3", "EMPEROR4"]


def make_fake_dataset(tmp_path, class_names=FAKE_CLASS_NAMES, per_class=PER_CLASS):
    # Unique content per file (NOT .touch()) -- every file needs a distinct hash by default so
    # the hash-collision test below can introduce exactly one controlled duplicate pair.
    for c, name in enumerate(class_names):
        side_a = tmp_path / f"{c:02d}_{name}" / "side_a"
        side_a.mkdir(parents=True)
        for i in range(per_class):
            (side_a / f"img_{i}.jpg").write_bytes(f"{name}-{i}".encode())
    return tmp_path


@pytest.fixture
def data_dir(tmp_path):
    return make_fake_dataset(tmp_path)


def original_split(data_dir):
    filepaths, all_labs, *_ = discover_dataset(data_dir)
    train_idx, val_idx, test_idx = split_dataset(all_labs, test_size=0.2, random_state=42)
    return filepaths, train_idx, val_idx, test_idx


def relpaths(data_dir, filepaths, indices):
    return {str(filepaths[i].relative_to(data_dir)) for i in indices}


def test_no_clean_list_matches_the_raw_split(data_dir):
    filepaths, _, _, test_idx = original_split(data_dir)
    ds = build_test_dataset(data_dir)
    assert set(ds.indices.tolist()) == set(test_idx.tolist())


def test_clean_list_only_removes_files_it_drops_from_test_side(data_dir):
    filepaths, train_idx, _, test_idx = original_split(data_dir)
    test_relpaths = relpaths(data_dir, filepaths, test_idx)
    train_relpaths = relpaths(data_dir, filepaths, train_idx)

    dropped_from_test = set(list(test_relpaths)[:3])
    survivors = (test_relpaths | train_relpaths) - dropped_from_test
    clean_list = data_dir / "clean_files.txt"
    clean_list.write_text("\n".join(sorted(survivors)) + "\n")

    ds = build_test_dataset(data_dir, clean_list=clean_list)
    result_relpaths = relpaths(data_dir, ds.filepaths, ds.indices.tolist())

    assert result_relpaths == test_relpaths - dropped_from_test


def test_dropping_train_side_files_never_changes_the_test_set(data_dir):
    # The whole point: clean_list entries on the TRAIN side must have zero effect on what ends
    # up in the test set -- if the split were being recomputed, removing train-side files would
    # reshuffle the boundary and this would fail.
    filepaths, train_idx, _, test_idx = original_split(data_dir)
    test_relpaths = relpaths(data_dir, filepaths, test_idx)
    train_relpaths = relpaths(data_dir, filepaths, train_idx)

    # Drop half the train side, keep the test side untouched.
    dropped_from_train = set(sorted(train_relpaths)[: len(train_relpaths) // 2])
    survivors = (test_relpaths | train_relpaths) - dropped_from_train
    clean_list = data_dir / "clean_files.txt"
    clean_list.write_text("\n".join(sorted(survivors)) + "\n")

    ds = build_test_dataset(data_dir, clean_list=clean_list)
    result_relpaths = relpaths(data_dir, ds.filepaths, ds.indices.tolist())

    assert result_relpaths == test_relpaths


def test_clean_list_that_keeps_everything_is_a_no_op(data_dir):
    filepaths, _, _, test_idx = original_split(data_dir)
    all_relpaths = {str(p.relative_to(data_dir)) for p in Path(data_dir).glob("*/side_a/*.jpg")}
    clean_list = data_dir / "clean_files.txt"
    clean_list.write_text("\n".join(sorted(all_relpaths)) + "\n")

    ds = build_test_dataset(data_dir, clean_list=clean_list)
    assert set(ds.indices.tolist()) == set(test_idx.tolist())


def test_hash_twin_in_train_is_excluded_even_when_test_copy_is_the_clean_list_survivor(data_dir):
    # The exact production scenario: give a test-side file and a train-side file identical
    # bytes, then have clean_list keep the TEST-side copy (as clean_duplicates.py's filename
    # tie-break might). Membership alone would wrongly keep it; hash-disjointness must not.
    filepaths, train_idx, val_idx, test_idx = original_split(data_dir)
    test_relpaths = relpaths(data_dir, filepaths, test_idx)
    train_relpaths = relpaths(data_dir, filepaths, train_idx)

    test_victim = sorted(test_relpaths)[0]
    train_twin = sorted(train_relpaths)[0]
    (data_dir / train_twin).write_bytes((data_dir / test_victim).read_bytes())

    all_relpaths = {str(p.relative_to(data_dir)) for p in Path(data_dir).glob("*/side_a/*.jpg")}
    survivors = all_relpaths - {train_twin}  # clean_duplicates.py kept the TEST-side copy
    clean_list = data_dir / "clean_files.txt"
    clean_list.write_text("\n".join(sorted(survivors)) + "\n")

    ds = build_test_dataset(data_dir, clean_list=clean_list)
    result_relpaths = relpaths(data_dir, ds.filepaths, ds.indices.tolist())

    assert test_victim not in result_relpaths
    assert result_relpaths == test_relpaths - {test_victim}


def test_hash_collision_within_test_side_only_is_not_removed(data_dir):
    # Two test-side files sharing content is metric redundancy, not train/test leakage -- the
    # hash-disjointness check is specifically against train/val, so it must leave these alone.
    filepaths, train_idx, val_idx, test_idx = original_split(data_dir)
    test_relpaths = sorted(relpaths(data_dir, filepaths, test_idx))
    a, b = test_relpaths[0], test_relpaths[1]
    (data_dir / b).write_bytes((data_dir / a).read_bytes())

    all_relpaths = {str(p.relative_to(data_dir)) for p in Path(data_dir).glob("*/side_a/*.jpg")}
    clean_list = data_dir / "clean_files.txt"
    clean_list.write_text("\n".join(sorted(all_relpaths)) + "\n")  # nothing dropped from the corpus

    ds = build_test_dataset(data_dir, clean_list=clean_list)
    result_relpaths = relpaths(data_dir, ds.filepaths, ds.indices.tolist())

    assert result_relpaths == set(test_relpaths)


def test_frozen_split_filters_train_and_val_by_membership(data_dir):
    filepaths, train_idx, val_idx, test_idx = original_split(data_dir)
    train_relpaths = relpaths(data_dir, filepaths, train_idx)
    val_relpaths = relpaths(data_dir, filepaths, val_idx)

    dropped = set(sorted(train_relpaths)[:3]) | set(sorted(val_relpaths)[:2])
    all_relpaths = {str(p.relative_to(data_dir)) for p in Path(data_dir).glob("*/side_a/*.jpg")}
    survivors = all_relpaths - dropped
    clean_list = data_dir / "clean_files.txt"
    clean_list.write_text("\n".join(sorted(survivors)) + "\n")

    new_filepaths, _, _, _, _, new_train_idx, new_val_idx, new_test_idx = frozen_split(
        data_dir, clean_list=clean_list
    )
    assert relpaths(data_dir, new_filepaths, new_train_idx) == train_relpaths - dropped
    assert relpaths(data_dir, new_filepaths, new_val_idx) == val_relpaths - dropped


def test_frozen_split_train_and_test_end_up_hash_disjoint(data_dir):
    # The end-to-end guarantee train.py now relies on: after frozen_split, nothing in the final
    # train (or val) set shares content with anything in the final test set.
    filepaths, train_idx, val_idx, test_idx = original_split(data_dir)
    train_relpaths = sorted(relpaths(data_dir, filepaths, train_idx))
    test_relpaths = sorted(relpaths(data_dir, filepaths, test_idx))

    # Inject a duplicate spanning train/test, with the TEST copy as the corpus-wide "canonical"
    # -- clean_duplicates.py's filename tie-break has no idea which split either copy is in.
    victim_test = test_relpaths[0]
    twin_train = train_relpaths[0]
    (data_dir / twin_train).write_bytes((data_dir / victim_test).read_bytes())

    all_relpaths = {str(p.relative_to(data_dir)) for p in Path(data_dir).glob("*/side_a/*.jpg")}
    survivors = all_relpaths - {twin_train}  # clean_duplicates.py kept the test-side copy
    clean_list = data_dir / "clean_files.txt"
    clean_list.write_text("\n".join(sorted(survivors)) + "\n")

    new_filepaths, _, _, _, _, new_train_idx, new_val_idx, new_test_idx = frozen_split(
        data_dir, clean_list=clean_list
    )
    train_hashes = {file_hash(new_filepaths[i]) for i in new_train_idx}
    val_hashes = {file_hash(new_filepaths[i]) for i in new_val_idx}
    test_hashes = {file_hash(new_filepaths[i]) for i in new_test_idx}

    assert train_hashes.isdisjoint(test_hashes)
    assert val_hashes.isdisjoint(test_hashes)
