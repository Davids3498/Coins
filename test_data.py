"""Tests for coin_clf.data.build_test_dataset's clean_list handling.

The point being tested: clean_list must filter the ALREADY-COMPUTED test_idx, not shrink the
universe before splitting. Recomputing the split on a smaller universe would move the train/test
boundary (sklearn's stratified split consumes one shared RNG stream class by class, so dropping
even one image from an early class reshuffles every class split after it) -- which would risk
scoring an already-trained model on images that were on the TRAIN side of the boundary it was
actually trained under. Filtering test_idx after the fact can only shrink the holdout, never pull
an image across the line.
"""
from pathlib import Path

import pytest

from coin_clf.data import build_test_dataset, discover_dataset, split_dataset

PER_CLASS = 40
FAKE_CLASS_NAMES = ["GORDIAN I", "GORDIAN II", "EMPEROR2", "EMPEROR3", "EMPEROR4"]


def make_fake_dataset(tmp_path, class_names=FAKE_CLASS_NAMES, per_class=PER_CLASS):
    for c, name in enumerate(class_names):
        side_a = tmp_path / f"{c:02d}_{name}" / "side_a"
        side_a.mkdir(parents=True)
        for i in range(per_class):
            (side_a / f"img_{i}.jpg").touch()
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
