"""Tests for splits.py -- the train / frozen-holdout / future-pool carve and its batch iterator.

No real coin images needed: discover_dataset only globs `*/side_a/*.jpg` and parses folder
names, it never opens the files, so a fake dataset is just empty files in the right tree shape.
"""
import numpy as np
import pytest
import torch

from splits import Batch, DatasetSplits, FuturePool, carve

PER_CLASS = 40  # -> holdout 8/class, future 8/class, train 24/class at the defaults
# discover_dataset() unconditionally applies GORDIAN_MERGES (GORDIAN II -> GORDIAN I), so any
# fixture tree it reads must contain both names or it KeyErrors -- include them here rather than
# touching data.py's real-dataset assumption.
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


# --- carve() ----------------------------------------------------------------

def test_carve_is_disjoint_and_covers_everything(data_dir):
    s = carve(data_dir, allow_raw_tree=True)
    train, hold, fut = set(s.train_idx.tolist()), set(s.holdout_idx.tolist()), set(s.future_idx.tolist())
    assert not (train & hold)
    assert not (train & fut)
    assert not (hold & fut)
    assert train | hold | fut == set(range(len(s.filepaths)))


def test_carve_sizes_match_requested_fractions(data_dir):
    s = carve(data_dir, allow_raw_tree=True, holdout_size=0.2, future_size=0.2)
    total = len(s.filepaths)
    tol = s.num_classes
    assert s.sizes["holdout"] == pytest.approx(0.2 * total, abs=tol)
    assert s.sizes["future_pool"] == pytest.approx(0.2 * total, abs=tol)
    assert s.sizes["train"] == pytest.approx(0.6 * total, abs=tol)


def test_carve_is_deterministic(data_dir):
    a = carve(data_dir, allow_raw_tree=True, seed=42)
    b = carve(data_dir, allow_raw_tree=True, seed=42)
    assert np.array_equal(a.train_idx, b.train_idx)
    assert np.array_equal(a.holdout_idx, b.holdout_idx)
    assert np.array_equal(a.future_idx, b.future_idx)


def test_different_seeds_give_different_future_pools(data_dir):
    a = carve(data_dir, allow_raw_tree=True, seed=42)
    b = carve(data_dir, allow_raw_tree=True, seed=7)
    assert not np.array_equal(a.future_idx, b.future_idx)


def test_carve_holdout_is_split_dataset_test_idx_not_a_second_definition(data_dir):
    # The whole point: carve must not invent a holdout of its own. Its stage 1 IS
    # coin_clf.data.split_dataset over the same universe, and the manifest it writes is the
    # single definition every scorer resolves to via build_manifest_holdout.
    from coin_clf.data import discover_dataset, split_dataset

    _, all_labs, _, _, _ = discover_dataset(data_dir, allow_raw_tree=True)
    _, _, expected_test_idx = split_dataset(all_labs, test_size=0.2, random_state=42)

    s = carve(data_dir, allow_raw_tree=True, holdout_size=0.2, seed=42)
    assert np.array_equal(s.holdout_idx, np.sort(expected_test_idx))


def test_carve_clean_list_drops_files_before_splitting(data_dir):
    from coin_clf.data import discover_dataset

    _, all_labs, *_ = discover_dataset(data_dir, allow_raw_tree=True)
    unfiltered_total = len(all_labs)

    filepaths = sorted(data_dir.glob("*/side_a/*.jpg"))
    dropped = {str(p.relative_to(data_dir)) for p in filepaths[:10]}
    clean_list = data_dir / "clean_files.txt"
    survivors = [str(p.relative_to(data_dir)) for p in filepaths if str(p.relative_to(data_dir)) not in dropped]
    clean_list.write_text("\n".join(survivors) + "\n")

    s = carve(data_dir, clean_list=clean_list)

    total_kept = len(s.train_idx) + len(s.holdout_idx) + len(s.future_idx)
    assert total_kept == unfiltered_total - len(dropped)
    kept_relpaths = {str(s.filepaths[i].relative_to(data_dir)) for i in range(len(s.filepaths))}
    assert dropped.isdisjoint(kept_relpaths)


def test_carve_with_allow_raw_tree_keeps_everything(data_dir):
    # allow_raw_tree=True is the ONLY way to reach unfiltered data; clean_list=None now means
    # 'use data/clean_files.txt', not 'no filtering'.
    s = carve(data_dir, allow_raw_tree=True)
    total = len(s.train_idx) + len(s.holdout_idx) + len(s.future_idx)
    assert total == len(list(data_dir.glob("*/side_a/*.jpg")))


@pytest.mark.parametrize("holdout_size,future_size", [(0.0, 0.2), (1.0, 0.2), (0.2, 0.0), (0.6, 0.6)])
def test_carve_rejects_impossible_fractions(data_dir, holdout_size, future_size):
    with pytest.raises(ValueError):
        carve(data_dir, allow_raw_tree=True, holdout_size=holdout_size, future_size=future_size)


def test_datasetsplits_rejects_overlapping_indices():
    with pytest.raises(ValueError):
        DatasetSplits(
            filepaths=[f"f{i}" for i in range(6)],
            all_labs=torch.zeros(6, dtype=torch.long),
            label_encoder={"A": 0},
            idx_to_label={0: "A"},
            train_idx=np.array([0, 1, 2]),
            holdout_idx=np.array([2, 3]),  # overlaps train
            future_idx=np.array([4, 5]),
        )


# --- save / load --------------------------------------------------------------

def test_save_load_roundtrip(data_dir, tmp_path):
    s = carve(data_dir, allow_raw_tree=True)
    manifest = tmp_path / "manifest.json"
    s.save(manifest, data_dir=data_dir)

    loaded = DatasetSplits.load(manifest, data_dir=data_dir, allow_raw_tree=True)
    assert set(loaded.train_idx.tolist()) == set(s.train_idx.tolist())
    assert set(loaded.holdout_idx.tolist()) == set(s.holdout_idx.tolist())
    # future-pool ARRIVAL ORDER must survive the round trip, not just set membership.
    assert loaded.future_idx.tolist() == s.future_idx.tolist()
    assert loaded.label_encoder == s.label_encoder


def test_save_refuses_to_overwrite(data_dir, tmp_path):
    s = carve(data_dir, allow_raw_tree=True)
    manifest = tmp_path / "manifest.json"
    s.save(manifest, data_dir=data_dir)
    with pytest.raises(FileExistsError):
        s.save(manifest, data_dir=data_dir)


def test_load_detects_label_drift(data_dir, tmp_path):
    s = carve(data_dir, allow_raw_tree=True)
    manifest = tmp_path / "manifest.json"
    s.save(manifest, data_dir=data_dir)

    # A new emperor folder shows up in data_dir after the carve -> label space has drifted.
    new_side_a = data_dir / "99_NEWEMPEROR" / "side_a"
    new_side_a.mkdir(parents=True)
    (new_side_a / "img_0.jpg").touch()

    with pytest.raises(ValueError, match="drifted"):
        DatasetSplits.load(manifest, data_dir=data_dir, allow_raw_tree=True)


def test_load_detects_missing_files(data_dir, tmp_path):
    s = carve(data_dir, allow_raw_tree=True)
    manifest = tmp_path / "manifest.json"
    s.save(manifest, data_dir=data_dir)

    # Delete a file that the manifest references.
    victim = s.filepaths[int(s.train_idx[0])]
    victim.unlink()

    with pytest.raises(FileNotFoundError):
        DatasetSplits.load(manifest, data_dir=data_dir, allow_raw_tree=True)


# --- FuturePool batch iterator ------------------------------------------------

def test_future_pool_batches_partition_future_idx_exactly(data_dir):
    s = carve(data_dir, allow_raw_tree=True)
    pool = s.future_pool(batch_size=7)  # deliberately does not divide evenly
    seen = []
    for batch in pool:
        seen.extend(batch.indices.tolist())
    assert seen == s.future_idx.tolist()  # exact order, no gaps, no duplicates


def test_future_pool_len_is_ceil_division(data_dir):
    s = carve(data_dir, allow_raw_tree=True)
    pool = s.future_pool(batch_size=7)
    assert len(pool) == -(-len(s.future_idx) // 7)


def test_future_pool_last_batch_is_the_remainder(data_dir):
    s = carve(data_dir, allow_raw_tree=True)
    pool = s.future_pool(batch_size=7)
    last = pool[len(pool) - 1]
    expected_last_size = len(s.future_idx) - 7 * (len(pool) - 1)
    assert len(last) == expected_last_size


def test_future_pool_indexing_matches_iteration(data_dir):
    s = carve(data_dir, allow_raw_tree=True)
    pool = s.future_pool(batch_size=5)
    via_iter = list(pool)
    via_index = [pool[i] for i in range(len(pool))]
    for a, b in zip(via_iter, via_index):
        assert a.indices.tolist() == b.indices.tolist()


def test_future_pool_negative_indexing(data_dir):
    s = carve(data_dir, allow_raw_tree=True)
    pool = s.future_pool(batch_size=5)
    assert pool[-1].indices.tolist() == pool[len(pool) - 1].indices.tolist()


def test_future_pool_out_of_range_raises(data_dir):
    s = carve(data_dir, allow_raw_tree=True)
    pool = s.future_pool(batch_size=5)
    with pytest.raises(IndexError):
        pool[len(pool)]


def test_future_pool_rejects_bad_batch_size(data_dir):
    s = carve(data_dir, allow_raw_tree=True)
    with pytest.raises(ValueError):
        s.future_pool(batch_size=0)


def test_batch_carries_matching_filepaths_and_labels(data_dir):
    s = carve(data_dir, allow_raw_tree=True)
    batch = s.future_pool(batch_size=5)[0]
    assert isinstance(batch, Batch)
    assert len(batch.filepaths) == len(batch.indices)
    assert torch.equal(batch.labels, s.all_labs[batch.indices])


def test_ingested_through_is_cumulative(data_dir):
    s = carve(data_dir, allow_raw_tree=True)
    pool = s.future_pool(batch_size=5)
    assert pool.ingested_through(0).tolist() == pool[0].indices.tolist()
    combined = pool[0].indices.tolist() + pool[1].indices.tolist()
    assert pool.ingested_through(1).tolist() == combined
