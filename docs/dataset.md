# Dataset and its boundaries

[← back to the README](../README.md)

How 62,134 raw images became 57,792 clean ones carved into three provably disjoint partitions, and the three rules the code enforces so they stay that way. Every number here is re-derived from file bytes by `verify_data_integrity.py`.

Every number below is re-derived from file bytes by `verify_data_integrity.py`;
the full output of the last run is tracked at `verify_data_integrity_run.txt`.

```
62,134  original images
 4,342  quarantined by clean_duplicates.py (byte-duplicates + cross-label conflicts)
57,792  clean corpus (data/clean_files.txt) — hash-unique, all decode end-to-end

carved once by splits.py, seed 42, into data/splits_manifest.json:
  34,674  train        (~60%)
  11,559  frozen holdout (~20%)  fingerprint c47d3f321a0e13c2 — THE holdout
  11,559  future pool  (~20%)    withheld; stands in for data arriving after launch
```

The future pool is released in batches of 5,000 (`BATCH_SIZE` in
`dags/retrain_coin_clf.py`); `data/future_pool_cursor.json` counts batch
*numbers*, so the size must stay fixed for as long as one cursor is in use.
Two batches are released, which is why `data/active_train.txt` currently holds
44,674 files (34,674 manifest train + 10,000 released) and 1,559 future-pool
images remain unreleased.

Three rules the code enforces rather than documents:

* **Clean by default.** Every entry point in `coin_clf.data` resolves to the
  clean list; a missing clean list or manifest raises `RawTreeError` instead of
  falling back to the raw tree. Reaching unfiltered data requires typing
  `allow_raw_tree=True` at the call site.
* **One holdout.** `build_manifest_holdout` is the only definition in live code.
  The two that used to compete with it (`frozen_split`, `build_test_dataset`) are
  deleted, and there is a test plus an integrity check asserting they stay deleted.
* **Disjointness by content hash, not filename.** `active_split` re-hashes both
  sides and refuses to return if any training image is byte-identical to a
  holdout image. The same physical coin uploaded twice under two filenames is a
  leak that index bookkeeping cannot see.
