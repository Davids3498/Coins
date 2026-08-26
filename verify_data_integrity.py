"""verify_data_integrity.py -- prove, from the bytes on disk, that the coin dataset partitions
are what the code thinks they are.

READ-ONLY. Modifies nothing: no file is written, moved, or appended; no cursor advances; no
MLflow call is made. Entry points with side effects (release_batch.release_next_batch) are
reconstructed read-only rather than invoked -- see B/release_batch.

TRUSTS NOTHING BUT THE BYTES. Every hash in this report is recomputed this run, in this process,
from the raw file contents via coin_clf.hashing. drop_log.csv is never read. clean_files.txt,
active_train.txt, splits_manifest.json and future_pool_cursor.json ARE read -- they are the claims
under test (which files each partition says it contains) -- but every statement about what those
files hold is verified against freshly computed content hashes, never against a previously logged
count or a prior assertion.

Three sections:

  A. PARTITION INTEGRITY -- sizes, pairwise content-hash disjointness, clean-list membership,
     quarantine accounting, and the full arithmetic reconciliation against the clean corpus.

  B. DATA-LOADING ENTRY POINTS -- every function in the repo that turns data_dir into a set of
     images for training / evaluation / the gate is actually CALLED here, and the set it returns
     is fingerprinted by content hash. Consumers that claim to return "the holdout" are checked
     to return the IDENTICAL set; consumers that feed training are checked to be hash-disjoint
     from it.

  C. UNREACHABILITY -- the raw-tree and rival-holdout code paths that caused the leakage are
     checked to be GONE, not merely unused: deleted symbols stay deleted, missing clean list or
     manifest raises instead of falling back, and no CLI still offers an escape hatch.

Usage:
    PYTHONPATH=src python3 verify_data_integrity.py
    PYTHONPATH=src python3 verify_data_integrity.py --jobs 8
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from coin_clf.hashing import fingerprint, hash_many  # noqa: E402

# The batch size the DAG is pinned to; the cursor counts batch NUMBERS, so released-count
# arithmetic is only meaningful against the same size that produced the cursor.
DAG_BATCH_SIZE = 200
CLEAN_CORPUS_EXPECTED = 57792  # what the reconciliation in A4 must land on


# --------------------------------------------------------------------------------------------
# reporting scaffolding
# --------------------------------------------------------------------------------------------
class Report:
    """Collects PASS/FAIL rows so a failure never short-circuits the rest of the picture."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = "", fail_detail: str = "") -> bool:
        """detail is printed either way (it carries the measured number). fail_detail explains
        what a failure would MEAN and is printed only when the check actually fails -- printing
        it on a pass makes a green report read like a red one.
        """
        shown = detail if detail else (fail_detail if not passed else "")
        if detail and fail_detail and not passed:
            shown = f"{detail} -- {fail_detail}"
        self.rows.append((name, passed, shown))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" -- {shown}" if shown else ""))
        return passed

    def note(self, text: str) -> None:
        print(f"  [note] {text}")

    @property
    def failures(self) -> list[tuple[str, bool, str]]:
        return [r for r in self.rows if not r[1]]


def section(title: str) -> None:
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def sub(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, 88 - len(title)))


def raises(fn, *args, **kwargs) -> Exception | None:
    try:
        fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        return e
    return None


# --------------------------------------------------------------------------------------------
# fresh hashing of the real tree
# --------------------------------------------------------------------------------------------
def hash_tree(data_dir: Path, jobs: int) -> dict[str, str]:
    """Recompute a SHA-256 for every image the dataset glob can see, this run, from the bytes.

    Keyed by path relative to data_dir -- the same key space clean_files.txt, active_train.txt
    and the splits manifest all use, so every later set operation is comparing like with like.
    """
    filepaths = sorted(data_dir.glob("*/side_a/*.jpg"))
    print(f"  globbing {data_dir}/*/side_a/*.jpg -> {len(filepaths)} files")
    print(f"  hashing all {len(filepaths)} files fresh with {jobs} process(es) "
          f"(nothing is read from drop_log.csv or any cache)...")
    t0 = time.time()
    by_abs = hash_many(filepaths, jobs=jobs, progress_every=10000, label="tree")
    out = {str(Path(p).relative_to(data_dir)): h for p, h in by_abs.items()}
    print(f"  hashed {len(out)} files in {time.time() - t0:.1f}s")
    return out


def hashes_of(relpaths, tree: dict[str, str]) -> set[str]:
    return {tree[r] for r in relpaths if r in tree}


def describe(name: str, relpaths, tree: dict[str, str]) -> dict:
    rel = list(relpaths)
    missing = [r for r in rel if r not in tree]
    hs = hashes_of(rel, tree)
    return {
        "name": name,
        "relpaths": rel,
        "n_files": len(rel),
        "n_unique_paths": len(set(rel)),
        "hashes": hs,
        "n_unique_hashes": len(hs),
        "n_missing_from_tree": len(missing),
        "fingerprint": fingerprint(hs),
    }


def print_pool(d: dict) -> None:
    dup_note = ""
    if d["n_unique_hashes"] != d["n_unique_paths"]:
        dup_note = f"  <-- {d['n_unique_paths'] - d['n_unique_hashes']} internal byte-duplicate(s)"
    miss = f"  MISSING_FROM_TREE={d['n_missing_from_tree']}" if d["n_missing_from_tree"] else ""
    print(f"    {d['name']:<52} files={d['n_files']:>6}  unique_hashes={d['n_unique_hashes']:>6}  "
          f"fp={d['fingerprint']}{dup_note}{miss}")


def pair_collisions(a: dict, b: dict, tree: dict[str, str]) -> tuple[int, list[tuple[str, str]]]:
    """Distinct content hashes present in BOTH pools, plus example (path_a, path_b) pairs."""
    shared = a["hashes"] & b["hashes"]
    if not shared:
        return 0, []
    by_hash_a: dict[str, list[str]] = defaultdict(list)
    for r in a["relpaths"]:
        if r in tree and tree[r] in shared:
            by_hash_a[tree[r]].append(r)
    by_hash_b: dict[str, list[str]] = defaultdict(list)
    for r in b["relpaths"]:
        if r in tree and tree[r] in shared:
            by_hash_b[tree[r]].append(r)
    examples = [(by_hash_a[h][0], by_hash_b[h][0]) for h in sorted(shared)[:5]]
    return len(shared), examples


# --------------------------------------------------------------------------------------------
# SECTION A -- partition integrity
# --------------------------------------------------------------------------------------------
def section_a(cfg, tree: dict[str, str], rep: Report) -> dict:
    section("SECTION A -- PARTITION INTEGRITY (by content hash; filenames are never the key)")

    clean_relpaths = [l for l in cfg.clean_list.read_text().splitlines() if l.strip()]
    clean_set = set(clean_relpaths)
    active_relpaths = [l for l in cfg.active_train_list.read_text().splitlines() if l.strip()]
    manifest = json.loads(cfg.manifest.read_text())
    m_train = manifest["splits"]["train"]
    m_holdout = manifest["splits"]["holdout"]
    m_future = manifest["splits"]["future_pool"]

    cursor = 0
    if cfg.cursor_file.exists():
        cursor = json.loads(cfg.cursor_file.read_text())["next_batch"]
    released = m_future[: cursor * DAG_BATCH_SIZE]
    unreleased = m_future[cursor * DAG_BATCH_SIZE :]

    sub("A1. exact size of each live partition")
    print(f"    cursor file says {cursor} batch(es) of {DAG_BATCH_SIZE} already released "
          f"({len(released)} images)")
    print()
    train = describe("TRAIN      (data/active_train.txt)", active_relpaths, tree)
    holdout = describe("HOLDOUT    (manifest splits.holdout)", m_holdout, tree)
    future_un = describe("FUTURE-POOL unreleased (manifest tail)", unreleased, tree)
    released_d = describe("future-pool released (already in train)", released, tree)
    print_pool(train)
    print_pool(holdout)
    print_pool(future_un)
    print_pool(released_d)

    sub("A2. pairwise content-hash disjointness (every count below must be 0)")
    pools = [train, holdout, future_un]
    for i in range(len(pools)):
        for j in range(i + 1, len(pools)):
            a, b = pools[i], pools[j]
            n, examples = pair_collisions(a, b, tree)
            label = f"{a['name'].split()[0]} x {b['name'].split()[0]}"
            rep.check(f"A2 hash-disjoint: {label}", n == 0, f"{n} colliding content hash(es)")
            for pa, pb in examples:
                print(f"        e.g. {pa}\n          == {pb}")

    sub("A3. clean_files.txt membership (every file in every live partition must be a survivor)")
    print(f"    clean_files.txt: {len(clean_relpaths)} lines, {len(clean_set)} unique")
    for pool in pools:
        outside = [r for r in pool["relpaths"] if r not in clean_set]
        rep.check(f"A3 all-in-clean-list: {pool['name'].split()[0]}", not outside,
                  f"{len(outside)} file(s) not on the clean list"
                  + (f" (e.g. {outside[0]})" if outside else ""))

    sub("A4. arithmetic reconciliation + quarantine accounting")
    raw_n = len(tree)
    raw_unique_h = len(set(tree.values()))
    clean_h = hashes_of(clean_set, tree)
    live_total = train["n_files"] + holdout["n_files"] + future_un["n_files"]
    quarantined = sorted(cfg.quarantine_dir.glob("*/side_a/*.jpg")) if cfg.quarantine_dir.exists() else []
    print(f"    training tree on disk                             {raw_n:>7}  "
          f"({raw_unique_h} unique hashes -> {raw_n - raw_unique_h} byte-duplicate files present)")
    print(f"    clean_files.txt survivors                         {len(clean_set):>7}  "
          f"({len(clean_h)} unique hashes)")
    print(f"    quarantined out of the tree ({cfg.quarantine_dir.name})       "
          f"{len(quarantined):>7}")
    print()
    print(f"    manifest train                                    {len(m_train):>7}")
    print(f"    manifest holdout                                  {len(m_holdout):>7}")
    print(f"    manifest future_pool                              {len(m_future):>7}")
    print(f"    manifest total                                    "
          f"{len(m_train) + len(m_holdout) + len(m_future):>7}")
    print()
    print(f"    TRAIN  active_train.txt                           {train['n_files']:>7}"
          f"   (= manifest train {len(m_train)} + {len(released)} released)")
    print(f"    HOLDOUT manifest                                  {holdout['n_files']:>7}")
    print(f"    FUTURE-POOL unreleased                            {future_un['n_files']:>7}"
          f"   (= manifest future {len(m_future)} - {len(released)} released)")
    print(f"    {'-' * 66}")
    print(f"    live total                                        {live_total:>7}")
    print(f"    target (clean corpus)                             {CLEAN_CORPUS_EXPECTED:>7}")
    print(f"    gap                                               "
          f"{live_total - CLEAN_CORPUS_EXPECTED:>7}")

    rep.check("A4 manifest sums to the clean corpus",
              len(m_train) + len(m_holdout) + len(m_future) == len(clean_set),
              f"{len(m_train) + len(m_holdout) + len(m_future)} vs {len(clean_set)} survivors")
    rep.check("A4 train+holdout+unreleased == clean corpus",
              live_total == CLEAN_CORPUS_EXPECTED,
              f"{live_total} vs {CLEAN_CORPUS_EXPECTED}")
    rep.check("A4 active_train.txt == manifest train + released batches",
              set(active_relpaths) == set(m_train) | set(released),
              f"{len(set(active_relpaths) ^ (set(m_train) | set(released)))} symmetric difference")
    rep.check("A4 clean corpus is internally hash-unique",
              len(clean_h) == len(clean_set),
              f"{len(clean_set) - len(clean_h)} survivor(s) share bytes with another survivor")

    # F1: the tree itself is now the clean corpus -- no condemned file is still reachable by a glob.
    rep.check("A4 training tree on disk == clean corpus exactly (F1 quarantine)",
              set(tree) == clean_set,
              f"{len(set(tree) ^ clean_set)} file(s) differ between the tree and the clean list")
    rep.check("A4 training tree is internally hash-unique (no byte-duplicates left in place)",
              raw_n == raw_unique_h,
              f"{raw_n - raw_unique_h} duplicate file(s) still in the tree")
    rep.check("A4 quarantine holds the condemned files, reversibly",
              len(quarantined) > 0 and len(quarantined) + len(clean_set) == 62134,
              f"{len(quarantined)} quarantined + {len(clean_set)} clean = "
              f"{len(quarantined) + len(clean_set)} (expected 62134 originals)")

    return {
        "clean_set": clean_set,
        "manifest": manifest,
        "train": train,
        "holdout": holdout,
        "future_un": future_un,
        "released": released,
        "cursor": cursor,
        "active_relpaths": active_relpaths,
    }


# --------------------------------------------------------------------------------------------
# SECTION B -- every data-loading entry point
# --------------------------------------------------------------------------------------------
def _ds_relpaths(ds, data_dir: Path) -> list[str]:
    """Relative paths a CoinImageDataset will actually open."""
    return [str(Path(ds.filepaths[i]).relative_to(data_dir)) for i in ds.indices]


def _idx_relpaths(filepaths, idx, data_dir: Path) -> list[str]:
    return [str(Path(filepaths[i]).relative_to(data_dir)) for i in idx]


def section_b(cfg, tree: dict[str, str], a: dict, rep: Report, jobs: int) -> None:
    section("SECTION B -- EVERY DATA-LOADING ENTRY POINT, CALLED FOR REAL")

    from coin_clf.data import active_split, build_manifest_holdout, discover_dataset
    import evaluate as evaluate_mod
    from splits import DatasetSplits, carve

    dd = cfg.data_dir
    manifest_p = str(cfg.manifest)
    active_p = str(cfg.active_train_list)

    results: list[dict] = []

    def record(name, role, consumer, relpaths):
        """role: universe | train | val | holdout | future -- what the caller treats it AS."""
        d = describe(name, relpaths, tree)
        d.update(role=role, consumer=consumer)
        results.append(d)
        return d

    sub("B1. calling each entry point (size + content-hash fingerprint of what it returns)")

    fps_clean, labs_clean, le_clean, _, nc_clean = discover_dataset(dd)
    record("discover_dataset(data_dir)  [clean by default]", "universe",
           "build_manifest_holdout, active_split, DatasetSplits.load, carve",
           [str(p.relative_to(dd)) for p in fps_clean])
    fps_raw, _, le_raw, _, nc_raw = discover_dataset(dd, allow_raw_tree=True)
    record("discover_dataset(data_dir, allow_raw_tree=True)", "universe",
           "explicit opt-in only (no production caller)",
           [str(p.relative_to(dd)) for p in fps_raw])

    ds = build_manifest_holdout(dd, manifest_p)
    record("build_manifest_holdout(data_dir, manifest)", "holdout",
           "THE holdout -- evaluate, promote, train, export", _ds_relpaths(ds, dd))

    for name, mf, consumer in [
        ("load_holdout(data_dir, manifest)", manifest_p, "evaluate.py / promote.py --manifest"),
        ("load_holdout(data_dir, None)", None, "bare programmatic evaluate_version(...) call"),
    ]:
        record(name, "holdout", consumer, _ds_relpaths(evaluate_mod.load_holdout(str(dd), mf), dd))

    fps, _, _, _, _, tri, vai, tei = active_split(
        dd, active_train_list=active_p, manifest_path=manifest_p, random_state=42, jobs=jobs
    )
    record("active_split(active_train.txt).train_idx", "train", "train.py, DAG train",
           _idx_relpaths(fps, tri, dd))
    record("active_split(active_train.txt).val_idx", "val", "train.py, DAG train",
           _idx_relpaths(fps, vai, dd))
    record("active_split(active_train.txt).test_idx", "holdout", "train.py's own test_acc",
           _idx_relpaths(fps, tei, dd))

    fps2, _, _, _, _, tri2, vai2, tei2 = active_split(
        dd, manifest_path=manifest_p, from_manifest_train=True, random_state=42, jobs=jobs
    )
    record("active_split(from_manifest_train).train_idx", "train", "train.py --train-from-manifest",
           _idx_relpaths(fps2, tri2, dd))
    record("active_split(from_manifest_train).val_idx", "val", "train.py --train-from-manifest",
           _idx_relpaths(fps2, vai2, dd))
    record("active_split(from_manifest_train).test_idx", "holdout", "train.py --train-from-manifest",
           _idx_relpaths(fps2, tei2, dd))

    sp = DatasetSplits.load(manifest_p, data_dir=dd)
    record("DatasetSplits.load(manifest).train_idx", "train", "release_batch bootstrap seed",
           _idx_relpaths(sp.filepaths, sp.train_idx, dd))
    record("DatasetSplits.load(manifest).holdout_idx", "holdout", "check_split_leakage --manifest",
           _idx_relpaths(sp.filepaths, sp.holdout_idx, dd))
    record("DatasetSplits.load(manifest).future_idx", "future", "release_batch FuturePool",
           _idx_relpaths(sp.filepaths, sp.future_idx, dd))

    recarve = carve(dd)
    record("carve(data_dir).train_idx", "train", "splits.py CLI (re-carve)",
           _idx_relpaths(recarve.filepaths, recarve.train_idx, dd))
    record("carve(data_dir).holdout_idx", "holdout", "splits.py CLI (re-carve)",
           _idx_relpaths(recarve.filepaths, recarve.holdout_idx, dd))
    record("carve(data_dir).future_idx", "future", "splits.py CLI (re-carve)",
           _idx_relpaths(recarve.filepaths, recarve.future_idx, dd))

    # -- release_batch.py: RECONSTRUCTED, never called (it appends + advances the cursor) ------
    pool = sp.future_pool(DAG_BATCH_SIZE)
    cursor = a["cursor"]
    rel_idx = pool.ingested_through(cursor - 1) if cursor > 0 else []
    record(f"release_batch: batches 0..{cursor - 1} already released", "train",
           "release_batch.release_next_batch (reconstructed read-only)",
           _idx_relpaths(sp.filepaths, list(rel_idx), dd))
    if cursor < len(pool):
        record(f"release_batch: NEXT batch ({cursor}) the DAG would release", "future",
               "release_batch.release_next_batch (reconstructed read-only)",
               [str(p.relative_to(dd)) for p in pool[cursor].filepaths])

    ref = (set(Path(active_p).read_text().splitlines())
           | set(a["manifest"]["splits"]["holdout"]))
    ref.discard("")
    record("validate_batch reference set (active_train + holdout)", "universe",
           "coin_clf.validate_batch --active-train-list/--manifest", sorted(ref))

    # export.py now resolves through build_manifest_holdout -- same call, so same set by
    # construction; recorded separately so the report still shows it agreeing.
    record("export.evaluate_int8_on_test_set loader", "holdout", "export.py --data-dir",
           _ds_relpaths(build_manifest_holdout(dd, manifest_p), dd))

    for d in results:
        print_pool(d)

    sub("B1b. label space")
    rep.check("B1b discover_dataset label space is 51 classes after GORDIAN merge",
              nc_clean == 51, f"clean={nc_clean}")
    rep.check("B1b clean and raw-tree discovery now agree (tree == clean corpus)",
              le_raw == le_clean and len(fps_raw) == len(fps_clean),
              f"raw={len(fps_raw)} clean={len(fps_clean)}")

    sub("B1c. data_dir aliasing (train.py's default path vs the DAG's)")
    alias = REPO_ROOT / "FOR_TRAINNING"
    if alias.exists():
        fps_alias, *_ = discover_dataset(alias)
        rep.check("B1c FOR_TRAINNING symlink and data/FOR_TRAINNING discover the same files",
                  {str(p.relative_to(alias)) for p in fps_alias}
                  == {str(p.relative_to(dd)) for p in fps_clean},
                  "relpath sets differ")
    else:
        rep.note("no FOR_TRAINNING symlink at repo root")

    sub("B2. do all 'holdout' consumers return the IDENTICAL set?")
    holdouts = [d for d in results if d["role"] == "holdout"]
    canonical = next(d for d in holdouts if d["name"].startswith("build_manifest_holdout"))
    print(f"    canonical = {canonical['name']}  ({canonical['n_files']} files, "
          f"fp={canonical['fingerprint']})")
    print()
    groups: dict[str, list[dict]] = defaultdict(list)
    for d in holdouts:
        groups[d["fingerprint"]].append(d)
    for fp, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        marker = "AGREES with canonical" if fp == canonical["fingerprint"] else "DISAGREES"
        print(f"    fp={fp}  n={members[0]['n_files']:>6}  {marker}")
        for d in members:
            print(f"        - {d['name']}")
            print(f"          consumer: {d['consumer']}")
    rep.check("B2 every 'the holdout' consumer returns one identical set",
              len(groups) == 1,
              f"{len(groups)} distinct holdout definition(s) in live code")

    sub("B3. does any training path overlap the holdout by content hash?")
    for d in [d for d in results if d["role"] in ("train", "val")]:
        n, examples = pair_collisions(d, canonical, tree)
        rep.check(f"B3 hash-disjoint from holdout: {d['name']}", n == 0,
                  f"{n} colliding content hash(es)")
        for pa, pb in examples[:3]:
            print(f"        train: {pa}\n        hold : {pb}")

    sub("B4. does re-running splits.carve still reproduce the pinned manifest?")
    m = a["manifest"]
    for key, attr in (("train", "train_idx"), ("holdout", "holdout_idx"), ("future_pool", "future_idx")):
        got = set(_idx_relpaths(recarve.filepaths, getattr(recarve, attr), dd))
        rep.check(f"B4 carve() reproduces manifest.{key}", got == set(m["splits"][key]),
                  f"{len(got ^ set(m['splits'][key]))} file(s) differ")

    sub("B5. DAG task -> entry point mapping (dags/retrain_coin_clf.py)")
    print("    release_batch  -> release_batch.py    => DatasetSplits.load(manifest).future_idx")
    print("    validate       -> coin_clf.validate_batch => reference set (active_train+holdout)")
    print("    train          -> train.py            => active_split(...)")
    print("    evaluate       -> evaluate.py         => build_manifest_holdout [canonical]")
    print("    promote        -> promote.py          => build_manifest_holdout [canonical]")


# --------------------------------------------------------------------------------------------
# SECTION C -- the dangerous paths are gone, not just unused
# --------------------------------------------------------------------------------------------
def section_c(cfg, rep: Report) -> None:
    section("SECTION C -- UNREACHABILITY OF THE RAW TREE AND THE RIVAL HOLDOUTS")

    import coin_clf.data as data_mod
    import evaluate as evaluate_mod
    from coin_clf.data import RawTreeError, resolve_clean_list, resolve_manifest

    missing = cfg.data_dir / "definitely-not-here.txt"

    sub("C1. deleted symbols stay deleted (F2 -- the 12,427 and 11,382 rivals)")
    for sym in ("frozen_split", "build_test_dataset", "CoinDistilDataset"):
        rep.check(f"C1 coin_clf.data.{sym} no longer exists", not hasattr(data_mod, sym),
                  fail_detail="still importable -- a rival holdout definition is back")

    sub("C2. missing clean list / manifest RAISES instead of falling back (F7)")
    for label, fn, fnargs in (
        ("resolve_clean_list(missing)", resolve_clean_list, (missing,)),
        ("resolve_manifest(missing)", resolve_manifest, (missing,)),
        ("discover_dataset(missing clean list)", data_mod.discover_dataset, (cfg.data_dir, missing)),
        ("build_manifest_holdout(missing manifest)", data_mod.build_manifest_holdout,
         (cfg.data_dir, missing)),
        ("active_split(missing active list)", data_mod.active_split,
         (cfg.data_dir, missing, cfg.manifest)),
    ):
        e = raises(fn, *fnargs)
        rep.check(f"C2 {label} raises RawTreeError", isinstance(e, RawTreeError),
                  f"got {type(e).__name__ if e else 'NO EXCEPTION -- it fell back'}")

    sub("C3. no scoring API can still select a different holdout (F5)")
    for fn, name in ((evaluate_mod.load_holdout, "load_holdout"),
                     (evaluate_mod.evaluate_version, "evaluate_version")):
        params = inspect.signature(fn).parameters
        rep.check(f"C3 evaluate.{name} has no clean_list parameter", "clean_list" not in params,
                  fail_detail="clean_list can still redirect the holdout")

    sub("C4. no CLI still offers a raw-tree escape hatch")
    cli_expectations = [
        ("evaluate.py", ["--clean-list"], []),
        ("promote.py", ["--clean-list"], []),
        ("train.py", ["--clean-list", "--distill-temp", "--distill-alpha", "--teacher-test-acc"], []),
        ("splits.py", [], ["--allow-raw-tree"]),
    ]
    for fname, forbidden, required in cli_expectations:
        text = (REPO_ROOT / fname).read_text()
        for flag in forbidden:
            rep.check(f"C4 {fname} no longer accepts {flag}", f'"{flag}"' not in text,
                      fail_detail="flag still wired up")
        for flag in required:
            rep.check(f"C4 {fname} exposes {flag} as an explicit opt-in", f'"{flag}"' in text,
                      fail_detail="explicit raw-tree flag missing")

    sub("C5. the discarded teacher is unreachable from the training path")
    # AST, not a substring scan: train.py's docstring legitimately EXPLAINS that distillation was
    # removed, and a grep for "teacher" cannot tell prose from a live import. What matters is
    # whether any name the interpreter would actually resolve still reaches the teacher.
    import ast

    tree_ast = ast.parse((REPO_ROOT / "train.py").read_text())
    imported: set[str] = set()
    identifiers: set[str] = set()
    for node in ast.walk(tree_ast):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)

    rep.check("C5 train.py imports nothing from coin_clf.teacher",
              not any("teacher" in m.lower() for m in imported),
              f"{sorted(m for m in imported if 'teacher' in m.lower())}",
              fail_detail="the compromised teacher is still imported")
    tainted = sorted(
        i for i in identifiers
        if any(t in i.lower() for t in ("teacher", "distill", "soft_label", "compression_gap"))
    )
    rep.check("C5 train.py resolves no teacher/distillation identifier at runtime",
              not tainted, f"{len(tainted)} tainted identifier(s)",
              fail_detail=f"{tainted}")
    rep.check("C5 coin_clf.teacher is imported by no live pipeline module",
              not any("teacher" in m.lower()
                      for f in ("evaluate.py", "promote.py", "export.py", "splits.py",
                                "release_batch.py", "src/coin_clf/data.py")
                      for m in _imports_of(REPO_ROOT / f)),
              fail_detail="a pipeline module still imports the teacher")


def _imports_of(path: Path) -> set[str]:
    import ast

    mods: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
            mods.update(a.name for a in node.names)
    return mods


# --------------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "FOR_TRAINNING"))
    p.add_argument("--clean-list", default=str(REPO_ROOT / "data" / "clean_files.txt"))
    p.add_argument("--manifest", default=str(REPO_ROOT / "data" / "splits_manifest.json"))
    p.add_argument("--active-train-list", default=str(REPO_ROOT / "data" / "active_train.txt"))
    p.add_argument("--cursor-file", default=str(REPO_ROOT / "data" / "future_pool_cursor.json"))
    p.add_argument("--quarantine-dir", default=str(REPO_ROOT / "data" / "quarantine"))
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()

    class Cfg:
        data_dir = Path(args.data_dir)
        clean_list = Path(args.clean_list)
        manifest = Path(args.manifest)
        active_train_list = Path(args.active_train_list)
        cursor_file = Path(args.cursor_file)
        quarantine_dir = Path(args.quarantine_dir)

    cfg = Cfg()
    rep = Report()

    section("SECTION 0 -- INPUTS (claims under test) + FRESH HASHING OF THE REAL TREE")
    for label, path in (("data_dir", cfg.data_dir), ("clean_list", cfg.clean_list),
                        ("manifest", cfg.manifest), ("active_train_list", cfg.active_train_list),
                        ("cursor_file", cfg.cursor_file), ("quarantine_dir", cfg.quarantine_dir)):
        print(f"  {label:<20} {path}  {'OK' if path.exists() else 'MISSING'}")
    print()
    tree = hash_tree(cfg.data_dir, args.jobs)

    a = section_a(cfg, tree, rep)
    section_b(cfg, tree, a, rep, args.jobs)
    section_c(cfg, rep)

    section("SUMMARY")
    passed = sum(1 for _, ok, _ in rep.rows if ok)
    print(f"  {passed}/{len(rep.rows)} checks passed")
    if rep.failures:
        print()
        print(f"  {len(rep.failures)} FAILURE(S):")
        for name, _, detail in rep.failures:
            print(f"    - {name}" + (f"  --  {detail}" if detail else ""))
        sys.exit(1)
    print("  ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
