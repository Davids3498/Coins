"""Shared content-hash helper -- SHA-256 over raw file bytes.

Extracted out of clean_duplicates.py so that module (dedup over the raw tree) and
coin_clf.validate_batch (the retraining gate's duplicate/leakage checks) hash files identically
instead of validate_batch reimplementing clean_duplicates.py's hashing logic. Lives in src/ rather
than being imported from the root-level script so both directions stay dependency-clean: root
scripts may import from src/coin_clf, src/coin_clf never imports from root scripts.

hash_many is the parallel form. coin_clf.data.active_split's leakage guard and
verify_data_integrity.py both need to hash tens of thousands of images before anything trains,
and both must agree byte-for-byte with the single-file path above -- so they share it here rather
than each rolling their own pool.
"""
from __future__ import annotations

import hashlib
from multiprocessing import Pool
from pathlib import Path

DEFAULT_JOBS = 8


def fingerprint(hashes) -> str:
    """Order-independent fingerprint of a SET of content hashes.

    Two loaders are looking at the same images iff these match -- filenames, ordering and index
    space are deliberately not part of it. verify_data_integrity.py uses this to prove the
    holdout has one definition; the training notebook prints it so a run is visibly on the
    canonical set. One definition, so those two can never disagree about what agreement means.
    """
    h = hashlib.sha256()
    for x in sorted(set(hashes)):
        h.update(x.encode())
    return h.hexdigest()[:16]


def file_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_one(path_str: str) -> tuple[str, str]:
    return path_str, file_hash(path_str)


def hash_many(
    paths, jobs: int = DEFAULT_JOBS, progress_every: int = 0, label: str = "hashing"
) -> dict[str, str]:
    """Hash an iterable of paths in parallel. Returns {str(path): sha256}.

    Identical output to {str(p): file_hash(p) for p in paths}, just faster -- callers rely on
    that equivalence, so this must never grow a shortcut (size/mtime stat, cache file) that
    file_hash doesn't also take. The whole point of these hashes is that they came from bytes
    read this run.
    """
    paths = [str(p) for p in paths]
    if not paths:
        return {}
    if jobs <= 1 or len(paths) < 64:
        return dict(_hash_one(p) for p in paths)

    out: dict[str, str] = {}
    with Pool(jobs) as pool:
        for i, (path_str, h) in enumerate(pool.imap_unordered(_hash_one, paths, chunksize=64), 1):
            out[path_str] = h
            if progress_every and i % progress_every == 0:
                print(f"    {label}: {i}/{len(paths)}")
    return out
