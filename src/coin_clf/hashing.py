"""Shared content-hash helper -- SHA-256 over raw file bytes.

Extracted out of clean_duplicates.py so that module (dedup over the raw tree) and
coin_clf.validate_batch (the retraining gate's duplicate/leakage checks) hash files identically
instead of validate_batch reimplementing clean_duplicates.py's hashing logic. Lives in src/ rather
than being imported from the root-level script so both directions stay dependency-clean: root
scripts may import from src/coin_clf, src/coin_clf never imports from root scripts.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
