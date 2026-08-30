"""Dataset discovery and the ONE training/holdout boundary.

CLEAN BY DEFAULT, RAW ONLY ON PURPOSE. Every entry point here resolves to the cleaned corpus
(data/clean_files.txt) unless a caller passes allow_raw_tree=True and says so out loud. If the
clean list or the splits manifest is missing, these functions RAISE (RawTreeError) -- they never
quietly fall back to the raw tree. That inversion is the point: for three rounds, leakage kept
reappearing because safety was opt-in at every call site and one caller always forgot.

ONE HOLDOUT. `build_manifest_holdout` (splits.py's carved manifest) is the only definition of
"the holdout" in this codebase. The two that used to compete with it -- frozen_split()'s
independent 80/20 draw and build_test_dataset() -- are gone, not deprecated: a model trained
under one boundary and scored under another is exactly the failure this module exists to prevent,
and leaving a second definition reachable is how it came back each time.

The training boundary is `active_split`: train/val carved out of the growing
data/active_train.txt (clean train UNION released future-pool batches), holdout pinned to the
manifest. Its disjointness guard is by CONTENT HASH, not filename -- see active_split.
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler

from coin_clf.hashing import DEFAULT_JOBS, hash_many
from coin_clf.transforms import val_transform

GORDIAN_MERGES = {"GORDIAN II": "GORDIAN I"}

# Repo-root-anchored, not cwd-anchored: Airflow, notebooks and ad-hoc scripts all run from
# different working directories, and a default that silently resolves to a non-existent relative
# path is how you end up back on the raw tree.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN_LIST = REPO_ROOT / "data" / "clean_files.txt"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "splits_manifest.json"


class RawTreeError(RuntimeError):
    """A caller would have silently fallen back to the raw, uncleaned image tree."""


def resolve_clean_list(clean_list=None, *, allow_raw_tree: bool = False) -> Path | None:
    """Resolve the survivor list, or raise rather than let a caller drift onto the raw tree.

    allow_raw_tree=True returns None (no filtering) -- the ONLY way to reach unfiltered data, and
    it has to be typed at the call site.
    """
    if allow_raw_tree:
        if clean_list is not None:
            raise ValueError("pass either clean_list or allow_raw_tree=True, not both")
        return None
    path = Path(clean_list) if clean_list is not None else DEFAULT_CLEAN_LIST
    if not path.exists():
        raise RawTreeError(
            f"clean list {path} not found. Refusing to fall back to the raw tree -- it still "
            f"contains byte-duplicates and cross-label conflicts. Run clean_duplicates.py to "
            f"regenerate it, or pass allow_raw_tree=True if you genuinely want raw data."
        )
    return path


def resolve_manifest(manifest_path=None) -> Path:
    """Resolve splits.py's carve manifest, or raise. There is no unmanifested holdout."""
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST
    if not path.exists():
        raise RawTreeError(
            f"splits manifest {path} not found. It is the only definition of the frozen holdout; "
            f"there is no fallback. Carve one with splits.py before training or evaluating."
        )
    return path


def folder_label(name: str) -> str:
    return re.sub(r"^\d+_", "", name)


def discover_dataset(data_dir: Path, clean_list=None, *, allow_raw_tree: bool = False):
    """Glob the labeled image tree, encode labels, and apply the GORDIAN II -> GORDIAN I merge.

    Filters to the clean survivor list BY DEFAULT (see resolve_clean_list). Raises if that list
    is missing. allow_raw_tree=True is the explicit, deliberate escape hatch.

    Also verifies the survivor list and the tree still agree: a file on the clean list that has
    vanished from disk means someone quarantined too much (or the tree moved), and every split
    derived from here would silently shrink. That must raise, not shrink.

    Returns (filepaths, all_labs, label_encoder, idx_to_label, num_classes).
    """
    image_dir = Path(data_dir)
    filepaths = sorted(image_dir.glob("*/side_a/*.jpg"))

    clean_path = resolve_clean_list(clean_list, allow_raw_tree=allow_raw_tree)
    if clean_path is not None:
        allowed = {l for l in clean_path.read_text().splitlines() if l.strip()}
        on_disk = {str(p.relative_to(image_dir)) for p in filepaths}
        vanished = allowed - on_disk
        if vanished:
            raise RawTreeError(
                f"{len(vanished)} file(s) on {clean_path} are missing from {image_dir} "
                f"(e.g. {sorted(vanished)[0]}). The clean list and the tree have diverged -- "
                f"every split derived from here would silently shrink."
            )
        filepaths = [p for p in filepaths if str(p.relative_to(image_dir)) in allowed]

    unique_labels = sorted({folder_label(p.parent.parent.name) for p in filepaths})
    label_encoder = {name: i for i, name in enumerate(unique_labels)}
    raw_labels = [folder_label(p.parent.parent.name) for p in filepaths]
    labels_int = [label_encoder[l] for l in raw_labels]

    kept_names = [n for n in unique_labels if n not in GORDIAN_MERGES]
    new_label_encoder = {name: i for i, name in enumerate(kept_names)}
    old_to_new = {label_encoder[name]: new_label_encoder[name] for name in kept_names}
    for src, tgt in GORDIAN_MERGES.items():
        if src in label_encoder:
            old_to_new[label_encoder[src]] = new_label_encoder[tgt]

    all_labs = torch.tensor([old_to_new[l] for l in labels_int], dtype=torch.long)
    label_encoder = new_label_encoder
    idx_to_label = {v: k for k, v in label_encoder.items()}
    num_classes = len(label_encoder)

    return filepaths, all_labs, label_encoder, idx_to_label, num_classes


def split_dataset(all_labs: torch.Tensor, test_size: float = 0.2, random_state: int = 42):
    """Stratified two-stage index split over a label tensor. Pure function of (labels, params).

    This is a PRIMITIVE, not a holdout definition: splits.carve() uses it (over the clean corpus)
    to carve the manifest, and that manifest is the holdout. Calling it yourself on some other
    universe produces some other partition, which is precisely what must not happen -- which is
    why discover_dataset no longer hands out a raw universe to call it on.
    """
    indices = np.arange(len(all_labs))
    labels_np = all_labs.numpy()
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, shuffle=True, stratify=labels_np, random_state=random_state)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=test_size, shuffle=True,
        stratify=labels_np[train_idx], random_state=random_state)
    return train_idx, val_idx, test_idx


class CoinImageDataset(Dataset):
    """Returns (image, label). One dataset class for train, val and holdout -- only the
    transform differs, so there is no way for training and scoring to disagree about decoding.
    """

    def __init__(self, filepaths, all_labs, indices, transform):
        self.filepaths = filepaths
        self.all_labs = all_labs
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = Image.open(self.filepaths[idx]).convert("RGB")
        return self.transform(img), self.all_labs[idx]


def _load_manifest(manifest_path, label_encoder: dict) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["label_encoder"] != label_encoder:
        raise ValueError(
            f"label_encoder in {manifest_path} no longer matches discover_dataset(data_dir) -- "
            "the label space has drifted since splits.py carved this manifest. Re-carve before "
            "trusting it."
        )
    return manifest


def _resolve_relpaths(filepaths, image_dir: Path, relpaths, what: str) -> np.ndarray:
    """Indices of `relpaths` within `filepaths`, raising if any are unaccounted for."""
    wanted = set(relpaths)
    idx = np.array(
        [i for i, p in enumerate(filepaths) if str(Path(p).relative_to(image_dir)) in wanted],
        dtype=int,
    )
    if len(idx) != len(wanted):
        found = {str(Path(filepaths[i]).relative_to(image_dir)) for i in idx}
        missing = sorted(wanted - found)
        raise RawTreeError(
            f"{len(missing)} file(s) listed in {what} were not found in the discovered dataset "
            f"(e.g. {missing[0]}). Refusing to proceed on a silently truncated {what}."
        )
    return idx


def build_manifest_holdout(data_dir: Path, manifest_path=None, *, clean_list=None) -> Dataset:
    """THE frozen holdout. The single definition in this codebase; everything that scores a model
    -- evaluate.py, promote.py, train.py's own test metric, export.py -- resolves to this set.

    Sourced from splits.py's carve() manifest, which carved train/holdout/future-pool together
    out of one clean universe so the three are disjoint by construction, and stays disjoint from
    data/active_train.txt as the retraining loop grows it. Raises if the manifest is absent or if
    any holdout file has gone missing -- a holdout that silently shrinks is a holdout that
    silently flatters whatever is being scored on it.
    """
    manifest_path = resolve_manifest(manifest_path)
    filepaths, all_labs, label_encoder, _, _ = discover_dataset(data_dir, clean_list=clean_list)
    manifest = _load_manifest(manifest_path, label_encoder)
    image_dir = Path(data_dir)
    test_idx = _resolve_relpaths(
        filepaths, image_dir, manifest["splits"]["holdout"], f"{manifest_path} holdout"
    )
    return CoinImageDataset(filepaths, all_labs, test_idx, val_transform)


def active_split(
    data_dir: Path,
    active_train_list=None,
    manifest_path=None,
    val_size: float = 0.2,
    random_state: int = 42,
    *,
    from_manifest_train: bool = False,
    clean_list=None,
    jobs: int = DEFAULT_JOBS,
):
    """The ONE training boundary: train/val carved from the growing active training list, holdout
    pinned to the manifest (build_manifest_holdout's set, identically).

    active_train_list defaults to data/active_train.txt. Pass from_manifest_train=True to train on
    the manifest's own train split instead -- the clean starting state, before any future-pool
    batch has been released. There is no third option and no raw-tree path.

    LEAKAGE GUARD IS BY CONTENT HASH, not by filename. The previous filename-set check could not
    see the case it existed to catch: the same coin photographed or re-uploaded under a different
    name lands in both lists with different relpaths and identical bytes, and the model gets
    tested on an image it trained on. This hashes both sides fresh and refuses to return if any
    byte-identical pair straddles the boundary.

    Returns (filepaths, all_labs, label_encoder, idx_to_label, num_classes, train_idx, val_idx,
    test_idx).
    """
    manifest_path = resolve_manifest(manifest_path)
    filepaths, all_labs, label_encoder, idx_to_label, num_classes = discover_dataset(
        data_dir, clean_list=clean_list
    )
    manifest = _load_manifest(manifest_path, label_encoder)
    image_dir = Path(data_dir)

    if from_manifest_train:
        if active_train_list is not None:
            raise ValueError("pass either active_train_list or from_manifest_train=True, not both")
        active = list(manifest["splits"]["train"])
        source = f"{manifest_path} (train split)"
    else:
        path = Path(active_train_list) if active_train_list is not None else (
            REPO_ROOT / "data" / "active_train.txt"
        )
        if not path.exists():
            raise RawTreeError(
                f"active training list {path} not found. release_batch.py bootstraps it from the "
                f"manifest's train split on first use; or pass from_manifest_train=True to train "
                f"on that split directly."
            )
        active = [l for l in path.read_text().splitlines() if l.strip()]
        source = str(path)

    holdout_relpaths = list(manifest["splits"]["holdout"])

    train_val_idx = _resolve_relpaths(filepaths, image_dir, active, source)
    test_idx = _resolve_relpaths(
        filepaths, image_dir, holdout_relpaths, f"{manifest_path} holdout"
    )

    # --- content-hash disjointness (F6): filenames are not the key ---------------------------
    print(f"active_split: hashing {len(train_val_idx)} train + {len(test_idx)} holdout image(s) "
          f"to verify content-hash disjointness...")
    train_hashes = hash_many([filepaths[i] for i in train_val_idx], jobs=jobs)
    holdout_hashes = hash_many([filepaths[i] for i in test_idx], jobs=jobs)
    holdout_by_hash: dict[str, str] = {}
    for p, h in holdout_hashes.items():
        holdout_by_hash.setdefault(h, p)
    collisions = [(p, holdout_by_hash[h]) for p, h in train_hashes.items() if h in holdout_by_hash]
    if collisions:
        sample = "\n".join(f"    train {t}\n    hold  {ho}" for t, ho in collisions[:5])
        raise ValueError(
            f"{len(collisions)} training image(s) are BYTE-IDENTICAL to a frozen-holdout image "
            f"(train list: {source}). Refusing to train on the evaluation set.\n{sample}"
        )
    print(f"active_split: OK -- {len(set(train_hashes.values()))} unique train hashes, "
          f"{len(set(holdout_hashes.values()))} unique holdout hashes, 0 collisions")

    labels_np = all_labs.numpy()
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size, shuffle=True,
        stratify=labels_np[train_val_idx], random_state=random_state,
    )
    return filepaths, all_labs, label_encoder, idx_to_label, num_classes, train_idx, val_idx, test_idx


def class_balanced_weights(labels: torch.Tensor, num_classes: int, beta: float = 0.9999) -> torch.Tensor:
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes
    return weights


def weighted_sampler(labels: torch.Tensor, weights: torch.Tensor, num_samples: int | None = None) -> WeightedRandomSampler:
    sample_weights = weights[labels]
    # The stub declares Sequence[float], but WeightedRandomSampler calls torch.as_tensor() on
    # this argument, so a Tensor is the intended input. Converting to a list purely to satisfy
    # the stub would copy len(labels) floats for nothing.
    return WeightedRandomSampler(
        sample_weights,  # type: ignore[arg-type]
        num_samples=num_samples or len(labels),
        replacement=True,
    )
