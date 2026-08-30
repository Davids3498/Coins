"""Generate a tiny, self-contained stand-in for the real 57,792-image coin tree.

verify_data_integrity.py is the strongest check in this repo and the one that cannot run in CI:
it re-hashes every file in a 51 GB tree. This builds a ~60-image version of the same structure
-- the labelled tree, clean_files.txt, the carved manifest, active_train.txt, the cursor and the
quarantine -- so the checker can be exercised in seconds on a hosted runner.

The manifest is produced by calling splits.carve() for real, not by hand-writing JSON. A
hand-written manifest would agree with whatever the fixture author believed carve() does, which
is exactly the class of mistake the checker exists to catch.

The four --inject-* flags each corrupt ONE property and leave every other one intact, so a test
that asserts a non-zero exit is asserting that the specific check fired, not that something
somewhere broke. That negative direction is the point: the last three data incidents were all
"the checker didn't cover this case", and a checker is only worth its runtime if a deliberately
broken tree actually fails it.

Usage:
    python tests/fixtures/make_tree.py --out /tmp/fixture
    python tests/fixtures/make_tree.py --out /tmp/fixture --inject-duplicate
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

# Deliberately NOT "GORDIAN I"/"GORDIAN II": coin_clf.data.GORDIAN_MERGES collapses those two
# into one label, so using them here would make the fixture's class count differ from its
# directory count for a reason that has nothing to do with what is being tested.
CLASS_NAMES = ("AUGUSTUS", "NERO", "TRAJAN")
IMAGE_SIZE = 32


def _write_image(path: Path, rng: random.Random) -> None:
    """One 32x32 JPEG of random noise. Noise, not a flat colour, so every file's bytes -- and
    therefore its content hash -- differ; the checker's whole vocabulary is content hashes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    img.putdata([
        (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        for _ in range(IMAGE_SIZE * IMAGE_SIZE)
    ])
    img.save(path, format="JPEG", quality=95)


def build_tree(data_dir: Path, per_class: int, rng: random.Random) -> list[str]:
    """Write the labelled tree and return relpaths, in discover_dataset's sorted-glob order."""
    for i, name in enumerate(CLASS_NAMES, start=1):
        for j in range(per_class):
            _write_image(data_dir / f"{i:02d}_{name}" / "side_a" / f"{name.lower()}_{j:04d}.jpg", rng)
    return [str(p.relative_to(data_dir)) for p in sorted(data_dir.glob("*/side_a/*.jpg"))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="root to build under (mirrors the repo's data/)")
    p.add_argument("--per-class", type=int, default=20, help="images per class (default 20 -> 60)")
    p.add_argument("--quarantined", type=int, default=6, help="files parked in quarantine/")
    p.add_argument("--batch-size", type=int, default=5, help="future-pool batch size to record")
    p.add_argument("--seed", type=int, default=1234)

    g = p.add_argument_group("fault injection (each breaks exactly one property)")
    g.add_argument("--inject-duplicate", action="store_true",
                   help="overwrite one train file with another train file's bytes "
                        "(counts unchanged; hash-uniqueness broken)")
    g.add_argument("--inject-cross-split-leak", action="store_true",
                   help="overwrite one train file with a HOLDOUT file's bytes "
                        "(counts unchanged; train/holdout disjointness broken)")
    g.add_argument("--inject-unknown-label", action="store_true",
                   help="add a 4th class to the tree and clean list but not the manifest "
                        "(label space no longer matches the carved manifest)")
    g.add_argument("--inject-corrupt", action="store_true",
                   help="truncate one train file (still listed, still unique, no longer decodable)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    data_dir = out / "FOR_TRAINNING"

    # --- 1. the clean tree and the survivor list ---------------------------------------------
    relpaths = build_tree(data_dir, args.per_class, rng)
    clean_list = out / "clean_files.txt"
    clean_list.write_text("\n".join(relpaths) + "\n")

    # --- 2. the manifest, carved for real ----------------------------------------------------
    from splits import carve

    splits = carve(data_dir, clean_list=clean_list)
    manifest_path = out / "splits_manifest.json"
    splits.save(manifest_path, data_dir)
    manifest = json.loads(manifest_path.read_text())
    train_rel = manifest["splits"]["train"]
    holdout_rel = manifest["splits"]["holdout"]

    # --- 3. the growing training list and the cursor ------------------------------------------
    # next_batch = 0: nothing released yet, so active_train.txt is exactly the manifest's train
    # split. That is the state release_batch.py bootstraps into on first use.
    (out / "active_train.txt").write_text("\n".join(train_rel) + "\n")
    (out / "future_pool_cursor.json").write_text(json.dumps({"next_batch": 0}, indent=2))

    # --- 4. quarantine -- condemned files, kept reversibly ------------------------------------
    quarantine = out / "quarantine"
    for k in range(args.quarantined):
        _write_image(quarantine / f"dup_{k:04d}" / "side_a" / f"condemned_{k:04d}.jpg", rng)

    # --- 5. what the checker should expect of THIS tree ---------------------------------------
    # Written out rather than hardcoded in the test, so changing --per-class cannot silently
    # leave the test asserting numbers from a tree that no longer exists.
    expectations = {
        "clean_corpus": len(relpaths),
        "originals": len(relpaths) + args.quarantined,
        "classes": len(CLASS_NAMES),
        "batch_size": args.batch_size,
    }

    # --- 6. fault injection, applied last so the clean artifacts above are already pinned ------
    faults: list[str] = []

    if args.inject_duplicate:
        # Same path, same count, same manifest -- only the BYTES now collide with another train
        # file. Isolates hash-uniqueness from every count-based check.
        victim, donor = data_dir / train_rel[0], data_dir / train_rel[1]
        victim.write_bytes(donor.read_bytes())
        faults.append(f"duplicate: {train_rel[0]} now byte-identical to {train_rel[1]}")

    if args.inject_cross_split_leak:
        # The failure this whole repo exists to prevent: a training image that IS an evaluation
        # image, under a different filename. Only content hashing can see it.
        victim, donor = data_dir / train_rel[2], data_dir / holdout_rel[0]
        victim.write_bytes(donor.read_bytes())
        faults.append(f"leak: train {train_rel[2]} now byte-identical to holdout {holdout_rel[0]}")

    if args.inject_unknown_label:
        # A class folder that appeared after the manifest was carved. On the real tree this is a
        # new coin type arriving in a delivery; the label space silently widens and the manifest's
        # label_encoder no longer describes the data.
        extra = []
        for j in range(3):
            rel = f"04_HADRIAN/side_a/hadrian_{j:04d}.jpg"
            _write_image(data_dir / rel, rng)
            extra.append(rel)
        clean_list.write_text("\n".join(sorted(relpaths + extra)) + "\n")
        expectations["clean_corpus"] += len(extra)
        expectations["originals"] += len(extra)
        faults.append(f"unknown label: added 04_HADRIAN ({len(extra)} files) outside the manifest")

    if args.inject_corrupt:
        # Truncated mid-scan. Still present, still listed, still a unique hash -- and undecodable.
        # Nothing that only reads bytes can see this; it takes an actual decode attempt.
        victim = data_dir / train_rel[3]
        raw = victim.read_bytes()
        victim.write_bytes(raw[: len(raw) // 2])
        faults.append(f"corrupt: {train_rel[3]} truncated to half its bytes")

    (out / "expectations.json").write_text(json.dumps(expectations, indent=2))

    print(f"fixture tree -> {out}")
    print(f"  data_dir       {data_dir}")
    print(f"  images         {len(relpaths)} across {len(CLASS_NAMES)} classes")
    print(f"  manifest       train={len(train_rel)} holdout={len(holdout_rel)} "
          f"future={len(manifest['splits']['future_pool'])}")
    print(f"  quarantined    {args.quarantined}")
    for f in faults:
        print(f"  INJECTED       {f}")
    if not faults:
        print("  INJECTED       nothing -- this tree is clean")


if __name__ == "__main__":
    main()
