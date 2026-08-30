"""Gate the data-integrity checker itself, on a tree small enough to run in CI.

verify_data_integrity.py re-hashes 57,792 real files, so it cannot run on a hosted runner and
therefore gates nothing today. These tests run it against tests/fixtures/make_tree.py's ~60-image
stand-in, which has the same structure -- labelled tree, clean list, carved manifest,
active_train.txt, cursor, quarantine -- and takes seconds.

The positive test (clean tree -> exit 0) only proves the checker still runs. The NEGATIVE tests
are the point: each injects one specific fault and asserts both that the run fails AND that the
check which is supposed to notice actually named it. Asserting the exit code alone would pass if
the script died for an unrelated reason -- a checker that crashes is not a checker that caught
something, and the three data incidents this file exists to prevent were all cases where the
checker ran clean over a problem it had no check for.

Deliberately untyped, per the convention in test_gate.py.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAKE_TREE = REPO_ROOT / "tests" / "fixtures" / "make_tree.py"
CHECKER = REPO_ROOT / "verify_data_integrity.py"


def build_fixture(dest, *inject):
    """Generate a fixture tree and return (root, expectations dict)."""
    cmd = [sys.executable, str(MAKE_TREE), "--out", str(dest), *inject]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"make_tree.py failed:\n{proc.stdout}\n{proc.stderr}"
    return dest, json.loads((dest / "expectations.json").read_text())


def run_checker(root, expectations):
    """Run verify_data_integrity.py against a fixture tree. Returns the CompletedProcess.

    --jobs 2 rather than the default 8: 60 files across 8 processes is pure pool startup, and CI
    runners are small.
    """
    cmd = [
        sys.executable, str(CHECKER),
        "--data-root", str(root),
        "--expect-clean-corpus", str(expectations["clean_corpus"]),
        "--expect-originals", str(expectations["originals"]),
        "--expect-classes", str(expectations["classes"]),
        "--batch-size", str(expectations["batch_size"]),
        "--jobs", "2",
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory):
    root, exp = build_fixture(tmp_path_factory.mktemp("clean") / "tree")
    return run_checker(root, exp)


# --- positive: an intact tree passes ---------------------------------------

def test_clean_tree_exits_zero(clean_run):
    assert clean_run.returncode == 0, clean_run.stdout[-4000:]


def test_clean_tree_reports_no_failures(clean_run):
    assert "ALL CHECKS PASSED" in clean_run.stdout
    assert "[FAIL]" not in clean_run.stdout


def test_clean_tree_actually_ran_all_three_sections(clean_run):
    # Guards against a fixture so broken that the script exits 0 having checked almost nothing.
    for marker in ("SECTION A", "SECTION B", "SECTION C"):
        assert marker in clean_run.stdout
    assert "0/0 checks passed" not in clean_run.stdout


# --- negative: each injected fault must be caught, by name -----------------
#
# (flag, substring of the check that must FAIL). The substring is what stops a test from passing
# because the script crashed for some other reason.
FAULTS = [
    pytest.param(
        "--inject-duplicate",
        "A4 training tree is internally hash-unique",
        id="duplicate",
    ),
    pytest.param(
        "--inject-cross-split-leak",
        "A2 hash-disjoint: TRAIN x HOLDOUT",
        id="cross_split_leak",
    ),
    pytest.param(
        "--inject-unknown-label",
        "A5 clean tree label space == manifest label_encoder",
        id="unknown_label",
    ),
    pytest.param(
        "--inject-corrupt",
        "A6 every file in the training tree decodes end-to-end",
        id="corrupt",
    ),
]


@pytest.mark.parametrize("flag,expected_check", FAULTS)
def test_injected_fault_fails_the_checker(tmp_path, flag, expected_check):
    root, exp = build_fixture(tmp_path / "tree", flag)
    proc = run_checker(root, exp)

    assert proc.returncode != 0, (
        f"{flag} produced a tree the checker accepted:\n{proc.stdout[-4000:]}"
    )
    assert f"[FAIL] {expected_check}" in proc.stdout, (
        f"{flag} failed the run, but not via {expected_check!r} -- so this test would also pass "
        f"if the checker merely crashed:\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}"
    )


def test_corruption_is_caught_only_by_the_decode_check(tmp_path):
    """The reason A6 exists.

    A truncated JPEG is present, uniquely hashed, on the clean list and in the right partition:
    every byte-level check in the report passes it. Nothing but an actual decode attempt sees it.
    If this test ever fails because another check also caught it, that is fine -- but A6 must not
    be the one that stops mattering.
    """
    root, exp = build_fixture(tmp_path / "tree", "--inject-corrupt")
    proc = run_checker(root, exp)

    failures = [l for l in proc.stdout.splitlines() if "[FAIL]" in l]
    assert len(failures) == 1, "expected exactly one failed check, got:\n" + "\n".join(failures)
    assert "A6 every file in the training tree decodes end-to-end" in failures[0]
