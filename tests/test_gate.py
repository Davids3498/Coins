"""Gate tests -- the differentiator. Two levels, both with ZERO infrastructure:

  * decide()   -- the pure promotion RULE.
  * run_gate() -- the actual gate BEHAVIOUR, including the side effect that matters (does the
    @champion alias move or not?). Collaborators are injected as fakes, so these exercise the
    branches that break in production -- idempotency, bootstrap, fail-safe, dry-run -- with no
    MLflow server, no S3, no GPU, and no dataset.

All of it runs in milliseconds in the fast CI path that gates every PR.
"""
import pytest

from promote import decide, run_gate


# --- pure policy: decide() -------------------------------------------------

def test_promotes_when_challenger_clears_margin():
    assert decide(challenger_acc=0.92, champion_acc=0.90, margin=0.005).promote is True


def test_holds_when_challenger_worse():
    assert decide(challenger_acc=0.88, champion_acc=0.90, margin=0.005).promote is False


def test_holds_on_exact_tie():
    # A tie must never promote -- otherwise the alias churns for zero gain.
    assert decide(challenger_acc=0.90, champion_acc=0.90, margin=0.005).promote is False


def test_holds_inside_margin():
    # Better, but not by enough -- noise, not a real win.
    assert decide(challenger_acc=0.9049, champion_acc=0.9048, margin=0.005).promote is False


def test_promotes_exactly_at_margin():
    # The boundary is inclusive and deliberate.
    assert decide(challenger_acc=0.905, champion_acc=0.900, margin=0.005).promote is True


def test_tie_never_promotes_even_at_zero_margin():
    assert decide(challenger_acc=0.90, champion_acc=0.90, margin=0.0).promote is False


def test_decide_survives_missing_champion_metric():
    # Champion score unavailable -> fail-safe hold, no exception.
    assert decide(challenger_acc=0.95, champion_acc=None, margin=0.005).promote is False


def test_negative_margin_rejected():
    with pytest.raises(ValueError):
        decide(challenger_acc=0.9, champion_acc=0.8, margin=-0.01)


# --- orchestration + side effects: run_gate() ------------------------------

class AliasSpy:
    """Records whether (and where) the @champion alias was moved."""

    def __init__(self):
        self.moved_to = None
        self.calls = 0

    def __call__(self, version):
        self.moved_to = version
        self.calls += 1


def scorer(table):
    """Fake score(version) backed by a dict; raises for versions not in it (unscoreable)."""
    def score(version):
        if version not in table:
            raise RuntimeError(f"cannot score v{version}")
        return table[version]
    return score


def _never(_version):
    # Passed as the scorer where scoring would be wasted work -- any call fails the test.
    raise AssertionError("score should not be called in this branch")


def test_promote_moves_alias_to_challenger():
    spy = AliasSpy()
    out = run_gate(
        "4",
        get_champion_version=lambda: "3",
        score=scorer({"4": 0.95, "3": 0.90}),
        set_champion=spy,
        margin=0.005,
    )
    assert out.action == "promote"
    assert out.moved is True
    assert spy.moved_to == "4"
    assert spy.calls == 1


def test_hold_leaves_champion_untouched():
    spy = AliasSpy()
    out = run_gate(
        "2",
        get_champion_version=lambda: "3",
        score=scorer({"2": 0.2555, "3": 0.9046}),  # the real v2-vs-v3 hold
        set_champion=spy,
        margin=0.005,
    )
    assert out.action == "hold"
    assert out.moved is False
    assert spy.calls == 0


def test_idempotent_noop_when_already_champion():
    spy = AliasSpy()
    out = run_gate(
        "3",
        get_champion_version=lambda: "3",
        score=_never,  # short-circuits before scoring -- cheap, safe to re-run
        set_champion=spy,
        margin=0.005,
    )
    assert out.action == "noop"
    assert spy.calls == 0


def test_bootstrap_promotes_when_no_champion():
    spy = AliasSpy()
    out = run_gate(
        "1",
        get_champion_version=lambda: None,
        score=_never,  # nothing to compare against, so nothing is scored
        set_champion=spy,
        margin=0.005,
    )
    assert out.action == "promote"
    assert out.moved is True
    assert spy.moved_to == "1"


def test_failsafe_holds_when_champion_unscoreable():
    spy = AliasSpy()
    # champion "3" is missing from the table -> score raises -> fail-safe hold, even though
    # the challenger looks excellent.
    out = run_gate(
        "4",
        get_champion_version=lambda: "3",
        score=scorer({"4": 0.99}),
        set_champion=spy,
        margin=0.005,
    )
    assert out.action == "hold"
    assert out.moved is False
    assert spy.calls == 0


def test_dry_run_decides_promote_but_does_not_move():
    spy = AliasSpy()
    out = run_gate(
        "4",
        get_champion_version=lambda: "3",
        score=scorer({"4": 0.95, "3": 0.90}),
        set_champion=spy,
        margin=0.005,
        dry_run=True,
    )
    assert out.action == "promote"
    assert out.moved is False
    assert spy.calls == 0