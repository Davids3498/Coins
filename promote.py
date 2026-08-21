"""Automated champion/challenger promotion gate for `coin-classifier`.

Moves the @champion alias to a challenger ONLY if it beats the current champion on the
frozen holdout by at least --margin. Otherwise the champion stays and the reason is logged.

Two testable seams, and NEITHER needs infrastructure to test:
  * decide(...)   — the pure policy (is this challenger good enough?). No I/O.
  * run_gate(...) — the orchestration (resolve champion, score both, move the alias) with its
    three collaborators INJECTED, so tests drive the whole decision + side-effect path with
    fakes: no MLflow server, no S3, no GPU, no dataset. main() wires the real collaborators.

Defensible by design: re-scores BOTH models fresh at gate time (never trusts a logged number),
idempotent (challenger already champion -> no-op), bootstraps when no champion exists, and
fail-safe holds if the champion can't be scored (won't replace a serving model on a broken
comparison).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Optional

MODEL_NAME = "coin-classifier"
CHAMPION_ALIAS = "champion"
TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MARGIN = 0.005  # challenger must clear the champion by >= this (0.5 pp) to promote


@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str


@dataclass(frozen=True)
class Outcome:
    action: str                        # "promote" | "hold" | "noop"
    moved: bool                        # did the alias actually change (False under dry-run)
    reason: str
    challenger_version: str
    champion_version: Optional[str] = None
    challenger_acc: Optional[float] = None
    champion_acc: Optional[float] = None


def decide(challenger_acc: float, champion_acc: Optional[float], margin: float) -> Decision:
    """Pure gate policy. No I/O, no torch, no mlflow -- just the rule.

      * champion_acc is None            -> HOLD (can't confirm improvement; fail-safe)
      * strictly better AND >= margin   -> PROMOTE
      * worse, equal, or inside margin  -> HOLD

    The `> 0` guard makes "a tie never promotes" total, even if margin is set to 0.
    """
    if margin < 0:
        raise ValueError("margin must be >= 0")
    if champion_acc is None:
        return Decision(False, "champion metric unavailable -- refusing to promote without a valid comparison")
    delta = challenger_acc - champion_acc
    if delta > 0 and delta >= margin:
        return Decision(True, f"challenger beats champion by {delta:+.4f} (>= margin {margin:.4f})")
    return Decision(False, f"challenger delta {delta:+.4f} does not clear margin {margin:.4f}")


def run_gate(
    challenger: str,
    *,
    get_champion_version: Callable[[], Optional[str]],
    score: Callable[[str], float],
    set_champion: Callable[[str], None],
    margin: float = DEFAULT_MARGIN,
    dry_run: bool = False,
) -> Outcome:
    """Orchestrate one gate run against injected collaborators. Returns what it decided and
    whether it moved the alias -- it does NOT print. main() supplies MLflow-backed
    collaborators; tests supply fakes.

    Collaborators:
      get_champion_version() -> current champion version, or None if there's no champion yet
      score(version)         -> accuracy on the frozen holdout (may raise if unscoreable)
      set_champion(version)  -> move the @champion alias
    """
    champion_version = get_champion_version()

    # Idempotency: challenger already champion -> no-op. We don't even score (cheap, safe re-run).
    if champion_version == challenger:
        return Outcome("noop", False, f"v{challenger} is already @{CHAMPION_ALIAS}", challenger, champion_version)

    # Bootstrap: no champion at all -> promote (nothing is serving, nothing to compare against).
    if champion_version is None:
        moved = not dry_run
        if moved:
            set_champion(challenger)
        return Outcome("promote", moved, f"no @{CHAMPION_ALIAS} set -- bootstrapping challenger", challenger, None)

    # A challenger we can't score is a hard error (you asked to gate a broken model) -- let it raise.
    challenger_acc = score(challenger)
    # A champion we can't score is NOT fatal -- fail-safe hold instead of replacing it blindly.
    try:
        champion_acc = score(champion_version)
    except Exception:  # noqa: BLE001
        champion_acc = None

    decision = decide(challenger_acc, champion_acc, margin)
    moved = decision.promote and not dry_run
    if moved:
        set_champion(challenger)
    action = "promote" if decision.promote else "hold"
    return Outcome(action, moved, decision.reason, challenger, champion_version, challenger_acc, champion_acc)


def main() -> None:
    # Heavy imports here so `from promote import decide, run_gate` stays torch/mlflow-free.
    import mlflow
    from mlflow.tracking import MlflowClient

    from evaluate import evaluate_version

    p = argparse.ArgumentParser(description="Promote a challenger to @champion if it clears the margin.")
    p.add_argument("--challenger", required=True, help="challenger model version to evaluate")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dry-run", action="store_true", help="decide and print, but do not move the alias")
    args = p.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    def get_champion_version():
        try:
            return client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version
        except Exception:
            return None

    def score(version):
        _, acc = evaluate_version(version=version, data_dir=args.data_dir, batch_size=args.batch_size)
        return acc

    def set_champion(version):
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)

    outcome = run_gate(
        args.challenger,
        get_champion_version=get_champion_version,
        score=score,
        set_champion=set_champion,
        margin=args.margin,
        dry_run=args.dry_run,
    )

    if outcome.challenger_acc is not None or outcome.champion_acc is not None:
        champ = "n/a" if outcome.champion_acc is None else f"{outcome.champion_acc:.4f}"
        chal = "n/a" if outcome.challenger_acc is None else f"{outcome.challenger_acc:.4f}"
        print(f"champion v{outcome.champion_version} acc={champ}  |  challenger v{outcome.challenger_version} acc={chal}")
    print(f"{outcome.action.upper()} -- {outcome.reason}")
    if outcome.moved:
        print(f"@{CHAMPION_ALIAS} -> v{outcome.challenger_version} (previous v{outcome.champion_version} retained for rollback)")
    elif outcome.action == "promote":
        print(f"[dry-run] would set @{CHAMPION_ALIAS} -> v{outcome.challenger_version}")


if __name__ == "__main__":
    main()