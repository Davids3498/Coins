"""Safe checkpoint saving for train.py — never silently clobber a checkpoint again.

Root cause of the footgun: train.py wrote every run to the same path
(weights/emp_model_distil_student.pth), so training v3 overwrote v1's local weights. The
registry copy of v1 was safe in S3, but the local .pth no longer mapped to v1 — re-seeding
from that path would silently reproduce the WRONG model.

Fix: write a unique, run-tied filename and refuse to overwrite an existing file. Tying the
name to the MLflow run_id means a local checkpoint always maps back to a known run/version.

In train.py, replace the bare `torch.save(best_state, "weights/emp_model_distil_student.pth")`
with, inside the `with mlflow.start_run() as run:` block:

    from checkpoint import save_checkpoint
    save_checkpoint(best_state, "weights", run.info.run_id)
"""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    state_dict: dict,
    weights_dir: str,
    run_id: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write weights to weights_dir/student_<run_id>.pth. Refuses to clobber unless overwrite=True."""
    out = Path(weights_dir) / f"student_{run_id}.pth"
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists — refusing to overwrite (pass overwrite=True to force)")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, out)
    return out
