"""Export the registry's current @champion into a directory the cloud image bakes in.

WHY THIS EXISTS. app/main.py's default path resolves models:/coin-classifier@champion against a
tracking server at startup, which is the right design when one is reachable: the alias is the
source of truth, and a promotion changes what a restart serves with no rebuild. On Fargate there
is no tracking server and no artifact store the task can reach, so that path cannot start at all.
This script freezes the alias into a directory; serve-cloud.Dockerfile COPYs it in and sets
MODEL_SOURCE=local.

WHAT IT WRITES (out_dir, default build/champion):

    champion.json   name, alias, version, run_id, source URI, export timestamp
    artifact/       the MLflow model directory, downloaded verbatim from the registry

The sidecar is the point, not a nicety. Freezing an alias throws away the indirection that made
the alias useful, so the resulting image MUST carry the version number it froze -- otherwise a
container in the cloud can serve predictions it cannot attribute to any registered model, and
comparing cloud output to local output proves nothing. app/main.py reads champion.json for the
version it reports on /health, and refuses to start without it.

The artifact is downloaded rather than re-saved (load_model -> save_model) on purpose: a
round-trip through torch would re-serialize the module with THIS machine's torch version and
silently drop the MLmodel metadata. Downloading copies the exact bytes the training run logged.

Requires: a reachable tracking server (MLFLOW_TRACKING_URI) and read access to its artifact
store (this project's is s3://davids-mlops-artifacts-8412/mlflow, so AWS credentials too).

    python scripts/export_champion.py                    # -> build/champion
    python scripts/export_champion.py --out-dir /tmp/x   # anywhere else
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "build" / "champion"
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def export_champion(
    tracking_uri: str, model_name: str, model_alias: str, out_dir: Path
) -> dict[str, str]:
    """Download the aliased model version into out_dir and record where it came from.

    Returns the provenance dict that was written to champion.json.
    """
    mlflow.set_tracking_uri(tracking_uri)
    version_info = mlflow.MlflowClient().get_model_version_by_alias(model_name, model_alias)

    artifact_dir = out_dir / "artifact"
    # Cleared, not merged into: a stale file from a previous export left beside a new model
    # directory is exactly the kind of thing that loads fine and serves the wrong weights.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    artifact_dir.mkdir(parents=True)

    model_uri = f"models:/{model_name}@{model_alias}"
    downloaded = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri, dst_path=str(artifact_dir)
    )
    # download_artifacts may nest the model under a subdirectory of dst_path. Normalize so the
    # image layout is always <out_dir>/artifact/MLmodel -- app/main.py hardcodes that path.
    downloaded_path = Path(downloaded).resolve()
    if downloaded_path != artifact_dir.resolve():
        staged = out_dir / "_staged"
        shutil.move(str(downloaded_path), str(staged))
        shutil.rmtree(artifact_dir)
        shutil.move(str(staged), str(artifact_dir))

    if not (artifact_dir / "MLmodel").is_file():
        raise RuntimeError(
            f"Export produced no MLmodel at {artifact_dir} -- {model_uri} is not an MLflow model "
            "directory, and mlflow.pytorch.load_model will not read it."
        )

    provenance = {
        "model_name": model_name,
        "alias": model_alias,
        "version": str(version_info.version),
        "run_id": str(version_info.run_id),
        "source_uri": model_uri,
        "registry_source": str(version_info.source),
        "tracking_uri": tracking_uri,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "champion.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "coin-classifier"))
    parser.add_argument("--model-alias", default=os.environ.get("MODEL_ALIAS", "champion"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    provenance = export_champion(
        args.tracking_uri, args.model_name, args.model_alias, args.out_dir
    )
    print(json.dumps(provenance, indent=2))
    print(f"\nExported to {args.out_dir}")


if __name__ == "__main__":
    main()
