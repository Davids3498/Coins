"""Seed the coin-classifier MobileNetV3 checkpoint into the MLflow registry as the champion.

Prereqs: the MLflow server is running (sqlite + S3, --no-serve-artifacts).
Run:      python seed_champion.py
"""
from __future__ import annotations

import mlflow
import mlflow.pytorch
import torch
from mlflow.models import infer_signature

from coin_clf.model import build_model

# --- edit these ---
CHECKPOINT_PATH: str = "path/to/your.pth"      # your fine-tuned state dict
TRACKING_URI: str = "http://127.0.0.1:5000"
INPUT_SIZE: int = 224                          # MUST match your training/inference transform

# --- stable ---
MODEL_NAME: str = "coin-classifier"
EXPERIMENT_NAME: str = "coin-classifier"
NUM_CLASSES: int = 51
ARCH: str = "mobilenet_v3_large"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = build_model(NUM_CLASSES)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)  # strict=True by default: fails loudly on any key mismatch
    model.eval()

    # signature + input example so the logged model is self-describing for serving
    example_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        example_output = model(example_input)
    signature = infer_signature(example_input.numpy(), example_output.numpy())

    with mlflow.start_run(run_name="seed-champion") as run:
        mlflow.log_params(
            {
                "arch": ARCH,
                "num_classes": NUM_CLASSES,
                "input_size": INPUT_SIZE,
                "source_checkpoint": CHECKPOINT_PATH,
            }
        )
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",  # if this errors on an older client: change to artifact_path="model"
            signature=signature,
            input_example=example_input.numpy(),
        )
        run_id = run.info.run_id

    mv = mlflow.register_model(model_info.model_uri, MODEL_NAME)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)

    champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
    print(f"run_id={run_id}")
    print(f"{MODEL_NAME} v{champ.version} registered, alias @champion set")


if __name__ == "__main__":
    main()
