"""FastAPI server for the distilled coin classifier.

Model source: the current @champion in the MLflow registry
(models:/coin-classifier@champion), loaded at startup — no baked-in checkpoint.
Because the full model was logged (not just weights), no architecture-rebuild code
is needed here; mlflow.pytorch.load_model reconstructs the nn.Module directly.

Preprocessing mirrors val_transform: Resize(256) -> CenterCrop(224) -> ImageNet normalize.

Endpoints:
    GET  /health   liveness + model info (incl. model_version)
    POST /predict  multipart image upload -> top-k predictions (incl. model_version)
"""
import io
import json
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict

APP_DIR = Path(__file__).resolve().parent
LABELS_PATH = Path(os.environ.get("LABELS_PATH", APP_DIR / "coin_labels.json"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "coin-classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_labels(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise RuntimeError(f"Label file not found: {path}")
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def load_model_from_registry(
    tracking_uri: str, model_name: str, model_alias: str, device: torch.device
) -> tuple[nn.Module, str]:
    mlflow.set_tracking_uri(tracking_uri)
    version = mlflow.MlflowClient().get_model_version_by_alias(model_name, model_alias).version
    model = mlflow.pytorch.load_model(f"models:/{model_name}@{model_alias}")
    model.to(device).eval()
    return model, version


idx_to_name = load_labels(LABELS_PATH)
model, MODEL_VERSION = load_model_from_registry(MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_ALIAS, DEVICE)

app = FastAPI(title="Coin Classifier", description="Roman emperor coin classifier (MobileNetV3-Large, distilled)")


class Prediction(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow the model_version field name
    model_version: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    device: str
    num_classes: int
    model_name: str
    model_version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(DEVICE),
        num_classes=len(idx_to_name),
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), topk: int = 3) -> PredictResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image, got content-type {file.content_type!r}")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image")

    topk = max(1, min(topk, len(idx_to_name)))
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_p, top_i = probs.topk(topk, dim=1)

    predictions = [
        Prediction(label=idx_to_name[int(idx)], probability=float(p))
        for p, idx in zip(top_p[0].tolist(), top_i[0].tolist())
    ]
    return PredictResponse(model_version=MODEL_VERSION, predictions=predictions)