"""FastAPI server for the distilled coin classifier.

Model source: the current @champion in the MLflow registry
(models:/coin-classifier@champion), loaded on startup via a FastAPI lifespan -- no baked-in
checkpoint. Because the full model was logged (not just weights), no architecture-rebuild code
is needed here; mlflow.pytorch.load_model reconstructs the nn.Module directly.

The load happens in `lifespan` (startup), NOT at import, and the endpoints receive the loaded
model via the `get_bundle` dependency. That keeps the module importable with no registry call,
so tests inject a fake bundle through app.dependency_overrides -- no MLflow server, no S3, and
no monkeypatching of mlflow internals.

Preprocessing uses coin_clf.transforms.val_transform: Resize(256) -> CenterCrop(224) -> ImageNet normalize.

Endpoints:
    GET  /health   liveness + model info (incl. model_version)
    POST /predict  multipart image upload -> top-k predictions (incl. model_version)
"""
import io
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict

from coin_clf.labels import load_labels
from coin_clf.transforms import val_transform as preprocess

APP_DIR = Path(__file__).resolve().parent
LABELS_PATH = Path(os.environ.get("LABELS_PATH", APP_DIR / "coin_labels.json"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "coin-classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")


@dataclass
class ModelBundle:
    """Everything an endpoint needs, resolved once on startup."""

    model: nn.Module
    version: str
    idx_to_name: dict


def load_model_from_registry(
    tracking_uri: str, model_name: str, model_alias: str, device: torch.device
) -> tuple[nn.Module, str]:
    mlflow.set_tracking_uri(tracking_uri)
    version = mlflow.MlflowClient().get_model_version_by_alias(model_name, model_alias).version
    model = mlflow.pytorch.load_model(f"models:/{model_name}@{model_alias}")
    model.to(device).eval()
    return model, version


def build_bundle() -> ModelBundle:
    """The real, resource-touching load. Called from lifespan; overridden in tests."""
    idx_to_name = load_labels(LABELS_PATH)
    model, version = load_model_from_registry(MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_ALIAS, DEVICE)
    return ModelBundle(model=model, version=version, idx_to_name=idx_to_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = build_bundle()  # startup, not import time
    yield


def get_bundle(request: Request) -> ModelBundle:
    """Dependency the endpoints use. Tests override this so the real load never runs."""
    return request.app.state.bundle


app = FastAPI(
    title="Coin Classifier",
    description="Roman emperor coin classifier (MobileNetV3-Large, distilled)",
    lifespan=lifespan,
)


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
def health(bundle: ModelBundle = Depends(get_bundle)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(DEVICE),
        num_classes=len(bundle.idx_to_name),
        model_name=MODEL_NAME,
        model_version=bundle.version,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    topk: int = 3,
    bundle: ModelBundle = Depends(get_bundle),
) -> PredictResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image, got content-type {file.content_type!r}")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image")

    topk = max(1, min(topk, len(bundle.idx_to_name)))
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = bundle.model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_p, top_i = probs.topk(topk, dim=1)

    predictions = [
        Prediction(label=bundle.idx_to_name[int(idx)], probability=float(p))
        for p, idx in zip(top_p[0].tolist(), top_i[0].tolist())
    ]
    return PredictResponse(model_version=bundle.version, predictions=predictions)