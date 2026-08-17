"""FastAPI server for the distilled coin classifier.

Model: emp_model_distil_student.pth — MobileNetV3-Large distilled from the v6
ensemble teacher (92.71% test acc), 89.61% test acc as a single forward pass.
Preprocessing mirrors gradcam.py / emp_model_knowledge_distilation.ipynb's
val_transform: Resize(256) -> CenterCrop(224) -> ImageNet normalize.

Endpoints:
    GET  /health   liveness + model info
    POST /predict  multipart image upload -> top-k emperor predictions
"""
import io
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", APP_DIR.parent / "emp_model_distil_student.pth"))
LABELS_PATH = Path(os.environ.get("LABELS_PATH", APP_DIR / "coin_labels.json"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
    model = tvm.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


idx_to_name = load_labels(LABELS_PATH)
model = load_model(CHECKPOINT_PATH, len(idx_to_name), DEVICE)

app = FastAPI(title="Coin Classifier", description="Roman emperor coin classifier (MobileNetV3-Large, distilled)")


class Prediction(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    device: str
    num_classes: int
    checkpoint: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(DEVICE),
        num_classes=len(idx_to_name),
        checkpoint=CHECKPOINT_PATH.name,
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
    return PredictResponse(predictions=predictions)
