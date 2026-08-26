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

Every /predict request is logged (one row: version, label, confidence, image metadata, latency)
to the SQLite prediction log, which is what the drift check reads -- nothing else observes this
model in production. That logging is strictly an observer: it is wrapped so no failure in it can
reach the caller, and it is the LAST thing the endpoint does. See coin_clf.prediction_log.

Endpoints:
    GET  /health   liveness + model info (incl. model_version)
    POST /predict  multipart image upload -> top-k predictions (incl. model_version)
"""
import io
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict

from coin_clf.image_meta import image_metadata
from coin_clf.labels import load_labels
from coin_clf.prediction_log import PredictionLog, PredictionRecord, utc_now_iso
from coin_clf.transforms import val_transform as preprocess

APP_DIR = Path(__file__).resolve().parent
LABELS_PATH = Path(os.environ.get("LABELS_PATH", APP_DIR / "coin_labels.json"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "coin-classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")

# Default resolves to <repo>/outputs/monitoring/predictions.db on the host and
# /srv/outputs/monitoring/predictions.db in the container. The container path MUST be a mounted
# volume (`make serve` mounts it) or the log dies with `docker run --rm` and the host-side drift
# check has nothing to read.
PREDICTION_LOG_PATH = Path(os.environ.get(
    "PREDICTION_LOG_PATH", APP_DIR.parent / "outputs" / "monitoring" / "predictions.db"
))
MAX_SOURCE_TAG_LEN = 64  # a tag is a label, not a payload; bounded so it can't bloat the log


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
    # Constructing a PredictionLog touches no disk, so a bad log path cannot break startup --
    # monitoring must not be able to take serving down. The file appears on the first write.
    app.state.prediction_log = PredictionLog(PREDICTION_LOG_PATH)
    yield


def get_bundle(request: Request) -> ModelBundle:
    """Dependency the endpoints use. Tests override this so the real load never runs."""
    return request.app.state.bundle


def get_prediction_log(request: Request) -> PredictionLog | None:
    """Dependency for the log. None when lifespan never ran (tests that inject only a bundle).

    getattr-with-default rather than attribute access on purpose: a dependency that RAISES is a
    500 returned before the endpoint body runs, so the endpoint's own try/except would never get
    the chance to swallow it. The one place a missing log could still break a prediction is
    here, so it is handled here.
    """
    return getattr(request.app.state, "prediction_log", None)


def _clean_source_tag(raw: str | None) -> str | None:
    """Normalize the X-Traffic-Source header. Absent, blank, or whitespace-only all mean NULL --
    real traffic sends no tag, and an empty string is absence, not a tag named "".
    """
    if raw is None:
        return None
    return raw.strip()[:MAX_SOURCE_TAG_LEN] or None


def _log_prediction(log: PredictionLog | None, **fields) -> None:
    """Best-effort write. Belt and braces over PredictionLog.log()'s own never-raise contract:
    this guard also covers building the record and anything a future edit adds around it. A
    monitoring write failing must never fail a prediction.
    """
    if log is None:
        return
    try:
        log.log(PredictionRecord(**fields))
    except Exception:
        pass


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
    x_traffic_source: str | None = Header(default=None, alias="X-Traffic-Source"),
    bundle: ModelBundle = Depends(get_bundle),
    prediction_log: PredictionLog | None = Depends(get_prediction_log),
) -> PredictResponse:
    """x_traffic_source is an optional monitoring tag (the replay script sets it to mark a
    normal vs. a deliberately skewed batch). A header rather than a query param so the tag stays
    out of the endpoint's functional contract: it changes nothing about the prediction, only how
    the resulting row is grouped later. Real traffic omits it and logs NULL.
    """
    started = time.perf_counter()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image, got content-type {file.content_type!r}")

    raw = await file.read()
    try:
        with Image.open(io.BytesIO(raw)) as src:
            # Metadata comes off the image AS UPLOADED. Reading it after the convert below would
            # log mode="RGB" for every request ever made -- including grayscale uploads -- which
            # silently destroys the mode drift signal. See coin_clf.image_meta.image_metadata.
            meta = image_metadata(src)
            img = src.convert("RGB")
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

    # Everything the caller is owed is now computed; only observation is left. latency_ms is
    # taken before the write so the log records the prediction's cost, not its own.
    latency_ms = (time.perf_counter() - started) * 1000
    _log_prediction(
        prediction_log,
        ts=utc_now_iso(),
        model_version=bundle.version,
        predicted_label=predictions[0].label,   # topk is clamped to >= 1, so this always exists
        confidence=predictions[0].probability,
        width=meta.width,
        height=meta.height,
        mode=meta.mode,
        latency_ms=latency_ms,
        source=_clean_source_tag(x_traffic_source),
    )
    return PredictResponse(model_version=bundle.version, predictions=predictions)