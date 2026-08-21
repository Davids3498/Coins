"""Fast API smoke tests -- no MLflow server, no S3, no real model, no mlflow monkeypatching.

app.main loads the champion in a FastAPI lifespan (startup) and hands it to the endpoints via
the get_bundle dependency. Tests override that dependency with a fake bundle and skip lifespan
(no `with` on TestClient), so the real registry load never runs. Importing app is safe now --
it triggers no registry call. Requires `coin_clf` importable (`pythonpath = ["src"]` under
[tool.pytest.ini_options]).
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_bundle, ModelBundle

STUB_VERSION = "3"
FAKE_LABELS = {0: "NERO", 1: "CLAUDIUS", 2: "COMMODUS"}
TOP_LEVEL_KEYS = {"model_version", "predictions"}
PREDICTION_KEYS = {"label", "probability"}


class _FakeModel:
    """Fixed logits sized to FAKE_LABELS -- deterministic argmax, no weights, no S3."""

    def __call__(self, x):
        import torch

        logits = torch.zeros(x.shape[0], len(FAKE_LABELS))
        logits[:, 0] = 10.0  # argmax -> class 0 -> "NERO"
        return logits


@pytest.fixture
def client():
    fake = ModelBundle(model=_FakeModel(), version=STUB_VERSION, idx_to_name=FAKE_LABELS)
    app.dependency_overrides[get_bundle] = lambda: fake
    c = TestClient(app)  # NO `with` -> lifespan (the real load) never runs
    yield c
    app.dependency_overrides.clear()


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (128, 128, 128)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_surfaces_model_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == STUB_VERSION  # version flows bundle -> response


def test_predict_returns_schema(client):
    r = client.post("/predict", files={"file": ("coin.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()

    assert TOP_LEVEL_KEYS.issubset(body.keys())
    assert body["model_version"] == STUB_VERSION

    preds = body["predictions"]
    assert isinstance(preds, list) and preds          # non-empty top-k
    top = preds[0]
    assert PREDICTION_KEYS.issubset(top.keys())
    assert isinstance(top["label"], str)
    assert 0.0 <= top["probability"] <= 1.0

    probs = [p["probability"] for p in preds]
    assert probs == sorted(probs, reverse=True)       # torch.topk -> highest-first