"""Tests for the prediction log -- coin_clf.prediction_log plus the /predict wiring that feeds it.

Same fixture shape as test_api.py: a fake bundle injected through app.dependency_overrides, and
NO `with` on TestClient, so the real MLflow registry load never runs. The prediction log is
injected the same way, pointed at tmp_path -- these tests write real SQLite files, because the
thing under test is whether a row actually lands.

Two properties matter more than the rest and each has a test that fails loudly if it regresses:

  * metadata is read from the image AS UPLOADED, before /predict's .convert("RGB"). Read it
    after and every row says mode="RGB" forever, which silently kills the mode drift signal --
    the one a grayscale-degraded replay is supposed to trip.
  * a broken log never breaks a prediction. Monitoring observes the serving path; it does not
    get to fail it.
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_bundle, get_prediction_log, ModelBundle
from coin_clf.prediction_log import PredictionLog, PredictionRecord, utc_now_iso

STUB_VERSION = "3"
FAKE_LABELS = {0: "NERO", 1: "CLAUDIUS", 2: "COMMODUS"}


class _FakeModel:
    """Fixed logits sized to FAKE_LABELS -- deterministic argmax, no weights, no S3."""

    def __call__(self, x):
        import torch

        logits = torch.zeros(x.shape[0], len(FAKE_LABELS))
        logits[:, 0] = 10.0  # argmax -> class 0 -> "NERO"
        return logits


class _BrokenLog:
    """A log whose every write blows up -- stands in for a full disk / unwritable mount."""

    def log(self, record):
        raise RuntimeError("disk on fire")


@pytest.fixture
def log(tmp_path):
    return PredictionLog(tmp_path / "predictions.db")


def make_client(log):
    fake = ModelBundle(model=_FakeModel(), version=STUB_VERSION, idx_to_name=FAKE_LABELS)
    app.dependency_overrides[get_bundle] = lambda: fake
    if log is not None:
        app.dependency_overrides[get_prediction_log] = lambda: log
    return TestClient(app)  # NO `with` -> lifespan (the real load) never runs


@pytest.fixture
def client(log):
    c = make_client(log)
    yield c
    app.dependency_overrides.clear()


def _png_bytes(size=(224, 224), mode="RGB", color=(128, 128, 128)):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    buf.seek(0)
    return buf


def post_image(client, image=None, **kwargs):
    files = {"file": ("coin.png", image or _png_bytes(), "image/png")}
    return client.post("/predict", files=files, **kwargs)


def only_row(log):
    rows = log.read_records()
    assert len(rows) == 1, f"expected exactly one logged row, got {len(rows)}"
    return rows[0]


# --- a prediction writes a row -------------------------------------------------

def test_predict_writes_a_row(client, log):
    r = post_image(client)
    assert r.status_code == 200
    body = r.json()

    row = only_row(log)
    assert row["model_version"] == body["model_version"] == STUB_VERSION
    assert row["predicted_label"] == body["predictions"][0]["label"] == "NERO"
    assert row["confidence"] == pytest.approx(body["predictions"][0]["probability"])
    assert row["ts"]                      # timestamped
    assert row["latency_ms"] > 0          # measured, not a placeholder


def test_each_request_writes_exactly_one_row(client, log):
    for _ in range(3):
        assert post_image(client).status_code == 200
    assert len(log.read_records()) == 3   # one row per REQUEST, not per top-k prediction


# --- metadata is the uploaded image's, not the preprocessed tensor's -----------

def test_logged_mode_is_pre_convert(client, log):
    """The signal a grayscale-degraded replay is meant to trip. If /predict logs metadata after
    .convert("RGB"), this row says RGB and mode drift can never fire again.
    """
    r = post_image(client, image=_png_bytes(mode="L", color=128))
    assert r.status_code == 200
    assert only_row(log)["mode"] == "L"


def test_logged_dimensions_are_the_uploaded_ones(client, log):
    """224x224 in this row would mean we logged the preprocessed tensor's shape -- constant for
    every request, so size drift would be dead the same way mode drift would.
    """
    r = post_image(client, image=_png_bytes(size=(300, 150)))
    assert r.status_code == 200
    row = only_row(log)
    assert (row["width"], row["height"]) == (300, 150)


# --- traffic tagging -----------------------------------------------------------

def test_untagged_request_logs_null_source(client, log):
    """Real traffic sends no tag and must be unaffected by tagging existing at all."""
    assert post_image(client).status_code == 200
    assert only_row(log)["source"] is None


def test_tagged_request_records_the_tag(client, log):
    r = post_image(client, headers={"X-Traffic-Source": "skewed-replay"})
    assert r.status_code == 200
    assert only_row(log)["source"] == "skewed-replay"


def test_blank_tag_is_stored_as_null(client, log):
    """An empty header is absence of a tag, not a tag whose name is the empty string."""
    assert post_image(client, headers={"X-Traffic-Source": "   "}).status_code == 200
    assert only_row(log)["source"] is None


# --- a broken log never breaks a prediction ------------------------------------

def test_logging_failure_does_not_break_predict():
    """The whole contract in one test: the log raises, the caller still gets its prediction."""
    try:
        client = make_client(_BrokenLog())
        r = post_image(client)
        assert r.status_code == 200
        body = r.json()
        assert body["model_version"] == STUB_VERSION
        assert body["predictions"][0]["label"] == "NERO"   # response is intact, not degraded
    finally:
        app.dependency_overrides.clear()


def test_predict_works_with_no_log_configured():
    """get_prediction_log resolves to None when the app's lifespan never ran. That must be a
    no-op, not an AttributeError -- a dependency that raises is a 500 the endpoint's own guard
    would never get to catch.
    """
    try:
        client = make_client(None)
        assert post_image(client).status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_unwritable_path_returns_false_instead_of_raising(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")
    log = PredictionLog(blocker / "predictions.db")  # parent mkdir cannot succeed

    ok = log.log(PredictionRecord(ts=utc_now_iso(), model_version="3",
                                  predicted_label="NERO", confidence=0.9))
    assert ok is False


# --- the read path the drift check consumes ------------------------------------

def test_read_records_roundtrip(log):
    for i in range(5):
        assert log.log(PredictionRecord(
            ts=utc_now_iso(), model_version="3", predicted_label="NERO",
            confidence=0.5 + i / 100, width=224, height=224, mode="RGB",
            latency_ms=12.5, source=None,
        )) is True

    rows = log.read_records()
    assert len(rows) == 5
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)   # oldest-first
    assert rows[0]["confidence"] == pytest.approx(0.5)
    assert isinstance(rows[0]["width"], int)
    assert rows[0]["source"] is None


def test_read_records_limit_returns_most_recent_oldest_first(log):
    for label in ["NERO", "CLAUDIUS", "COMMODUS"]:
        log.log(PredictionRecord(ts=utc_now_iso(), model_version="3",
                                 predicted_label=label, confidence=0.9))

    rows = log.read_records(limit=2)
    assert [r["predicted_label"] for r in rows] == ["CLAUDIUS", "COMMODUS"]


def test_read_records_since_filters_the_time_window(log):
    log.log(PredictionRecord(ts="2020-01-01T00:00:00+00:00", model_version="3",
                             predicted_label="NERO", confidence=0.9))
    cutoff = utc_now_iso()
    log.log(PredictionRecord(ts=cutoff, model_version="3",
                             predicted_label="COMMODUS", confidence=0.9))

    rows = log.read_records(since=cutoff)
    assert [r["predicted_label"] for r in rows] == ["COMMODUS"]


def test_read_records_raises_on_missing_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        PredictionLog(tmp_path / "never_written.db").read_records()
