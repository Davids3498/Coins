# Serving

[← back to the README](../README.md)

The FastAPI app: how it loads a model, what the two endpoints return, every environment variable, and how to run it with and without Docker. The cloud variant of this container is covered separately in [cloud-deploy.md](cloud-deploy.md).

`app/main.py` loads the model in a FastAPI **lifespan** (startup), not at import.
That is what lets the module be imported with no registry reachable, which is how
the tests inject a fake bundle through `app.dependency_overrides` and how CI's
`--network none` import smoke test works.

By default it loads from the MLflow registry — `models:/coin-classifier@champion`,
the full `nn.Module`, not a state dict. `MODEL_SOURCE=local` instead loads a
champion baked into the image at build time, which is what the Fargate deployment
runs and the only thing that differs about it; see [cloud-deploy.md](cloud-deploy.md).
Both paths report the same registry version on `/health`.

- `GET /health` — `status`, `device`, `num_classes`, `model_name`, `model_version`, `model_source`
- `POST /predict?topk=N` — multipart image upload → `{model_version, predictions: [{label, probability}]}`
  - optional `X-Traffic-Source` header: a monitoring tag (the replay script sets
    it to separate a normal from a deliberately skewed batch). It changes nothing
    about the prediction, only how the logged row is grouped later.

Preprocessing is `coin_clf.transforms.val_transform`, the same object training
uses: `Resize(256) -> CenterCrop(224) -> ImageNet normalize`. Labels come from
`app/coin_labels.json` (index → emperor name), overridable with `LABELS_PATH`.

Every request writes one row — version, label, confidence, width/height/mode,
latency, source — to the SQLite prediction log (`coin_clf.prediction_log`). That
log is the only thing observing this model in production and the sole input to the
drift check. It is strictly an observer: `log()` never raises, `/predict` wraps it
anyway, and it is the last thing the endpoint does. Image metadata is read from
the upload **before** `.convert("RGB")`, or every row would report `mode="RGB"`
forever and the mode drift signal would be permanently dead.

Configuration, all environment variables:
`MODEL_SOURCE=registry` (`registry` | `local`), `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`,
`MODEL_NAME=coin-classifier`, `MODEL_ALIAS=champion`, `BAKED_MODEL_DIR=<repo>/model`
(read only when `MODEL_SOURCE=local`), `LABELS_PATH=app/coin_labels.json`,
`PREDICTION_LOG_PATH=<repo>/outputs/monitoring/predictions.db`.

The container needs the tracking server (for the alias) and its artifact store
(for the weights), so `make serve` runs it on the host network with `~/.aws`
mounted read-only and the prediction-log directory bind-mounted — without that
mount the log dies with the `--rm` container and the host-side drift check has
nothing to read.

<details>
<summary>The equivalent raw <code>docker run</code></summary>

```bash
docker run --rm --network host \
  -v ~/.aws:/root/.aws:ro \
  -v "$PWD/outputs/monitoring:/srv/outputs/monitoring" \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e PREDICTION_LOG_PATH=/srv/outputs/monitoring/predictions.db \
  coin-classifier:latest
```

</details>

The image is CPU-only (`python:3.10-slim` + CPU torch/torchvision wheels): the
model is 4.27M parameters with the 51-class head, so single-image inference does
not need a GPU. For GPU serving, swap the base image for an `nvidia/cuda` runtime
and install the `cu121` wheels in `serve.Dockerfile`.

To run it without Docker, you need Python 3.10 with `torch`, `torchvision`, the
packages in `requirements-serve.txt`, and a reachable tracking server:

```bash
pip install -r requirements-serve.txt && pip install -e . --no-deps
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
