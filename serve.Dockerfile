# Serves whatever the MLflow registry currently aliases @champion
# (models:/coin-classifier@champion) behind a FastAPI app with /predict and /health.
#
# NO CHECKPOINT IS BAKED IN. Nothing under weights/ is COPYed here on purpose: the model
# is resolved by alias at startup, inside app/main.py's lifespan, so a promotion changes
# what this image serves without rebuilding it. Two consequences worth knowing before you
# run it:
#   * The container needs a reachable tracking server AND its artifact store at RUNTIME --
#     the alias lookup and the weights download both happen on startup. `make serve` wires
#     both (host network + ~/.aws mounted read-only); a bare `docker run` on the default
#     bridge network cannot reach a tracking server on the host's 127.0.0.1 and will fail
#     to start.
#   * The alias is resolved ONCE. After a promotion this container keeps serving -- and
#     logging -- the previous version until it is restarted.
#
# CPU-only image by default: the model is MobileNetV3-Large, 4.27M params with the
# 51-class head, so single-image inference is fast enough without a GPU and the image
# stays small/portable. For GPU inference, switch the base image to an nvidia/cuda runtime
# image and install torch from https://download.pytorch.org/whl/cu121 instead.

FROM python:3.10-slim

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-serve.txt

# coin_clf: shared model/transform code, installed before app/ so `import coin_clf`
# works at startup. --no-deps: torch/torchvision are already pinned above (CPU
# wheels); letting pip re-resolve coin_clf's deps from PyPI could pull in GPU
# wheels or drift the pinned versions.
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

COPY app/ app/

ENV LABELS_PATH=/srv/app/coin_labels.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
