# Serves emp_model_distil_student.pth (MobileNetV3-Large, distilled coin classifier)
# behind a FastAPI app with /predict and /health.
#
# CPU-only image by default (the student model is 5.4M params — CPU inference is
# fast enough for single-image requests and this keeps the image small/portable).
# For GPU inference, switch the base image to an nvidia/cuda runtime image and
# install torch from https://download.pytorch.org/whl/cu121 instead.

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

COPY app/ app/
COPY emp_model_distil_student.pth .

ENV CHECKPOINT_PATH=/srv/emp_model_distil_student.pth
ENV LABELS_PATH=/srv/app/coin_labels.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
