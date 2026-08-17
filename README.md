# Roman Coin Classifier

Classifies photos of Roman Imperial coins by emperor (51 classes, GORDIAN II
merged into GORDIAN I). Built as a series of notebooks that iterate on feature
extraction, head architecture, and ensembling, then compressed into a small
deployable model via knowledge distillation.

## Model history

| Notebook | Approach | Test acc |
|---|---|---|
| `emp_model_v2.ipynb` | Frozen C-RADIO v4-H + linear head | 84.99% |
| `emp_model_v3.ipynb` | + MLP/cosine head, mixup, class-balanced sampling, 5-head ensemble | 87.51% |
| `emp_model_v3_TTA.ipynb` | + TTA feature extraction, 10-head ensemble | 88.82% |
| `emp_model_v4.ipynb` | + ArcFace, patch tokens, hierarchical classifier (frozen backbone) | 88.69% |
| `emp_model_v4.1.ipynb` | + fine-tuned C-RADIO backbone | 92.05% |
| `emp_model_v5.ipynb` | Multi-stream (portrait/legend crops + DINOv2) — regressed | 91.73% |
| `emp_model_v6.ipynb` | + per-stream projection, hard-pair sub-classifiers | **92.71% (best)** |
| `emp_model_v7.ipynb` | Frozen DINOv2-only baseline (phase 1, no fine-tuning follow-up) | 85.78% |
| `emp_model_knowledge_distilation.ipynb` | MobileNetV3-Large distilled from the v6 ensemble | 89.61% |
| `emp_model_mobilenet_baseline.ipynb` | Same MobileNetV3, plain CE (no distillation) — comparison | 89.02% |

Full write-up of what worked and what didn't is in `improvement_or_not.md`
(kept locally, not pushed).

v6 has the best raw accuracy but is an ensemble + kNN + sub-classifier
pipeline that's impractical to serve. The **distilled MobileNetV3-Large**
(`emp_model_distil_student.pth`) trades ~3pp of accuracy for a single
forward pass over a 5.4M-param model, and is what the serving app below
loads.

## Serving

`app/main.py` is a FastAPI app that loads `emp_model_distil_student.pth` and
exposes:

- `GET /health` — status, device, class count, checkpoint name
- `POST /predict?topk=N` — multipart image upload → top-k `{label, probability}`

Preprocessing matches training: `Resize(256) -> CenterCrop(224) -> ImageNet normalize`.
Labels are read from `app/coin_labels.json` (index → emperor name).

### Run with Docker

```bash
docker build -f serve.Dockerfile -t coin-classifier .
docker run -d -p 8000:8000 --name coin-classifier coin-classifier
```

```bash
curl http://localhost:8000/health

curl -X POST "http://localhost:8000/predict?topk=3" \
  -F "file=@/path/to/coin.jpg;type=image/jpeg"
```

The image is CPU-only (`python:3.10-slim` + CPU torch/torchvision wheels) for
portability — inference on this model size is fast enough without a GPU. For
GPU serving, swap the base image for an `nvidia/cuda` runtime and install the
`cu121` torch wheels in `serve.Dockerfile`.

### Run locally without Docker

Requires Python 3.10 with `torch`, `torchvision`, and the packages in
`requirements-serve.txt`:

```bash
pip install -r requirements-serve.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Other scripts

- `train.py` / `prepare.py` — training entrypoint and fixed data/model infra (embeddings, splits, `CoinHead`).
- `predict.py` — batch CLI inference over a directory of images using a checkpoint.
- `gradcam.py` — Grad-CAM visualization for the distilled student, to sanity-check what pixels drive a prediction.
- `download.py` — fetches the training dataset.

## Data & weights

The training dataset (`FOR_TRAINNING/`) and all `.pth`/`.onnx` checkpoints are
gitignored — too large for the repo. Only notebooks, source files, and this
README are tracked.
