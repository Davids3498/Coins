# Coin classifier — project control panel.
# Override any variable on the command line, e.g. `make serve PORT=9000`.

BUCKET       ?= davids-mlops-artifacts-8412
AWS_REGION   ?= us-east-1
IMAGE        ?= coin-classifier:latest
MLFLOW_HOST  ?= 127.0.0.1
MLFLOW_PORT  ?= 5000
SERVE_PORT   ?= 8000
AIRFLOW_PORT ?= 8090
PY           ?= /usr/bin/python3

# Absolute path to the registry DB, derived from the repo root so it never
# depends on the directory you run `make` from. `sqlite:///` + an absolute
# /path yields the required four-slash form (sqlite:////home/.../mlflow.db).
MLFLOW_DB    := $(CURDIR)/mlflow/mlflow.db
TRACKING_URI := http://$(MLFLOW_HOST):$(MLFLOW_PORT)

# Host side of the serving container's prediction-log mount, and where the drift check looks.
MONITORING_DIR := $(CURDIR)/outputs/monitoring

.DEFAULT_GOAL := help
.PHONY: help mlflow airflow build serve train retrain install champion health \
        monitoring-install drift

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

mlflow:  ## start the MLflow tracking server (SQLite backend + S3 artifacts)
	@mkdir -p $(dir $(MLFLOW_DB))
	mlflow server \
	  --backend-store-uri sqlite:///$(MLFLOW_DB) \
	  --default-artifact-root s3://$(BUCKET)/mlflow \
	  --no-serve-artifacts \
	  --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

airflow:  ## start Airflow (webserver + scheduler + triggerer) — UI on AIRFLOW_PORT
	@bash -c 'source $(CURDIR)/.airflow_env.sh && \
	  AIRFLOW__WEBSERVER__WEB_SERVER_PORT=$(AIRFLOW_PORT) exec airflow standalone'

build:  ## build the serving image
	docker build -f serve.Dockerfile -t $(IMAGE) .

# The monitoring mount is not optional: the container runs --rm, so a prediction log written to
# the container filesystem is deleted the moment serving stops, and the host-side drift check
# would have nothing to read. Mounting it makes the log outlive the container AND land on the
# same path the drift check reads.
serve:  ## run the serving container (needs `make mlflow` running in another terminal)
	@mkdir -p $(MONITORING_DIR)
	docker run --rm --network host \
	  -v $(HOME)/.aws:/root/.aws:ro \
	  -v $(MONITORING_DIR):/srv/outputs/monitoring \
	  -e AWS_DEFAULT_REGION=$(AWS_REGION) \
	  -e PREDICTION_LOG_PATH=/srv/outputs/monitoring/predictions.db \
	  $(IMAGE)

train:  ## run a training run — pass ARGS="--epochs 1" for a smoke run
	PYTHONPATH=src $(PY) train.py $(ARGS)

# The same script, same flags and same interpreter the retrain_coin_clf DAG's train task shells
# out to — so a run reproduced by hand is the run Airflow would have made, not a near-miss.
# Registers a challenger; @champion is only moved by promote.py.
retrain:  ## run the DAG's training recipe (train_hard_labels.py) — needs `make mlflow`; ARGS="--epochs 1" for a smoke run
	PYTHONPATH=src $(PY) train_hard_labels.py $(ARGS)

install:  ## editable install of coin_clf WITHOUT touching your CUDA torch
	$(PY) -m pip install -e . --no-deps

champion:  ## print the version currently aliased @champion
	@$(PY) -c "import mlflow; mlflow.set_tracking_uri('$(TRACKING_URI)'); \
	print('coin-classifier @champion -> v' + mlflow.MlflowClient().get_model_version_by_alias('coin-classifier','champion').version)"

monitoring-install:  ## install evidently for the offline drift check (NOT in the serving image)
	$(PY) -m pip install --user -r requirements-monitoring.txt

# Needs `make mlflow` running: the champion version is resolved on every run so a reference
# cached for a previous champion is rebuilt rather than silently compared against.
# ARGS="--source skewed-replay" to scope the check to one replay run.
drift:  ## run the drift check -> HTML report + machine-readable verdict in outputs/monitoring
	PYTHONPATH=src $(PY) drift_report.py --data-dir data/FOR_TRAINNING \
	  --manifest data/splits_manifest.json $(ARGS)

health:  ## curl the serving container's /health
	@curl -s http://localhost:$(SERVE_PORT)/health | $(PY) -m json.tool