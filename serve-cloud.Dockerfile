# The cloud variant of serve.Dockerfile: same app, same code, model FROZEN INTO THE IMAGE.
#
# WHY A SECOND DOCKERFILE. serve.Dockerfile bakes in no checkpoint on purpose -- it resolves
# models:/coin-classifier@champion at startup, so a promotion changes what it serves without a
# rebuild. That requires a reachable tracking server AND its artifact store at runtime. An ECS
# Fargate task in a public subnet has neither: the tracking server is a local process on the
# developer's machine, backed by SQLite. The as-built serving image CANNOT START on Fargate.
#
# So the cloud image trades the alias indirection for portability. That is a real loss, and the
# mitigation is provenance: scripts/export_champion.py writes champion.json recording which
# registry version was frozen, and app/main.py refuses to start without it and reports it on
# /health. A cloud container that cannot name the champion it serves is an untraceable binary,
# and the whole point of the deploy is proving the cloud serves the SAME version as local.
#
# The proper fix is not a third Dockerfile -- it is a reachable tracking server, at which point
# the cloud task uses serve.Dockerfile and the alias stays the single source of truth in both
# places. See the README's "Cloud deploy" section.
#
# BUILD (the export must exist first -- it is gitignored, 24 MB of weights):
#   python scripts/export_champion.py                     # needs MLflow + S3, writes build/champion
#   docker build -f serve.Dockerfile -t coin-classifier:latest .
#   docker build -f serve-cloud.Dockerfile -t coin-classifier:cloud .
#
# Deliberately FROM the serving image rather than repeating its layers: the cloud image must be
# the local image plus a model, never a parallel build that can drift from it. If the base tag
# is stale, the cloud serves stale code -- build both, in that order.

ARG BASE_IMAGE=coin-classifier:latest
FROM ${BASE_IMAGE}

# build/champion/{champion.json, artifact/} -> /srv/model/{champion.json, artifact/}
# app/main.py hardcodes the artifact/ and champion.json names under BAKED_MODEL_DIR.
COPY build/champion/ /srv/model/

# The only behavioural difference between this image and its base. MODEL_SOURCE=local sends
# build_bundle() down load_model_from_baked(), which touches no network -- so the task starts
# with no tracking server, no S3, and no NAT gateway to reach either.
ENV MODEL_SOURCE=local \
    BAKED_MODEL_DIR=/srv/model

# No PREDICTION_LOG_PATH override and no volume: on Fargate the prediction log has nowhere
# durable to live, so it writes inside the container and dies with the task. That is acceptable
# ONLY because this deployment is a proving run, not the monitored one -- the drift check reads
# the log from `make serve`, which mounts outputs/monitoring from the host. A real cloud
# deployment needs the log on a mounted volume or shipped out; see the README's limitations.

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
