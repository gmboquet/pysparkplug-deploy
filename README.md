# mixle-mlops

A container + Kubernetes serving layer for **[mixle](https://github.com/gmboquet/mixle)**
models: a thin FastAPI server over `mixle.inference.ModelService`, seed / drift-retrain helpers, a
Dockerfile, and k8s manifests.

This lives **outside** the core library on purpose — mixle ships the model primitives
(`ModelRegistry`, `ModelService`, drift detection, provenance); the deployment opinions (HTTP, Docker,
Kubernetes) live here. It's for the **classical / probabilistic models mixle builds** (CPU, one-shot
scoring, retrain-on-drift) — not an LLM serving stack.

## Install

```sh
# until mixle is on PyPI, the core comes from git:
pip install "mixle @ git+https://github.com/gmboquet/mixle.git"
pip install "mixle-mlops @ git+https://github.com/gmboquet/mixle-mlops.git"
# or, for local dev:  pip install -e ".[dev]"
```

This installs three console scripts: **`mixle-serve`** (the API), **`mixle-seed`** (register+promote a model),
**`mixle-drift-retrain`** (drift check → retrain → swap).

## Run locally

```sh
export MIXLE_REGISTRY_ROOT=./models MIXLE_MODEL_NAME=model
mixle-seed                                    # train + register + promote("production")
export MIXLE_REFERENCE_PATH=./models/model/reference.json
mixle-serve                                   # uvicorn on :8000
```

```sh
curl localhost:8000/health
curl -X POST localhost:8000/score -H 'content-type: application/json' -d '{"records":[1.0,2.5,9.9]}'
curl -X POST localhost:8000/drift -H 'content-type: application/json' -d '{"records":[6.0,6.2,5.8]}'
curl localhost:8000/info                      # provenance header
```

| endpoint | role |
|---|---|
| `POST /score` | per-record log-density (non-finite → `null`, with `n_unscorable`) |
| `GET /health` | k8s liveness/readiness probe + activity summary |
| `GET /info` | the served model's provenance header |
| `POST /drift` | drift report vs the reference sample (needs `MIXLE_REFERENCE_PATH`) |
| `POST /reload` | hot model swap after `registry.promote(...)` |

Config is by env var: `MIXLE_REGISTRY_ROOT`, `MIXLE_MODEL_NAME`, `MIXLE_MODEL_ALIAS`, `MIXLE_REFERENCE_PATH`,
`MIXLE_ACTIVITY_LOG` (e.g. `/dev/stdout`).

## Kubernetes

```sh
docker build -t <registry>/mixle-mlops:latest . && docker push <registry>/mixle-mlops:latest
# set the image in k8s/*.yaml, then:
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/seed-job.yaml          # one-shot: populate the registry
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl apply -f k8s/drift-retrain-cronjob.yaml
```

Stateless replicas load the `production` alias from a shared `ModelRegistry`; a model swap is
`registry.promote(...)` (done by `mixle-drift-retrain`) then `kubectl rollout restart deployment/mixle-model`
(or `POST /reload`).

## Caveats (harden for real use)

- **Registry storage** is filesystem-backed; the manifests use a `ReadWriteMany` PVC. No RWX class? Back it
  with object storage (S3/GCS via a CSI mount, or adapt `ModelRegistry`).
- **Logging**: set `MIXLE_ACTIVITY_LOG=/dev/stdout` so the per-request activity log lands in container logs.
- **Record shape**: `/score` JSON arrays map to model records (inner arrays → tuples for composite models).
- **Auth / rate limiting / TLS**: add at the ingress; none is included here.
- **Estimator**: `mixle-seed` / `mixle-drift-retrain` use a Gaussian example — swap in your real model +
  estimator and wire `drift_retrain._recent_batch()` to your production data store.
