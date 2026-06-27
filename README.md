# pysparkplug-deploy

A container + Kubernetes serving layer for **[pysparkplug](https://github.com/gmboquet/pysparkplug)**
models: a thin FastAPI server over `pysp.inference.ModelService`, seed / drift-retrain helpers, a
Dockerfile, and k8s manifests.

This lives **outside** the core library on purpose — pysparkplug ships the model primitives
(`ModelRegistry`, `ModelService`, drift detection, provenance); the deployment opinions (HTTP, Docker,
Kubernetes) live here. It's for the **classical / probabilistic models pysparkplug builds** (CPU, one-shot
scoring, retrain-on-drift) — not an LLM serving stack.

## Install

```sh
# until pysp-learn is on PyPI, the core comes from git:
pip install "pysp-learn @ git+https://github.com/gmboquet/pysparkplug.git"
pip install "pysparkplug-deploy @ git+https://github.com/gmboquet/pysparkplug-deploy.git"
# or, for local dev:  pip install -e ".[dev]"
```

This installs three console scripts: **`pysp-serve`** (the API), **`pysp-seed`** (register+promote a model),
**`pysp-drift-retrain`** (drift check → retrain → swap).

## Run locally

```sh
export PYSP_REGISTRY_ROOT=./models PYSP_MODEL_NAME=model
pysp-seed                                    # train + register + promote("production")
export PYSP_REFERENCE_PATH=./models/model/reference.json
pysp-serve                                   # uvicorn on :8000
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
| `POST /drift` | drift report vs the reference sample (needs `PYSP_REFERENCE_PATH`) |
| `POST /reload` | hot model swap after `registry.promote(...)` |

Config is by env var: `PYSP_REGISTRY_ROOT`, `PYSP_MODEL_NAME`, `PYSP_MODEL_ALIAS`, `PYSP_REFERENCE_PATH`,
`PYSP_ACTIVITY_LOG` (e.g. `/dev/stdout`).

## Kubernetes

```sh
docker build -t <registry>/pysparkplug-deploy:latest . && docker push <registry>/pysparkplug-deploy:latest
# set the image in k8s/*.yaml, then:
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/seed-job.yaml          # one-shot: populate the registry
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl apply -f k8s/drift-retrain-cronjob.yaml
```

Stateless replicas load the `production` alias from a shared `ModelRegistry`; a model swap is
`registry.promote(...)` (done by `pysp-drift-retrain`) then `kubectl rollout restart deployment/pysp-model`
(or `POST /reload`).

## Caveats (harden for real use)

- **Registry storage** is filesystem-backed; the manifests use a `ReadWriteMany` PVC. No RWX class? Back it
  with object storage (S3/GCS via a CSI mount, or adapt `ModelRegistry`).
- **Logging**: set `PYSP_ACTIVITY_LOG=/dev/stdout` so the per-request activity log lands in container logs.
- **Record shape**: `/score` JSON arrays map to model records (inner arrays → tuples for composite models).
- **Auth / rate limiting / TLS**: add at the ingress; none is included here.
- **Estimator**: `pysp-seed` / `pysp-drift-retrain` use a Gaussian example — swap in your real model +
  estimator and wire `drift_retrain._recent_batch()` to your production data store.
