# Universal cloud compute

mixle-mlops keeps **state** and **compute** separate:

- State plane: database, object store, Redis. The existing cloud profiles provision this for AWS, Azure, GCP, and
  Alibaba Cloud.
- Compute plane: the gateway talks to any OpenAI-compatible model server. That can be Ollama, vLLM, llama.cpp, TGI, a
  hosted API, a GPU VM, a rented marketplace instance, managed Kubernetes, or an on-prem box.

This is the portability boundary. To support another compute provider, you normally do **not** need provider-specific
code; you need a reachable `/v1` model endpoint.

## Option A: External model compute

Run the model server wherever the GPUs are, then point the gateway at it:

```bash
MIXLE_LLM_BASE_URL=https://your-gpu-host.example.com/v1
MIXLE_LLM_API_KEY=change-me
MIXLE_LLM_MODELS='["qwen2.5-72b"]'
```

For several compute pools or different upstream model ids, use `MIXLE_LLM_BACKENDS`:

```bash
MIXLE_LLM_BACKENDS='{
  "fast-local": {
    "base_url": "https://gpu-a.example.com/v1",
    "api_key": "change-me",
    "upstream_model": "llama3.2"
  },
  "large-rented": {
    "base_url": "https://gpu-b.example.com/v1",
    "api_key": "change-me",
    "upstream_model": "Qwen/Qwen2.5-72B-Instruct"
  }
}'
```

The backend should provide:

- `POST /v1/chat/completions`
- `GET /v1/models` when model discovery is desired
- Bearer-token auth or network-level protection if it is exposed outside a private network

This covers Vast.ai, RunPod, Lambda Cloud, Paperspace/CoreWeave-style GPU boxes, managed vLLM endpoints, and private
clusters without adding a provider adapter to mixle-mlops.

## Option B: Single GPU host

Use this when the rented machine runs both the gateway and the local model server.

```bash
cp deploy/.env.example deploy/.env
# edit deploy/.env: set MIXLE_SECRET_KEY and model ids
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.gpu.yml \
  --env-file deploy/.env up -d gateway ollama
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.gpu.yml \
  --env-file deploy/.env exec ollama ollama pull llama3.2
```

The GPU override is intentionally generic NVIDIA Compose wiring. The host still needs a working NVIDIA driver and
container runtime.

For ephemeral or preemptible compute, avoid local-only state for production. Put the database, object store, and Redis
on managed services or an external durable host, then use the GPU node only for inference.

## Option C: Kubernetes GPU compute

The Helm chart can either point to an external model endpoint:

```bash
helm upgrade --install mixle deploy/helm/mixle-mlops \
  --set llm.baseUrl=https://your-model-server.example.com/v1 \
  --set llm.apiKey=change-me \
  --set llm.models='["qwen2.5-72b"]'
```

For multiple compute providers through Helm, set `llm.backends` to the same JSON object used by
`MIXLE_LLM_BACKENDS`.

Or run the demo Ollama backend inside the cluster and schedule it onto GPU nodes:

```bash
helm upgrade --install mixle deploy/helm/mixle-mlops \
  -f deploy/helm/mixle-mlops/values-compute-gpu.yaml
```

Edit `values-compute-gpu.yaml` for your cluster's node labels, tolerations, storage class, and GPU resource name. Most
NVIDIA clusters use `nvidia.com/gpu`; some managed providers use custom labels for node selection.

## Provider checklist

For any compute provider, verify:

- Docker or Kubernetes can access the GPU.
- The model server binds on a network address the gateway can reach.
- The endpoint path includes `/v1`, for example `http://host:8000/v1`.
- `MIXLE_LLM_MODELS` uses the ids exposed by the backend, or `MIXLE_LLM_BACKENDS` maps Mixle-facing ids to
  `upstream_model`.
- Durable state is outside ephemeral GPU instances for production.
