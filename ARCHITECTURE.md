# mixle-mlops — an all-in-one AI platform

mixle-mlops deploys and operates AI products. It hosts **mixle's own probabilistic models** *and* **open LLMs**
(Llama, DeepSeek, …) behind one gateway, lets mixle **compose** them, and ships the full product surface around
them — accounts, API keys, a chat UI + landing page, multimodal I/O, an MCP server, and a principled
**RLHF / human-feedback loop**. Its differentiator is the **mixle core**: every served model can speak
*distributions and decisions* (calibrated intervals, tail probabilities, abstention, Bayes-optimal actions),
and the feedback loop is a *real, calibrated, actively-elicited preference model* — not thumbs-up logging.

## The thesis

Generic AI-serving platforms (KServe, Seldon, BentoML, vLLM, Ollama, OpenWebUI, LibreChat) serve a function
`x → ŷ` and stop there. mixle-mlops adds the two things mixle uniquely makes possible:

1. **Serve distributions + decisions, not just tokens/points.** A mixle model exposes a calibrated predictive
   distribution and, given a loss, the Bayes-optimal action with a tail-risk profile; the platform monitors
   PIT/coverage/proper-scores and the model's own log-density drift, and abstains out-of-support.
2. **A principled feedback loop.** RLHF here = fit a mixle preference/reward model (Bradley–Terry /
   Plackett–Luce / Mallows) with **calibrated uncertainty** over collected human comparisons, **actively elicit**
   the most informative next comparison (mixle DoE), and close the loop. That works for *any* hosted model —
   including LLMs — and is the thing OpenWebUI/LibreChat/etc. do not have.

Everything else (auth, keys, chat UI, multimodal, MCP, k8s) is the table-stakes product surface that makes it
shippable as something that "looks like Claude/ChatGPT."

## Layered architecture

```
                ┌─────────────────────────────── Frontend (Next.js) ───────────────────────────────┐
                │  landing page   ·   chat UI (streaming, history, multimodal upload, 👍/👎/edit)     │
                │  account/login  ·   API-key management   ·   model picker   ·   usage dashboard     │
                └───────────────────────────────────────┬──────────────────────────────────────────┘
                                                         │  HTTPS (OpenAI-compatible + platform API)
  ┌──────────────────────────────────────── Gateway (FastAPI) ───────────────────────────────────────┐
  │  auth & API keys · rate limit / usage metering · request validation · SSE streaming · multimodal   │
  │  routes:  /v1/chat/completions  /v1/completions  /v1/models  /v1/embeddings   (OpenAI-compatible)   │
  │           /predict /decide /score /calibration /latent /capabilities          (mixle distribution)  │
  │           /feedback  /rlhf/*                                                   (human-feedback loop) │
  │           /accounts/* /keys/*                                                  (identity)            │
  └───────┬───────────────────────┬──────────────────────────┬──────────────────────┬────────────────┘
          │                       │                          │                       │
   ┌──────▼──────┐        ┌───────▼────────┐        ┌────────▼─────────┐     ┌───────▼────────┐
   │ Model layer │        │  mixle core    │        │  Feedback / RLHF │     │   MCP server   │
   │  adapters:  │        │ (differentiator)│       │  collect → reward │     │ expose models  │
   │  · Mixle    │        │ · predictive    │       │  (Bradley-Terry,  │     │ + tools as MCP;│
   │  · OpenAI-  │        │   intervals/CDF │       │   Plackett-Luce)  │     │ models can call│
   │    compat   │        │ · calibration & │       │  · active elicit  │     │ MCP tools too  │
   │    (vLLM/   │        │   coverage      │       │   (mixle.doe)     │     └────────────────┘
   │    Ollama/  │        │ · proper-score  │       │  · close-the-loop │
   │    llama.cpp│        │   champ/chall   │       │   (promote/DPO)   │
   │    /OpenAI) │        │ · Bayes decision│       └──────────────────┘
   │  · Composite│        │ · density OOD/  │
   │   (mix      │        │   abstention    │
   │   mixle+LLM)│        │ · drift monitor │
   └──────┬──────┘        └────────┬────────┘
          │                        │
  ┌───────▼────────────────────────▼─────────────────────────────────────────────────────────────────┐
  │ Storage (pluggable: LOCAL = SQLite + filesystem   |   CLOUD = Postgres + S3/object store)           │
  │  model registry & versions · accounts/orgs/keys/usage · conversations · feedback/preferences · blobs│
  └────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Deploy:  Docker Compose (dev: gateway + frontend + db + Ollama)   ·   Kubernetes manifests (prod)
```

## Components (Python package `mixle_mlops/`)

- `config.py` — env-driven settings; `deployment=local|cloud` switches storage/auth backends.
- `core/` — the model contract and the mixle differentiator:
  - `adapters.py` — `ModelAdapter` ABC + `ChatMessage`/`ChatResponse`/`Prediction`/`Decision` types. Every
    backend implements the same interface (`chat`, `complete`, `embed`, and — when supported — `predict`,
    `decide`, `score`, `latent`, `capabilities`).
  - `registry.py` — `ModelRegistry`: the catalog of hosted models (mixle artifacts + configured LLM backends),
    versioning, provenance; lists to `/v1/models`.
  - `predictive.py` — predictive adapter over a mixle `Distribution` (intervals, quantiles, ensemble, CDF,
    tail-probability) feeding calibration/decision. *(serving-core design P0-A — upstreamed to mixle later)*
  - `decision.py` — `bayes_action(posterior, loss, actions)` → action + expected loss + risk profile. *(P1-A)*
  - `capability.py` — route a query to the right method via `mixle.capability.supports`, 422 otherwise.
- `models/` — adapters: `mixle_model.py` (`MixleAdapter`), `openai_compat.py` (`OpenAICompatAdapter` → any
  OpenAI-compatible server: vLLM, Ollama, llama.cpp, TGI, hosted APIs), `composite.py` (`CompositeAdapter` — a
  mixle model gating/reranking/calibrating an LLM, or an LLM tool-using a mixle model), `echo.py` (test stub).
- `gateway/` — `app.py` (FastAPI factory), `auth.py` (API-key + session dependencies, rate limit/usage), and
  `routes/` (chat, models, mixle, accounts, feedback, mcp).
- `accounts/` — `User`/`Org`/`ApiKey`/`Usage` models, password+token auth, optional OAuth/cloud SSO.
- `feedback/` — `collect.py` (ingest ratings/preferences/edits), `reward.py` (mixle preference model with UQ),
  `elicit.py` (active comparison selection via `mixle.doe`), `loop.py` (close-the-loop promotion / DPO export).
- `monitoring/` — calibration, drift, decision-quality monitors (the serving-core design's P0-B/C, P1-B).
- `mcp/` — an MCP server exposing hosted models + platform tools; and an MCP *client* so hosted models can call
  external MCP tools.
- `storage/` — `base.py` (ABC: relational + blob + kv), `local.py` (SQLite + fs), `cloud.py` (Postgres + S3).

## Frontend (`frontend/`, Next.js + TypeScript)

Landing page; a Claude/ChatGPT-style chat (SSE streaming, conversation history, multimodal upload, model picker,
per-message 👍/👎/edit/regenerate feeding the RLHF loop); auth (login/signup); API-key management; usage view.
Talks to the gateway over the OpenAI-compatible API + the platform API.

## Design rules

- **OpenAI-compatible by default** so any existing client/SDK/UI works and LLMs drop in via their standard
  servers. Mixle-specific power lives on dedicated routes, advertised per-model via `/v1/models` + `/capabilities`.
- **mixle math upstreams to mixle-core** (predictive adapter, decision module, calibration monitors,
  preference-reward helpers); the package keeps only deployment opinions (HTTP, config, storage, k8s). Until
  upstreamed, those live in `core/` here and are written to move cleanly.
- **Local-first, cloud-optional.** Everything runs on a laptop with SQLite + a local Ollama; the *same code*
  scales to Postgres + S3 + a cluster by config, no rewrite.
- **Don't rebuild a serving runtime.** Proxy LLMs to vLLM/Ollama/llama.cpp; the platform is the gateway + the
  mixle UQ/feedback logic + the product surface, not a new inference kernel.

## Build phases (each phase ends runnable + tested)

1. **Foundation** *(this commit)* — config, `ModelAdapter` + registry, echo + OpenAI-compat adapters, SQLite
   storage, accounts + API keys + auth, FastAPI gateway with OpenAI-compatible streaming `/v1/chat/completions`
   + `/v1/models` + account/key routes. Runnable end-to-end against a local Ollama or the echo model.
2. **LLM backends + multimodal** — wire vLLM/Ollama/llama.cpp/hosted; multimodal content parts (images/files);
   embeddings; model registry of real backends.
3. **mixle differentiator** — `MixleAdapter` serving distributions/decisions; the calibration/drift/decision
   monitors; `/predict /decide /score /calibration /latent /capabilities`; composite (mixle+LLM) models.
4. **RLHF / feedback loop** — feedback capture from chat → mixle preference-reward model with UQ → active
   elicitation → close-the-loop promotion / DPO export.
5. **MCP server** — expose hosted models + platform tools as MCP; MCP-client so models can use external tools.
6. **Frontend** — landing + chat UI + accounts + keys + feedback + usage.
7. **Deploy + cloud** — Docker Compose, k8s, Postgres/S3 cloud backends, usage/billing, observability.
