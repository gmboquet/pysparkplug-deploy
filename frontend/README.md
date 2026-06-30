# mixle-mlops — frontend

A Next.js (App Router) + TypeScript + Tailwind web app for the mixle-mlops platform: a landing
page, auth (signup/login), a Claude/ChatGPT-style streaming chat, and API-key management. It talks
to the FastAPI gateway over its OpenAI-compatible + platform API.

## What's here

- **Landing page** (`/`) — hero ("host mixle + open LLMs, calibrated, with a feedback loop"),
  feature cards, a "Try the chat" CTA, login/signup links.
- **Auth** (`/login`, `/signup`) — call `POST /auth/login` and `POST /auth/signup`, store the
  returned token / api_key in `localStorage`, expose it via a small auth context (`lib/auth.tsx`).
- **Chat** (`/chat`) — model picker from `GET /v1/models`, a message list, a composer with
  multimodal file upload, **SSE streaming** by POSTing to `/v1/chat/completions` with `stream:true`
  and parsing `data:` lines incrementally, and per-assistant-message **👍 / 👎 / edit / regenerate**
  that POST to `/feedback`.
- **API keys** (`/keys`) — `GET` / `POST` / `DELETE /keys` with one-time key reveal.

## Run

```bash
cd frontend
cp .env.example .env.local        # optional; defaults to http://localhost:8000
npm install
npm run dev                       # http://localhost:3000
```

Production build:

```bash
npm run build && npm run start
```

### Point at the gateway

The browser calls the gateway directly. Set the base URL via env (defaults to
`http://localhost:8000`):

```
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

Start the gateway separately, e.g.:

```bash
cd ..  # repo root (mixle-mlops/)
uvicorn mixle_mlops.gateway.app:app --reload --port 8000
```

The gateway ships with an `echo` model, so the chat works end-to-end with **no LLM backend**.
The gateway already enables permissive CORS (`CORSMiddleware` with `allow_origins` from settings);
make sure `http://localhost:3000` is allowed (or `*`) for browser calls to succeed.

## Gateway API used

| UI surface | Method & path | Notes |
|---|---|---|
| signup | `POST /auth/signup` | → `{ user, api_key }` (key stored as Bearer token) |
| login | `POST /auth/login` | → `{ user, token }` |
| current user | `GET /auth/me` | Bearer |
| model picker | `GET /v1/models` | → `{ data: [ModelInfo] }` |
| chat (stream) | `POST /v1/chat/completions` `{stream:true}` | SSE `data:` lines, `[DONE]` terminator |
| file upload | `POST /v1/files` (multipart `file`) | **optional** — see below |
| feedback | `POST /feedback` | **optional** — see below |
| keys | `GET` / `POST` / `DELETE /keys` | Bearer |

### Graceful degradation for routes built in parallel

Two routes are owned by other builders and may not be wired yet:

- **`POST /v1/files`** (multimodal upload). The chat reads each file into a base64 **data URL** in
  the browser and embeds images directly as OpenAI-compatible `image_url` content parts, so
  vision-style requests work even without the files route. If `/v1/files` exists, the returned file
  id is also attached. Expected response shape: `{ "id": "file_..." }` (also accepts `file_id`).
- **`POST /feedback`** (RLHF loop). 👍/👎/edit/regenerate fire a `POST /feedback` and silently
  no-op on 404. Expected request body:

  ```json
  {
    "message_id": "a_...",
    "model": "echo",
    "rating": "up" | "down",
    "action": "rate" | "edit" | "regenerate",
    "edited_text": "…",
    "prompt": "…",
    "response": "…"
  }
  ```

When the gateway adds these routes, no frontend change is required.

## Layout

```
frontend/
├── app/
│   ├── layout.tsx           # root layout, wraps AuthProvider
│   ├── globals.css          # Tailwind + theme tokens (dark/light)
│   ├── page.tsx             # landing
│   ├── login/page.tsx
│   ├── signup/page.tsx
│   ├── chat/page.tsx        # streaming chat (the centerpiece)
│   ├── keys/page.tsx        # API-key management
│   └── components/
│       ├── NavBar.tsx
│       ├── AuthForm.tsx
│       └── Message.tsx      # bubble + 👍/👎/edit/regenerate
├── lib/
│   ├── api.ts               # gateway client + SSE stream parser
│   ├── auth.tsx             # localStorage-backed auth context
│   ├── files.ts             # File → data-URL attachment
│   └── types.ts
├── package.json
├── tsconfig.json
├── tailwind.config.ts
├── postcss.config.mjs
├── next.config.mjs
└── .env.example
```

## Assumptions

- The token returned by `/auth/login` and the `api_key` from `/auth/signup` are both usable as a
  `Bearer` token against `/v1/*` and `/keys` (confirmed by the gateway's `auth.py` resolving any
  API key, `kind` "api" or "session").
- `GET /v1/models` and `GET /v1/chat/completions` work **without** auth (the gateway's
  `current_user` returns `None` unless `require_auth` is set), so the chat is usable logged-out
  against `echo`. Logged-in requests just add the Bearer header.
- The SSE stream emits OpenAI-compatible chunks (`choices[0].delta.content`) terminated by
  `data: [DONE]`, matching `gateway/routes/chat.py`.
