// Thin client for the mixle-mlops gateway (OpenAI-compatible + platform API).
// All calls are made directly from the browser to NEXT_PUBLIC_API_BASE.

import type {
  ApiKeyInfo,
  Attachment,
  ConversationDetail,
  ConversationSummary,
  DatasetArtifact,
  EvolutionRun,
  EvolvePolicy,
  ExportFormat,
  GeneratedImage,
  ModelInfo,
  OpenAIMessage,
  PublicUser,
  RagDocument,
  RagHit,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ||
  "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

function authHeaders(token: string | null): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (body?.detail) return JSON.stringify(body.detail);
    if (body?.error?.message) return body.error.message;
    return JSON.stringify(body);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) throw new ApiError(res.status, await parseError(res));
  return (await res.json()) as T;
}

// --- auth ---

export interface SignupResult {
  user: PublicUser;
  api_key: string;
}
export interface LoginResult {
  user: PublicUser;
  token: string;
}

export async function signup(
  email: string,
  password: string
): Promise<SignupResult> {
  const res = await fetch(`${API_BASE}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return jsonOrThrow<SignupResult>(res);
}

export async function login(
  email: string,
  password: string
): Promise<LoginResult> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return jsonOrThrow<LoginResult>(res);
}

export async function me(token: string): Promise<PublicUser> {
  const res = await fetch(`${API_BASE}/auth/me`, {
    headers: authHeaders(token),
  });
  return jsonOrThrow<PublicUser>(res);
}

// --- models ---

export async function listModels(token: string | null): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/v1/models`, {
    headers: authHeaders(token),
  });
  const body = await jsonOrThrow<{ data: ModelInfo[] }>(res);
  return body.data ?? [];
}

// --- api keys ---

export async function listKeys(token: string): Promise<ApiKeyInfo[]> {
  const res = await fetch(`${API_BASE}/keys`, { headers: authHeaders(token) });
  return jsonOrThrow<ApiKeyInfo[]>(res);
}

export async function createKey(
  token: string,
  name: string
): Promise<{ api_key: string; name: string }> {
  const res = await fetch(`${API_BASE}/keys`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ name }),
  });
  return jsonOrThrow<{ api_key: string; name: string }>(res);
}

export async function deleteKey(token: string, keyId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/keys/${keyId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new ApiError(res.status, await parseError(res));
}

// --- files (multimodal upload) ---
// The gateway /v1/files route is OpenAI-compatible (multipart "file" field) and is
// added by the multimodal builder. If it's not present yet we degrade gracefully:
// the caller still has the data URL and can inline the image as an image_url part.

export async function uploadFile(
  token: string | null,
  file: File
): Promise<string | null> {
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("purpose", "vision");
    const res = await fetch(`${API_BASE}/v1/files`, {
      method: "POST",
      headers: authHeaders(token), // do NOT set Content-Type; browser sets multipart boundary
      body: form,
    });
    if (!res.ok) return null;
    const body = await res.json();
    return body?.id ?? body?.file_id ?? null;
  } catch {
    return null;
  }
}

// --- feedback (RLHF loop) ---

export interface FeedbackPayload {
  message_id: string;
  model?: string;
  rating?: "up" | "down";
  edited_text?: string;
  action?: "rate" | "edit" | "regenerate";
  prompt?: string;
  response?: string;
}

// POST /feedback is added by the feedback/RLHF builder. We fire-and-report; a 404
// just means the route isn't wired yet, which the UI surfaces softly.
export async function sendFeedback(
  token: string | null,
  payload: FeedbackPayload
): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(token) },
      body: JSON.stringify(payload),
    });
    return res.ok;
  } catch {
    return false;
  }
}

// --- chat completions (SSE streaming) ---

export interface StreamHandlers {
  onDelta: (text: string) => void;
  onError?: (message: string) => void;
  signal?: AbortSignal;
}

// Convert UI messages (with attachments) into OpenAI-compatible wire messages.
export function toWireMessages(
  history: { role: string; content: string; attachments?: Attachment[] }[]
): OpenAIMessage[] {
  return history.map((m) => {
    const imgs = (m.attachments ?? []).filter((a) => a.isImage);
    if (m.role === "user" && imgs.length > 0) {
      const parts: OpenAIMessage["content"] = [];
      if (m.content) parts.push({ type: "text", text: m.content });
      for (const a of imgs) {
        parts.push({ type: "image_url", image_url: { url: a.dataUrl } });
      }
      return { role: "user", content: parts };
    }
    return { role: m.role as OpenAIMessage["role"], content: m.content };
  });
}

// Cascade router config (chat.py reads extra.cascade={frontier, threshold, n}):
// answer locally when self-consistent enough, else escalate to a frontier model.
export interface CascadeConfig {
  frontier: string;
  threshold?: number;
  n?: number;
}

// Mixture-of-Agents config (chat.py reads extra.moa={proposers, aggregator, layers}):
// several models propose, an aggregator synthesizes the final answer.
export interface MoaConfig {
  proposers: string[];
  aggregator: string;
  layers?: number;
}

// Backend-specific passthrough options forwarded as the request's `extra` object.
// `rag` opts into document/conversation retrieval augmentation; `conversation_id`
// threads turns into a persisted conversation (chat.py reads both off `req.extra`).
// The advanced strategies below are all opt-in and non-streaming (see chat.py):
//   agent      → server-side tool/agent loop
//   best_of_n  → self-consistency sampling (response sets X-Self-Consistency)
//   cascade    → local→frontier escalation (response sets X-Cascade-Escalated)
//   moa        → mixture-of-agents ensemble
export interface ChatExtra {
  rag?: boolean;
  conversation_id?: string;
  agent?: boolean;
  best_of_n?: number;
  cascade?: CascadeConfig;
  moa?: MoaConfig;
  [key: string]: unknown;
}

// Stream a chat completion, parsing SSE `data:` lines incrementally.
export async function streamChat(
  token: string | null,
  model: string,
  messages: OpenAIMessage[],
  handlers: StreamHandlers,
  extra?: ChatExtra
): Promise<void> {
  const body: Record<string, unknown> = { model, messages, stream: true };
  if (extra && Object.keys(extra).length > 0) body.extra = extra;
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify(body),
    signal: handlers.signal,
  });

  if (!res.ok || !res.body) {
    handlers.onError?.(await parseError(res));
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  // SSE events are separated by a blank line; each event has one or more `data:` lines.
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of rawEvent.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const data = trimmed.slice(5).trim();
        if (data === "[DONE]") return;
        try {
          const json = JSON.parse(data);
          if (json.error) {
            handlers.onError?.(json.error.message ?? "stream error");
            return;
          }
          const delta = json.choices?.[0]?.delta?.content;
          if (typeof delta === "string" && delta.length) handlers.onDelta(delta);
        } catch {
          // ignore non-JSON keepalive lines
        }
      }
    }
  }
}

// Headers the advanced (non-streaming) strategies surface on the chat response.
//   selfConsistency  → X-Self-Consistency: best-of-N / cascade local confidence (0..1)
//   cascadeEscalated  → X-Cascade-Escalated: "1" escalated to frontier, "0" answered locally
//   conversationId    → X-Conversation-Id: the persisted conversation this turn threaded into
export interface CompletionHeaders {
  selfConsistency: number | null;
  cascadeEscalated: boolean | null;
  conversationId: string | null;
}

export interface CompletionResult {
  completion: ChatCompletion;
  headers: CompletionHeaders;
}

// Minimal view of the OpenAI-compatible non-streaming chat completion we consume.
export interface ChatCompletion {
  id?: string;
  model?: string;
  choices?: {
    index?: number;
    message?: { role?: string; content?: string | null };
    finish_reason?: string | null;
  }[];
}

// Non-streaming chat completion. Used for the advanced strategies (agent, best_of_n,
// cascade, moa) which chat.py handles non-streaming; returns the JSON plus the relevant
// response headers (X-Self-Consistency / X-Cascade-Escalated / X-Conversation-Id).
export async function chatCompletion(
  token: string | null,
  model: string,
  messages: OpenAIMessage[],
  extra?: ChatExtra,
  signal?: AbortSignal
): Promise<CompletionResult> {
  const body: Record<string, unknown> = { model, messages, stream: false };
  if (extra && Object.keys(extra).length > 0) body.extra = extra;
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify(body),
    signal,
  });
  const completion = await jsonOrThrow<ChatCompletion>(res);
  const sc = res.headers.get("X-Self-Consistency");
  const esc = res.headers.get("X-Cascade-Escalated");
  return {
    completion,
    headers: {
      selfConsistency: sc !== null ? Number(sc) : null,
      cascadeEscalated: esc !== null ? esc === "1" : null,
      conversationId: res.headers.get("X-Conversation-Id"),
    },
  };
}

// --- self-evolution (admin) ---

export async function triggerEvolution(
  token: string,
  modelId: string,
  policy: EvolvePolicy,
  opts: { records?: unknown[]; promote?: boolean } = {}
): Promise<EvolutionRun> {
  const res = await fetch(`${API_BASE}/v1/evolve/${modelId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({
      records: opts.records ?? [],
      policy,
      promote: opts.promote ?? true,
    }),
  });
  return jsonOrThrow<EvolutionRun>(res);
}

export async function listEvolutionRuns(
  token: string,
  modelId: string
): Promise<EvolutionRun[]> {
  const res = await fetch(`${API_BASE}/v1/evolve/${modelId}/runs`, {
    headers: authHeaders(token),
  });
  const body = await jsonOrThrow<{ data: EvolutionRun[] }>(res);
  return body.data ?? [];
}

export async function getEvolutionRun(
  token: string,
  runId: string
): Promise<EvolutionRun> {
  const res = await fetch(`${API_BASE}/v1/evolve/runs/${runId}`, {
    headers: authHeaders(token),
  });
  return jsonOrThrow<EvolutionRun>(res);
}

export async function rollbackEvolution(
  token: string,
  modelId: string
): Promise<{ model_id: string; rolled_back: boolean }> {
  const res = await fetch(`${API_BASE}/v1/evolve/${modelId}/rollback`, {
    method: "POST",
    headers: authHeaders(token),
  });
  return jsonOrThrow<{ model_id: string; rolled_back: boolean }>(res);
}

// Resolve a gateway-relative path (e.g. "/v1/files/abc/content") against API_BASE.
// Absolute http(s) URLs are returned unchanged.
export function resolveUrl(url: string): string {
  if (/^https?:\/\//i.test(url) || url.startsWith("data:")) return url;
  return `${API_BASE}${url.startsWith("/") ? "" : "/"}${url}`;
}

// Fetch a (possibly auth-protected) blob and trigger a browser download.
// Blob endpoints like /v1/files/{id}/content and the conversation export require the
// Bearer token, so a plain <a download> won't work — we fetch with auth then save.
export async function downloadBlob(
  token: string | null,
  url: string,
  filename: string
): Promise<void> {
  const res = await fetch(resolveUrl(url), { headers: authHeaders(token) });
  if (!res.ok) throw new ApiError(res.status, await parseError(res));
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(objectUrl);
}

// --- conversations ---

export async function listConversations(
  token: string
): Promise<ConversationSummary[]> {
  const res = await fetch(`${API_BASE}/v1/conversations`, {
    headers: authHeaders(token),
  });
  const body = await jsonOrThrow<{ data: ConversationSummary[] }>(res);
  return body.data ?? [];
}

export async function getConversation(
  token: string,
  id: string
): Promise<ConversationDetail> {
  const res = await fetch(`${API_BASE}/v1/conversations/${id}`, {
    headers: authHeaders(token),
  });
  return jsonOrThrow<ConversationDetail>(res);
}

export async function deleteConversation(
  token: string,
  id: string
): Promise<void> {
  const res = await fetch(`${API_BASE}/v1/conversations/${id}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new ApiError(res.status, await parseError(res));
}

// Download a conversation export (json | markdown | pdf). Uses an authenticated
// fetch because the export route is behind require_user.
export async function exportConversation(
  token: string,
  id: string,
  format: ExportFormat
): Promise<void> {
  const suffix = format === "markdown" ? "md" : format;
  await downloadBlob(
    token,
    `/v1/conversations/${id}/export?format=${format}`,
    `conversation-${id}.${suffix}`
  );
}

// --- documents + RAG ---

export async function listDocuments(token: string): Promise<RagDocument[]> {
  const res = await fetch(`${API_BASE}/v1/documents`, {
    headers: authHeaders(token),
  });
  const body = await jsonOrThrow<{ data: RagDocument[] }>(res);
  return body.data ?? [];
}

// POST /v1/documents is a multipart upload (the route takes an UploadFile "file"),
// not JSON base64. Do not set Content-Type — the browser adds the multipart boundary.
export async function uploadDocument(
  token: string,
  file: File
): Promise<RagDocument> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/v1/documents`, {
    method: "POST",
    headers: authHeaders(token),
    body: form,
  });
  return jsonOrThrow<RagDocument>(res);
}

export async function ragSearch(
  token: string,
  query: string,
  opts: { k?: number; namespace?: string | null } = {}
): Promise<RagHit[]> {
  const res = await fetch(`${API_BASE}/v1/rag/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({
      query,
      k: opts.k ?? 5,
      namespace: opts.namespace ?? null,
    }),
  });
  const body = await jsonOrThrow<{ data: RagHit[] }>(res);
  return body.data ?? [];
}

// --- images ---

export interface ImageGenParams {
  prompt: string;
  model?: string;
  n?: number;
  size?: string;
  response_format?: "url" | "b64_json";
}

export async function generateImages(
  token: string,
  params: ImageGenParams
): Promise<GeneratedImage[]> {
  const res = await fetch(`${API_BASE}/v1/images/generations`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({
      prompt: params.prompt,
      model: params.model ?? "",
      n: params.n ?? 1,
      size: params.size,
      response_format: params.response_format ?? "url",
    }),
  });
  const body = await jsonOrThrow<{ data: GeneratedImage[] }>(res);
  return body.data ?? [];
}

// --- datasets ---

export interface DatasetGenerateParams {
  source: "mixle" | "llm";
  model: string;
  n?: number;
  seed?: number;
  schema?: Record<string, string> | null;
  prompt?: string | null;
  format?: "jsonl" | "csv" | "parquet";
  columns?: string[] | null;
}

export async function listDatasets(token: string): Promise<DatasetArtifact[]> {
  const res = await fetch(`${API_BASE}/v1/datasets`, {
    headers: authHeaders(token),
  });
  const body = await jsonOrThrow<{ data: DatasetArtifact[] }>(res);
  return body.data ?? [];
}

export async function getDataset(
  token: string,
  id: string
): Promise<DatasetArtifact> {
  const res = await fetch(`${API_BASE}/v1/datasets/${id}`, {
    headers: authHeaders(token),
  });
  return jsonOrThrow<DatasetArtifact>(res);
}

export async function generateDataset(
  token: string,
  params: DatasetGenerateParams
): Promise<DatasetArtifact> {
  const res = await fetch(`${API_BASE}/v1/datasets/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({
      source: params.source,
      model: params.model,
      n: params.n ?? 100,
      seed: params.seed ?? 0,
      schema: params.schema ?? null,
      prompt: params.prompt ?? null,
      format: params.format ?? "jsonl",
      columns: params.columns ?? null,
    }),
  });
  return jsonOrThrow<DatasetArtifact>(res);
}
