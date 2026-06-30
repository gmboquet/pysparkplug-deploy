// Thin client for the mixle-mlops gateway (OpenAI-compatible + platform API).
// All calls are made directly from the browser to NEXT_PUBLIC_API_BASE.

import type {
  ApiKeyInfo,
  Attachment,
  ModelInfo,
  OpenAIMessage,
  PublicUser,
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

// Stream a chat completion, parsing SSE `data:` lines incrementally.
export async function streamChat(
  token: string | null,
  model: string,
  messages: OpenAIMessage[],
  handlers: StreamHandlers
): Promise<void> {
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ model, messages, stream: true }),
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
