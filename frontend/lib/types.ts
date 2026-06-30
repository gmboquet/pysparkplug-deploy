// Shared API + UI types for the mixle-mlops frontend.

export type Role = "system" | "user" | "assistant" | "tool";

export interface ModelInfo {
  id: string;
  object?: string;
  created?: number;
  owned_by?: string;
  kind?: "llm" | "mixle" | "composite";
  capabilities?: string[];
}

export interface PublicUser {
  id: string;
  email: string;
  is_admin: boolean;
}

export interface ApiKeyInfo {
  id: string;
  name: string;
  prefix: string;
  kind: string;
  created_at: string;
  last_used: string | null;
}

// An uploaded attachment. We keep a data URL so it can be embedded as an
// OpenAI-compatible image_url part even if the gateway /v1/files route is absent.
export interface Attachment {
  id: string;
  name: string;
  mime: string;
  size: number;
  dataUrl: string; // data:<mime>;base64,...
  // Gateway file id, when POST /v1/files succeeded.
  fileId?: string;
  isImage: boolean;
}

export type Feedback = "up" | "down" | null;

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  // Attachments only meaningful on user messages.
  attachments?: Attachment[];
  // Model that produced an assistant message.
  model?: string;
  // Streaming-in-progress flag for assistant messages.
  pending?: boolean;
  feedback?: Feedback;
}

// OpenAI-compatible content part shapes the gateway expects.
export type ContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

export interface OpenAIMessage {
  role: Role;
  content: string | ContentPart[];
  name?: string;
}
