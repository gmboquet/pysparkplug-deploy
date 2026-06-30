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

// --- Wave 2: conversations / documents / RAG / images / datasets ---

// Summary shape returned by GET /v1/conversations (and the head of the detail view).
export interface ConversationSummary {
  id: string;
  title: string;
  model: string | null;
  created_at: string;
  updated_at: string;
}

// A persisted message inside a conversation detail.
export interface ConversationMessage {
  id: string;
  role: Role;
  content: string;
  created_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: ConversationMessage[];
}

export type ExportFormat = "json" | "markdown" | "pdf";

// An ingested RAG document (rag/models.py Document.to_dict).
export interface RagDocument {
  id: string;
  object: "rag.document";
  filename: string;
  content_type: string;
  blob_id: string | null;
  n_chunks: number;
  n_chars: number;
  created_at: string | null;
}

// A retrieved snippet (rag/vectorstore.py Hit.to_dict).
export interface RagHit {
  id: string;
  text: string;
  score: number;
  namespace: string;
  source_id: string | null;
  meta: Record<string, unknown>;
}

// One generated image entry (images.py returns url OR b64_json).
export interface GeneratedImage {
  url?: string;
  b64_json?: string;
}

// --- Wave 3: advanced inference strategies + self-evolution ---

// Objectives the self-evolution policy can optimize (evolve/policy.py OBJECTIVES).
export type EvolveObjective =
  | "nll"
  | "log_score"
  | "crps"
  | "interval"
  | "calibration";

// One self-evolution run (evolve/models.py EvolutionRecord.to_dict).
export interface EvolutionRun {
  id: string;
  model_id?: string;
  objective: string;
  operator: string | null;
  verified: boolean;
  promoted: boolean;
  delta: number;
  n_data?: number;
  verdict: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
}

// The serving-side knob set for an evolution run (evolve/policy.py EvolutionPolicy).
export interface EvolvePolicy {
  objective: EvolveObjective;
  alpha?: number;
  min_effect?: number;
  holdout?: number;
  operators?: string[] | null;
  approval?: "none" | "required";
}

// A generated dataset artifact (datasets/models.py DatasetArtifact.to_dict).
export interface DatasetArtifact {
  id: string;
  object: "dataset";
  source: string; // "mixle" | "llm"
  model: string | null;
  format: string; // "jsonl" | "csv" | "parquet"
  n_rows: number;
  seed: number | null;
  prompt: string | null;
  blob_id: string | null;
  url: string | null;
  schema: Record<string, string>;
  created_at: string | null;
  // generate returns the artifact ref nested under "artifact"
  artifact?: { id?: string; url?: string };
}
