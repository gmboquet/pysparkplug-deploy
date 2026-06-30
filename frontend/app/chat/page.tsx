"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { toAttachment } from "@/lib/files";
import type { Attachment, ChatMessage, ModelInfo } from "@/lib/types";
import { NavBar } from "../components/NavBar";
import { Message } from "../components/Message";

let _mid = 0;
function newId(prefix: string): string {
  _mid += 1;
  return `${prefix}_${Date.now().toString(36)}_${_mid}`;
}

const KIND_BADGE: Record<string, string> = {
  mixle: "calibrated",
  composite: "composite",
  llm: "LLM",
};

export default function ChatPage() {
  const { token } = useAuth();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState<string>("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [banner, setBanner] = useState<string | null>(null);
  const [useRag, setUseRag] = useState(false);

  // --- advanced inference strategies (all opt-in passthrough `extra.*` options the
  // gateway reads; each is non-streaming, so enabling any one routes through the
  // non-streaming completion endpoint instead of streamChat). ---
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [agentMode, setAgentMode] = useState(false);
  const [bestOfN, setBestOfN] = useState(1);
  const [cascadeOn, setCascadeOn] = useState(false);
  const [cascadeFrontier, setCascadeFrontier] = useState("");
  const [cascadeThreshold, setCascadeThreshold] = useState(0.6);
  const [moaOn, setMoaOn] = useState(false);
  const [moaProposers, setMoaProposers] = useState<string[]>([]);
  const [moaAggregator, setMoaAggregator] = useState("");
  // Per-strategy result surfaced under the latest assistant turn.
  const [lastInfo, setLastInfo] = useState<api.CompletionHeaders | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  // Persisted conversation this thread is writing to. The streaming route does not
  // return a conversation id, so after the first authenticated turn we discover the
  // freshly-created conversation (most-recently-updated) and thread subsequent turns.
  const conversationRef = useRef<string | null>(null);

  // load models (works unauthenticated against the default echo model)
  useEffect(() => {
    api
      .listModels(token)
      .then((m) => {
        setModels(m);
        if (m.length) setModel((cur) => cur || m[0].id);
      })
      .catch(() => setBanner("Could not load models from the gateway."));
  }, [token]);

  // autoscroll
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const selectedModel =
    models.find((m) => m.id === model) ?? (model ? { id: model } : undefined);

  // The advanced strategies require auth (the gateway gates conversation/RAG/agent
  // tooling on a real user). A strategy is "active" only when its toggle is on AND
  // its required inputs are present (e.g. cascade needs a frontier, moa needs both
  // proposers and an aggregator) — otherwise we leave the default chat path alone.
  const cascadeReady = cascadeOn && !!cascadeFrontier;
  const moaReady = moaOn && moaProposers.length > 0 && !!moaAggregator;
  const anyAdvanced =
    !!token &&
    (agentMode || bestOfN > 1 || cascadeReady || moaReady);

  // Build the `extra` passthrough for a turn, layering in whichever advanced
  // strategies are active. Only one branch is needed by the gateway, but it picks
  // the first it sees; we keep the set minimal to avoid surprising precedence.
  const buildExtra = useCallback((): api.ChatExtra => {
    const extra: api.ChatExtra = {};
    if (token && useRag) extra.rag = true;
    if (token && conversationRef.current) {
      extra.conversation_id = conversationRef.current;
    }
    if (token) {
      if (agentMode) extra.agent = true;
      else if (cascadeReady) {
        extra.cascade = { frontier: cascadeFrontier, threshold: cascadeThreshold };
      } else if (moaReady) {
        extra.moa = { proposers: moaProposers, aggregator: moaAggregator };
      } else if (bestOfN > 1) extra.best_of_n = bestOfN;
    }
    return extra;
  }, [
    token,
    useRag,
    agentMode,
    cascadeReady,
    cascadeFrontier,
    cascadeThreshold,
    moaReady,
    moaProposers,
    moaAggregator,
    bestOfN,
  ]);

  // --- core: run a completion given the conversation up to (and including) the
  // user turn, appending a streaming assistant message. ---
  const runCompletion = useCallback(
    async (history: ChatMessage[]) => {
      const assistantId = newId("a");
      setMessages([
        ...history,
        {
          id: assistantId,
          role: "assistant",
          content: "",
          model,
          pending: true,
          feedback: null,
        },
      ]);
      setStreaming(true);
      setBanner(null);
      setLastInfo(null);

      const controller = new AbortController();
      abortRef.current = controller;

      const wire = api.toWireMessages(
        history.map((m) => ({
          role: m.role,
          content: m.content,
          attachments: m.attachments,
        }))
      );

      const extra = buildExtra();

      // Advanced strategies (agent / best_of_n / cascade / moa) are non-streaming on
      // the gateway, so when any is active we hit the non-streaming endpoint and render
      // the single result, surfacing its X-* headers as badges. Otherwise stream.
      if (anyAdvanced) {
        try {
          const { completion, headers } = await api.chatCompletion(
            token,
            model,
            wire,
            extra,
            controller.signal
          );
          const text = completion.choices?.[0]?.message?.content ?? "";
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: text, pending: false }
                : m
            )
          );
          setLastInfo(headers);
          if (headers.conversationId) {
            conversationRef.current = headers.conversationId;
          }
        } catch (err) {
          if ((err as Error)?.name !== "AbortError") {
            setBanner(err instanceof Error ? err.message : "completion failed");
          }
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, pending: false } : m
            )
          );
        } finally {
          setStreaming(false);
          abortRef.current = null;
        }
        return;
      }

      let acc = "";
      let streamErrored = false;
      await api.streamChat(
        token,
        model,
        wire,
        {
          signal: controller.signal,
          onDelta: (d) => {
            acc += d;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: acc } : m
              )
            );
          },
          onError: (msg) => {
            streamErrored = true;
            setBanner(msg);
          },
        },
        extra
      );

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, pending: false } : m
        )
      );
      setStreaming(false);
      abortRef.current = null;

      // Discover the conversation this turn was persisted into so subsequent turns
      // thread into it. Best-effort: failures here never affect the chat.
      if (token && !streamErrored && !conversationRef.current) {
        try {
          const convs = await api.listConversations(token);
          if (convs.length > 0) conversationRef.current = convs[0].id;
        } catch {
          /* ignore */
        }
      }
    },
    [model, token, anyAdvanced, buildExtra]
  );

  async function onSend() {
    const text = input.trim();
    if ((!text && attachments.length === 0) || streaming || !model) return;

    // best-effort upload to /v1/files (degrades to inline data URL if absent)
    const uploaded = await Promise.all(
      attachments.map(async (a) => ({
        ...a,
        fileId: (await api.uploadFile(token, dataUrlToFile(a))) ?? undefined,
      }))
    );

    const userMsg: ChatMessage = {
      id: newId("u"),
      role: "user",
      content: text,
      attachments: uploaded,
    };
    setInput("");
    setAttachments([]);
    await runCompletion([...messages, userMsg]);
  }

  function stop() {
    abortRef.current?.abort();
    setStreaming(false);
  }

  // --- feedback / edit / regenerate (RLHF loop) ---

  function findPriorUserPrompt(history: ChatMessage[], idx: number): string {
    for (let i = idx - 1; i >= 0; i--) {
      if (history[i].role === "user") return history[i].content;
    }
    return "";
  }

  function onFeedback(id: string, rating: "up" | "down") {
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === id);
      const target = prev[idx];
      void api.sendFeedback(token, {
        message_id: id,
        model: target?.model ?? model,
        rating,
        action: "rate",
        prompt: findPriorUserPrompt(prev, idx),
        response: target?.content,
      });
      return prev.map((m) =>
        m.id === id
          ? { ...m, feedback: m.feedback === rating ? null : rating }
          : m
      );
    });
  }

  async function onEditUser(id: string, text: string) {
    // Edit a user message and re-run from that point (an RLHF "edit" signal).
    const idx = messages.findIndex((m) => m.id === id);
    if (idx === -1) return;
    void api.sendFeedback(token, {
      message_id: id,
      model,
      action: "edit",
      edited_text: text,
    });
    const edited: ChatMessage = { ...messages[idx], content: text };
    const history = [...messages.slice(0, idx), edited];
    await runCompletion(history);
  }

  async function onRegenerate(id: string) {
    const idx = messages.findIndex((m) => m.id === id);
    if (idx === -1) return;
    void api.sendFeedback(token, {
      message_id: id,
      model: messages[idx]?.model ?? model,
      action: "regenerate",
    });
    // drop the assistant message (and anything after) and re-run from prior turns
    const history = messages.slice(0, idx);
    await runCompletion(history);
  }

  // --- attachments ---
  async function onFiles(files: FileList | null) {
    if (!files) return;
    const next = await Promise.all(Array.from(files).map(toAttachment));
    setAttachments((cur) => [...cur, ...next]);
    if (fileRef.current) fileRef.current.value = "";
  }

  return (
    <div className="flex h-screen flex-col">
      <NavBar />

      {/* model picker bar */}
      <div className="border-b border-border bg-bg/80 px-4 py-2 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center gap-2">
          <label className="text-xs text-muted">Model</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="rounded-lg border border-border bg-surface px-2 py-1 text-sm outline-none focus:border-accent"
          >
            {models.length === 0 && <option value="">(loading…)</option>}
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id}
              </option>
            ))}
          </select>
          {selectedModel && "kind" in selectedModel && selectedModel.kind && (
            <span className="rounded-full border border-border bg-surface px-2 py-0.5 text-[11px] text-muted">
              {KIND_BADGE[selectedModel.kind] ?? selectedModel.kind}
            </span>
          )}
          {selectedModel &&
            "capabilities" in selectedModel &&
            (selectedModel.capabilities?.length ?? 0) > 0 && (
              <span className="hidden text-[11px] text-muted sm:inline">
                {selectedModel.capabilities!.join(" · ")}
              </span>
            )}
          {token && (
            <label
              className="ml-auto flex cursor-pointer items-center gap-1.5 text-xs text-muted"
              title="Augment replies with retrieval from your uploaded documents + past conversations"
            >
              <input
                type="checkbox"
                checked={useRag}
                onChange={(e) => setUseRag(e.target.checked)}
                className="accent-accent"
              />
              Use my documents (RAG)
            </label>
          )}
          {token && (
            <button
              onClick={() => setAdvancedOpen((v) => !v)}
              className={`${
                token ? "" : "ml-auto "
              }flex items-center gap-1 rounded-md px-2 py-1 text-xs ${
                anyAdvanced
                  ? "text-accent"
                  : "text-muted hover:bg-surface-2 hover:text-fg"
              }`}
              title="Advanced inference strategies (agent, best-of-N, cascade, mixture-of-agents)"
            >
              Advanced
              {anyAdvanced && (
                <span className="grid h-1.5 w-1.5 place-items-center rounded-full bg-accent" />
              )}
              <span className="text-[10px]">{advancedOpen ? "▴" : "▾"}</span>
            </button>
          )}
          {messages.length > 0 && (
            <button
              onClick={() => {
                setMessages([]);
                conversationRef.current = null;
                setLastInfo(null);
              }}
              className={`${
                token ? "" : "ml-auto "
              }rounded-md px-2 py-1 text-xs text-muted hover:bg-surface-2`}
            >
              New chat
            </button>
          )}
        </div>

        {/* advanced inference strategies panel */}
        {token && advancedOpen && (
          <div className="mx-auto mt-2 max-w-3xl rounded-xl border border-border bg-surface p-3 text-sm">
            <div className="grid gap-3 sm:grid-cols-2">
              {/* agent mode */}
              <div className="flex flex-col gap-1 rounded-lg border border-border bg-surface-2 p-3">
                <label className="flex cursor-pointer items-center gap-2 font-medium">
                  <input
                    type="checkbox"
                    checked={agentMode}
                    onChange={(e) => setAgentMode(e.target.checked)}
                    className="accent-accent"
                  />
                  Agent mode
                </label>
                <p className="text-xs text-muted">
                  Let the model use the platform&apos;s tools (RAG, mixle
                  decide/predict) in a server-side loop.
                </p>
              </div>

              {/* best-of-N */}
              <div className="flex flex-col gap-1 rounded-lg border border-border bg-surface-2 p-3">
                <label className="flex items-center justify-between gap-2 font-medium">
                  Best-of-N
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={bestOfN}
                    onChange={(e) =>
                      setBestOfN(
                        Math.max(1, Math.min(10, Number(e.target.value) || 1))
                      )
                    }
                    className="w-16 rounded-lg border border-border bg-surface px-2 py-1 text-sm outline-none focus:border-accent"
                  />
                </label>
                <p className="text-xs text-muted">
                  Sample N answers and return the self-consistent majority with a
                  calibrated confidence.
                </p>
              </div>

              {/* cascade */}
              <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface-2 p-3">
                <label className="flex cursor-pointer items-center gap-2 font-medium">
                  <input
                    type="checkbox"
                    checked={cascadeOn}
                    onChange={(e) => setCascadeOn(e.target.checked)}
                    className="accent-accent"
                  />
                  Cascade
                </label>
                <p className="text-xs text-muted">
                  Answer locally when confident; otherwise escalate to a frontier
                  model.
                </p>
                {cascadeOn && (
                  <div className="flex flex-col gap-2">
                    <label className="flex flex-col gap-1 text-xs text-muted">
                      Frontier model
                      <select
                        value={cascadeFrontier}
                        onChange={(e) => setCascadeFrontier(e.target.value)}
                        className="rounded-lg border border-border bg-surface px-2 py-1 text-sm text-fg outline-none focus:border-accent"
                      >
                        <option value="">(pick a frontier model)</option>
                        {models
                          .filter((m) => m.id !== model)
                          .map((m) => (
                            <option key={m.id} value={m.id}>
                              {m.id}
                            </option>
                          ))}
                      </select>
                    </label>
                    <label className="flex flex-col gap-1 text-xs text-muted">
                      <span className="flex justify-between">
                        Escalation threshold
                        <span className="text-fg">
                          {cascadeThreshold.toFixed(2)}
                        </span>
                      </span>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={cascadeThreshold}
                        onChange={(e) =>
                          setCascadeThreshold(Number(e.target.value))
                        }
                        className="accent-accent"
                      />
                    </label>
                    {!cascadeFrontier && (
                      <p className="text-xs text-amber-400">
                        Pick a frontier model to enable cascade.
                      </p>
                    )}
                  </div>
                )}
              </div>

              {/* mixture-of-agents */}
              <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface-2 p-3">
                <label className="flex cursor-pointer items-center gap-2 font-medium">
                  <input
                    type="checkbox"
                    checked={moaOn}
                    onChange={(e) => setMoaOn(e.target.checked)}
                    className="accent-accent"
                  />
                  Mixture-of-Agents
                </label>
                <p className="text-xs text-muted">
                  Several models propose; an aggregator synthesizes the final
                  answer.
                </p>
                {moaOn && (
                  <div className="flex flex-col gap-2">
                    <div className="flex flex-col gap-1 text-xs text-muted">
                      <span>Proposers</span>
                      <div className="flex flex-wrap gap-1.5">
                        {models.length === 0 && (
                          <span className="text-muted">(no models)</span>
                        )}
                        {models.map((m) => {
                          const on = moaProposers.includes(m.id);
                          return (
                            <button
                              key={m.id}
                              type="button"
                              onClick={() =>
                                setMoaProposers((cur) =>
                                  on
                                    ? cur.filter((x) => x !== m.id)
                                    : [...cur, m.id]
                                )
                              }
                              className={`rounded-full border px-2 py-0.5 text-[11px] ${
                                on
                                  ? "border-accent text-accent"
                                  : "border-border text-muted hover:bg-surface"
                              }`}
                            >
                              {m.id}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    <label className="flex flex-col gap-1 text-xs text-muted">
                      Aggregator
                      <select
                        value={moaAggregator}
                        onChange={(e) => setMoaAggregator(e.target.value)}
                        className="rounded-lg border border-border bg-surface px-2 py-1 text-sm text-fg outline-none focus:border-accent"
                      >
                        <option value="">(pick an aggregator)</option>
                        {models.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.id}
                          </option>
                        ))}
                      </select>
                    </label>
                    {!moaReady && (
                      <p className="text-xs text-amber-400">
                        Select at least one proposer and an aggregator.
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>
            <p className="mt-3 text-[11px] text-muted">
              These strategies are non-streaming — when one is active the reply
              arrives as a single result. Agent mode takes precedence, then
              cascade, then mixture-of-agents, then best-of-N.
            </p>
          </div>
        )}
      </div>

      {/* message list */}
      <div ref={scrollRef} className="scroll-thin flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-4 px-4 py-6">
          {messages.length === 0 && (
            <div className="mt-24 text-center text-muted">
              <p className="text-lg font-medium text-fg">
                Chat with mixle + open LLMs
              </p>
              <p className="mt-1 text-sm">
                Pick a model, attach an image, and start. Responses stream token by
                token; rate them to train the feedback loop.
              </p>
            </div>
          )}
          {messages.map((m) => (
            <Message
              key={m.id}
              message={m}
              onFeedback={onFeedback}
              onEdit={onEditUser}
              onRegenerate={onRegenerate}
            />
          ))}
        </div>
      </div>

      {/* advanced-strategy result badges (best-of-N confidence / cascade routing) */}
      {lastInfo &&
        (lastInfo.selfConsistency !== null ||
          lastInfo.cascadeEscalated !== null) && (
          <div className="mx-auto w-full max-w-3xl px-4">
            <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
              {lastInfo.selfConsistency !== null && (
                <span
                  className="rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 text-accent"
                  title="Self-consistency confidence (X-Self-Consistency)"
                >
                  confidence {Math.round(lastInfo.selfConsistency * 100)}%
                </span>
              )}
              {lastInfo.cascadeEscalated !== null && (
                <span
                  className="rounded-full border border-border bg-surface px-2 py-0.5 text-muted"
                  title="Cascade routing (X-Cascade-Escalated)"
                >
                  {lastInfo.cascadeEscalated
                    ? `escalated to ${cascadeFrontier || "frontier"}`
                    : "answered locally"}
                </span>
              )}
            </div>
          </div>
        )}

      {/* banner */}
      {banner && (
        <div className="mx-auto w-full max-w-3xl px-4">
          <div className="mb-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-400">
            {banner}
          </div>
        </div>
      )}

      {/* composer */}
      <div className="border-t border-border bg-bg px-4 py-3">
        <div className="mx-auto max-w-3xl">
          {attachments.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {attachments.map((a) => (
                <span
                  key={a.id}
                  className="flex items-center gap-2 rounded-lg border border-border bg-surface px-2 py-1 text-xs"
                >
                  {a.isImage ? "🖼" : "📎"} {a.name}
                  <button
                    onClick={() =>
                      setAttachments((cur) => cur.filter((x) => x.id !== a.id))
                    }
                    className="text-muted hover:text-fg"
                  >
                    ✕
                  </button>
                </span>
              ))}
            </div>
          )}
          <div className="flex items-end gap-2 rounded-2xl border border-border bg-surface p-2">
            <button
              title="Attach file"
              onClick={() => fileRef.current?.click()}
              className="grid h-9 w-9 place-items-center rounded-lg text-muted hover:bg-surface-2"
            >
              +
            </button>
            <input
              ref={fileRef}
              type="file"
              multiple
              accept="image/*,.pdf,.txt,.md,.csv,.json"
              className="hidden"
              onChange={(e) => onFiles(e.target.files)}
            />
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void onSend();
                }
              }}
              rows={1}
              placeholder="Message mixle…  (Enter to send, Shift+Enter for newline)"
              className="max-h-40 flex-1 resize-none bg-transparent px-1 py-2 text-sm outline-none"
            />
            {streaming ? (
              <button
                onClick={stop}
                className="rounded-lg border border-border px-3 py-2 text-sm hover:bg-surface-2"
              >
                Stop
              </button>
            ) : (
              <button
                onClick={() => void onSend()}
                disabled={!model || (!input.trim() && attachments.length === 0)}
                className="rounded-lg px-4 py-2 text-sm font-medium text-accent-fg disabled:opacity-50"
                style={{ background: "var(--accent)" }}
              >
                Send
              </button>
            )}
          </div>
          <p className="mt-1 text-center text-[11px] text-muted">
            mixle serves calibrated distributions + decisions. Your 👍/👎/edits feed
            the preference model.
          </p>
        </div>
      </div>
    </div>
  );
}

// Reconstruct a File from a stored data URL so we can POST it to /v1/files.
function dataUrlToFile(a: Attachment): File {
  const [meta, b64] = a.dataUrl.split(",");
  const mime = /data:([^;]+)/.exec(meta)?.[1] || a.mime;
  const bin = atob(b64 ?? "");
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new File([bytes], a.name, { type: mime });
}
