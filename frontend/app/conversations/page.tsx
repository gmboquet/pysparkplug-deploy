"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import * as api from "@/lib/api";
import { useAuth } from "@/lib/auth";
import type {
  ConversationDetail,
  ConversationSummary,
  ExportFormat,
} from "@/lib/types";
import { NavBar } from "../components/NavBar";

const EXPORT_FORMATS: ExportFormat[] = ["json", "markdown", "pdf"];

export default function ConversationsPage() {
  const router = useRouter();
  const { token, ready } = useAuth();

  const [convs, setConvs] = useState<ConversationSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ConversationDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [exporting, setExporting] = useState<ExportFormat | null>(null);

  useEffect(() => {
    if (ready && !token) router.replace("/login");
  }, [ready, token, router]);

  const refresh = useCallback(async () => {
    if (!token) return;
    setLoadingList(true);
    try {
      setConvs(await api.listConversations(token));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to load conversations");
    } finally {
      setLoadingList(false);
    }
  }, [token]);

  useEffect(() => {
    if (token) void refresh();
  }, [token, refresh]);

  const openConversation = useCallback(
    async (id: string) => {
      if (!token) return;
      setSelectedId(id);
      setDetail(null);
      setLoadingDetail(true);
      try {
        setDetail(await api.getConversation(token, id));
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "failed to load conversation");
      } finally {
        setLoadingDetail(false);
      }
    },
    [token]
  );

  async function onExport(id: string, format: ExportFormat) {
    if (!token) return;
    setExporting(format);
    try {
      await api.exportConversation(token, id, format);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : `failed to export as ${format}`);
    } finally {
      setExporting(null);
    }
  }

  async function onDelete(id: string) {
    if (!token) return;
    try {
      await api.deleteConversation(token, id);
      if (selectedId === id) {
        setSelectedId(null);
        setDetail(null);
      }
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to delete conversation");
    }
  }

  if (!ready) return null;

  return (
    <div className="min-h-screen">
      <NavBar />
      <div className="mx-auto max-w-6xl px-4 py-10">
        <h1 className="text-2xl font-semibold">Conversations</h1>
        <p className="mt-1 text-sm text-muted">
          Your saved chat threads, most recent first. Open one to read its
          messages, export it (JSON / Markdown / PDF), or delete it.
        </p>

        {error && (
          <div className="mt-4 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-400">
            {error}
          </div>
        )}

        <div className="mt-6 grid gap-6 md:grid-cols-[20rem_1fr]">
          {/* list */}
          <div className="overflow-hidden rounded-xl border border-border">
            <div className="flex items-center justify-between border-b border-border bg-surface-2 px-4 py-2 text-sm text-muted">
              <span>{convs.length} thread{convs.length === 1 ? "" : "s"}</span>
              <button
                onClick={() => void refresh()}
                className="rounded-md px-2 py-1 text-xs hover:bg-surface"
              >
                Refresh
              </button>
            </div>
            <ul className="max-h-[60vh] overflow-y-auto">
              {loadingList && convs.length === 0 && (
                <li className="px-4 py-6 text-center text-sm text-muted">
                  Loading…
                </li>
              )}
              {!loadingList && convs.length === 0 && (
                <li className="px-4 py-6 text-center text-sm text-muted">
                  No conversations yet. Start one in the chat.
                </li>
              )}
              {convs.map((c) => (
                <li key={c.id} className="border-t border-border first:border-t-0">
                  <button
                    onClick={() => void openConversation(c.id)}
                    className={`flex w-full flex-col gap-0.5 px-4 py-3 text-left hover:bg-surface-2 ${
                      selectedId === c.id ? "bg-surface-2" : ""
                    }`}
                  >
                    <span className="truncate text-sm font-medium">
                      {c.title || "Untitled"}
                    </span>
                    <span className="text-xs text-muted">
                      {c.model ? `${c.model} · ` : ""}
                      {new Date(c.updated_at).toLocaleString()}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          {/* detail */}
          <div className="rounded-xl border border-border bg-surface p-5">
            {!selectedId && (
              <div className="grid h-full place-items-center py-16 text-center text-sm text-muted">
                Select a conversation to read it.
              </div>
            )}

            {selectedId && (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="mr-auto text-lg font-semibold">
                    {detail?.title ??
                      convs.find((c) => c.id === selectedId)?.title ??
                      "Conversation"}
                  </h2>
                  {EXPORT_FORMATS.map((fmt) => (
                    <button
                      key={fmt}
                      onClick={() => void onExport(selectedId, fmt)}
                      disabled={exporting !== null}
                      className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-surface-2 disabled:opacity-50"
                    >
                      {exporting === fmt ? "Exporting…" : `Export ${fmt}`}
                    </button>
                  ))}
                  <button
                    onClick={() => void onDelete(selectedId)}
                    className="rounded-md px-2.5 py-1 text-xs text-red-400 hover:bg-red-500/10"
                  >
                    Delete
                  </button>
                </div>

                {detail && (
                  <p className="mt-1 text-xs text-muted">
                    {detail.model ? `${detail.model} · ` : ""}
                    {detail.messages.length} message
                    {detail.messages.length === 1 ? "" : "s"} · created{" "}
                    {new Date(detail.created_at).toLocaleString()}
                  </p>
                )}

                <div className="mt-4 flex flex-col gap-3">
                  {loadingDetail && (
                    <div className="text-sm text-muted">Loading messages…</div>
                  )}
                  {detail && detail.messages.length === 0 && !loadingDetail && (
                    <div className="text-sm text-muted">
                      This conversation has no messages.
                    </div>
                  )}
                  {detail?.messages.map((m) => {
                    const isUser = m.role === "user";
                    return (
                      <div
                        key={m.id}
                        className={`flex w-full ${
                          isUser ? "justify-end" : "justify-start"
                        }`}
                      >
                        <div className="max-w-[85%]">
                          <div
                            className={`rounded-2xl px-4 py-3 text-sm ${
                              isUser
                                ? "rounded-br-sm text-accent-fg"
                                : "rounded-bl-sm border border-border bg-surface-2"
                            }`}
                            style={
                              isUser ? { background: "var(--accent)" } : undefined
                            }
                          >
                            <div className="msg-body">{m.content}</div>
                          </div>
                          <div
                            className={`mt-1 text-[11px] text-muted ${
                              isUser ? "text-right" : "text-left"
                            }`}
                          >
                            {m.role} · {new Date(m.created_at).toLocaleString()}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
