"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "@/lib/api";
import { useAuth } from "@/lib/auth";
import type { RagDocument, RagHit } from "@/lib/types";
import { NavBar } from "../components/NavBar";

const ACCEPT = ".pdf,.docx,.pptx,.txt,.md";

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} chars`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}k chars`;
  return `${(n / (1024 * 1024)).toFixed(1)}M chars`;
}

export default function DocumentsPage() {
  const router = useRouter();
  const { token, ready } = useAuth();

  const [docs, setDocs] = useState<RagDocument[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [hits, setHits] = useState<RagHit[] | null>(null);

  useEffect(() => {
    if (ready && !token) router.replace("/login");
  }, [ready, token, router]);

  const refresh = useCallback(async () => {
    if (!token) return;
    try {
      setDocs(await api.listDocuments(token));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to load documents");
    }
  }, [token]);

  useEffect(() => {
    if (token) void refresh();
  }, [token, refresh]);

  async function onUpload(files: FileList | null) {
    if (!token || !files || files.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      for (const f of Array.from(files)) {
        await api.uploadDocument(token, f);
      }
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "upload failed");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function onSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!token || !query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      setHits(await api.ragSearch(token, query.trim(), { k: 8, namespace: "document" }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "search failed");
      setHits(null);
    } finally {
      setSearching(false);
    }
  }

  if (!ready) return null;

  return (
    <div className="min-h-screen">
      <NavBar />
      <div className="mx-auto max-w-4xl px-4 py-10">
        <h1 className="text-2xl font-semibold">Documents</h1>
        <p className="mt-1 text-sm text-muted">
          Upload PDF / DOCX / PPTX / TXT files. Each is parsed, chunked, and
          indexed so the chat can retrieve from it (enable{" "}
          <strong className="text-fg">Use my documents</strong> there). Search
          your indexed chunks below.
        </p>

        {error && (
          <div className="mt-4 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* upload */}
        <div className="mt-6 flex flex-wrap items-center gap-3 rounded-xl border border-border bg-surface p-4">
          <input
            ref={fileRef}
            type="file"
            multiple
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => void onUpload(e.target.files)}
          />
          <button
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            className="rounded-lg px-4 py-2 font-medium text-accent-fg disabled:opacity-60"
            style={{ background: "var(--accent)" }}
          >
            {uploading ? "Uploading…" : "Upload document"}
          </button>
          <span className="text-xs text-muted">PDF, DOCX, PPTX, TXT, MD</span>
        </div>

        {/* document list */}
        <div className="mt-8 overflow-hidden rounded-xl border border-border">
          <table className="w-full text-left text-sm">
            <thead className="bg-surface-2 text-muted">
              <tr>
                <th className="px-4 py-2 font-medium">Filename</th>
                <th className="px-4 py-2 font-medium">Type</th>
                <th className="px-4 py-2 font-medium">Chunks</th>
                <th className="px-4 py-2 font-medium">Size</th>
                <th className="px-4 py-2 font-medium">Uploaded</th>
              </tr>
            </thead>
            <tbody>
              {docs.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-6 text-center text-muted">
                    No documents yet. Upload one to start building your RAG index.
                  </td>
                </tr>
              )}
              {docs.map((d) => (
                <tr key={d.id} className="border-t border-border">
                  <td className="px-4 py-2">{d.filename}</td>
                  <td className="px-4 py-2 text-muted">{d.content_type || "—"}</td>
                  <td className="px-4 py-2 text-muted">{d.n_chunks}</td>
                  <td className="px-4 py-2 text-muted">{fmtBytes(d.n_chars)}</td>
                  <td className="px-4 py-2 text-muted">
                    {d.created_at
                      ? new Date(d.created_at).toLocaleDateString()
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* search */}
        <h2 className="mt-10 text-lg font-semibold">Search your documents</h2>
        <form onSubmit={onSearch} className="mt-3 flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask something about your uploaded documents…"
            className="flex-1 rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <button
            type="submit"
            disabled={searching || !query.trim()}
            className="rounded-lg px-4 py-2 text-sm font-medium text-accent-fg disabled:opacity-60"
            style={{ background: "var(--accent)" }}
          >
            {searching ? "Searching…" : "Search"}
          </button>
        </form>

        {hits !== null && (
          <div className="mt-4 flex flex-col gap-3">
            {hits.length === 0 && (
              <div className="rounded-lg border border-border bg-surface px-4 py-6 text-center text-sm text-muted">
                No matching chunks. Try a different query or upload more documents.
              </div>
            )}
            {hits.map((h) => {
              const filename =
                typeof h.meta?.filename === "string" ? h.meta.filename : null;
              return (
                <div
                  key={h.id}
                  className="rounded-xl border border-border bg-surface p-4"
                >
                  <div className="flex items-center justify-between text-xs text-muted">
                    <span>
                      {filename ? `${filename} · ` : ""}
                      {h.namespace}
                    </span>
                    <span className="rounded-full border border-border px-2 py-0.5">
                      score {h.score.toFixed(3)}
                    </span>
                  </div>
                  <p className="msg-body mt-2 text-sm">{h.text}</p>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
