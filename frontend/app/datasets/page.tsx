"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import * as api from "@/lib/api";
import { useAuth } from "@/lib/auth";
import type { DatasetArtifact, ModelInfo } from "@/lib/types";
import { NavBar } from "../components/NavBar";

type Source = "mixle" | "llm";
type Format = "jsonl" | "csv" | "parquet";

interface SchemaRow {
  name: string;
  type: string;
}

const FORMATS: Format[] = ["jsonl", "csv", "parquet"];

export default function DatasetsPage() {
  const router = useRouter();
  const { token, ready } = useAuth();

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetArtifact[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // form state
  const [source, setSource] = useState<Source>("mixle");
  const [model, setModel] = useState("");
  const [n, setN] = useState(100);
  const [seed, setSeed] = useState(0);
  const [format, setFormat] = useState<Format>("jsonl");
  const [prompt, setPrompt] = useState("");
  const [schemaRows, setSchemaRows] = useState<SchemaRow[]>([
    { name: "", type: "" },
  ]);

  useEffect(() => {
    if (ready && !token) router.replace("/login");
  }, [ready, token, router]);

  // mixle source → only mixle/composite models; llm source → llm models.
  const mixleModels = models.filter(
    (m) => m.kind === "mixle" || m.kind === "composite"
  );
  const llmModels = models.filter((m) => m.kind === "llm" || !m.kind);
  const candidates = source === "mixle" ? mixleModels : llmModels;

  useEffect(() => {
    api
      .listModels(token)
      .then(setModels)
      .catch(() => setModels([]));
  }, [token]);

  // keep a sensible default model selected for the chosen source
  useEffect(() => {
    if (candidates.length && !candidates.some((m) => m.id === model)) {
      setModel(candidates[0].id);
    }
  }, [candidates, model]);

  const refresh = useCallback(async () => {
    if (!token) return;
    try {
      setDatasets(await api.listDatasets(token));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to load datasets");
    }
  }, [token]);

  useEffect(() => {
    if (token) void refresh();
  }, [token, refresh]);

  function setRow(i: number, patch: Partial<SchemaRow>) {
    setSchemaRows((rows) =>
      rows.map((r, idx) => (idx === i ? { ...r, ...patch } : r))
    );
  }
  function addRow() {
    setSchemaRows((rows) => [...rows, { name: "", type: "" }]);
  }
  function removeRow(i: number) {
    setSchemaRows((rows) =>
      rows.length === 1 ? rows : rows.filter((_, idx) => idx !== i)
    );
  }

  function buildSchema(): Record<string, string> | null {
    const entries = schemaRows
      .map((r) => [r.name.trim(), r.type.trim()] as const)
      .filter(([k, v]) => k && v);
    if (entries.length === 0) return null;
    return Object.fromEntries(entries);
  }

  async function onGenerate(e: React.FormEvent) {
    e.preventDefault();
    if (!token || !model) return;
    setBusy(true);
    setError(null);
    try {
      await api.generateDataset(token, {
        source,
        model,
        n,
        seed,
        format,
        schema: buildSchema(),
        prompt: source === "llm" && prompt.trim() ? prompt.trim() : null,
      });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "generation failed");
    } finally {
      setBusy(false);
    }
  }

  async function onDownload(d: DatasetArtifact) {
    if (!token) return;
    const url = d.url ?? d.artifact?.url;
    if (!url) {
      setError("This dataset has no downloadable artifact.");
      return;
    }
    try {
      await api.downloadBlob(token, url, `${d.id}.${d.format}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "download failed");
    }
  }

  if (!ready) return null;

  return (
    <div className="min-h-screen">
      <NavBar />
      <div className="mx-auto max-w-4xl px-4 py-10">
        <h1 className="text-2xl font-semibold">Datasets</h1>
        <p className="mt-1 text-sm text-muted">
          Generate labeled data with verifiable labels — sample a mixle
          generative model, or drive an LLM against a small JSON schema. Datasets
          are materialized to the blob store with a download link.
        </p>

        {error && (
          <div className="mt-4 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-400">
            {error}
          </div>
        )}

        <form
          onSubmit={onGenerate}
          className="mt-6 flex flex-col gap-4 rounded-xl border border-border bg-surface p-4"
        >
          <div className="flex flex-wrap items-end gap-4">
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-muted">Source</span>
              <select
                value={source}
                onChange={(e) => setSource(e.target.value as Source)}
                className="rounded-lg border border-border bg-surface-2 px-2 py-1.5 text-sm outline-none focus:border-accent"
              >
                <option value="mixle">mixle model</option>
                <option value="llm">LLM</option>
              </select>
            </label>

            <label className="flex flex-1 flex-col gap-1 text-sm">
              <span className="text-muted">Model</span>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="rounded-lg border border-border bg-surface-2 px-2 py-1.5 text-sm outline-none focus:border-accent"
              >
                {candidates.length === 0 && (
                  <option value="">(no {source} models)</option>
                )}
                {candidates.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex flex-col gap-1 text-sm">
              <span className="text-muted">Rows</span>
              <input
                type="number"
                min={1}
                max={100000}
                value={n}
                onChange={(e) =>
                  setN(Math.max(1, Math.min(100000, Number(e.target.value) || 1)))
                }
                className="w-28 rounded-lg border border-border bg-surface-2 px-2 py-1.5 text-sm outline-none focus:border-accent"
              />
            </label>

            <label className="flex flex-col gap-1 text-sm">
              <span className="text-muted">Seed</span>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value) || 0)}
                className="w-24 rounded-lg border border-border bg-surface-2 px-2 py-1.5 text-sm outline-none focus:border-accent"
              />
            </label>

            <label className="flex flex-col gap-1 text-sm">
              <span className="text-muted">Format</span>
              <select
                value={format}
                onChange={(e) => setFormat(e.target.value as Format)}
                className="rounded-lg border border-border bg-surface-2 px-2 py-1.5 text-sm outline-none focus:border-accent"
              >
                {FORMATS.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {source === "llm" && (
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-muted">Prompt (LLM source)</span>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={2}
                placeholder="Generate realistic customer-support tickets…"
                className="resize-none rounded-lg border border-border bg-surface-2 px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </label>
          )}

          {/* schema editor: column -> type (dict[str, str]) */}
          <div className="flex flex-col gap-2">
            <span className="text-sm text-muted">
              Schema (column → type){" "}
              <span className="text-xs">optional — leave blank to infer</span>
            </span>
            {schemaRows.map((r, i) => (
              <div key={i} className="flex items-center gap-2">
                <input
                  value={r.name}
                  onChange={(e) => setRow(i, { name: e.target.value })}
                  placeholder="column"
                  className="flex-1 rounded-lg border border-border bg-surface-2 px-3 py-1.5 text-sm outline-none focus:border-accent"
                />
                <input
                  value={r.type}
                  onChange={(e) => setRow(i, { type: e.target.value })}
                  placeholder="type (e.g. int, float, string)"
                  className="flex-1 rounded-lg border border-border bg-surface-2 px-3 py-1.5 text-sm outline-none focus:border-accent"
                />
                <button
                  type="button"
                  onClick={() => removeRow(i)}
                  className="rounded-md px-2 py-1 text-muted hover:bg-surface-2 hover:text-fg"
                  title="Remove"
                >
                  ✕
                </button>
              </div>
            ))}
            <button
              type="button"
              onClick={addRow}
              className="self-start rounded-md border border-border px-2.5 py-1 text-xs hover:bg-surface-2"
            >
              + Add column
            </button>
          </div>

          <button
            type="submit"
            disabled={busy || !model}
            className="self-start rounded-lg px-4 py-2 text-sm font-medium text-accent-fg disabled:opacity-60"
            style={{ background: "var(--accent)" }}
          >
            {busy ? "Generating…" : "Generate dataset"}
          </button>
        </form>

        {/* dataset list */}
        <div className="mt-8 overflow-hidden rounded-xl border border-border">
          <table className="w-full text-left text-sm">
            <thead className="bg-surface-2 text-muted">
              <tr>
                <th className="px-4 py-2 font-medium">Source</th>
                <th className="px-4 py-2 font-medium">Model</th>
                <th className="px-4 py-2 font-medium">Rows</th>
                <th className="px-4 py-2 font-medium">Format</th>
                <th className="px-4 py-2 font-medium">Created</th>
                <th className="px-4 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {datasets.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-4 py-6 text-center text-muted">
                    No datasets yet. Generate one above.
                  </td>
                </tr>
              )}
              {datasets.map((d) => (
                <tr key={d.id} className="border-t border-border">
                  <td className="px-4 py-2">{d.source}</td>
                  <td className="px-4 py-2 text-muted">{d.model ?? "—"}</td>
                  <td className="px-4 py-2 text-muted">{d.n_rows}</td>
                  <td className="px-4 py-2 text-muted">{d.format}</td>
                  <td className="px-4 py-2 text-muted">
                    {d.created_at
                      ? new Date(d.created_at).toLocaleString()
                      : "—"}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <button
                      onClick={() => void onDownload(d)}
                      disabled={!d.url && !d.artifact?.url}
                      className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-surface-2 disabled:opacity-40"
                    >
                      Download
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
