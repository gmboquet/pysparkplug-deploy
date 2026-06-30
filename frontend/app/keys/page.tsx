"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import * as api from "@/lib/api";
import { useAuth } from "@/lib/auth";
import type { ApiKeyInfo } from "@/lib/types";
import { NavBar } from "../components/NavBar";

export default function KeysPage() {
  const router = useRouter();
  const { token, ready } = useAuth();
  const [keys, setKeys] = useState<ApiKeyInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [created, setCreated] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (ready && !token) router.replace("/login");
  }, [ready, token, router]);

  const refresh = useCallback(async () => {
    if (!token) return;
    try {
      setKeys(await api.listKeys(token));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to load keys");
    }
  }, [token]);

  useEffect(() => {
    if (token) void refresh();
  }, [token, refresh]);

  async function onCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!token) return;
    setBusy(true);
    setCreated(null);
    try {
      const res = await api.createKey(token, newName.trim() || "default");
      setCreated(res.api_key);
      setNewName("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to create key");
    } finally {
      setBusy(false);
    }
  }

  async function onDelete(id: string) {
    if (!token) return;
    try {
      await api.deleteKey(token, id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to revoke key");
    }
  }

  if (!ready) return null;

  return (
    <div className="min-h-screen">
      <NavBar />
      <div className="mx-auto max-w-3xl px-4 py-10">
        <h1 className="text-2xl font-semibold">API keys</h1>
        <p className="mt-1 text-sm text-muted">
          Use a key as a Bearer token against the OpenAI-compatible{" "}
          <code>/v1</code> routes. The full key is shown once at creation.
        </p>

        {error && (
          <div className="mt-4 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-400">
            {error}
          </div>
        )}

        <form
          onSubmit={onCreate}
          className="mt-6 flex flex-wrap items-end gap-3 rounded-xl border border-border bg-surface p-4"
        >
          <label className="flex flex-1 flex-col gap-1 text-sm">
            <span className="text-muted">New key name</span>
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="e.g. production"
              className="rounded-lg border border-border bg-surface-2 px-3 py-2 outline-none focus:border-accent"
            />
          </label>
          <button
            type="submit"
            disabled={busy}
            className="rounded-lg px-4 py-2 font-medium text-accent-fg disabled:opacity-60"
            style={{ background: "var(--accent)" }}
          >
            Create key
          </button>
        </form>

        {created && (
          <div className="mt-4 rounded-xl border border-accent/40 bg-surface p-4">
            <p className="text-sm font-medium">Your new API key (copy it now):</p>
            <div className="mt-2 flex items-center gap-2">
              <code className="flex-1 overflow-x-auto rounded-lg border border-border bg-surface-2 px-3 py-2 text-xs">
                {created}
              </code>
              <button
                onClick={() => navigator.clipboard?.writeText(created)}
                className="rounded-lg border border-border px-3 py-2 text-sm hover:bg-surface-2"
              >
                Copy
              </button>
            </div>
          </div>
        )}

        <div className="mt-8 overflow-hidden rounded-xl border border-border">
          <table className="w-full text-left text-sm">
            <thead className="bg-surface-2 text-muted">
              <tr>
                <th className="px-4 py-2 font-medium">Name</th>
                <th className="px-4 py-2 font-medium">Prefix</th>
                <th className="px-4 py-2 font-medium">Kind</th>
                <th className="px-4 py-2 font-medium">Created</th>
                <th className="px-4 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {keys.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-6 text-center text-muted">
                    No active keys yet.
                  </td>
                </tr>
              )}
              {keys.map((k) => (
                <tr key={k.id} className="border-t border-border">
                  <td className="px-4 py-2">{k.name}</td>
                  <td className="px-4 py-2 font-mono text-xs">{k.prefix}…</td>
                  <td className="px-4 py-2 text-muted">{k.kind}</td>
                  <td className="px-4 py-2 text-muted">
                    {new Date(k.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <button
                      onClick={() => onDelete(k.id)}
                      className="rounded-md px-2 py-1 text-xs text-red-400 hover:bg-red-500/10"
                    >
                      Revoke
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
