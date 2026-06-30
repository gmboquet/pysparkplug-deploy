"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import * as api from "@/lib/api";
import { ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import type {
  EvolutionRun,
  EvolveObjective,
  EvolvePolicy,
  ModelInfo,
} from "@/lib/types";
import { NavBar } from "../components/NavBar";

const OBJECTIVES: EvolveObjective[] = [
  "nll",
  "log_score",
  "crps",
  "interval",
  "calibration",
];

// Human label for each objective so the dropdown reads clearly.
const OBJECTIVE_LABEL: Record<EvolveObjective, string> = {
  nll: "nll — negative log-likelihood",
  log_score: "log_score — log scoring rule",
  crps: "crps — continuous ranked probability score",
  interval: "interval — interval score",
  calibration: "calibration — calibration error",
};

export default function EvolvePage() {
  const router = useRouter();
  const { token, user, ready } = useAuth();

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState("");
  const [runs, setRuns] = useState<EvolutionRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [forbidden, setForbidden] = useState(false);
  const [busy, setBusy] = useState(false);
  const [lastRun, setLastRun] = useState<EvolutionRun | null>(null);

  // form state
  const [objective, setObjective] = useState<EvolveObjective>("nll");
  const [records, setRecords] = useState("");
  const [promote, setPromote] = useState(true);

  useEffect(() => {
    if (ready && !token) router.replace("/login");
  }, [ready, token, router]);

  // mixle + composite models can self-evolve (the gateway exposes `kind`).
  const mixleModels = models.filter(
    (m) => m.kind === "mixle" || m.kind === "composite"
  );

  useEffect(() => {
    api
      .listModels(token)
      .then(setModels)
      .catch(() => setModels([]));
  }, [token]);

  // keep a valid mixle model selected
  useEffect(() => {
    if (mixleModels.length && !mixleModels.some((m) => m.id === model)) {
      setModel(mixleModels[0].id);
    }
  }, [mixleModels, model]);

  const refresh = useCallback(async () => {
    if (!token || !model) return;
    try {
      setRuns(await api.listEvolutionRuns(token, model));
      setError(null);
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        setForbidden(true);
        return;
      }
      setError(err instanceof Error ? err.message : "failed to load runs");
    }
  }, [token, model]);

  useEffect(() => {
    if (token && model) void refresh();
  }, [token, model, refresh]);

  function parseRecords(): unknown[] | null {
    const text = records.trim();
    if (!text) return [];
    try {
      const parsed = JSON.parse(text);
      if (!Array.isArray(parsed)) {
        setError("Records must be a JSON array.");
        return null;
      }
      return parsed;
    } catch {
      setError("Records is not valid JSON.");
      return null;
    }
  }

  async function onTrigger(e: React.FormEvent) {
    e.preventDefault();
    if (!token || !model) return;
    const parsed = parseRecords();
    if (parsed === null) return;
    setBusy(true);
    setError(null);
    try {
      const policy: EvolvePolicy = { objective };
      const run = await api.triggerEvolution(token, model, policy, {
        records: parsed,
        promote,
      });
      setLastRun(run);
      await refresh();
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        setForbidden(true);
      } else {
        setError(err instanceof Error ? err.message : "evolution run failed");
      }
    } finally {
      setBusy(false);
    }
  }

  async function onRollback() {
    if (!token || !model) return;
    setBusy(true);
    setError(null);
    try {
      await api.rollbackEvolution(token, model);
      await refresh();
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        setForbidden(true);
      } else {
        setError(err instanceof Error ? err.message : "rollback failed");
      }
    } finally {
      setBusy(false);
    }
  }

  if (!ready) return null;

  // Admin-only feature. We detect the flag client-side (user.is_admin) for the
  // happy path, but the gateway is the source of truth — a 403 from any call flips
  // `forbidden` and shows the same graceful state.
  const isAdmin = !!user?.is_admin;

  return (
    <div className="min-h-screen">
      <NavBar />
      <div className="mx-auto max-w-4xl px-4 py-10">
        <h1 className="text-2xl font-semibold">Self-evolution</h1>
        <p className="mt-1 text-sm text-muted">
          Run a measure → propose → verify → promote loop on a hosted mixle model.
          Each run tries an improvement operator, statistically verifies it against a
          holdout, and (optionally) promotes a verified win to serving. Admin only.
        </p>

        {(!isAdmin || forbidden) && (
          <div className="mt-6 rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-400">
            Self-evolution requires an admin account. You can view this page, but
            triggering runs and rollbacks is restricted. Ask an administrator to
            evolve a model on your behalf.
          </div>
        )}

        {error && (
          <div className="mt-4 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-400">
            {error}
          </div>
        )}

        {mixleModels.length === 0 ? (
          <div className="mt-6 rounded-xl border border-border bg-surface px-4 py-6 text-center text-sm text-muted">
            No self-evolvable models available. Only mixle and composite models can
            self-evolve.
          </div>
        ) : (
          <form
            onSubmit={onTrigger}
            className="mt-6 flex flex-col gap-4 rounded-xl border border-border bg-surface p-4"
          >
            <div className="flex flex-wrap items-end gap-4">
              <label className="flex flex-1 flex-col gap-1 text-sm">
                <span className="text-muted">Model</span>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="rounded-lg border border-border bg-surface-2 px-2 py-1.5 text-sm outline-none focus:border-accent"
                >
                  {mixleModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.id}
                    </option>
                  ))}
                </select>
              </label>

              <label className="flex flex-1 flex-col gap-1 text-sm">
                <span className="text-muted">Objective</span>
                <select
                  value={objective}
                  onChange={(e) =>
                    setObjective(e.target.value as EvolveObjective)
                  }
                  className="rounded-lg border border-border bg-surface-2 px-2 py-1.5 text-sm outline-none focus:border-accent"
                >
                  {OBJECTIVES.map((o) => (
                    <option key={o} value={o}>
                      {OBJECTIVE_LABEL[o]}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <label className="flex flex-col gap-1 text-sm">
              <span className="text-muted">
                Records (JSON array){" "}
                <span className="text-xs">
                  optional — new data to improve on, combined with retained fit data
                </span>
              </span>
              <textarea
                value={records}
                onChange={(e) => setRecords(e.target.value)}
                rows={4}
                placeholder='[ {"x": 1.0, "y": 2.0}, ... ]'
                className="resize-none rounded-lg border border-border bg-surface-2 px-3 py-2 font-mono text-xs outline-none focus:border-accent"
              />
            </label>

            <label className="flex cursor-pointer items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={promote}
                onChange={(e) => setPromote(e.target.checked)}
                className="accent-accent"
              />
              Promote a verified win to serving
            </label>

            <div className="flex flex-wrap gap-2">
              <button
                type="submit"
                disabled={busy || !isAdmin || !model}
                className="rounded-lg px-4 py-2 text-sm font-medium text-accent-fg disabled:opacity-60"
                style={{ background: "var(--accent)" }}
                title={isAdmin ? "" : "Admin only"}
              >
                {busy ? "Running…" : "Run evolution"}
              </button>
              <button
                type="button"
                onClick={onRollback}
                disabled={busy || !isAdmin || !model}
                className="rounded-lg border border-border px-4 py-2 text-sm hover:bg-surface-2 disabled:opacity-60"
                title={
                  isAdmin
                    ? "Restore the previous champion after a promotion"
                    : "Admin only"
                }
              >
                Roll back
              </button>
            </div>
          </form>
        )}

        {/* last run verdict */}
        {lastRun && (
          <div className="mt-6 rounded-xl border border-border bg-surface p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium">Latest run</span>
              <Verdict run={lastRun} />
            </div>
            <dl className="mt-3 grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
              <Field label="Operator" value={lastRun.operator ?? "—"} />
              <Field label="Objective" value={lastRun.objective} />
              <Field
                label="Δ (improvement)"
                value={formatDelta(lastRun.delta)}
              />
              <Field
                label="Records used"
                value={String(lastRun.n_data ?? 0)}
              />
            </dl>
            {lastRun.error && (
              <p className="mt-2 text-sm text-red-400">{lastRun.error}</p>
            )}
          </div>
        )}

        {/* lineage table */}
        <h2 className="mt-10 text-lg font-medium">Lineage</h2>
        <div className="mt-3 overflow-hidden rounded-xl border border-border">
          <table className="w-full text-left text-sm">
            <thead className="bg-surface-2 text-muted">
              <tr>
                <th className="px-4 py-2 font-medium">When</th>
                <th className="px-4 py-2 font-medium">Operator</th>
                <th className="px-4 py-2 font-medium">Objective</th>
                <th className="px-4 py-2 font-medium">Δ</th>
                <th className="px-4 py-2 font-medium">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {runs.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-6 text-center text-muted">
                    No evolution runs yet for this model.
                  </td>
                </tr>
              )}
              {runs.map((r) => (
                <tr key={r.id} className="border-t border-border">
                  <td className="px-4 py-2 text-muted">
                    {new Date(r.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-2">{r.operator ?? "—"}</td>
                  <td className="px-4 py-2 text-muted">{r.objective}</td>
                  <td className="px-4 py-2 text-muted">{formatDelta(r.delta)}</td>
                  <td className="px-4 py-2">
                    <Verdict run={r} />
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

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-muted">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  );
}

// A compact verified/promoted/error status pill.
function Verdict({ run }: { run: EvolutionRun }) {
  if (run.error) {
    return (
      <span className="rounded-full border border-red-500/40 bg-red-500/10 px-2 py-0.5 text-xs text-red-400">
        error
      </span>
    );
  }
  if (run.promoted) {
    return (
      <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-400">
        promoted
      </span>
    );
  }
  if (run.verified) {
    return (
      <span className="rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 text-xs text-accent">
        verified
      </span>
    );
  }
  return (
    <span className="rounded-full border border-border bg-surface px-2 py-0.5 text-xs text-muted">
      no change
    </span>
  );
}

function formatDelta(delta: number): string {
  if (!Number.isFinite(delta) || delta === 0) return "0";
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta.toFixed(4)}`;
}
