"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import * as api from "@/lib/api";
import { useAuth } from "@/lib/auth";
import type { GeneratedImage, ModelInfo } from "@/lib/types";
import { NavBar } from "../components/NavBar";

const SIZES = ["256x256", "512x512", "1024x1024"];

function imageSrc(img: GeneratedImage): string | null {
  if (img.b64_json) return `data:image/png;base64,${img.b64_json}`;
  if (img.url) return api.resolveUrl(img.url);
  return null;
}

export default function ImagesPage() {
  const router = useRouter();
  const { token, ready } = useAuth();

  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState("");
  const [imageModels, setImageModels] = useState<ModelInfo[]>([]);
  const [size, setSize] = useState("512x512");
  const [n, setN] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [images, setImages] = useState<GeneratedImage[]>([]);

  useEffect(() => {
    if (ready && !token) router.replace("/login");
  }, [ready, token, router]);

  // Discover image-capable models so the picker only offers valid ones (optional —
  // an empty model lets the gateway pick the first image-capable model).
  useEffect(() => {
    if (!token) return;
    api
      .listModels(token)
      .then((all) =>
        setImageModels(
          all.filter((m) => m.capabilities?.includes("image_generation"))
        )
      )
      .catch(() => setImageModels([]));
  }, [token]);

  async function onGenerate(e: React.FormEvent) {
    e.preventDefault();
    if (!token || !prompt.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const out = await api.generateImages(token, {
        prompt: prompt.trim(),
        model: model || undefined,
        n,
        size,
      });
      setImages(out);
      if (out.length === 0) {
        setError("The image backend returned no images.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "image generation failed");
    } finally {
      setBusy(false);
    }
  }

  if (!ready) return null;

  return (
    <div className="min-h-screen">
      <NavBar />
      <div className="mx-auto max-w-4xl px-4 py-10">
        <h1 className="text-2xl font-semibold">Image generation</h1>
        <p className="mt-1 text-sm text-muted">
          Describe an image and the gateway routes it to an image-capable model.
          The default deployment ships a stub model, so results may be
          placeholders until a real image backend is registered.
        </p>

        {error && (
          <div className="mt-4 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-400">
            {error}
          </div>
        )}

        <form
          onSubmit={onGenerate}
          className="mt-6 flex flex-col gap-3 rounded-xl border border-border bg-surface p-4"
        >
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            placeholder="A calibrated probability distribution as a watercolor landscape…"
            className="resize-none rounded-lg border border-border bg-surface-2 px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <div className="flex flex-wrap items-end gap-4">
            {imageModels.length > 0 && (
              <label className="flex flex-col gap-1 text-sm">
                <span className="text-muted">Model</span>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="rounded-lg border border-border bg-surface-2 px-2 py-1.5 text-sm outline-none focus:border-accent"
                >
                  <option value="">auto</option>
                  {imageModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.id}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-muted">Size</span>
              <select
                value={size}
                onChange={(e) => setSize(e.target.value)}
                className="rounded-lg border border-border bg-surface-2 px-2 py-1.5 text-sm outline-none focus:border-accent"
              >
                {SIZES.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-muted">Count</span>
              <input
                type="number"
                min={1}
                max={4}
                value={n}
                onChange={(e) =>
                  setN(Math.max(1, Math.min(4, Number(e.target.value) || 1)))
                }
                className="w-20 rounded-lg border border-border bg-surface-2 px-2 py-1.5 text-sm outline-none focus:border-accent"
              />
            </label>
            <button
              type="submit"
              disabled={busy || !prompt.trim()}
              className="ml-auto rounded-lg px-4 py-2 text-sm font-medium text-accent-fg disabled:opacity-60"
              style={{ background: "var(--accent)" }}
            >
              {busy ? "Generating…" : "Generate"}
            </button>
          </div>
        </form>

        {images.length > 0 && (
          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            {images.map((img, i) => {
              const src = imageSrc(img);
              return (
                <div
                  key={i}
                  className="overflow-hidden rounded-xl border border-border bg-surface"
                >
                  {src ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={src}
                      alt={`generated ${i + 1}`}
                      className="w-full"
                    />
                  ) : (
                    <div className="grid h-48 place-items-center text-sm text-muted">
                      (no image data returned)
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
