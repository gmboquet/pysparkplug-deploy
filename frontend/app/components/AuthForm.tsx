"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import Link from "next/link";

import { ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { NavBar } from "./NavBar";

export function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const router = useRouter();
  const { login, signup } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const isSignup = mode === "signup";

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (isSignup) await signup(email, password);
      else await login(email, password);
      router.push("/chat");
    } catch (err) {
      if (err instanceof ApiError) setError(err.message);
      else
        setError(
          "Could not reach the gateway. Is it running at the configured API base?"
        );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen">
      <NavBar />
      <div className="mx-auto flex max-w-md flex-col px-4 pt-16">
        <h1 className="text-2xl font-semibold">
          {isSignup ? "Create your account" : "Welcome back"}
        </h1>
        <p className="mt-1 text-sm text-muted">
          {isSignup
            ? "Sign up to get an API key and start chatting."
            : "Log in to continue to the chat."}
        </p>

        <form onSubmit={onSubmit} className="mt-8 flex flex-col gap-3">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-muted">Email</span>
            <input
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="rounded-lg border border-border bg-surface px-3 py-2 outline-none focus:border-accent"
              placeholder="you@example.com"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-muted">Password</span>
            <input
              type="password"
              required
              minLength={6}
              autoComplete={isSignup ? "new-password" : "current-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="rounded-lg border border-border bg-surface px-3 py-2 outline-none focus:border-accent"
              placeholder="••••••••"
            />
          </label>

          {error && (
            <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-400">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={busy}
            className="mt-2 rounded-lg px-3 py-2 font-medium text-accent-fg disabled:opacity-60"
            style={{ background: "var(--accent)" }}
          >
            {busy ? "Please wait…" : isSignup ? "Sign up" : "Log in"}
          </button>
        </form>

        <p className="mt-6 text-center text-sm text-muted">
          {isSignup ? (
            <>
              Already have an account?{" "}
              <Link href="/login" className="text-accent">
                Log in
              </Link>
            </>
          ) : (
            <>
              New here?{" "}
              <Link href="/signup" className="text-accent">
                Create an account
              </Link>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
