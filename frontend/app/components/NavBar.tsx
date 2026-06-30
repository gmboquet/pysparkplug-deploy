"use client";

import Link from "next/link";
import { useAuth } from "@/lib/auth";

export function Logo({ className = "" }: { className?: string }) {
  return (
    <Link href="/" className={`flex items-center gap-2 font-semibold ${className}`}>
      <span
        className="grid h-7 w-7 place-items-center rounded-lg text-sm font-bold text-accent-fg"
        style={{ background: "var(--accent)" }}
      >
        m
      </span>
      <span>mixle</span>
    </Link>
  );
}

export function NavBar() {
  const { user, ready, logout } = useAuth();

  return (
    <header className="sticky top-0 z-20 border-b border-border bg-bg/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        <Logo />
        <nav className="flex items-center gap-1 text-sm">
          <Link
            href="/chat"
            className="rounded-md px-3 py-1.5 text-muted hover:bg-surface-2 hover:text-fg"
          >
            Chat
          </Link>
          {ready && user ? (
            <>
              <Link
                href="/keys"
                className="rounded-md px-3 py-1.5 text-muted hover:bg-surface-2 hover:text-fg"
              >
                API keys
              </Link>
              <span className="hidden px-2 text-muted sm:inline">{user.email}</span>
              <button
                onClick={logout}
                className="rounded-md px-3 py-1.5 text-muted hover:bg-surface-2 hover:text-fg"
              >
                Log out
              </button>
            </>
          ) : (
            <>
              <Link
                href="/login"
                className="rounded-md px-3 py-1.5 text-muted hover:bg-surface-2 hover:text-fg"
              >
                Log in
              </Link>
              <Link
                href="/signup"
                className="rounded-md px-3 py-1.5 font-medium text-accent-fg"
                style={{ background: "var(--accent)" }}
              >
                Sign up
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
