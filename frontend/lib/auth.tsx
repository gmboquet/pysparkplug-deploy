"use client";

// Minimal auth context: persists a Bearer token (+ user) in localStorage and
// exposes login/signup/logout. The same token works for the OpenAI-compatible
// routes and the platform API (the gateway resolves it as an API key).

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";

import * as api from "./api";
import type { PublicUser } from "./types";

const TOKEN_KEY = "mixle.token";
const USER_KEY = "mixle.user";

interface AuthState {
  token: string | null;
  user: PublicUser | null;
  ready: boolean;
  signup: (email: string, password: string) => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<PublicUser | null>(null);
  const [ready, setReady] = useState(false);

  // Hydrate from localStorage on mount.
  useEffect(() => {
    try {
      const t = localStorage.getItem(TOKEN_KEY);
      const u = localStorage.getItem(USER_KEY);
      if (t) setToken(t);
      if (u) setUser(JSON.parse(u));
    } catch {
      /* ignore */
    }
    setReady(true);
  }, []);

  const persist = useCallback((t: string, u: PublicUser) => {
    setToken(t);
    setUser(u);
    try {
      localStorage.setItem(TOKEN_KEY, t);
      localStorage.setItem(USER_KEY, JSON.stringify(u));
    } catch {
      /* ignore */
    }
  }, []);

  const signup = useCallback(
    async (email: string, password: string) => {
      const res = await api.signup(email, password);
      persist(res.api_key, res.user);
    },
    [persist]
  );

  const login = useCallback(
    async (email: string, password: string) => {
      const res = await api.login(email, password);
      persist(res.token, res.user);
    },
    [persist]
  );

  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
    try {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    } catch {
      /* ignore */
    }
  }, []);

  const value = useMemo<AuthState>(
    () => ({ token, user, ready, signup, login, logout }),
    [token, user, ready, signup, login, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within <AuthProvider>");
  return ctx;
}
