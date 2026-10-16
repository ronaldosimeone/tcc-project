/**
 * Sessão de autenticação (JWT do backend, `POST /v1/auth/login`).
 *
 * O token vive em sessionStorage (some ao fechar a aba — postura mais segura
 * para estação compartilhada de manutenção). `apiFetch` injeta o header
 * `Authorization` em toda chamada à API e, num 401, leva ao /login.
 * Com AUTH_ENABLED=false no backend nada disso é exigido: sem token, as
 * chamadas seguem funcionando como antes.
 */

import { resolveBaseUrl } from "@/lib/api-client";

const TOKEN_KEY = "predictiq:auth";

export type Role = "operator" | "technician" | "engineer" | "admin";

export interface Session {
  accessToken: string;
  username: string;
  role: Role;
  expiresAt: string;
}

interface TokenResponse {
  access_token: string;
  username: string;
  role: Role;
  expires_at: string;
}

export const ROLE_LABEL: Record<Role, string> = {
  operator: "Operador",
  technician: "Técnico",
  engineer: "Engenheiro",
  admin: "Administrador",
};

export const getSession = (): Session | null => {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(TOKEN_KEY);
    if (!raw) return null;
    const session = JSON.parse(raw) as Session;
    if (new Date(session.expiresAt).getTime() <= Date.now()) {
      window.sessionStorage.removeItem(TOKEN_KEY);
      return null;
    }
    return session;
  } catch {
    return null;
  }
};

export const clearSession = (): void => {
  if (typeof window !== "undefined")
    window.sessionStorage.removeItem(TOKEN_KEY);
};

export const login = async ({
  username,
  password,
}: {
  username: string;
  password: string;
}): Promise<Session> => {
  const response = await fetch(`${resolveBaseUrl()}/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username, password }).toString(),
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(
      response.status === 401
        ? "Usuário ou senha inválidos."
        : `Falha no login (HTTP ${response.status}).`,
    );
  }
  const body = (await response.json()) as TokenResponse;
  const session: Session = {
    accessToken: body.access_token,
    username: body.username,
    role: body.role,
    expiresAt: body.expires_at,
  };
  window.sessionStorage.setItem(TOKEN_KEY, JSON.stringify(session));
  return session;
};

/** `fetch` com o token da sessão; 401 limpa a sessão e leva ao /login. */
export const apiFetch = async (
  input: string,
  init: RequestInit = {},
): Promise<Response> => {
  const session = getSession();
  // Sem sessão a chamada fica idêntica a um fetch comum (AUTH_ENABLED=false).
  const request: RequestInit = session
    ? {
        ...init,
        headers: {
          ...Object.fromEntries(new Headers(init.headers).entries()),
          Authorization: `Bearer ${session.accessToken}`,
        },
      }
    : init;
  const response = await fetch(input, request);
  if (
    response.status === 401 &&
    typeof window !== "undefined" &&
    !window.location.pathname.startsWith("/login")
  ) {
    clearSession();
    // Fora do React não há router; o reload completo também descarta o
    // estado em memória da sessão expirada.
    // eslint-disable-next-line @next/next/no-location-assign-relative-destination
    window.location.assign(
      `/login?next=${encodeURIComponent(window.location.pathname)}`,
    );
  }
  return response;
};
