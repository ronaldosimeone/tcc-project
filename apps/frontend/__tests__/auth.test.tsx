/**
 * Sessão / login (RBAC) — lib/auth.ts e LoginForm.
 *
 * 1. Login grava a sessão e `apiFetch` passa a enviar `Authorization: Bearer`.
 * 2. Sem sessão, `apiFetch` é idêntico a um fetch comum (AUTH_ENABLED=false).
 * 3. Sessão expirada é descartada.
 * 4. 401 da API limpa a sessão e redireciona ao /login com `next`.
 * 5. LoginForm mostra erro único em credencial inválida.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import LoginForm from "@/components/auth/login-form";
import { apiFetch, clearSession, getSession, login } from "@/lib/auth";

vi.mock("next/navigation", () => ({ useRouter: () => ({ replace: vi.fn() }) }));

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_API_URL", "/api");
  vi.stubGlobal("fetch", fetchMock);
  clearSession();
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  fetchMock.mockReset();
});

describe("lib/auth", () => {
  it("login guarda a sessão e apiFetch envia o bearer token", async () => {
    const expires = new Date(Date.now() + 3_600_000).toISOString();
    fetchMock.mockResolvedValueOnce(
      json({
        access_token: "tok",
        username: "tec",
        role: "technician",
        expires_at: expires,
      }),
    );
    await login({ username: "tec", password: "pw" });

    expect(getSession()).toMatchObject({ username: "tec", role: "technician" });
    const [, init] = fetchMock.mock.calls[0];
    expect(init.body).toBe("username=tec&password=pw");

    fetchMock.mockResolvedValueOnce(json({}));
    await apiFetch("/api/v1/events", {
      headers: { "Content-Type": "application/json" },
    });
    expect(fetchMock.mock.calls[1][1].headers).toEqual({
      "content-type": "application/json",
      Authorization: "Bearer tok",
    });
  });

  it("sem sessão apiFetch repassa a requisição sem alterar", async () => {
    fetchMock.mockResolvedValueOnce(json({}));
    const init = {
      headers: { "Content-Type": "application/json" },
      cache: "no-store" as const,
    };
    await apiFetch("/api/x", init);
    expect(fetchMock).toHaveBeenCalledWith("/api/x", init);
  });

  it("sessão expirada é descartada", () => {
    window.sessionStorage.setItem(
      "predictiq:auth",
      JSON.stringify({
        accessToken: "t",
        username: "a",
        role: "operator",
        expiresAt: "2000-01-01T00:00:00Z",
      }),
    );
    expect(getSession()).toBeNull();
  });

  it("401 limpa a sessão e redireciona ao login", async () => {
    const assign = vi.fn();
    vi.stubGlobal("location", {
      ...window.location,
      pathname: "/history",
      assign,
    });
    window.sessionStorage.setItem(
      "predictiq:auth",
      JSON.stringify({
        accessToken: "t",
        username: "a",
        role: "operator",
        expiresAt: "2999-01-01T00:00:00Z",
      }),
    );
    fetchMock.mockResolvedValueOnce(json({ detail: "expired" }, 401));
    await apiFetch("/api/v1/events/1");
    expect(getSession()).toBeNull();
    expect(assign).toHaveBeenCalledWith("/login?next=%2Fhistory");
  });
});

describe("LoginForm", () => {
  it("mostra erro em credencial inválida", async () => {
    fetchMock.mockResolvedValueOnce(
      json({ detail: "Invalid credentials." }, 401),
    );
    render(<LoginForm next="/" />);
    fireEvent.change(screen.getByLabelText("Usuário"), {
      target: { value: "x" },
    });
    fireEvent.change(screen.getByLabelText("Senha"), {
      target: { value: "y" },
    });
    fireEvent.click(screen.getByRole("button", { name: /entrar/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Usuário ou senha inválidos.",
    );
  });
});
