/**
 * RF-38 — voto do técnico por trecho citado (SuggestionFeedback).
 *
 * 1. Um voto por arquivo+página (trechos repetidos da mesma página viram um).
 * 2. Enviar exige a resposta "o plano foi útil?".
 * 3. O payload leva só os trechos votados, o status RN-04 e o comentário.
 * 4. 401/403 mostra a mensagem de permissão e mantém o formulário.
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SuggestionFeedback } from "@/components/maintenance-assistant/suggestion-feedback";
import type { ManualReference } from "@/lib/api-client";

const ref = (
  file_name: string,
  page: number,
  chunk_index: number,
): ManualReference => ({
  file_name,
  page,
  chunk_index,
  source: `${file_name}#${chunk_index}`,
  score: 0.8,
});

const REFERENCES: ManualReference[] = [
  ref("apu.pdf", 12, 0),
  ref("apu.pdf", 12, 1),
  ref("dryer.pdf", 3, 0),
];

const REQUEST = {
  failure_probability: 0.91,
  equipment_name: "Compressor da APU",
  symptom_description: "queda de pressão",
};

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_API_URL", "/api");
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  fetchMock.mockReset();
});

const renderFeedback = () =>
  render(
    <SuggestionFeedback
      request={REQUEST}
      references={REFERENCES}
      groundingStatus="grounded"
    />,
  );

describe("SuggestionFeedback", () => {
  it("mostra um voto por arquivo+página", () => {
    renderFeedback();
    expect(screen.getAllByRole("group")).toHaveLength(3); // 2 trechos + plano
    expect(screen.getByText("apu.pdf — p. 12")).toBeInTheDocument();
    expect(screen.getByText("dryer.pdf — p. 3")).toBeInTheDocument();
  });

  it("só habilita o envio depois da resposta sobre a utilidade", () => {
    renderFeedback();
    const send = screen.getByRole("button", { name: "Enviar avaliação" });
    expect(send).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Plano: relevante" }));
    expect(send).toBeEnabled();
  });

  it("envia só os trechos votados, com status RN-04 e comentário", async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({ id: 7, created_at: "2026-10-16T12:00:00Z" }),
        {
          status: 201,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );
    renderFeedback();
    fireEvent.click(
      screen.getByRole("button", {
        name: "dryer.pdf, página 3: não relevante",
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Plano: relevante" }));
    fireEvent.change(screen.getByLabelText("Comentário opcional"), {
      target: { value: "  faltou o torque  " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Enviar avaliação" }));

    await screen.findByTestId("suggestion-feedback-sent");
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/v1/maintenance/feedback");
    expect(JSON.parse(String(init.body))).toEqual({
      equipment_name: "Compressor da APU",
      symptom_description: "queda de pressão",
      grounding_status: "grounded",
      useful: true,
      reference_votes: [{ file_name: "dryer.pdf", page: 3, helpful: false }],
      comment: "faltou o torque",
    });
  });

  it("mostra a mensagem de permissão em 403", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 403 }));
    renderFeedback();
    fireEvent.click(
      screen.getByRole("button", { name: "Plano: não relevante" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Enviar avaliação" }));

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Entre como técnico para registrar o feedback.",
      ),
    );
    expect(screen.getByTestId("suggestion-feedback")).toBeInTheDocument();
  });
});
