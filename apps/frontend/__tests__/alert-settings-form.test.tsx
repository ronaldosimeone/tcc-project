/**
 * Testes de AlertSettingsForm — RF-25 / RNF-49.
 *
 * `@/lib/api-client` é mockado (mesmo padrão de model-status-card.test.tsx)
 * — nenhuma chamada de rede real. O slider Radix expõe `role="slider"` e o
 * switch expõe `role="switch"`, ambos com `aria-*` — usados aqui em vez de
 * simular arraste de mouse/clique físico em pixel (não confiável em
 * jsdom), o padrão recomendado para testar primitivos Radix.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AlertSettingsForm } from "@/components/alert-settings-form";
import type { AlertSettingsResponse } from "@/lib/api-client";

vi.mock("@/lib/api-client", () => ({
  getAlertSettings: vi.fn(),
  updateAlertSettings: vi.fn(),
  testAlertNotification: vi.fn(),
}));

import {
  getAlertSettings,
  testAlertNotification,
  updateAlertSettings,
} from "@/lib/api-client";

const mockedGet = getAlertSettings as unknown as ReturnType<typeof vi.fn>;
const mockedUpdate = updateAlertSettings as unknown as ReturnType<typeof vi.fn>;
const mockedTest = testAlertNotification as unknown as ReturnType<typeof vi.fn>;

const DEFAULTS: AlertSettingsResponse = {
  alert_threshold: 0.85,
  telegram_enabled: true,
  email_enabled: false,
  alert_email: null,
};

beforeEach(() => {
  vi.clearAllMocks();
  mockedGet.mockResolvedValue(DEFAULTS);
  mockedUpdate.mockImplementation(
    async (payload: AlertSettingsResponse) => payload,
  );
  mockedTest.mockResolvedValue({ message: "Telegram: enviado" });
});

async function renderAndWaitForSlider() {
  render(<AlertSettingsForm />);
  const slider = await screen.findByRole("slider");
  return slider;
}

describe("AlertSettingsForm", () => {
  // A) Renderização inicial ---------------------------------------------
  it("renderiza título, slider, switches, input de e-mail e botões", async () => {
    mockedGet.mockResolvedValue({
      alert_threshold: 0.75,
      telegram_enabled: true,
      email_enabled: true,
      alert_email: "alertas@empresa.com.br",
    });
    await renderAndWaitForSlider();

    expect(screen.getByRole("slider")).toBeInTheDocument();
    expect(
      screen.getByRole("switch", { name: /ativar notificações por telegram/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("switch", { name: /ativar notificações por e-mail/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/e-mail para receber alertas/i)).toHaveValue(
      "alertas@empresa.com.br",
    );
    expect(
      screen.getByRole("button", { name: /salvar configurações/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /testar notificação/i }),
    ).toBeInTheDocument();
  });

  // E) GET inicial preenche todos os campos --------------------------------
  it("chama getAlertSettings() no mount e reflete threshold + canais + e-mail", async () => {
    mockedGet.mockResolvedValue({
      alert_threshold: 0.75,
      telegram_enabled: true,
      email_enabled: true,
      alert_email: "alertas@empresa.com.br",
    });
    const slider = await renderAndWaitForSlider();

    expect(mockedGet).toHaveBeenCalledTimes(1);
    expect(slider).toHaveAttribute("aria-valuenow", "0.75");
    expect(screen.getByTestId("alert-threshold-value")).toHaveTextContent(
      "75%",
    );
    expect(
      screen.getByRole("switch", { name: /ativar notificações por telegram/i }),
    ).toHaveAttribute("aria-checked", "true");
    expect(
      screen.getByRole("switch", { name: /ativar notificações por e-mail/i }),
    ).toHaveAttribute("aria-checked", "true");
  });

  // I) Loading -------------------------------------------------------------
  it("mostra estado de carregamento antes do GET resolver", async () => {
    let resolveGet: (v: AlertSettingsResponse) => void = () => {};
    mockedGet.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveGet = resolve;
        }),
    );
    render(<AlertSettingsForm />);
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();

    resolveGet(DEFAULTS);
    await waitFor(() => {
      expect(screen.getByRole("slider")).toBeInTheDocument();
    });
  });

  // B/C) Slider mínimo/máximo -----------------------------------------------
  it("slider tem mínimo 0.5 e máximo 0.95", async () => {
    const slider = await renderAndWaitForSlider();
    expect(slider).toHaveAttribute("aria-valuemin", "0.5");
    expect(slider).toHaveAttribute("aria-valuemax", "0.95");
  });

  // D) Alteração do valor ---------------------------------------------------
  it("altera o valor exibido ao mover o slider (0.85 -> 0.75)", async () => {
    const slider = await renderAndWaitForSlider();
    slider.focus();
    for (let i = 0; i < 10; i += 1) {
      fireEvent.keyDown(slider, { key: "ArrowLeft" });
    }
    expect(slider).toHaveAttribute("aria-valuenow", "0.75");
    expect(screen.getByTestId("alert-threshold-value")).toHaveTextContent(
      "75%",
    );
  });

  // Canais — Telegram ON/OFF -------------------------------------------------
  it("desliga Telegram e reflete no switch", async () => {
    await renderAndWaitForSlider();
    const telegramSwitch = screen.getByRole("switch", {
      name: /ativar notificações por telegram/i,
    });
    expect(telegramSwitch).toHaveAttribute("aria-checked", "true");
    fireEvent.click(telegramSwitch);
    expect(telegramSwitch).toHaveAttribute("aria-checked", "false");
  });

  // Canais — E-mail ON/OFF + input -------------------------------------------
  it("habilita e-mail e mostra o campo de endereço", async () => {
    await renderAndWaitForSlider();
    expect(
      screen.queryByLabelText(/e-mail para receber alertas/i),
    ).not.toBeInTheDocument();

    const emailSwitch = screen.getByRole("switch", {
      name: /ativar notificações por e-mail/i,
    });
    fireEvent.click(emailSwitch);

    expect(
      screen.getByLabelText(/e-mail para receber alertas/i),
    ).toBeInTheDocument();
  });

  it("insere um endereço de e-mail e o valor reflete no input", async () => {
    await renderAndWaitForSlider();
    fireEvent.click(
      screen.getByRole("switch", { name: /ativar notificações por e-mail/i }),
    );
    const emailInput = screen.getByLabelText(/e-mail para receber alertas/i);
    fireEvent.change(emailInput, {
      target: { value: "alertas@empresa.com.br" },
    });
    expect(emailInput).toHaveValue("alertas@empresa.com.br");
  });

  // 18) Validação de e-mail (frontend) --------------------------------------
  it.each([
    ["alertas@empresa.com.br", true],
    ["teste", false],
    ["teste@", false],
    ["@empresa.com", false],
  ])(
    "e-mail '%s' -> válido=%s (bloqueia/libera Salvar)",
    async (email, valid) => {
      await renderAndWaitForSlider();
      fireEvent.click(
        screen.getByRole("switch", { name: /ativar notificações por e-mail/i }),
      );
      fireEvent.change(screen.getByLabelText(/e-mail para receber alertas/i), {
        target: { value: email },
      });

      const saveButton = screen.getByRole("button", {
        name: /salvar configurações/i,
      });
      if (valid) {
        expect(saveButton).toBeEnabled();
      } else {
        expect(saveButton).toBeDisabled();
      }
    },
  );

  it("bloqueia Salvar quando e-mail está habilitado sem endereço", async () => {
    await renderAndWaitForSlider();
    fireEvent.click(
      screen.getByRole("switch", { name: /ativar notificações por e-mail/i }),
    );
    expect(
      screen.getByRole("button", { name: /salvar configurações/i }),
    ).toBeDisabled();
    expect(screen.getByText(/informe um e-mail válido/i)).toBeInTheDocument();
  });

  it("bloqueia Salvar quando nenhum canal está habilitado", async () => {
    await renderAndWaitForSlider();
    fireEvent.click(
      screen.getByRole("switch", { name: /ativar notificações por telegram/i }),
    );
    expect(
      screen.getByRole("button", { name: /salvar configurações/i }),
    ).toBeDisabled();
    expect(screen.getByText(/habilite ao menos um canal/i)).toBeInTheDocument();
  });

  // F/G) PUT ao salvar + sucesso ---------------------------------------------
  it("chama updateAlertSettings com o payload correto e mostra confirmação", async () => {
    await renderAndWaitForSlider();
    fireEvent.click(
      screen.getByRole("switch", { name: /ativar notificações por e-mail/i }),
    );
    fireEvent.change(screen.getByLabelText(/e-mail para receber alertas/i), {
      target: { value: "alertas@empresa.com.br" },
    });

    fireEvent.click(
      screen.getByRole("button", { name: /salvar configurações/i }),
    );

    await waitFor(() => {
      expect(mockedUpdate).toHaveBeenCalledWith({
        alert_threshold: 0.85,
        telegram_enabled: true,
        email_enabled: true,
        alert_email: "alertas@empresa.com.br",
      });
    });
    await waitFor(() => {
      expect(screen.getByText(/configuração salva/i)).toBeInTheDocument();
    });
  });

  it("envia alert_email null quando e-mail está desabilitado", async () => {
    await renderAndWaitForSlider();
    fireEvent.click(
      screen.getByRole("button", { name: /salvar configurações/i }),
    );
    await waitFor(() => {
      expect(mockedUpdate).toHaveBeenCalledWith(
        expect.objectContaining({ email_enabled: false, alert_email: null }),
      );
    });
  });

  // K) Botão desabilitado durante o save --------------------------------------
  it("desabilita o botão Salvar enquanto o PUT está em andamento", async () => {
    let resolveUpdate: (v: AlertSettingsResponse) => void = () => {};
    mockedUpdate.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveUpdate = resolve;
        }),
    );
    await renderAndWaitForSlider();
    const saveButton = screen.getByRole("button", {
      name: /salvar configurações/i,
    });
    fireEvent.click(saveButton);

    await waitFor(() => expect(saveButton).toBeDisabled());
    resolveUpdate(DEFAULTS);
    await waitFor(() => expect(saveButton).not.toBeDisabled());
  });

  // H) Erro do backend ao salvar -----------------------------------------------
  it("mostra erro e mantém o valor local quando o PUT falha", async () => {
    mockedUpdate.mockRejectedValue(new Error("HTTP 422"));
    const slider = await renderAndWaitForSlider();

    fireEvent.click(
      screen.getByRole("button", { name: /salvar configurações/i }),
    );

    await waitFor(() => {
      expect(screen.getByText(/HTTP 422/i)).toBeInTheDocument();
    });
    expect(slider).toHaveAttribute("aria-valuenow", "0.85");
  });

  // J) Valor persistido reaparece em um novo GET (reabrir a página) -----------
  it("reflete a configuração mais recente em um novo GET (reabrir a página)", async () => {
    const { unmount } = render(<AlertSettingsForm />);
    await screen.findByRole("slider");
    unmount();

    mockedGet.mockResolvedValue({
      alert_threshold: 0.9,
      telegram_enabled: false,
      email_enabled: true,
      alert_email: "ops@empresa.com",
    });
    render(<AlertSettingsForm />);
    const slider = await screen.findByRole("slider");
    expect(slider).toHaveAttribute("aria-valuenow", "0.9");
    expect(
      screen.getByRole("switch", { name: /ativar notificações por telegram/i }),
    ).toHaveAttribute("aria-checked", "false");
    expect(screen.getByLabelText(/e-mail para receber alertas/i)).toHaveValue(
      "ops@empresa.com",
    );
  });

  // Testar Notificação ----------------------------------------------------------
  it("clica em Testar Notificação e mostra estados testando/sucesso com a mensagem retornada", async () => {
    let resolveTest: (v: { message: string }) => void = () => {};
    mockedTest.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveTest = resolve;
        }),
    );
    await renderAndWaitForSlider();

    fireEvent.click(
      screen.getByRole("button", { name: /testar notificação/i }),
    );
    expect(mockedTest).toHaveBeenCalledTimes(1);
    expect(
      screen.getByRole("button", { name: /testando/i }),
    ).toBeInTheDocument();

    resolveTest({ message: "Telegram: enviado · E-mail: enviado" });
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /notificação enviada/i }),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/telegram: enviado · e-mail: enviado/i),
    ).toBeInTheDocument();
  });

  it("mostra erro e permite retry quando o teste de notificação falha", async () => {
    mockedTest.mockRejectedValueOnce(
      new Error("Nenhum canal de notificação está habilitado."),
    );
    await renderAndWaitForSlider();

    fireEvent.click(
      screen.getByRole("button", { name: /testar notificação/i }),
    );

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /tentar novamente/i }),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/nenhum canal de notificação está habilitado/i),
    ).toBeInTheDocument();

    mockedTest.mockResolvedValueOnce({ message: "Telegram: enviado" });
    fireEvent.click(screen.getByRole("button", { name: /tentar novamente/i }));
    expect(mockedTest).toHaveBeenCalledTimes(2);
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /notificação enviada/i }),
      ).toBeInTheDocument();
    });
  });

  it("nunca exibe token/API key na mensagem de erro do teste", async () => {
    mockedTest.mockRejectedValue(new Error("HTTP 502"));
    await renderAndWaitForSlider();
    fireEvent.click(
      screen.getByRole("button", { name: /testar notificação/i }),
    );
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /tentar novamente/i }),
      ).toBeInTheDocument();
    });
    const bodyText = document.body.textContent?.toLowerCase() ?? "";
    expect(bodyText).not.toContain("telegram_bot_token");
    expect(bodyText).not.toContain("resend_api_key");
  });
});
