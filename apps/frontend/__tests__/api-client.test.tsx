/**
 * Testes do cliente HTTP tipado (lib/api-client.ts) — RNF-59.
 *
 * `fetch` é substituído diretamente (sem MSW): cada teste assume controle
 * total da URL/método/corpo/resposta, o que dá assertions mais precisas do
 * que um handler MSW genérico para uma superfície tão mecânica (8 funções,
 * todas seguindo o mesmo padrão fetch → checar `ok` → lançar ou retornar
 * json). `NEXT_PUBLIC_API_URL` é controlada via `vi.stubEnv` (lido em
 * runtime por `resolveBaseUrl()`, não no import do módulo).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getAlertSettings,
  getSimulatorMode,
  listModels,
  predict,
  resolveBaseUrl,
  setSimulatorMode,
  swapActiveModel,
  testAlertNotification,
  updateAlertSettings,
  type PredictPayload,
} from "@/lib/api-client";

const PAYLOAD: PredictPayload = {
  TP2: 8.1,
  TP3: 7.9,
  H1: 8.5,
  DV_pressure: 1.2,
  Reservoirs: 7.0,
  Motor_current: 5.1,
  Oil_temperature: 72.0,
  COMP: 1,
  DV_eletric: 0,
  Towers: 1,
  MPG: 0,
  Oil_level: 1,
};

function mockFetchOnce(status: number, body: unknown, statusText = "OK") {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText,
    json: () => Promise.resolve(body),
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_API_URL", "http://test-api");
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("resolveBaseUrl", () => {
  it("lança erro descritivo quando NEXT_PUBLIC_API_URL não está definida", () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "");
    expect(() => resolveBaseUrl()).toThrow(/NEXT_PUBLIC_API_URL/);
  });

  it("remove a barra final da URL configurada", () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://localhost:8000/");
    expect(resolveBaseUrl()).toBe("http://localhost:8000");
  });

  it("mantém a URL como está quando não tem barra final", () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "/api");
    expect(resolveBaseUrl()).toBe("/api");
  });
});

describe("predict", () => {
  it("faz POST em /predict/ com o payload serializado e retorna a predição", async () => {
    const response = {
      predicted_class: 1,
      failure_probability: 0.91,
      timestamp: "2026-01-01T00:00:00.000Z",
    };
    const fetchMock = mockFetchOnce(200, response);

    const result = await predict(PAYLOAD);

    expect(result).toEqual(response);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://test-api/predict/",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(PAYLOAD),
        cache: "no-store",
      }),
    );
  });

  it("lança erro com o status HTTP quando a resposta não é ok", async () => {
    mockFetchOnce(500, {}, "Internal Server Error");
    await expect(predict(PAYLOAD)).rejects.toThrow(/predict\(\).*500/);
  });
});

describe("listModels / swapActiveModel", () => {
  it("GET /models retorna a lista de modelos", async () => {
    const response = {
      active_model: "random_forest_v2",
      models: [
        { name: "random_forest_v2", active: true, artefact_ready: true },
      ],
    };
    const fetchMock = mockFetchOnce(200, response);

    const result = await listModels();

    expect(result).toEqual(response);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://test-api/models",
      expect.objectContaining({ method: "GET", cache: "no-store" }),
    );
  });

  it("listModels() lança erro quando a resposta falha", async () => {
    mockFetchOnce(503, {}, "Service Unavailable");
    await expect(listModels()).rejects.toThrow(/listModels\(\).*503/);
  });

  it("PUT /models/active envia { model_name } e retorna a troca", async () => {
    const response = {
      previous_model: "random_forest_v2",
      active_model: "xgboost_v1",
      message: "ok",
    };
    const fetchMock = mockFetchOnce(200, response);

    const result = await swapActiveModel("xgboost_v1");

    expect(result).toEqual(response);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://test-api/models/active",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ model_name: "xgboost_v1" }),
      }),
    );
  });

  it("swapActiveModel() lança erro quando a resposta falha", async () => {
    mockFetchOnce(404, {}, "Not Found");
    await expect(swapActiveModel("inexistente")).rejects.toThrow(
      /swapActiveModel\(\).*404/,
    );
  });
});

describe("getSimulatorMode / setSimulatorMode", () => {
  it("GET /simulator/mode retorna o cenário atual", async () => {
    const response = { mode: "NORMAL" as const, message: "ok" };
    const fetchMock = mockFetchOnce(200, response);

    const result = await getSimulatorMode();

    expect(result).toEqual(response);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://test-api/simulator/mode",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("getSimulatorMode() lança erro quando a resposta falha", async () => {
    mockFetchOnce(500, {});
    await expect(getSimulatorMode()).rejects.toThrow(/getSimulatorMode\(\)/);
  });

  it("PUT /simulator/mode envia { mode } e retorna a confirmação", async () => {
    const response = { mode: "FAILURE" as const, message: "ok" };
    const fetchMock = mockFetchOnce(200, response);

    const result = await setSimulatorMode("FAILURE");

    expect(result).toEqual(response);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://test-api/simulator/mode",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ mode: "FAILURE" }),
      }),
    );
  });

  it("setSimulatorMode() lança erro quando a resposta falha", async () => {
    mockFetchOnce(400, {});
    await expect(setSimulatorMode("DEGRADATION")).rejects.toThrow(
      /setSimulatorMode\(\)/,
    );
  });
});

describe("getAlertSettings / updateAlertSettings / testAlertNotification", () => {
  it("GET /v1/settings/alerts retorna a configuração atual", async () => {
    const response = {
      alert_threshold: 0.85,
      telegram_enabled: true,
      email_enabled: false,
      alert_email: null,
    };
    const fetchMock = mockFetchOnce(200, response);

    const result = await getAlertSettings();

    expect(result).toEqual(response);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://test-api/v1/settings/alerts",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("getAlertSettings() lança erro quando a resposta falha", async () => {
    mockFetchOnce(500, {});
    await expect(getAlertSettings()).rejects.toThrow(/getAlertSettings\(\)/);
  });

  it("PUT /v1/settings/alerts envia o payload completo e retorna a config salva", async () => {
    const payload = {
      alert_threshold: 0.7,
      telegram_enabled: true,
      email_enabled: true,
      alert_email: "alertas@empresa.com.br",
    };
    const fetchMock = mockFetchOnce(200, payload);

    const result = await updateAlertSettings(payload);

    expect(result).toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://test-api/v1/settings/alerts",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify(payload),
      }),
    );
  });

  it("updateAlertSettings() lança erro quando a resposta falha", async () => {
    mockFetchOnce(422, {});
    await expect(
      updateAlertSettings({
        alert_threshold: 0.7,
        telegram_enabled: false,
        email_enabled: false,
        alert_email: null,
      }),
    ).rejects.toThrow(/updateAlertSettings\(\)/);
  });

  it("POST /v1/settings/alerts/test retorna a mensagem de confirmação", async () => {
    const response = { message: "Notificação de teste enviada." };
    const fetchMock = mockFetchOnce(200, response);

    const result = await testAlertNotification();

    expect(result).toEqual(response);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://test-api/v1/settings/alerts/test",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("testAlertNotification() lança erro quando a resposta falha", async () => {
    mockFetchOnce(500, {}, "Internal Server Error");
    await expect(testAlertNotification()).rejects.toThrow(
      /testAlertNotification\(\)/,
    );
  });
});
