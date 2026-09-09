/**
 * Cliente HTTP tipado para o backend FastAPI.
 * RNF-12: Strict TypeScript — proibido `any`.
 * RNF-13: URL lida exclusivamente de NEXT_PUBLIC_API_URL.
 */

// ── Tipos exportados ─────────────────────────────────────────────────────────

export interface PredictPayload {
  /** Pressão a jusante do compressor (bar) */
  TP2: number;
  /** Pressão no painel pneumático (bar) */
  TP3: number;
  /** Pressão no circuito consumidor (bar) */
  H1: number;
  /** Pressão diferencial no secador (bar) */
  DV_pressure: number;
  /** Pressão no reservatório de ar (bar) */
  Reservoirs: number;
  /** Corrente do motor elétrico (A) */
  Motor_current: number;
  /** Temperatura do óleo do compressor (°C) */
  Oil_temperature: number;
  /** Estado do compressor (0 = OFF · 1 = ON) */
  COMP: number;
  /** Válvula de descarga elétrica (0 = OFF · 1 = ON) */
  DV_eletric: number;
  /** Comutador de torre dessecante (0 = OFF · 1 = ON) */
  Towers: number;
  /** Válvula MPG (0 = OFF · 1 = ON) */
  MPG: number;
  /** Nível de óleo (0 = OFF · 1 = ON) */
  Oil_level: number;
}

export interface PredictResponse {
  /** 0 = operação normal · 1 = falha detectada */
  predicted_class: number;
  /** Confiança do modelo para classe 1 [0.0 – 1.0] */
  failure_probability: number;
  /** Timestamp ISO 8601 UTC da inferência */
  timestamp: string;
}

// ── Domínio: MLOps & Simulator ───────────────────────────────────────────────

/** Status de um único modelo registado (RF-11). */
export interface ModelSummary {
  /** Nome bruto enviado pelo backend (ex.: `random_forest_v2`). */
  name: string;
  /** `true` se for o modelo activo na inferência. */
  active: boolean;
  /** `true` se o artefato `.onnx` existe em disco. */
  artefact_ready: boolean;
}

/** Resposta de `GET /models`. */
export interface ModelsListResponse {
  active_model: string;
  models: ModelSummary[];
}

/** Resposta de `PUT /models/active`. */
export interface SwapModelResponse {
  previous_model: string;
  active_model: string;
  message: string;
}

/** Os 3 cenários do simulador de dados (RNF-29). */
export type SimulatorMode = "NORMAL" | "DEGRADATION" | "FAILURE";

export interface SimulatorModeResponse {
  mode: SimulatorMode;
  message: string;
}

// ── Domínio: Sugestão de manutenção via RAG (RF-22 / RF-23) ──────────────────

/** Um trecho de manual citado no plano — metadados preservados do MCP/RF-21. */
export interface ManualReference {
  file_name: string;
  page: number;
  chunk_index: number;
  source: string;
  score: number;
}

/** Corpo de `POST /v1/maintenance/suggest[/stream]`. */
export interface MaintenanceSuggestionPayload {
  failure_probability: number;
  equipment_name: string;
  symptom_description?: string;
}

// ── Domínio: Configuração de alertas (RF-25 / RNF-49) ────────────────────────

/** Corpo/resposta de `GET|PUT /v1/settings/alerts`. Configuração GLOBAL do
 * sistema — não há usuários individuais no projeto (ver README §4.11). */
export interface AlertSettingsResponse {
  /** Limiar efetivo de alerta crítico, em [0.5, 0.95]. */
  alert_threshold: number;
  /** Notificação crítica também enviada via Telegram. */
  telegram_enabled: boolean;
  /** Notificação crítica também enviada por e-mail (Resend). */
  email_enabled: boolean;
  /** Endereço de destino — obrigatório quando email_enabled=true. */
  alert_email: string | null;
}

/** Corpo de `PUT /v1/settings/alerts`. */
export interface AlertSettingsUpdatePayload {
  alert_threshold: number;
  telegram_enabled: boolean;
  email_enabled: boolean;
  alert_email: string | null;
}

/** Resposta de `POST /v1/settings/alerts/test`. */
export interface NotificationTestResponse {
  message: string;
}

// ── Helpers internos ─────────────────────────────────────────────────────────

export function resolveBaseUrl(): string {
  const url = process.env.NEXT_PUBLIC_API_URL;
  if (!url) {
    throw new Error(
      "[api-client] NEXT_PUBLIC_API_URL não está definida.\n" +
        "Crie o arquivo apps/frontend/.env.local com uma das opções:\n" +
        "  NEXT_PUBLIC_API_URL=/api                   # docker compose (nginx proxy)\n" +
        "  NEXT_PUBLIC_API_URL=http://localhost:8000  # FastAPI standalone",
    );
  }
  return url.replace(/\/$/, ""); // remove trailing slash
}

// ── Funções públicas ─────────────────────────────────────────────────────────

/**
 * POST /predict/
 * Envia uma leitura de sensores e retorna a predição de falha.
 */
export async function predict(
  payload: PredictPayload,
): Promise<PredictResponse> {
  const baseUrl = resolveBaseUrl();

  const response = await fetch(`${baseUrl}/predict/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    // Não cacheia — leitura em tempo real
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `[api-client] predict() falhou — HTTP ${response.status} ${response.statusText}`,
    );
  }

  return response.json() as Promise<PredictResponse>;
}

// ── MLOps: gestão dos modelos (RF-11) ────────────────────────────────────────

/** GET /models — devolve o modelo activo + status de todos os modelos. */
export async function listModels(): Promise<ModelsListResponse> {
  const baseUrl = resolveBaseUrl();

  const response = await fetch(`${baseUrl}/models`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `[api-client] listModels() falhou — HTTP ${response.status} ${response.statusText}`,
    );
  }

  return response.json() as Promise<ModelsListResponse>;
}

/** PUT /models/active — troca atomicamente o modelo de inferência. */
export async function swapActiveModel(
  modelName: string,
): Promise<SwapModelResponse> {
  const baseUrl = resolveBaseUrl();

  const response = await fetch(`${baseUrl}/models/active`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: modelName }),
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `[api-client] swapActiveModel() falhou — HTTP ${response.status} ${response.statusText}`,
    );
  }

  return response.json() as Promise<SwapModelResponse>;
}

// ── Simulator: controlo dos cenários (RNF-29) ────────────────────────────────

/** GET /simulator/mode — devolve o cenário activo. */
export async function getSimulatorMode(): Promise<SimulatorModeResponse> {
  const baseUrl = resolveBaseUrl();

  const response = await fetch(`${baseUrl}/simulator/mode`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `[api-client] getSimulatorMode() falhou — HTTP ${response.status} ${response.statusText}`,
    );
  }

  return response.json() as Promise<SimulatorModeResponse>;
}

/** PUT /simulator/mode — troca o cenário de simulação em tempo real. */
export async function setSimulatorMode(
  mode: SimulatorMode,
): Promise<SimulatorModeResponse> {
  const baseUrl = resolveBaseUrl();

  const response = await fetch(`${baseUrl}/simulator/mode`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `[api-client] setSimulatorMode() falhou — HTTP ${response.status} ${response.statusText}`,
    );
  }

  return response.json() as Promise<SimulatorModeResponse>;
}

// ── Configuração de alertas (RF-25 / RNF-49) ─────────────────────────────────

/** GET /v1/settings/alerts — configuração global atual (ou o default 0.85). */
export async function getAlertSettings(): Promise<AlertSettingsResponse> {
  const baseUrl = resolveBaseUrl();

  const response = await fetch(`${baseUrl}/v1/settings/alerts`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `[api-client] getAlertSettings() falhou — HTTP ${response.status} ${response.statusText}`,
    );
  }

  return response.json() as Promise<AlertSettingsResponse>;
}

/** PUT /v1/settings/alerts — salva a configuração global (limiar + canais). */
export async function updateAlertSettings(
  payload: AlertSettingsUpdatePayload,
): Promise<AlertSettingsResponse> {
  const baseUrl = resolveBaseUrl();

  const response = await fetch(`${baseUrl}/v1/settings/alerts`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `[api-client] updateAlertSettings() falhou — HTTP ${response.status} ${response.statusText}`,
    );
  }

  return response.json() as Promise<AlertSettingsResponse>;
}

/** POST /v1/settings/alerts/test — envia uma notificação de teste ao Telegram. */
export async function testAlertNotification(): Promise<NotificationTestResponse> {
  const baseUrl = resolveBaseUrl();

  const response = await fetch(`${baseUrl}/v1/settings/alerts/test`, {
    method: "POST",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `[api-client] testAlertNotification() falhou — HTTP ${response.status} ${response.statusText}`,
    );
  }

  return response.json() as Promise<NotificationTestResponse>;
}
