"use client";

// ── Hook de estado — RNF-58: extraído de alert-settings-form.tsx ────────────
//
// Toda a lógica de estado/efeitos/validação do formulário de configuração de
// alertas (RF-25/RNF-49) vive aqui — o componente fica só com JSX de
// apresentação. Nenhuma mudança de comportamento/contrato de API.

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getAlertSettings,
  testAlertNotification,
  updateAlertSettings,
  type AlertSettingsResponse,
} from "@/lib/api-client";

export const MIN_THRESHOLD = 0.5;
export const MAX_THRESHOLD = 0.95;
export const STEP = 0.01;

// Mesmo espírito do EmailStr do backend — sem regex excessivamente
// restritiva, só a forma básica local@dominio.tld (RF-25 §5).
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/** Arredonda para 2 casas — evita `0.7000000000000001` vindo do slider. */
function round2(value: number): number {
  return Math.round(value * 100) / 100;
}

export function toPercentLabel(value: number): string {
  return `${Math.round(value * 100)}%`;
}

type SaveState = "idle" | "saving" | "saved" | "error";
type TestState = "idle" | "testing" | "success" | "error";

export function useAlertSettingsForm() {
  const [threshold, setThreshold] = useState<number>(0.85);
  const [telegramEnabled, setTelegramEnabled] = useState<boolean>(true);
  const [emailEnabled, setEmailEnabled] = useState<boolean>(false);
  const [alertEmail, setAlertEmail] = useState<string>("");

  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState<string | null>(null);

  const [testState, setTestState] = useState<TestState>("idle");
  const [testError, setTestError] = useState<string | null>(null);
  const [testResultMessage, setTestResultMessage] = useState<string | null>(
    null,
  );

  const applySettings = useCallback((data: AlertSettingsResponse) => {
    setThreshold(data.alert_threshold);
    setTelegramEnabled(data.telegram_enabled);
    setEmailEnabled(data.email_enabled);
    setAlertEmail(data.alert_email ?? "");
  }, []);

  const loadSettings = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const current = await getAlertSettings();
      applySettings(current);
    } catch (err) {
      setLoadError(
        err instanceof Error
          ? err.message
          : "Falha ao carregar a configuração de alertas.",
      );
    } finally {
      setLoading(false);
    }
  }, [applySettings]);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  // Qualquer alteração invalida a confirmação/erro do save anterior — o
  // rótulo "salvo" só vale para o valor de fato persistido.
  const clearSaveFeedback = useCallback(() => {
    setSaveState("idle");
    setSaveError(null);
  }, []);

  const handleSliderChange = useCallback(
    (values: number[]) => {
      const next = values[0];
      if (next === undefined) return;
      setThreshold(round2(next));
      clearSaveFeedback();
    },
    [clearSaveFeedback],
  );

  const handleTelegramToggle = useCallback(
    (checked: boolean) => {
      setTelegramEnabled(checked === true);
      clearSaveFeedback();
    },
    [clearSaveFeedback],
  );

  const handleEmailToggle = useCallback(
    (checked: boolean) => {
      setEmailEnabled(checked === true);
      clearSaveFeedback();
    },
    [clearSaveFeedback],
  );

  const handleEmailChange = useCallback(
    (value: string) => {
      setAlertEmail(value);
      clearSaveFeedback();
    },
    [clearSaveFeedback],
  );

  // ── Validação frontend (RF-25 §20) — UX; backend é a autoridade final ──
  const emailTrimmed = alertEmail.trim();
  const emailFormatValid = !emailEnabled || EMAIL_PATTERN.test(emailTrimmed);
  const hasChannelEnabled = telegramEnabled || emailEnabled;
  const canSave = emailFormatValid && hasChannelEnabled;

  const validationMessage = useMemo(() => {
    if (!hasChannelEnabled) {
      return "Habilite ao menos um canal de notificação.";
    }
    if (emailEnabled && !emailFormatValid) {
      return "Informe um e-mail válido para receber os alertas.";
    }
    return null;
  }, [hasChannelEnabled, emailEnabled, emailFormatValid]);

  const handleSave = useCallback(async () => {
    if (!canSave) return;
    setSaveState("saving");
    setSaveError(null);
    try {
      const saved = await updateAlertSettings({
        alert_threshold: threshold,
        telegram_enabled: telegramEnabled,
        email_enabled: emailEnabled,
        alert_email: emailEnabled ? emailTrimmed : null,
      });
      applySettings(saved);
      setSaveState("saved");
    } catch (err) {
      setSaveState("error");
      setSaveError(
        err instanceof Error
          ? err.message
          : "Falha ao salvar a configuração. Tente novamente.",
      );
    }
  }, [
    canSave,
    threshold,
    telegramEnabled,
    emailEnabled,
    emailTrimmed,
    applySettings,
  ]);

  const handleTest = useCallback(async () => {
    setTestState("testing");
    setTestError(null);
    setTestResultMessage(null);
    try {
      const result = await testAlertNotification();
      setTestResultMessage(result.message);
      setTestState("success");
    } catch (err) {
      setTestState("error");
      setTestError(
        err instanceof Error
          ? err.message
          : "Não foi possível enviar a notificação. Verifique a configuração do Telegram.",
      );
    }
  }, []);

  return {
    threshold,
    telegramEnabled,
    emailEnabled,
    alertEmail,
    loading,
    loadError,
    saveState,
    saveError,
    testState,
    testError,
    testResultMessage,
    emailFormatValid,
    canSave,
    validationMessage,
    loadSettings,
    handleSliderChange,
    handleTelegramToggle,
    handleEmailToggle,
    handleEmailChange,
    handleSave,
    handleTest,
  };
}
