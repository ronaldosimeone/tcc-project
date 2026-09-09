"use client";

/**
 * Formulário de configuração de alertas — RF-25 / RNF-49.
 *
 * Configuração GLOBAL do sistema (não há usuários individuais no projeto —
 * ver README §4.11). Fluxo: GET no mount -> preenche slider + canais ->
 * usuário altera (só estado local) -> "Salvar configurações" -> PUT ->
 * backend confirma -> UI mostra sucesso. "Testar Notificação" é
 * independente do fluxo de salvar — testa a configuração JÁ PERSISTIDA no
 * backend (não o rascunho não salvo na tela).
 *
 * O Bot Token do Telegram e a API key do Resend NUNCA aparecem aqui — só
 * os toggles ON/OFF de cada canal e o endereço de e-mail de destino. As
 * credenciais continuam vindo exclusivamente da configuração segura do
 * backend (`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`/`RESEND_API_KEY`).
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bell,
  CircleAlert,
  CircleCheck,
  Loader2,
  Mail,
  Send,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  getAlertSettings,
  testAlertNotification,
  updateAlertSettings,
  type AlertSettingsResponse,
} from "@/lib/api-client";

const MIN_THRESHOLD = 0.5;
const MAX_THRESHOLD = 0.95;
const STEP = 0.01;

// Mesmo espírito do EmailStr do backend — sem regex excessivamente
// restritiva, só a forma básica local@dominio.tld (RF-25 §5).
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/** Arredonda para 2 casas — evita `0.7000000000000001` vindo do slider. */
function round2(value: number): number {
  return Math.round(value * 100) / 100;
}

function toPercentLabel(value: number): string {
  return `${Math.round(value * 100)}%`;
}

type SaveState = "idle" | "saving" | "saved" | "error";
type TestState = "idle" | "testing" | "success" | "error";

export function AlertSettingsForm() {
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

  if (loading) {
    return (
      <div className="flex max-w-lg flex-col gap-4">
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-2 w-full" />
        <Skeleton className="h-9 w-48" />
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="flex max-w-lg flex-col gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-4">
        <p className="flex items-start gap-2 text-sm text-destructive">
          <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" />
          {loadError}
        </p>
        <Button
          variant="outline"
          size="sm"
          onClick={loadSettings}
          className="w-fit"
        >
          Tentar novamente
        </Button>
      </div>
    );
  }

  return (
    <div className="flex max-w-lg flex-col gap-8">
      {/* ── Limite de alerta ──────────────────────────────────────────── */}
      <section className="flex flex-col gap-4">
        <div className="flex items-baseline justify-between">
          <Label
            htmlFor="alert-threshold-slider"
            className="text-sm font-medium"
          >
            Limite de alerta crítico
          </Label>
          <span
            data-testid="alert-threshold-value"
            className="font-mono text-lg font-semibold text-slate-900"
          >
            {toPercentLabel(threshold)}
          </span>
        </div>

        <Slider
          id="alert-threshold-slider"
          aria-label="Limite de alerta crítico"
          min={MIN_THRESHOLD}
          max={MAX_THRESHOLD}
          step={STEP}
          value={[threshold]}
          onValueChange={handleSliderChange}
          disabled={saveState === "saving"}
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{toPercentLabel(MIN_THRESHOLD)}</span>
          <span>{toPercentLabel(MAX_THRESHOLD)}</span>
        </div>

        <p className="text-xs text-muted-foreground">
          Alertas críticos serão disparados quando a probabilidade de falha
          ultrapassar {toPercentLabel(threshold)}.
        </p>
      </section>

      {/* ── Canais de notificação ─────────────────────────────────────── */}
      <section className="flex flex-col gap-4 border-t border-slate-200 pt-6">
        <Label className="text-sm font-medium">Canais de notificação</Label>

        <div className="flex items-center justify-between gap-4">
          <div className="flex flex-col">
            <span className="text-sm text-slate-900">Telegram</span>
            <span className="text-xs text-muted-foreground">
              Notificações críticas serão enviadas para o Telegram configurado
              no servidor.
            </span>
          </div>
          <Switch
            aria-label="Ativar notificações por Telegram"
            checked={telegramEnabled}
            onCheckedChange={(checked) => {
              setTelegramEnabled(checked === true);
              clearSaveFeedback();
            }}
            disabled={saveState === "saving"}
          />
        </div>

        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-4">
            <div className="flex flex-col">
              <span className="text-sm text-slate-900">E-mail</span>
              <span className="text-xs text-muted-foreground">
                O endereço configurado abaixo receberá os alertas.
              </span>
            </div>
            <Switch
              aria-label="Ativar notificações por e-mail"
              checked={emailEnabled}
              onCheckedChange={(checked) => {
                setEmailEnabled(checked === true);
                clearSaveFeedback();
              }}
              disabled={saveState === "saving"}
            />
          </div>

          {emailEnabled && (
            <div className="flex flex-col gap-1.5 pl-1">
              <Label
                htmlFor="alert-email"
                className="text-xs text-muted-foreground"
              >
                E-mail para receber alertas
              </Label>
              <div className="relative">
                <Mail className="pointer-events-none absolute top-1/2 left-2.5 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="alert-email"
                  type="email"
                  placeholder="alertas@empresa.com.br"
                  value={alertEmail}
                  onChange={(e) => {
                    setAlertEmail(e.target.value);
                    clearSaveFeedback();
                  }}
                  disabled={saveState === "saving"}
                  className="pl-8"
                  aria-invalid={!emailFormatValid}
                />
              </div>
            </div>
          )}
        </div>

        {validationMessage && (
          <p className="flex items-center gap-1.5 text-xs text-destructive">
            <CircleAlert className="h-3.5 w-3.5 shrink-0" />
            {validationMessage}
          </p>
        )}
      </section>

      {/* ── Salvar ────────────────────────────────────────────────────── */}
      <section className="flex items-center gap-3">
        <Button
          onClick={handleSave}
          disabled={!canSave || saveState === "saving"}
          className="w-fit"
        >
          {saveState === "saving" && (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          )}
          Salvar configurações
        </Button>

        {saveState === "saved" && (
          <p className="flex items-center gap-1.5 text-xs text-emerald-600">
            <CircleCheck className="h-3.5 w-3.5" />
            Configuração salva.
          </p>
        )}
        {saveState === "error" && saveError && (
          <p className="flex items-center gap-1.5 text-xs text-destructive">
            <CircleAlert className="h-3.5 w-3.5" />
            {saveError}
          </p>
        )}
      </section>

      {/* ── Testar notificação ────────────────────────────────────────── */}
      <section className="flex flex-col gap-3 border-t border-slate-200 pt-6">
        <div className="flex items-center gap-2">
          <Bell className="h-4 w-4 text-muted-foreground" />
          <Label className="text-sm font-medium">Testar notificação</Label>
        </div>
        <p className="text-xs text-muted-foreground">
          Envia uma mensagem de teste aos canais habilitados (configuração já
          salva), sem depender de uma predição real.
        </p>

        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            onClick={handleTest}
            disabled={testState === "testing"}
            className="w-fit"
          >
            {testState === "testing" ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Send className="h-3.5 w-3.5" />
            )}
            {testState === "testing"
              ? "Testando..."
              : testState === "success"
              ? "Notificação enviada"
              : testState === "error"
              ? "Tentar novamente"
              : "Testar Notificação"}
          </Button>

          {testState === "success" && (
            <p className="flex items-center gap-1.5 text-xs text-emerald-600">
              <CircleCheck className="h-3.5 w-3.5" />
              {testResultMessage ?? "Notificação de teste enviada."}
            </p>
          )}
          {testState === "error" && testError && (
            <p className="flex items-center gap-1.5 text-xs text-destructive">
              <CircleAlert className="h-3.5 w-3.5" />
              {testError}
            </p>
          )}
        </div>
      </section>
    </div>
  );
}
