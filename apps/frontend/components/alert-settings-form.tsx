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
 *
 * RNF-58: decomposto — estado/efeitos/validação em `useAlertSettingsForm`
 * (hooks/use-alert-settings-form.ts), JSX em
 * `components/alert-settings-form/*`. Este arquivo é só o orquestrador.
 * Nenhuma mudança de comportamento/contrato de API.
 */

import { CircleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useAlertSettingsForm } from "@/hooks/use-alert-settings-form";
import { ChannelsSection } from "./alert-settings-form/channels-section";
import {
  SaveSection,
  TestNotificationSection,
} from "./alert-settings-form/save-and-test-sections";
import { ThresholdSection } from "./alert-settings-form/threshold-section";

export function AlertSettingsForm() {
  const {
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
  } = useAlertSettingsForm();

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

  const isSaving = saveState === "saving";

  return (
    <div className="flex max-w-lg flex-col gap-8">
      <ThresholdSection
        threshold={threshold}
        isSaving={isSaving}
        onSliderChange={handleSliderChange}
      />

      <ChannelsSection
        telegramEnabled={telegramEnabled}
        emailEnabled={emailEnabled}
        alertEmail={alertEmail}
        emailFormatValid={emailFormatValid}
        validationMessage={validationMessage}
        isSaving={isSaving}
        onTelegramToggle={handleTelegramToggle}
        onEmailToggle={handleEmailToggle}
        onEmailChange={handleEmailChange}
      />

      <SaveSection
        saveState={saveState}
        saveError={saveError}
        canSave={canSave}
        onSave={handleSave}
      />

      <TestNotificationSection
        testState={testState}
        testError={testError}
        testResultMessage={testResultMessage}
        onTest={handleTest}
      />
    </div>
  );
}
