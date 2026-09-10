// ── Seções "Salvar" e "Testar notificação" — RNF-58: extraído de
// alert-settings-form.tsx ────────────────────────────────────────────────────

import { Bell, CircleAlert, CircleCheck, Loader2, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";

export interface SaveSectionProps {
  saveState: "idle" | "saving" | "saved" | "error";
  saveError: string | null;
  canSave: boolean;
  onSave: () => void;
}

export function SaveSection({
  saveState,
  saveError,
  canSave,
  onSave,
}: SaveSectionProps) {
  return (
    <section className="flex items-center gap-3">
      <Button
        onClick={onSave}
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
  );
}

export interface TestNotificationSectionProps {
  testState: "idle" | "testing" | "success" | "error";
  testError: string | null;
  testResultMessage: string | null;
  onTest: () => void;
}

export function TestNotificationSection({
  testState,
  testError,
  testResultMessage,
  onTest,
}: TestNotificationSectionProps) {
  return (
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
          onClick={onTest}
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
  );
}
