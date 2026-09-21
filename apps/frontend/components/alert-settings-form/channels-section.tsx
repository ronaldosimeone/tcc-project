// ── Seção "Canais de notificação" — RNF-58: extraído de
// alert-settings-form.tsx ────────────────────────────────────────────────────

import { CircleAlert, Mail } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

export interface ChannelsSectionProps {
  telegramEnabled: boolean;
  emailEnabled: boolean;
  alertEmail: string;
  emailFormatValid: boolean;
  validationMessage: string | null;
  isSaving: boolean;
  onTelegramToggle: (checked: boolean) => void;
  onEmailToggle: (checked: boolean) => void;
  onEmailChange: (value: string) => void;
}

export function ChannelsSection({
  telegramEnabled,
  emailEnabled,
  alertEmail,
  emailFormatValid,
  validationMessage,
  isSaving,
  onTelegramToggle,
  onEmailToggle,
  onEmailChange,
}: ChannelsSectionProps) {
  return (
    <section className="flex flex-col gap-4 border-t border-slate-200 pt-6">
      <Label className="text-sm font-medium">Canais de notificação</Label>

      <div className="flex items-center justify-between gap-4">
        <div className="flex flex-col">
          <span className="text-sm text-slate-900">Telegram</span>
          <span className="text-xs text-muted-foreground">
            Notificações críticas serão enviadas para o Telegram configurado no
            servidor.
          </span>
        </div>
        <Switch
          aria-label="Ativar notificações por Telegram"
          checked={telegramEnabled}
          onCheckedChange={(checked) => onTelegramToggle(checked === true)}
          disabled={isSaving}
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
            onCheckedChange={(checked) => onEmailToggle(checked === true)}
            disabled={isSaving}
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
                onChange={(e) => onEmailChange(e.target.value)}
                disabled={isSaving}
                className="pl-8"
                aria-invalid={!emailFormatValid}
              />
            </div>
          </div>
        )}
      </div>

      {validationMessage && (
        <p
          role="alert"
          className="flex items-center gap-1.5 text-xs text-destructive"
        >
          <CircleAlert className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          {validationMessage}
        </p>
      )}
    </section>
  );
}
