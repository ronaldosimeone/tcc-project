import Sidebar from "@/components/sidebar";
import { AlertSettingsForm } from "@/components/alert-settings-form";
import { ThemeSettings } from "@/components/theme-settings";

export default function AlertSettingsPage() {
  return (
    <div className="flex h-full bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-8">
        <div className="mx-auto flex max-w-2xl flex-col gap-1 pb-6">
          <h1 className="text-xl font-semibold tracking-tight text-foreground">
            Configurações de Alertas
          </h1>
          <p className="text-sm text-muted-foreground">
            Defina quando e como o PredictIQ deve enviar notificações críticas.
          </p>
        </div>
        <div className="mx-auto flex max-w-2xl flex-col gap-6">
          <AlertSettingsForm />
          <ThemeSettings />
        </div>
      </main>
    </div>
  );
}
