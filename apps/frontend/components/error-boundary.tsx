"use client";

/**
 * ErrorBoundary — envoltório de segurança para Client Components.
 *
 * Captura erros de renderização (ex: falha no Recharts, dados malformados)
 * e exibe uma UI de fallback em vez de quebrar a tela inteira.
 *
 * Uso:
 *   <ErrorBoundary>
 *     <SensorMonitor />
 *   </ErrorBoundary>
 *
 * Uso com fallback customizado:
 *   <ErrorBoundary fallback={<MinhaFallback />}>
 *     <SensorMonitor />
 *   </ErrorBoundary>
 */

import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

// ── Props e State ─────────────────────────────────────────────────────────

interface ErrorBoundaryProps {
  children: ReactNode;
  /** Nó React opcional exibido em lugar do fallback padrão. */
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

// ── Fallback padrão ───────────────────────────────────────────────────────

interface DefaultFallbackProps {
  error: Error | null;
  onReset: () => void;
}

function DefaultFallback({ error, onReset }: DefaultFallbackProps) {
  return (
    <div
      className="flex min-h-[320px] flex-col items-center justify-center gap-5 rounded-xl border border-destructive/30 bg-destructive/5 p-8"
      role="alert"
      aria-live="assertive"
    >
      <div className="rounded-full bg-destructive/10 p-4 ring-1 ring-destructive/30">
        <AlertTriangle className="h-8 w-8 text-destructive" />
      </div>

      <div className="max-w-sm text-center">
        <h2 className="text-base font-semibold text-foreground">
          Erro ao renderizar o painel
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {error?.message ??
            "Ocorreu um erro inesperado nos componentes de visualização."}
        </p>
      </div>

      <Button variant="outline" size="sm" onClick={onReset} className="gap-2">
        <RefreshCw className="h-4 w-4" />
        Tentar novamente
      </Button>
    </div>
  );
}

// ── Classe ErrorBoundary ──────────────────────────────────────────────────

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Em produção, enviar para um serviço de observabilidade (ex: Sentry).
    console.error("[ErrorBoundary] Erro capturado:", error.message);
    console.error("[ErrorBoundary] Stack do componente:", info.componentStack);
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    if (!this.state.hasError) {
      return this.props.children;
    }

    if (this.props.fallback !== undefined) {
      return this.props.fallback;
    }

    return (
      <DefaultFallback error={this.state.error} onReset={this.handleReset} />
    );
  }
}
