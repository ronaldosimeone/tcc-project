import Header from "@/components/header";
import Sidebar from "@/components/sidebar";
import SensorMonitor from "@/components/sensor-monitor";
import { ErrorBoundary } from "@/components/error-boundary";

/**
 * Página raiz — Server Component.
 *
 * Hierarquia:
 *   page.tsx (Server)
 *     └─ ErrorBoundary  (Client — captura falhas de renderização)
 *          └─ SensorMonitor (Client — polling, gráficos, gauge)
 *
 * ErrorBoundary garante que uma falha nos gráficos (ex: Recharts, dados
 * malformados) não quebre a tela inteira — RF-07 + boa prática de prod.
 */
export default function HomePage() {
  return (
    <div className="flex h-full flex-col bg-background text-foreground">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <ErrorBoundary>
            <SensorMonitor />
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
