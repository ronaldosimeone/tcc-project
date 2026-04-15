import Header from "@/components/header";
import Sidebar from "@/components/sidebar";
import SensorMonitor from "@/components/sensor-monitor";

/**
 * Página raiz — Server Component.
 * A interatividade (polling, estado, gráficos) fica isolada em SensorMonitor ("use client").
 */
export default function HomePage() {
  return (
    <div className="flex h-full flex-col bg-background text-foreground">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <SensorMonitor />
        </main>
      </div>
    </div>
  );
}
