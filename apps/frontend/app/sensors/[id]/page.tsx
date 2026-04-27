import Header from "@/components/header";
import Sidebar from "@/components/sidebar";
import SensorMonitor from "@/components/sensor-monitor";
import { ErrorBoundary } from "@/components/error-boundary";

export default function SensorDetailPage() {
  return (
    <div className="flex h-full flex-col bg-background text-foreground">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-hidden">
          <ErrorBoundary>
            <SensorMonitor />
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
