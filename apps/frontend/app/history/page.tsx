import Sidebar from "@/components/sidebar";
import HistoryDashboard from "@/components/history/HistoryDashboard";

export default function HistoryPage() {
  return (
    <div className="flex h-full bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <HistoryDashboard />
      </main>
    </div>
  );
}
