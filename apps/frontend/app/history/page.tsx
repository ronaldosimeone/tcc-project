import Header from "@/components/header";
import Sidebar from "@/components/sidebar";
import HistoryDashboard from "@/components/history/HistoryDashboard";

export default function HistoryPage() {
  return (
    <div className="flex h-full flex-col bg-background text-foreground">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <HistoryDashboard />
        </main>
      </div>
    </div>
  );
}
