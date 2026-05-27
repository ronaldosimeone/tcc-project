import Sidebar from "@/components/sidebar";
import FleetDashboard from "@/components/dashboard/FleetDashboard";

export default function HomePage() {
  return (
    <div className="flex h-full bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <FleetDashboard />
      </main>
    </div>
  );
}
