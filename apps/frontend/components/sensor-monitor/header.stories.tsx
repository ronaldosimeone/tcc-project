import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { SensorMonitorHeader } from "./header";

const meta: Meta<typeof SensorMonitorHeader> = {
  title: "Sensor Monitor/SensorMonitorHeader",
  component: SensorMonitorHeader,
  parameters: { layout: "padded" },
  tags: ["autodocs"],
  args: {
    isLive: true,
    isLoading: false,
    isOffline: false,
    sseStatus: "connected",
    wsStatus: "open",
  },
};

export default meta;
type Story = StoryObj<typeof SensorMonitorHeader>;

export const Normal: Story = {
  args: { riskLevel: "NORMAL", isCriticalState: false, isAlertState: false },
};

export const Alerta: Story = {
  args: { riskLevel: "ALERTA", isCriticalState: false, isAlertState: true },
};

export const Critico: Story = {
  name: "Crítico (RF-08 — FALHA CRÍTICA DETECTADA)",
  args: { riskLevel: "CRÍTICO", isCriticalState: true, isAlertState: false },
};

export const Loading: Story = {
  args: {
    riskLevel: "NORMAL",
    isCriticalState: false,
    isAlertState: false,
    isLive: false,
    isLoading: true,
    sseStatus: "connecting",
    wsStatus: "connecting",
  },
};

export const Offline: Story = {
  name: "Offline (backend indisponível, degradado — RNF-35)",
  args: {
    riskLevel: "NORMAL",
    isCriticalState: false,
    isAlertState: false,
    isLive: false,
    isOffline: true,
    sseStatus: "closed",
    wsStatus: "closed",
  },
};
