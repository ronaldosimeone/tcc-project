import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import FleetKPIs from "./FleetKPIs";

const meta: Meta<typeof FleetKPIs> = {
  title: "Dashboard/FleetKPIs",
  component: FleetKPIs,
  parameters: { layout: "padded" },
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof FleetKPIs>;

export const Normal: Story = {
  args: {
    liveRisk: 0.12,
    rawProbability: 0.12,
    effectiveRiskLevel: "NORMAL",
    latencyTelemetry: { messageId: "msg-1", latencyMs: 38 },
    isLoading: false,
  },
};

export const Alerta: Story = {
  args: {
    liveRisk: 0.48,
    rawProbability: 0.48,
    effectiveRiskLevel: "ALERTA",
    latencyTelemetry: { messageId: "msg-2", latencyMs: 52 },
    isLoading: false,
  },
};

export const Critico: Story = {
  args: {
    liveRisk: 0.91,
    rawProbability: 0.91,
    effectiveRiskLevel: "CRÍTICO",
    latencyTelemetry: { messageId: "msg-3", latencyMs: 74 },
    isLoading: false,
  },
};

export const Loading: Story = {
  args: {
    liveRisk: 0,
    rawProbability: 0,
    effectiveRiskLevel: "NORMAL",
    latencyTelemetry: null,
    isLoading: true,
  },
};

export const AguardandoPrimeiraInferencia: Story = {
  name: "Aguardando primeira inferência (sem latência ainda)",
  args: {
    liveRisk: 0.2,
    rawProbability: 0.2,
    effectiveRiskLevel: "NORMAL",
    latencyTelemetry: null,
    isLoading: false,
  },
};
