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
    liveProbability: 0.12,
    effectiveRiskLevel: "NORMAL",
    latencyTelemetry: { messageId: "msg-1", latencyMs: 38 },
    isLoading: false,
  },
};

export const Alerta: Story = {
  args: {
    liveProbability: 0.48,
    effectiveRiskLevel: "ALERTA",
    latencyTelemetry: { messageId: "msg-2", latencyMs: 52 },
    isLoading: false,
  },
};

export const Critico: Story = {
  args: {
    liveProbability: 0.91,
    effectiveRiskLevel: "CRÍTICO",
    latencyTelemetry: { messageId: "msg-3", latencyMs: 74 },
    isLoading: false,
  },
};

export const Loading: Story = {
  args: {
    liveProbability: 0,
    effectiveRiskLevel: "NORMAL",
    latencyTelemetry: null,
    isLoading: true,
  },
};

export const AguardandoPrimeiraInferencia: Story = {
  name: "Aguardando primeira inferência (sem latência ainda)",
  args: {
    liveProbability: 0.2,
    effectiveRiskLevel: "NORMAL",
    latencyTelemetry: null,
    isLoading: false,
  },
};
