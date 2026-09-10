import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import FleetHealthTable from "./FleetHealthTable";

const meta: Meta<typeof FleetHealthTable> = {
  title: "Dashboard/FleetHealthTable",
  component: FleetHealthTable,
  parameters: { layout: "padded" },
  tags: ["autodocs"],
  args: {
    selectedId: "APU-Trem-042",
    onSelect: (id) => console.log("onSelect", id),
  },
};

export default meta;
type Story = StoryObj<typeof FleetHealthTable>;

export const Normal: Story = {
  args: {
    effectiveRiskLevel: "NORMAL",
    effectiveProb: 0.12,
    isLoading: false,
  },
};

export const Alerta: Story = {
  args: {
    effectiveRiskLevel: "ALERTA",
    effectiveProb: 0.48,
    isLoading: false,
  },
};

export const Critico: Story = {
  args: {
    effectiveRiskLevel: "CRÍTICO",
    effectiveProb: 0.91,
    isLoading: false,
  },
};

export const Loading: Story = {
  args: {
    effectiveRiskLevel: "NORMAL",
    effectiveProb: 0,
    isLoading: true,
  },
};
