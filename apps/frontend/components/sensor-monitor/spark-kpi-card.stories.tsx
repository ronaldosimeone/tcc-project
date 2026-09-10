import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { Gauge } from "lucide-react";
import type { SensorDataPoint } from "@/hooks/use-sensor-data";
import { SparkKpiCard } from "./spark-kpi-card";

const SPARK_DATA: SensorDataPoint[] = Array.from({ length: 12 }, (_, i) => ({
  time: `${10 + i}:00`,
  TP2: 8 + Math.sin(i / 2) * 0.6,
  TP3: 7.8 + Math.cos(i / 2) * 0.5,
  Motor_current: 5 + Math.sin(i / 3) * 1.2,
  Oil_temperature: 72 + i * 0.3,
  failure_probability: 0.1 + i * 0.02,
  predicted_class: 0,
}));

const meta: Meta<typeof SparkKpiCard> = {
  title: "Sensor Monitor/SparkKpiCard",
  component: SparkKpiCard,
  parameters: { layout: "centered" },
  tags: ["autodocs"],
  args: {
    icon: Gauge,
    sparkData: SPARK_DATA,
    sparkKey: "TP2",
    sparkColor: "#3b82f6",
  },
  decorators: [
    (Story) => (
      <div style={{ width: 220 }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof SparkKpiCard>;

export const Default: Story = {
  args: { title: "Pressão TP2", value: "8.12", unit: "bar" },
};

export const Critical: Story = {
  name: "Valor em estado crítico",
  args: {
    title: "Corrente do motor",
    value: "9.80",
    unit: "A",
    alertColor: true,
    sparkKey: "Motor_current",
    sparkColor: "#ef4444",
  },
};

export const Loading: Story = {
  args: { title: "Pressão TP2", value: "8.12", unit: "bar", isLoading: true },
};
