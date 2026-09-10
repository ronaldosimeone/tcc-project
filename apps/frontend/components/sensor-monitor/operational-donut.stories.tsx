import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { SensorDataPoint } from "@/hooks/use-sensor-data";
import { OperationalDonut } from "./operational-donut";

// Motor_current cobrindo as 4 faixas do OP_THRESHOLDS (constants.ts):
// off < 1.0 · noLoad < 5.5 · load < 8.5 · partida ≥ 8.5.
const MIXED_HISTORY: SensorDataPoint[] = [
  ...Array.from({ length: 3 }, (_, i) => ({ Motor_current: 0.2 + i * 0.1 })),
  ...Array.from({ length: 8 }, (_, i) => ({ Motor_current: 3 + i * 0.2 })),
  ...Array.from({ length: 15 }, (_, i) => ({ Motor_current: 6 + i * 0.1 })),
  ...Array.from({ length: 2 }, (_, i) => ({ Motor_current: 9 + i * 0.3 })),
].map((partial, i) => ({
  time: `${i}:00`,
  TP2: 8,
  TP3: 7.8,
  Oil_temperature: 72,
  failure_probability: 0.1,
  predicted_class: 0,
  ...partial,
}));

const meta: Meta<typeof OperationalDonut> = {
  title: "Sensor Monitor/OperationalDonut",
  component: OperationalDonut,
  parameters: { layout: "centered" },
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div style={{ width: 260 }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof OperationalDonut>;

export const Default: Story = {
  args: { history: MIXED_HISTORY, isLoading: false },
};

export const Loading: Story = {
  args: { history: [], isLoading: true },
};

export const Empty: Story = {
  name: "Sem dados",
  args: { history: [], isLoading: false },
};
