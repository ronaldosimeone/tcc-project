import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { Skeleton } from "./skeleton";

const meta: Meta<typeof Skeleton> = {
  title: "UI/Skeleton",
  component: Skeleton,
  parameters: { layout: "centered" },
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof Skeleton>;

export const Default: Story = {
  args: { className: "h-4 w-40" },
};

// Composição real usada nos KPI cards durante o carregamento (RNF-21).
export const KpiCardLoading: Story = {
  render: () => (
    <div className="flex flex-col gap-2">
      <Skeleton className="h-3 w-24" />
      <Skeleton className="h-7 w-20" />
    </div>
  ),
};

export const CircularGauge: Story = {
  render: () => <Skeleton className="h-[120px] w-[120px] rounded-full" />,
};
