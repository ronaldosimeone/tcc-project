import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { ThresholdSection } from "./threshold-section";

const meta: Meta<typeof ThresholdSection> = {
  title: "Alert Settings/ThresholdSection",
  component: ThresholdSection,
  parameters: { layout: "padded" },
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof ThresholdSection>;

export const Default: Story = {
  args: { threshold: 0.85, isSaving: false },
  render: (args) => {
    // Wrapper com estado local — o Slider é controlado, então a story
    // precisa refletir o `onSliderChange` de volta para ser interativa.
    function Wrapper() {
      const [threshold, setThreshold] = useState(args.threshold);
      return (
        <ThresholdSection
          {...args}
          threshold={threshold}
          onSliderChange={(values) => {
            const next = values[0];
            if (next !== undefined) setThreshold(next);
          }}
        />
      );
    }
    return <Wrapper />;
  },
};

export const Saving: Story = {
  args: { threshold: 0.7, isSaving: true, onSliderChange: () => {} },
};
