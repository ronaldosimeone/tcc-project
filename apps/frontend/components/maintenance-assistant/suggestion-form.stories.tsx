import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { SuggestionForm } from "./suggestion-form";

const meta: Meta<typeof SuggestionForm> = {
  title: "Maintenance Assistant/SuggestionForm",
  component: SuggestionForm,
  parameters: { layout: "padded" },
  tags: ["autodocs"],
  args: { onSubmit: (input) => console.log("onSubmit", input) },
};

export default meta;
type Story = StoryObj<typeof SuggestionForm>;

export const Default: Story = {
  args: { disabled: false },
};

export const Disabled: Story = {
  name: "Desabilitado (gerando sugestão)",
  args: { disabled: true },
};
