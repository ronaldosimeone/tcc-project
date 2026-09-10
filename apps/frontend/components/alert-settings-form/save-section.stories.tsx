import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { SaveSection } from "./save-and-test-sections";

const meta: Meta<typeof SaveSection> = {
  title: "Alert Settings/SaveSection",
  component: SaveSection,
  parameters: { layout: "padded" },
  tags: ["autodocs"],
  args: { canSave: true, onSave: () => {} },
};

export default meta;
type Story = StoryObj<typeof SaveSection>;

export const Idle: Story = {
  args: { saveState: "idle", saveError: null },
};

export const Saving: Story = {
  args: { saveState: "saving", saveError: null },
};

export const Saved: Story = {
  args: { saveState: "saved", saveError: null },
};

export const SaveError: Story = {
  name: "Erro ao salvar",
  args: {
    saveState: "error",
    saveError: "Falha ao salvar a configuração. Tente novamente.",
  },
};

export const CannotSave: Story = {
  name: "Desabilitado (validação pendente)",
  args: { saveState: "idle", saveError: null, canSave: false },
};
