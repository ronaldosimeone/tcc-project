import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { TestNotificationSection } from "./save-and-test-sections";

const meta: Meta<typeof TestNotificationSection> = {
  title: "Alert Settings/TestNotificationSection",
  component: TestNotificationSection,
  parameters: { layout: "padded" },
  tags: ["autodocs"],
  args: { onTest: () => {} },
};

export default meta;
type Story = StoryObj<typeof TestNotificationSection>;

export const Idle: Story = {
  args: { testState: "idle", testError: null, testResultMessage: null },
};

export const Testing: Story = {
  args: { testState: "testing", testError: null, testResultMessage: null },
};

export const Success: Story = {
  args: {
    testState: "success",
    testError: null,
    testResultMessage: "Notificação de teste enviada ao Telegram.",
  },
};

export const TestError: Story = {
  name: "Erro ao testar",
  args: {
    testState: "error",
    testError:
      "Não foi possível enviar a notificação. Verifique a configuração do Telegram.",
    testResultMessage: null,
  },
};
