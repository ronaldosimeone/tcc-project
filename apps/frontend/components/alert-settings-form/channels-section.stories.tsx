import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { ChannelsSection } from "./channels-section";

const meta: Meta<typeof ChannelsSection> = {
  title: "Alert Settings/ChannelsSection",
  component: ChannelsSection,
  parameters: { layout: "padded" },
  tags: ["autodocs"],
  args: {
    isSaving: false,
    onTelegramToggle: () => {},
    onEmailToggle: () => {},
    onEmailChange: () => {},
  },
};

export default meta;
type Story = StoryObj<typeof ChannelsSection>;

export const TelegramOnly: Story = {
  args: {
    telegramEnabled: true,
    emailEnabled: false,
    alertEmail: "",
    emailFormatValid: true,
    validationMessage: null,
  },
};

export const EmailValido: Story = {
  name: "E-mail habilitado (válido)",
  args: {
    telegramEnabled: true,
    emailEnabled: true,
    alertEmail: "alertas@empresa.com.br",
    emailFormatValid: true,
    validationMessage: null,
  },
};

export const Error: Story = {
  name: "E-mail inválido (erro de validação)",
  args: {
    telegramEnabled: false,
    emailEnabled: true,
    alertEmail: "email-invalido",
    emailFormatValid: false,
    validationMessage: "Informe um e-mail válido para receber os alertas.",
  },
};

export const NenhumCanal: Story = {
  name: "Nenhum canal habilitado (erro de validação)",
  args: {
    telegramEnabled: false,
    emailEnabled: false,
    alertEmail: "",
    emailFormatValid: true,
    validationMessage: "Habilite ao menos um canal de notificação.",
  },
};

export const Saving: Story = {
  args: {
    telegramEnabled: true,
    emailEnabled: true,
    alertEmail: "alertas@empresa.com.br",
    emailFormatValid: true,
    validationMessage: null,
    isSaving: true,
  },
};
