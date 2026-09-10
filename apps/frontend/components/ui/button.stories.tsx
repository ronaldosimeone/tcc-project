import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { Loader2, Trash2 } from "lucide-react";
import { Button } from "./button";

const meta: Meta<typeof Button> = {
  title: "UI/Button",
  component: Button,
  parameters: { layout: "centered" },
  tags: ["autodocs"],
  argTypes: {
    variant: {
      control: "select",
      options: [
        "default",
        "outline",
        "secondary",
        "ghost",
        "destructive",
        "link",
      ],
    },
    size: {
      control: "select",
      options: ["default", "xs", "sm", "lg", "icon"],
    },
  },
};

export default meta;
type Story = StoryObj<typeof Button>;

export const Default: Story = {
  args: { children: "Salvar configurações", variant: "default" },
};

export const Outline: Story = {
  args: { children: "Testar Notificação", variant: "outline" },
};

export const Destructive: Story = {
  args: {
    children: (
      <>
        <Trash2 className="h-3 w-3" />
        Limpar
      </>
    ),
    variant: "ghost",
  },
};

export const Disabled: Story = {
  args: { children: "Salvar configurações", disabled: true },
};

export const Loading: Story = {
  name: "Loading (spinner inline)",
  args: {
    children: (
      <>
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Salvando…
      </>
    ),
    disabled: true,
  },
};
