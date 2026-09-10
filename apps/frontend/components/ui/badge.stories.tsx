import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import { Badge } from "./badge";

const meta: Meta<typeof Badge> = {
  title: "UI/Badge",
  component: Badge,
  parameters: { layout: "centered" },
  tags: ["autodocs"],
  argTypes: {
    variant: {
      control: "select",
      options: [
        "default",
        "secondary",
        "destructive",
        "outline",
        "ghost",
        "link",
      ],
    },
  },
};

export default meta;
type Story = StoryObj<typeof Badge>;

// Estados reais usados no projeto para o nível de risco (RF-08/RF-25).
export const Normal: Story = {
  args: {
    variant: "outline",
    className: "gap-1 border-green-500/40 bg-green-500/10 text-green-400",
    children: (
      <>
        <CheckCircle2 className="h-3 w-3" />
        NORMAL
      </>
    ),
  },
};

export const Alerta: Story = {
  args: {
    variant: "outline",
    className: "gap-1 border-amber-500/40 bg-amber-500/10 text-amber-400",
    children: (
      <>
        <AlertTriangle className="h-3 w-3" />
        ALERTA
      </>
    ),
  },
};

export const Critico: Story = {
  args: {
    variant: "outline",
    className: "gap-1 border-red-500/40 bg-red-500/10 text-red-400",
    children: (
      <>
        <XCircle className="h-3 w-3" />
        CRÍTICO
      </>
    ),
  },
};

export const Outline: Story = {
  args: { variant: "outline", children: "SIMULADO" },
};
