import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { AssistantBody } from "./assistant-body";

const meta: Meta<typeof AssistantBody> = {
  title: "Maintenance Assistant/AssistantBody",
  component: AssistantBody,
  parameters: { layout: "padded" },
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof AssistantBody>;

export const Idle: Story = {
  args: { status: "idle", markdown: "", references: [], message: null },
};

export const Searching: Story = {
  args: { status: "searching", markdown: "", references: [], message: null },
};

export const Generating: Story = {
  args: {
    status: "generating",
    markdown:
      "## Plano de manutenção\n\n1. Isolar o compressor e aliviar a pressão residual.\n2. Inspecionar",
    references: [],
    message: null,
  },
};

export const WithReferences: Story = {
  name: "Done (com referências)",
  args: {
    status: "done",
    markdown:
      "## Plano de manutenção\n\n1. Isolar o compressor e aliviar a pressão residual.\n2. Inspecionar o rolamento principal em busca de desgaste.\n3. Verificar o nível de óleo e substituir se necessário.\n\n> Prioridade: **alta** — probabilidade de falha acima de 85%.",
    references: [
      {
        file_name: "manual_compressor_atlas_copco.pdf",
        page: 42,
        chunk_index: 3,
        source: "chroma",
        score: 0.91,
      },
      {
        file_name: "manual_compressor_atlas_copco.pdf",
        page: 45,
        chunk_index: 1,
        source: "chroma",
        score: 0.87,
      },
    ],
    message: null,
  },
};

export const Skipped: Story = {
  args: {
    status: "skipped",
    markdown: "",
    references: [],
    message:
      "Sugestão automática só é gerada quando a probabilidade excede 0.7.",
  },
};

export const ErrorState: Story = {
  name: "Erro",
  args: {
    status: "error",
    markdown: "",
    references: [],
    message: "Não foi possível gerar a sugestão. Tente novamente.",
  },
};

export const Offline: Story = {
  args: {
    status: "offline",
    markdown: "",
    references: [],
    message: "O assistente de IA está indisponível no momento.",
  },
};
