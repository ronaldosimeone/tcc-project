import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { ReferencesList } from "./references-list";

const meta: Meta<typeof ReferencesList> = {
  title: "Maintenance Assistant/ReferencesList",
  component: ReferencesList,
  parameters: { layout: "padded" },
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof ReferencesList>;

export const Empty: Story = {
  args: { references: [] },
};

export const WithReferences: Story = {
  args: {
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
      {
        file_name: "guia_manutencao_preventiva.pdf",
        page: 12,
        chunk_index: 0,
        source: "chroma",
        score: 0.74,
      },
    ],
  },
};
