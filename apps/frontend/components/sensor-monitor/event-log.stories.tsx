import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import {
  PREDICTION_HISTORY_STORAGE_KEY,
  type PredictionHistoryEntry,
} from "@/hooks/use-prediction-history";
import { EventLog } from "./event-log";

const SEED: PredictionHistoryEntry[] = [
  {
    id: "1",
    timestamp: new Date(Date.now() - 5_000).toISOString(),
    failure_probability: 0.12,
    predicted_class: 0,
    riskLevel: "NORMAL",
  },
  {
    id: "2",
    timestamp: new Date(Date.now() - 60_000).toISOString(),
    failure_probability: 0.48,
    predicted_class: 0,
    riskLevel: "ALERTA",
  },
  {
    id: "3",
    timestamp: new Date(Date.now() - 180_000).toISOString(),
    failure_probability: 0.91,
    predicted_class: 1,
    riskLevel: "CRÍTICO",
  },
];

/** Pré-popula o localStorage (RNF-14) — o hook `usePredictionHistory`
 * hidrata a partir dele no mount. Seed acontece no render (antes de
 * qualquer effect), garantindo que a hidratação já encontre os dados. */
function withSeededHistory(seed: PredictionHistoryEntry[] | null) {
  return function Decorator(Story: () => React.ReactElement) {
    useState(() => {
      try {
        if (seed) {
          localStorage.setItem(
            PREDICTION_HISTORY_STORAGE_KEY,
            JSON.stringify(seed),
          );
        } else {
          localStorage.removeItem(PREDICTION_HISTORY_STORAGE_KEY);
        }
      } catch {
        // localStorage indisponível (ex.: preview isolado) — ignora.
      }
      return null;
    });
    return <Story />;
  };
}

const meta: Meta<typeof EventLog> = {
  title: "Sensor Monitor/EventLog",
  component: EventLog,
  parameters: { layout: "padded" },
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div style={{ maxWidth: 360 }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof EventLog>;

export const Empty: Story = {
  decorators: [withSeededHistory(null)],
  args: { latest: null, riskLevel: "NORMAL" },
};

export const WithHistory: Story = {
  decorators: [withSeededHistory(SEED)],
  args: {
    latest: {
      predicted_class: 1,
      failure_probability: 0.91,
      timestamp: new Date().toISOString(),
    },
    riskLevel: "CRÍTICO",
  },
};
