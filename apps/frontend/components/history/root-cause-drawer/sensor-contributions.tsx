// ── Contribuição por sensor (oclusão, RF-32) congelada na abertura do evento ──

import type { SensorContribution } from "@/lib/history-types";
import { cn } from "@/lib/utils";

const SENSOR_LABEL: Record<string, string> = {
  TP2: "Pressão de saída do compressor",
  TP3: "Pressão do painel pneumático",
  H1: "Filtro separador ciclônico",
  DV_pressure: "Descarga das torres do secador",
  Reservoirs: "Reservatório principal",
  Oil_temperature: "Temperatura do óleo",
  Motor_current: "Corrente do motor",
  COMP: "Válvula de admissão",
  DV_eletric: "Válvula de saída",
  Towers: "Torres do secador",
  MPG: "Pressostato de partida",
  Oil_level: "Nível de óleo",
};

interface SensorContributionsProps {
  contributions: SensorContribution[];
  limit?: number;
}

export function SensorContributions({
  contributions,
  limit = 6,
}: SensorContributionsProps) {
  const shown = contributions.slice(0, limit);
  const max = Math.max(...shown.map((c) => Math.abs(c.contribution)), 1e-9);

  return (
    <section aria-label="Contribuição dos sensores">
      <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
        Sensores que explicam o alarme · oclusão no momento da abertura
      </p>
      <ul className="space-y-1.5">
        {shown.map((c) => {
          const width = `${(Math.abs(c.contribution) / max) * 100}%`;
          const towardFailure = c.contribution > 0;
          return (
            <li key={c.sensor} className="text-xs">
              <div className="flex items-baseline justify-between gap-2">
                <span className="truncate text-foreground">
                  <span className="font-mono font-semibold">{c.sensor}</span>{" "}
                  <span className="text-muted-foreground">
                    {SENSOR_LABEL[c.sensor] ?? ""}
                  </span>
                </span>
                <span
                  className={cn(
                    "font-mono tabular-nums",
                    towardFailure
                      ? "text-destructive"
                      : "text-muted-foreground",
                  )}
                >
                  {c.contribution >= 0 ? "+" : ""}
                  {c.contribution.toFixed(3)}
                </span>
              </div>
              <div className="mt-0.5 h-1.5 rounded-full bg-muted">
                <div
                  className={cn(
                    "h-1.5 rounded-full",
                    towardFailure ? "bg-destructive" : "bg-chart-2",
                  )}
                  style={{ width }}
                />
              </div>
            </li>
          );
        })}
      </ul>
      <p className="mt-2 text-[10px] leading-snug text-muted-foreground">
        Queda de P(falha) quando a janela do sensor é substituída pela mediana
        de operação normal. Positivo = empurra para falha.
      </p>
    </section>
  );
}
