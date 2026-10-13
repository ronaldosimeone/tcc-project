// ── Timeline do incidente (3 passos) — RNF-58: extraído de RootCauseDrawer ──

import { cn } from "@/lib/utils";
import { TIMELINE, TIMELINE_DOT } from "./constants";

export function IncidentTimeline() {
  return (
    <section>
      <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
        Linha do tempo do incidente
      </p>
      <ol className="relative space-y-3">
        {/* Linha vertical conectora */}
        <span
          aria-hidden="true"
          className="absolute left-[5px] top-2 bottom-2 w-px bg-muted"
        />
        {TIMELINE.map((step, i) => (
          <li key={step.label} className="relative flex items-start gap-3 pl-1">
            <span
              className={cn(
                "mt-1 inline-flex h-2.5 w-2.5 shrink-0 rounded-full ring-4",
                TIMELINE_DOT[step.tone],
              )}
              aria-hidden="true"
            />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-foreground">
                {i + 1}. {step.label}
              </p>
              <p className="text-xs text-muted-foreground">{step.detail}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
