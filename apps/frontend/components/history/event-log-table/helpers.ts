// ── Helpers puros — RNF-58: extraído de EventLogTable ───────────────────────

export function formatTimestamp(iso: string): string {
  return new Intl.DateTimeFormat("pt-BR", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  })
    .format(new Date(iso))
    .replace(".", "");
}
