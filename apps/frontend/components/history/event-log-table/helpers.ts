// ── Helpers puros — RNF-58: extraído de EventLogTable ───────────────────────

import { formatDayMonthTime } from "@/lib/format";

/** "20 de out, 01:02" no fuso do operador (a API envia UTC). */
export const formatTimestamp = (iso: string): string => formatDayMonthTime(iso);
