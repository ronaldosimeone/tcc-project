/**
 * lib/format — datas no fuso do operador (dayjs) e números sem zero negativo.
 */
import dayjs from "dayjs";
import { describe, expect, it } from "vitest";

import {
  formatClock,
  formatDayMonth,
  formatDayMonthTime,
  formatFixed,
  localDayKey,
} from "@/lib/format";

describe("formatFixed", () => {
  it("não mostra zero negativo para ruído do sensor", () => {
    expect(formatFixed(-0.014, 1)).toBe("0.0");
    expect(formatFixed(-0.0004, 2)).toBe("0.00");
  });

  it("mantém negativos de verdade e arredonda", () => {
    expect(formatFixed(-0.06, 1)).toBe("-0.1");
    expect(formatFixed(8.977999687194824, 2)).toBe("8.98");
  });
});

describe("datas no fuso local", () => {
  // Instante UTC como a API envia; o esperado é calculado no fuso do
  // ambiente de teste, então o teste vale em qualquer fuso.
  const iso = "2026-10-20T01:02:03Z";
  const local = dayjs(iso);

  it("formatClock usa HH:mm:ss local", () => {
    expect(formatClock(iso)).toBe(local.format("HH:mm:ss"));
  });

  it("formatDayMonthTime em pt-BR", () => {
    expect(formatDayMonthTime(iso)).toMatch(
      /^\d{2} de [a-zç]{3}, \d{2}:\d{2}$/,
    );
    expect(formatDayMonthTime(iso)).toContain(local.format("HH:mm"));
  });

  it("formatDayMonth e localDayKey seguem o dia local", () => {
    expect(formatDayMonth(iso)).toBe(local.format("DD/MM"));
    expect(localDayKey(iso)).toBe(local.format("YYYY-MM-DD"));
  });
});
