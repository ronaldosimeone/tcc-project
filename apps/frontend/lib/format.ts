/**
 * Formatação de datas e números da UI — fonte única.
 *
 * Datas: EXCLUSIVO `dayjs` (CLAUDE.md), sempre no fuso do navegador do
 * operador. A API envia instantes em UTC (`…Z`); mostrar em UTC fazia o
 * histórico e o feed ficarem 3 h à frente das demais telas no Brasil.
 */

import dayjs from "dayjs";
import "dayjs/locale/pt-br";

type Instant = string | number | Date;

/** Relógio de log industrial: `HH:mm:ss`. */
export const formatClock = (ts: Instant): string =>
  dayjs(ts).format("HH:mm:ss");

/** Data curta com hora: `20 de out, 01:02`. */
export const formatDayMonthTime = (ts: Instant): string =>
  dayjs(ts).locale("pt-br").format("DD [de] MMM, HH:mm");

/** Dia e mês: `20/10`. */
export const formatDayMonth = (ts: Instant): string =>
  dayjs(ts).format("DD/MM");

/** Chave de dia local `YYYY-MM-DD` (agrupamentos por dia). */
export const localDayKey = (ts: Instant): string =>
  dayjs(ts).format("YYYY-MM-DD");

/**
 * `toFixed` sem zero negativo: ruído do sensor (ex.: −0,014 bar) virava
 * "-0.0" na tela.
 */
export const formatFixed = (value: number, digits: number): string => {
  const text = value.toFixed(digits);
  return Number(text) === 0 ? (0).toFixed(digits) : text;
};
