export type Severity = "NORMAL" | "ALERTA" | "CRÍTICO";
export type EventType = "Diagnóstico" | "Alerta" | "Falha";

export interface HistoryEvent {
  id: string;
  timestamp: string;
  equipment: string;
  type: EventType;
  severity: Severity;
  duration: string;
  description: string;
}

export interface DayAlertCount {
  date: string;
  alerta: number;
  critico: number;
}

export const HISTORY_EVENTS: HistoryEvent[] = [
  {
    id: "EVT-001",
    timestamp: "2026-04-27T14:32:00.000Z",
    equipment: "APU-Trem-042",
    type: "Falha",
    severity: "CRÍTICO",
    duration: "1h 12min",
    description:
      "Vazamento de ar detectado no compressor de estágio 2 — queda abrupta de TP2",
  },
  {
    id: "EVT-002",
    timestamp: "2026-04-27T09:15:00.000Z",
    equipment: "APU-Trem-042",
    type: "Alerta",
    severity: "ALERTA",
    duration: "22min",
    description:
      "Queda de pressão TP2: 7.8 → 6.4 bar — abaixo do limiar operacional",
  },
  {
    id: "EVT-003",
    timestamp: "2026-04-26T22:45:00.000Z",
    equipment: "APU-Trem-023",
    type: "Alerta",
    severity: "ALERTA",
    duration: "47min",
    description:
      "Anomalia na válvula diferenciadora (ΔP = 0.9 bar) — ciclo de descarga irregular",
  },
  {
    id: "EVT-004",
    timestamp: "2026-04-26T16:20:00.000Z",
    equipment: "APU-Trem-042",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "< 1min",
    description:
      "Diagnóstico preventivo automático concluído — sem anomalias detectadas",
  },
  {
    id: "EVT-005",
    timestamp: "2026-04-25T11:05:00.000Z",
    equipment: "APU-Trem-031",
    type: "Alerta",
    severity: "ALERTA",
    duration: "35min",
    description:
      "Temperatura do óleo elevada: 87.3°C (limiar: 85°C) — risco de degradação do lubrificante",
  },
  {
    id: "EVT-006",
    timestamp: "2026-04-25T08:30:00.000Z",
    equipment: "APU-Trem-042",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "< 1min",
    description: "Lubrificação automática do compressor executada com sucesso",
  },
  {
    id: "EVT-007",
    timestamp: "2026-04-24T20:10:00.000Z",
    equipment: "APU-Trem-042",
    type: "Falha",
    severity: "CRÍTICO",
    duration: "3h 20min",
    description:
      "Sobrecorrente crítico no motor de tração: 44.2 A (máximo: 40 A) — parada automática acionada",
  },
  {
    id: "EVT-008",
    timestamp: "2026-04-24T19:55:00.000Z",
    equipment: "APU-Trem-042",
    type: "Alerta",
    severity: "ALERTA",
    duration: "15min",
    description:
      "Pré-alerta: corrente do motor em ascensão rápida (39.8 A) — tendência de sobrecarga",
  },
  {
    id: "EVT-009",
    timestamp: "2026-04-24T14:00:00.000Z",
    equipment: "APU-Trem-015",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "< 1min",
    description:
      "Calibração do sensor TP3 realizada — desvio corrigido: +0.02 bar",
  },
  {
    id: "EVT-010",
    timestamp: "2026-04-23T23:50:00.000Z",
    equipment: "APU-Trem-023",
    type: "Falha",
    severity: "CRÍTICO",
    duration: "52min",
    description:
      "Reservatório de ar em pressão crítica: 4.8 bar (mínimo: 6.0 bar) — serviço interrompido",
  },
  {
    id: "EVT-011",
    timestamp: "2026-04-23T21:30:00.000Z",
    equipment: "APU-Trem-023",
    type: "Alerta",
    severity: "ALERTA",
    duration: "40min",
    description:
      "Ciclo de compressão com frequência anômala: 18 ciclos/hora (esperado: ≤12)",
  },
  {
    id: "EVT-012",
    timestamp: "2026-04-23T10:15:00.000Z",
    equipment: "APU-Trem-055",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "2min",
    description:
      "Teste de pressão de retenção aprovado — válvula de retenção íntegra",
  },
  {
    id: "EVT-013",
    timestamp: "2026-04-22T18:44:00.000Z",
    equipment: "APU-Trem-042",
    type: "Alerta",
    severity: "ALERTA",
    duration: "28min",
    description:
      "Falha no secador de ar detectada: H1 excedeu 3.2 kVA — umidade residual elevada",
  },
  {
    id: "EVT-014",
    timestamp: "2026-04-22T12:00:00.000Z",
    equipment: "APU-Trem-031",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "< 1min",
    description:
      "Verificação de pressão diferencial concluída — dentro dos parâmetros esperados",
  },
  {
    id: "EVT-015",
    timestamp: "2026-04-21T15:30:00.000Z",
    equipment: "APU-Trem-042",
    type: "Alerta",
    severity: "ALERTA",
    duration: "1h 5min",
    description:
      "Queda gradual de pressão TP2: tendência de −0.3 bar/hora — possível micro-vazamento",
  },
  {
    id: "EVT-016",
    timestamp: "2026-04-20T09:20:00.000Z",
    equipment: "APU-Trem-015",
    type: "Falha",
    severity: "CRÍTICO",
    duration: "2h 15min",
    description:
      "Vazamento de ar no acoplamento flexível identificado — parada técnica não planejada",
  },
  {
    id: "EVT-017",
    timestamp: "2026-04-20T07:45:00.000Z",
    equipment: "APU-Trem-015",
    type: "Alerta",
    severity: "ALERTA",
    duration: "30min",
    description:
      "Pressão TP2 instável: oscilação ±0.8 bar em 10 minutos — acoplamento suspeito",
  },
  {
    id: "EVT-018",
    timestamp: "2026-04-20T06:00:00.000Z",
    equipment: "APU-Trem-042",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "< 1min",
    description:
      "Relatório diário gerado — 0 anomalias nos últimos 24h para APU-Trem-042",
  },
  {
    id: "EVT-019",
    timestamp: "2026-04-19T21:10:00.000Z",
    equipment: "APU-Trem-023",
    type: "Alerta",
    severity: "ALERTA",
    duration: "44min",
    description:
      "Temperatura do óleo: pico de 91.2°C registrado — risco de coqueificação do lubrificante",
  },
  {
    id: "EVT-020",
    timestamp: "2026-04-19T14:35:00.000Z",
    equipment: "APU-Trem-031",
    type: "Alerta",
    severity: "ALERTA",
    duration: "20min",
    description:
      "Motor_current com ruído elétrico elevado (±2.1 A) — possível interferência no inversor",
  },
  {
    id: "EVT-021",
    timestamp: "2026-04-18T17:50:00.000Z",
    equipment: "APU-Trem-055",
    type: "Falha",
    severity: "CRÍTICO",
    duration: "4h 30min",
    description:
      "Falha crítica do sistema pneumático — compressor não responde a comando de pressurização",
  },
  {
    id: "EVT-022",
    timestamp: "2026-04-18T17:40:00.000Z",
    equipment: "APU-Trem-055",
    type: "Alerta",
    severity: "ALERTA",
    duration: "10min",
    description:
      "Pré-alerta: válvula solenoide de abertura não responde em 3 tentativas consecutivas",
  },
  {
    id: "EVT-023",
    timestamp: "2026-04-18T10:00:00.000Z",
    equipment: "APU-Trem-042",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "3min",
    description:
      "Ajuste de parâmetros PID do controlador de pressão realizado — resposta otimizada",
  },
  {
    id: "EVT-024",
    timestamp: "2026-04-17T22:20:00.000Z",
    equipment: "APU-Trem-042",
    type: "Alerta",
    severity: "ALERTA",
    duration: "55min",
    description:
      "Pico de corrente no motor: 41.5 A — monitoramento intensificado ativado",
  },
  {
    id: "EVT-025",
    timestamp: "2026-04-17T16:45:00.000Z",
    equipment: "APU-Trem-023",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "< 1min",
    description:
      "Verificação do nível de óleo — complementação de 0.3 L realizada conforme protocolo",
  },
  {
    id: "EVT-026",
    timestamp: "2026-04-15T13:10:00.000Z",
    equipment: "APU-Trem-015",
    type: "Alerta",
    severity: "ALERTA",
    duration: "38min",
    description:
      "Desvio de pressão no reservatório: TP3 em 6.9 bar (esperado: 8.0 ± 0.5 bar)",
  },
  {
    id: "EVT-027",
    timestamp: "2026-04-15T08:25:00.000Z",
    equipment: "APU-Trem-031",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "1min",
    description:
      "Sensor DV_pressure recalibrado após substituição de filtro de linha anti-poluentes",
  },
  {
    id: "EVT-028",
    timestamp: "2026-04-14T20:05:00.000Z",
    equipment: "APU-Trem-042",
    type: "Falha",
    severity: "CRÍTICO",
    duration: "1h 45min",
    description:
      "Alarme: temperatura do óleo atingiu 97.8°C — parada de emergência por proteção térmica",
  },
  {
    id: "EVT-029",
    timestamp: "2026-04-14T19:50:00.000Z",
    equipment: "APU-Trem-042",
    type: "Alerta",
    severity: "ALERTA",
    duration: "15min",
    description:
      "Temperatura do óleo em escalada: 88.3°C → 97.8°C em 15 minutos — resfriamento insuficiente",
  },
  {
    id: "EVT-030",
    timestamp: "2026-04-14T11:00:00.000Z",
    equipment: "APU-Trem-055",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "< 1min",
    description:
      "Inspeção visual do acoplamento flexível — sem desgaste visível ou folga excessiva",
  },
  {
    id: "EVT-031",
    timestamp: "2026-04-12T16:30:00.000Z",
    equipment: "APU-Trem-023",
    type: "Alerta",
    severity: "ALERTA",
    duration: "1h 10min",
    description:
      "Ciclo de compressão degradado: rendimento volumétrico em 71% (esperado: ≥85%)",
  },
  {
    id: "EVT-032",
    timestamp: "2026-04-12T09:45:00.000Z",
    equipment: "APU-Trem-031",
    type: "Falha",
    severity: "CRÍTICO",
    duration: "2h 40min",
    description:
      "Vazamento severo no tubo de descarga — perda de 1.2 bar em 5 minutos — serviço suspenso",
  },
  {
    id: "EVT-033",
    timestamp: "2026-04-10T14:20:00.000Z",
    equipment: "APU-Trem-042",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "5min",
    description:
      "Relatório semanal de manutenção preventiva gerado — 2 alertas, 0 falhas na semana",
  },
  {
    id: "EVT-034",
    timestamp: "2026-04-10T07:00:00.000Z",
    equipment: "APU-Trem-015",
    type: "Alerta",
    severity: "ALERTA",
    duration: "25min",
    description:
      "Oscilação de corrente do motor: ±3.2 A em regime constante — investigar conexões",
  },
  {
    id: "EVT-035",
    timestamp: "2026-04-09T19:15:00.000Z",
    equipment: "APU-Trem-023",
    type: "Falha",
    severity: "CRÍTICO",
    duration: "6h 20min",
    description:
      "Falha no dreno automático — acúmulo de condensado no reservatório (nível crítico)",
  },
  {
    id: "EVT-036",
    timestamp: "2026-04-09T18:50:00.000Z",
    equipment: "APU-Trem-023",
    type: "Alerta",
    severity: "ALERTA",
    duration: "25min",
    description:
      "Sensor de drenagem não responde — possível bloqueio mecânico por sedimento",
  },
  {
    id: "EVT-037",
    timestamp: "2026-04-07T15:00:00.000Z",
    equipment: "APU-Trem-055",
    type: "Alerta",
    severity: "ALERTA",
    duration: "40min",
    description:
      "Vibração anômala detectada na carcaça do compressor: 14.2 mm/s (máx: 12 mm/s)",
  },
  {
    id: "EVT-038",
    timestamp: "2026-04-07T10:30:00.000Z",
    equipment: "APU-Trem-042",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "< 1min",
    description:
      "Análise de vibrações concluída — dentro dos parâmetros de operação ISO 10816",
  },
  {
    id: "EVT-039",
    timestamp: "2026-04-05T22:00:00.000Z",
    equipment: "APU-Trem-031",
    type: "Alerta",
    severity: "ALERTA",
    duration: "1h 30min",
    description:
      "Pressão diferencial elevada no filtro de ar: 0.8 bar (máx: 0.5 bar) — troca urgente",
  },
  {
    id: "EVT-040",
    timestamp: "2026-04-05T12:10:00.000Z",
    equipment: "APU-Trem-015",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "2min",
    description:
      "Filtro de ar trocado — queda de pressão diferencial normalizada (0.12 bar)",
  },
  {
    id: "EVT-041",
    timestamp: "2026-04-03T18:40:00.000Z",
    equipment: "APU-Trem-042",
    type: "Alerta",
    severity: "ALERTA",
    duration: "50min",
    description:
      "Queda de pressão TP2 durante pico de demanda: 8.1 → 7.0 bar — capacidade insuficiente",
  },
  {
    id: "EVT-042",
    timestamp: "2026-04-03T14:25:00.000Z",
    equipment: "APU-Trem-023",
    type: "Falha",
    severity: "CRÍTICO",
    duration: "3h 10min",
    description:
      "Sobrecarga térmica do motor — desligamento por proteção automática (classe F atingida)",
  },
  {
    id: "EVT-043",
    timestamp: "2026-04-01T11:30:00.000Z",
    equipment: "APU-Trem-031",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "3min",
    description:
      "Manutenção preventiva trimestral — troca de óleo ISO VG 46 e filtros realizada",
  },
  {
    id: "EVT-044",
    timestamp: "2026-04-01T08:00:00.000Z",
    equipment: "APU-Trem-055",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "< 1min",
    description:
      "Verificação de torque dos parafusos de fixação — todos dentro do especificado (Nm)",
  },
  {
    id: "EVT-045",
    timestamp: "2026-03-30T20:15:00.000Z",
    equipment: "APU-Trem-015",
    type: "Falha",
    severity: "CRÍTICO",
    duration: "5h 0min",
    description:
      "Ruptura da gaxeta do compressor (estágio 1) — parada não planejada, reparo emergencial",
  },
  {
    id: "EVT-046",
    timestamp: "2026-03-30T20:00:00.000Z",
    equipment: "APU-Trem-015",
    type: "Alerta",
    severity: "ALERTA",
    duration: "15min",
    description:
      "Queda súbita de pressão no estágio 1: 7.9 → 5.2 bar em apenas 3 minutos",
  },
  {
    id: "EVT-047",
    timestamp: "2026-03-29T15:45:00.000Z",
    equipment: "APU-Trem-042",
    type: "Alerta",
    severity: "ALERTA",
    duration: "35min",
    description:
      "Corrente do motor com desvio de +8% acima da linha de base histórica — investigar rolamentos",
  },
  {
    id: "EVT-048",
    timestamp: "2026-03-29T10:20:00.000Z",
    equipment: "APU-Trem-023",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "1min",
    description:
      "Reset do contador de ciclos após manutenção preventiva — sistema reiniciado",
  },
  {
    id: "EVT-049",
    timestamp: "2026-03-28T16:00:00.000Z",
    equipment: "APU-Trem-055",
    type: "Falha",
    severity: "CRÍTICO",
    duration: "2h 50min",
    description:
      "Contaminação do óleo por partículas metálicas (>100 ppm Fe) — análise espectrométrica",
  },
  {
    id: "EVT-050",
    timestamp: "2026-03-28T09:30:00.000Z",
    equipment: "APU-Trem-031",
    type: "Diagnóstico",
    severity: "NORMAL",
    duration: "< 1min",
    description:
      "Início do período de monitoramento estendido — todos os 12 sensores online",
  },
];

export function generateDailyAlertCounts(
  events: HistoryEvent[],
  days: number,
): DayAlertCount[] {
  const result: DayAlertCount[] = [];
  const base = new Date("2026-04-27T00:00:00.000Z");

  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(base);
    d.setUTCDate(d.getUTCDate() - i);
    const dayStr = d.toISOString().slice(0, 10);

    let alerta = 0;
    let critico = 0;

    for (const e of events) {
      if (e.timestamp.slice(0, 10) !== dayStr) continue;
      if (e.severity === "CRÍTICO") critico++;
      else if (e.severity === "ALERTA") alerta++;
    }

    result.push({
      date: `${String(d.getUTCDate()).padStart(2, "0")}/${String(
        d.getUTCMonth() + 1,
      ).padStart(2, "0")}`,
      alerta,
      critico,
    });
  }

  return result;
}
