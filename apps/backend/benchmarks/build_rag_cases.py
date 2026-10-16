"""
Builds the annotated RAG benchmark (RF-36) — ``rag_cases_v1.json``.

Every case is an operator-style failure description (PT-BR) with:
    gold_refs   manual file + page(s) that contain the answer
    gold_facts  key facts the plan must contain; each fact is a list of
                accepted surface forms (EN manual wording / PT translation)
    answerable  False = the corpus has NO procedure for it: the correct
                behaviour is "Limitações" (used to measure hallucination)
    sensors     optional XAI evidence (MetroPT-3 sensor -> contribution),
                consistent with the physical component of the case (RF-35)

Annotation status: DRAFT written from the manual text by the author's
assistant. Every case carries ``validation: "pending"`` until a maintenance
specialist reviews it (thesis requirement for RF-36); the review outcome is
recorded per case in ``validated_by`` / ``validation_notes``.

Run:  python benchmarks/build_rag_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

WAP5 = "wap5-scheduled-maintenance-vol-b.pdf"
CX500 = "manual-compressor-cx500.pdf"
MOTOR = "manual-motor-eletrico.pdf"
PUMP = "manual-bomba-centrifuga.pdf"

EQUIP = "Compressor de ar da unidade de produção de ar (APU)"

# (id, query, refs[(file, page)], facts[[alternatives]], sensors{sensor: contrib}, task)
ANSWERABLE: list[
    tuple[str, str, list[tuple[str, int]], list[list[str]], dict[str, float], str]
] = [
    (
        "A01",
        "nível de óleo do compressor baixo",
        [(WAP5, 170)],
        [
            ["top-up", "completar"],
            ["do not over fill", "não encher demais", "não exceder"],
        ],
        {"Oil_level": 0.35},
        "601.1W.02",
    ),
    (
        "A02",
        "vazamento de óleo no compressor principal",
        [(WAP5, 170)],
        [
            ["oil leakage", "vazamento de óleo"],
            ["replace the compressor", "substituir o compressor"],
        ],
        {"Oil_level": 0.2, "Oil_temperature": 0.15},
        "601.1W.03",
    ),
    (
        "A03",
        "motor do compressor superaquecendo, ventilação obstruída",
        [(WAP5, 170)],
        [
            ["cooling vents", "aberturas de ventilação", "ventilação"],
            ["dirt", "sujeira"],
            ["bearings", "rolamentos"],
        ],
        {"Motor_current": 0.3, "Oil_temperature": 0.2},
        "601.1M.01",
    ),
    (
        "A04",
        "pressão de óleo do compressor fora da especificação",
        [(WAP5, 170)],
        [["oil pressure", "pressão do óleo"], ["adjust", "ajustar"]],
        {"Oil_temperature": 0.25},
        "601.1M.02",
    ),
    (
        "A05",
        "ruído anormal no compressor durante operação",
        [(WAP5, 170)],
        [["unusual noises", "ruídos anormais", "ruído"], ["replace", "substituir"]],
        {"Motor_current": 0.2, "COMP": 0.1},
        "601.1M.03",
    ),
    (
        "A06",
        "vibração do compressor, coxins desgastados",
        [(WAP5, 170)],
        [
            ["resilient mounting", "coxins", "suportes resilientes"],
            ["worn", "desgastad"],
        ],
        {"Motor_current": 0.1},
        "601.3M.01",
    ),
    (
        "A07",
        "mangueira de descarga do compressor danificada",
        [(WAP5, 170)],
        [["delivery hose", "mangueira de descarga"], ["replace", "substituir"]],
        {"TP2": 0.3, "Reservoirs": 0.2},
        "601.3M.02",
    ),
    (
        "A08",
        "filtro de admissão de ar do compressor sujo",
        [(WAP5, 171)],
        [["air intake filter", "filtro de admissão"], ["clean", "limpar"]],
        {"COMP": 0.3, "Motor_current": 0.2},
        "601.3M.04",
    ),
    (
        "A09",
        "respiro do cárter do compressor obstruído",
        [(WAP5, 171)],
        [["crankcase breather", "respiro do cárter"], ["clean", "limpar"]],
        {"Oil_temperature": 0.2},
        "601.3M.05",
    ),
    (
        "A10",
        "compressor demora para encher os reservatórios",
        [(WAP5, 171)],
        [
            ["time taken", "tempo"],
            ["fill the main reservoirs", "encher os reservatórios"],
            ["replace the compressor", "substituir o compressor"],
        ],
        {"Reservoirs": 0.4, "TP2": 0.3},
        "601.3M.07",
    ),
    (
        "A11",
        "fixações do compressor soltas",
        [(WAP5, 171)],
        [["fasteners", "fixações", "parafusos"], ["tighten", "apertar"]],
        {"Motor_current": 0.1},
        "601.3M.08",
    ),
    (
        "A12",
        "filtro do copo de dreno (drip cup) sujo",
        [(WAP5, 171)],
        [["drip cup filter", "copo de dreno", "drip cup"], ["clean", "limpar"]],
        {"H1": 0.3},
        "601.6M.01",
    ),
    (
        "A13",
        "inspeção anual do compressor: motor, ventilador e cárter",
        [(WAP5, 171)],
        [["cooling fan", "ventilador"], ["oil sump", "cárter"]],
        {},
        "601.1Y.01",
    ),
    (
        "A14",
        "filtro de óleo primário do compressor entupido",
        [(WAP5, 171)],
        [["primary oil filter", "filtro de óleo primário"], ["clean", "limpar"]],
        {"Oil_temperature": 0.3},
        "601.1Y.02",
    ),
    (
        "A15",
        "trocar filtro de óleo secundário do compressor",
        [(WAP5, 171)],
        [["secondary oil filter", "filtro de óleo secundário"], ["change", "trocar"]],
        {"Oil_temperature": 0.2},
        "601.1Y.03",
    ),
    (
        "A16",
        "válvulas concêntricas do compressor com depósitos",
        [(WAP5, 172)],
        [["concentric valves", "válvulas concêntricas"], ["clean", "limpar"]],
        {"TP2": 0.25, "COMP": 0.2},
        "601.1Y.04",
    ),
    (
        "A17",
        "válvulas de dreno automático de baixa pressão sujas",
        [(WAP5, 172)],
        [
            ["low pressure automatic drain valves", "dreno automático"],
            ["clean", "limpar"],
        ],
        {"H1": 0.3, "DV_pressure": 0.1},
        "601.1Y.06",
    ),
    (
        "A18",
        "lubrificar rolamentos do motor do compressor",
        [(WAP5, 172)],
        [["motor bearings", "rolamentos do motor"], ["lubricate", "lubrificar"]],
        {"Motor_current": 0.25},
        "601.18M.01",
    ),
    (
        "A19",
        "pressostato do compressor não aciona",
        [(WAP5, 172)],
        [
            ["pressure switches", "pressostato"],
            ["test", "testar"],
            ["brake unit", "unidade de freio"],
        ],
        {"MPG": 0.4, "Reservoirs": 0.2},
        "601.4Y.01",
    ),
    (
        "A20",
        "trinca no ponto de fixação do compressor no chassi",
        [(WAP5, 173)],
        [["mounting points", "pontos de fixação"], ["cracks", "trincas"]],
        {},
        "601.5Y.01",
    ),
    (
        "A21",
        "vazamento na saída do compressor, junta de cobre",
        [(WAP5, 173)],
        [["copper gasket", "junta de cobre"], ["replace", "substituir"]],
        {"TP2": 0.3, "Reservoirs": 0.25},
        "601.5Y.02",
    ),
    (
        "A22",
        "secador de ar com dano visível",
        [(WAP5, 173)],
        [
            ["air drier", "secador de ar"],
            ["replace the air drier", "substituir o secador"],
        ],
        {"Towers": 0.3, "DV_pressure": 0.2},
        "602.1W.O1",
    ),
    (
        "A23",
        "secador de ar não funciona com o compressor ligado",
        [(WAP5, 173)],
        [
            ["compressor operating", "compressor em operação", "compressor ligado"],
            ["operation of the air drier", "funcionamento do secador"],
        ],
        {"Towers": 0.4, "DV_pressure": 0.3},
        "602.1M.01",
    ),
    (
        "A24",
        "vazamento de ar nas tubulações do secador",
        [(WAP5, 173)],
        [["pneumatic pipes", "tubulações pneumáticas"], ["leakage", "vazamento"]],
        {"DV_pressure": 0.35, "Reservoirs": 0.3},
        "602.3M.01",
    ),
    (
        "A25",
        "aletas de resfriamento do secador sujas",
        [(WAP5, 173)],
        [["cooling fins", "aletas"], ["steam clean", "limpeza a vapor", "vapor"]],
        {"Towers": 0.2},
        "602.1Y.01",
    ),
    (
        "A26",
        "conduíte elétrico do secador danificado",
        [(WAP5, 174)],
        [["electrical conduit", "conduíte"], ["replace", "substituir"]],
        {"Towers": 0.15},
        "602.1Y.02",
    ),
    (
        "A27",
        "pré-filtro coalescente do secador de torre dupla sujo",
        [(WAP5, 174)],
        [
            ["pre-coalescer", "coalescente"],
            ["regenerating orifice", "orifício de regeneração"],
        ],
        {"Towers": 0.35, "H1": 0.2},
        "602.1Y.03",
    ),
    (
        "A28",
        "registros de isolamento do secador defeituosos",
        [(WAP5, 174)],
        [
            ["isolating cocks", "registros de isolamento", "torneiras de isolamento"],
            ["replace", "substituir"],
        ],
        {"Towers": 0.2},
        "602.1Y.04",
    ),
    (
        "A29",
        "dessecante das torres contaminado, umidade no ar comprimido",
        [(WAP5, 174)],
        [["desiccant", "dessecante"], ["oil or water", "óleo ou água"]],
        {"Towers": 0.5, "DV_pressure": 0.2},
        "602.2Y.01",
    ),
    (
        "A30",
        "água e óleo acumulados nos reservatórios",
        [(WAP5, 174)],
        [
            ["drain cocks", "torneiras de dreno", "drenos"],
            ["water and oil", "água e óleo"],
        ],
        {"Reservoirs": 0.3, "H1": 0.2},
        "603.1W.O1",
    ),
    (
        "A31",
        "vazamento nas tubulações pneumáticas dos reservatórios",
        [(WAP5, 174)],
        [["pneumatic pipes", "tubulações pneumáticas"], ["leakage", "vazamento"]],
        {"Reservoirs": 0.45, "TP3": 0.2},
        "603.3M.01",
    ),
    (
        "A32",
        "válvula de dreno automático do reservatório faltando ou com defeito",
        [(WAP5, 174)],
        [
            ["automatic drain valves", "dreno automático"],
            ["missing or defective", "faltando ou defeituos", "ausente"],
        ],
        {"Reservoirs": 0.3},
        "603.3M.02",
    ),
    (
        "A33",
        "registro de isolamento do reservatório principal desgastado",
        [(WAP5, 175)],
        [["isolating cocks", "registros de isolamento"], ["worn", "desgastad"]],
        {"Reservoirs": 0.25},
        "603.3M.03",
    ),
    (
        "A34",
        "correntes de segurança do reservatório soltas",
        [(WAP5, 175)],
        [["safety chains", "correntes de segurança"]],
        {},
        "603.3M.05",
    ),
    (
        "A35",
        "filtro de peneira da linha de ar sujo",
        [(WAP5, 175)],
        [
            ["air line sieve", "peneira da linha de ar", "filtro de peneira"],
            ["clean", "limpar"],
        ],
        {"TP3": 0.3, "H1": 0.2},
        "603.6M.02",
    ),
    (
        "A36",
        "válvulas de dreno automático não drenam",
        [(WAP5, 175)],
        [
            ["automatic drain valves", "dreno automático"],
            ["test", "testar"],
            ["replace", "substituir"],
        ],
        {"H1": 0.3, "Reservoirs": 0.2},
        "603.6M.03",
    ),
    (
        "A37",
        "pressão do reservatório cai com o compressor desligado (válvula de retenção)",
        [(WAP5, 175)],
        [["check valves", "válvulas de retenção"], ["test", "testar"]],
        {"Reservoirs": 0.5, "TP2": 0.2},
        "603.1Y.01",
    ),
    (
        "A38",
        "válvula de segurança do reservatório principal precisa de revisão",
        [(WAP5, 176)],
        [["safety valves", "válvulas de segurança"], ["overhaul", "revis"]],
        {"Reservoirs": 0.3},
        "603.4Y.05",
    ),
    (
        "A39",
        "nível de óleo do compressor auxiliar baixo",
        [(WAP5, 176)],
        [
            ["auxiliary", "auxiliar"],
            ["oil level", "nível de óleo"],
            ["top-up", "completar"],
        ],
        {},
        "604.3M.01",
    ),
    (
        "A40",
        "escovas do motor do compressor auxiliar gastas",
        [(WAP5, 177)],
        [["brushes", "escovas"], ["worn", "gastas", "desgastad"]],
        {},
        "604.1Y.02",
    ),
    (
        "A41",
        "trocar o óleo do compressor CX-500",
        [(CX500, 2)],
        [["400 horas", "400 hours"]],
        {"Oil_temperature": 0.2},
        "CX500-p2",
    ),
    (
        "A42",
        "filtro de ar do compressor CX-500",
        [(CX500, 3)],
        [["semanalmente", "weekly"]],
        {"COMP": 0.2},
        "CX500-p3",
    ),
]

# Failure descriptions the corpus has NO procedure for: correct answer = Limitações.
UNANSWERABLE: list[tuple[str, str, dict[str, float]]] = [
    ("U01", "qual o torque de aperto dos parafusos do cabeçote do compressor?", {}),
    (
        "U02",
        "pressão de abertura exata da válvula de segurança em bar",
        {"Reservoirs": 0.3},
    ),
    (
        "U03",
        "temperatura máxima admissível do óleo do compressor em °C",
        {"Oil_temperature": 0.5},
    ),
    (
        "U04",
        "corrente nominal do motor do compressor em ampères",
        {"Motor_current": 0.4},
    ),
    (
        "U05",
        "procedimento para regenerar o dessecante em campo sem trocar",
        {"Towers": 0.4},
    ),
    ("U06", "qual o part number do kit de vedação do compressor?", {}),
    (
        "U07",
        "intervalo de troca do elemento separador ar-óleo do compressor de parafuso",
        {},
    ),
    ("U08", "como calibrar o sensor de pressão TP2", {"TP2": 0.4}),
    ("U09", "folga de válvulas do compressor alternativo em milímetros", {}),
    ("U10", "procedimento de descarte do óleo usado conforme norma ambiental", {}),
    ("U11", "especificação do fusível do circuito do secador de ar", {"Towers": 0.2}),
    (
        "U12",
        "tempo de ciclo de comutação entre as torres do secador",
        {"Towers": 0.5, "DV_pressure": 0.2},
    ),
]


def build() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for cid, query, refs, facts, sensors, task in ANSWERABLE:
        cases.append(
            {
                "id": cid,
                "equipment_name": EQUIP,
                "symptom_description": query,
                "sensor_contributions": [
                    {"sensor": s, "contribution": c} for s, c in sensors.items()
                ],
                "answerable": True,
                "gold_refs": [{"file_name": f, "page": p} for f, p in refs],
                "gold_facts": facts,
                "source_task": task,
                "validation": "pending",
                "validated_by": None,
                "validation_notes": None,
            }
        )
    for cid, query, sensors in UNANSWERABLE:
        cases.append(
            {
                "id": cid,
                "equipment_name": EQUIP,
                "symptom_description": query,
                "sensor_contributions": [
                    {"sensor": s, "contribution": c} for s, c in sensors.items()
                ],
                "answerable": False,
                "gold_refs": [],
                "gold_facts": [],
                "source_task": None,
                "validation": "pending",
                "validated_by": None,
                "validation_notes": None,
            }
        )
    return {
        "version": "1.0",
        "requirement": "RF-36",
        "corpus": [WAP5 + " (sidecar p.170-177)", CX500, MOTOR, PUMP],
        "annotation": "draft — pending maintenance-specialist validation",
        "n_answerable": len(ANSWERABLE),
        "n_unanswerable": len(UNANSWERABLE),
        "cases": cases,
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("rag_cases_v1.json")
    out.write_text(json.dumps(build(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
