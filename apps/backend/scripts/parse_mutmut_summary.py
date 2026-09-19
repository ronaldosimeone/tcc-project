"""
Extrai o resumo final de uma execução de `mutmut run --CI` (RNF-64).

`mutmut run` não expõe um comando "export stats" nesta versão (2.5.1) — o
único lugar com killed/timeout/suspicious/survived/skipped já agregados é a
própria linha de progresso final impressa no terminal (formato
`N/TOTAL  killed  timeout  suspicious  survived  skipped`, com emojis
como marcador de cada campo). Este script lê esse log (stdout+stderr de
`mutmut run`, com os caracteres de carriage-return do spinner) e escreve um
JSON com os 6 números — usado pelo job de agregação do score de mutação no
CI (ver .github/workflows/ci.yml).

Uso:
    python scripts/parse_mutmut_summary.py <log_file> <output_json>
"""

from __future__ import annotations

import json
import re
import sys

# Emojis exatos usados pelo mutmut 2.5.1 na linha de progresso, nesta ordem:
# killed(🎉) timeout(⏰) suspicious(🤔) survived(🙁) skipped(🔇)
_LINE_RE = re.compile(
    r"(\d+)/(\d+)\s+\U0001f389\s+(\d+)\s+⏰\s+(\d+)\s+\U0001f914\s+(\d+)"
    r"\s+\U0001f641\s+(\d+)\s+\U0001f507\s+(\d+)"
)


def parse_summary(log_text: str) -> dict[str, int]:
    matches = list(_LINE_RE.finditer(log_text))
    if not matches:
        raise ValueError(
            "Nenhuma linha de progresso do mutmut encontrada no log — "
            "a execução pode ter falhado antes de rodar qualquer mutante."
        )
    last = matches[-1]
    _done, total, killed, timeout, suspicious, survived, skipped = (
        int(g) for g in last.groups()
    )
    return {
        "total": total,
        "killed": killed,
        "timeout": timeout,
        "suspicious": suspicious,
        "survived": survived,
        "skipped": skipped,
    }


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Uso: python parse_mutmut_summary.py <log_file> <output_json>",
            file=sys.stderr,
        )
        sys.exit(1)

    log_path, output_path = sys.argv[1], sys.argv[2]
    with open(log_path, encoding="utf-8", errors="replace") as f:
        log_text = f.read()

    summary = parse_summary(log_text)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    tested = summary["total"] - summary["skipped"]
    score = (
        0.0 if tested <= 0 else (summary["killed"] + summary["timeout"]) / tested * 100
    )
    print(f"Mutation summary: {summary}")
    print(f"Mutation score: {score:.2f}%")


if __name__ == "__main__":
    main()
