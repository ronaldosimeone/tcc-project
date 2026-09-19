"""
Agrega os JSONs de resumo de mutação (um por grupo do matrix de CI) e falha
o build se o mutation score agregado ficar abaixo do piso da RNF-64 (70%).

Fórmula idêntica à usada pelo próprio `mutmut badge` (mutmut/__main__.py):
    tested = total - skipped
    score  = (killed + timeout) / tested * 100
`timeout` conta como morto — o mutante travou/tornou o teste
anormalmente lento, o que também é uma detecção real de comportamento
alterado, não um "não sei".

Uso:
    python scripts/check_mutation_score.py <threshold> <summary1.json> [summary2.json ...]
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Uso: python check_mutation_score.py <threshold> <summary.json> [...]",
            file=sys.stderr,
        )
        sys.exit(1)

    threshold = float(sys.argv[1])
    summary_paths = sys.argv[2:]

    totals = {
        "total": 0,
        "killed": 0,
        "timeout": 0,
        "suspicious": 0,
        "survived": 0,
        "skipped": 0,
    }
    for path in summary_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key in totals:
            totals[key] += data[key]

    tested = totals["total"] - totals["skipped"]
    score = (
        0.0 if tested <= 0 else (totals["killed"] + totals["timeout"]) / tested * 100
    )

    print(f"Agregado de {len(summary_paths)} grupo(s): {totals}")
    print(f"Mutantes testados (total - skipped): {tested}")
    print(f"Mutation score: {score:.2f}%  (piso RNF-64: {threshold:.1f}%)")

    if score < threshold:
        print(
            f"FALHOU: mutation score {score:.2f}% abaixo do piso de {threshold:.1f}% (RNF-64).",
            file=sys.stderr,
        )
        sys.exit(1)

    print("OK: mutation score dentro do piso exigido pela RNF-64.")


if __name__ == "__main__":
    main()
