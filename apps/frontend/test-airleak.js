/**
 * Teste de integração: POST /predict/ com dados reais do MetroPT-3.
 *
 * Execução (dentro do container): node test-airleak.js
 * Requer: Node >= 18 (fetch nativo) e stack Docker rodando
 */

const API_URL = "http://api:8000/predict/";

// ── Payloads com os 12 campos exigidos pelo PredictRequest (Pydantic v2) ─────

const CENARIO_NORMAL = {
  label:   "NORMAL (22/04/2020)",
  payload: {
    TP2:             10.1,
    TP3:             10.1,
    H1:               9.2,
    DV_pressure:      2.1,
    Reservoirs:       9.0,
    Motor_current:    3.8,
    Oil_temperature:  64.0,
    COMP:             1.0,
    DV_eletric:       0.0,
    Towers:           1.0,
    MPG:              1.0,
    Oil_level:        1.0,
  },
};

const CENARIO_AIR_LEAK = {
  label:   "AIR LEAK (18/04/2020)",
  payload: {
    TP2:             8.9,
    TP3:             8.7,
    H1:              8.4,
    DV_pressure:     1.8,
    Reservoirs:      8.3,
    Motor_current:   5.8,
    Oil_temperature: 76.5,
    COMP:            1.0,
    DV_eletric:      0.0,
    Towers:          1.0,
    MPG:             1.0,
    Oil_level:       1.0,
  },
};

// ── Helpers ───────────────────────────────────────────────────────────────────

const RESET  = "\x1b[0m";
const BOLD   = "\x1b[1m";
const RED    = "\x1b[31m";
const GREEN  = "\x1b[32m";
const YELLOW = "\x1b[33m";
const CYAN   = "\x1b[36m";
const DIM    = "\x1b[2m";

function riskLevel(prob) {
  if (prob < 0.30) return { label: "LOW",    color: GREEN  };
  if (prob < 0.70) return { label: "MÉDIO",  color: YELLOW };
  return               { label: "CRÍTICO", color: RED    };
}

async function runTest({ label, payload }) {
  let response;
  try {
    response = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
  } catch (err) {
    console.error(`${RED}[ERRO]${RESET} Não foi possível conectar em ${API_URL}`);
    console.error(`${DIM}       Certifique-se de que a stack está no ar: docker compose up${RESET}`);
    process.exit(1);
  }

  if (!response.ok) {
    const body = await response.text();
    console.error(`${RED}[ERRO ${response.status}]${RESET} ${label}`);
    console.error(`${DIM}${body}${RESET}`);
    return;
  }

  const data = await response.json();
  const prob = data.failure_probability;
  const pct  = (prob * 100).toFixed(1);
  const risk = riskLevel(prob);
  const tag  = label.startsWith("NORMAL") ? "TESTE NORMAL  " : "TESTE AIR LEAK";

  console.log(
    `${BOLD}[${tag}]${RESET}` +
    `  Probabilidade de Falha: ${BOLD}${pct}%${RESET}` +
    `  Classe: ${data.predicted_class === 1 ? `${RED}FALHA${RESET}` : `${GREEN}NORMAL${RESET}`}` +
    `  Risco: ${risk.color}${BOLD}${risk.label}${RESET}` +
    `  ${DIM}(${data.timestamp})${RESET}`
  );
}

// ── Execução ──────────────────────────────────────────────────────────────────

(async () => {
  console.log();
  console.log(`${CYAN}${BOLD}══════════════════════════════════════════════════════${RESET}`);
  console.log(`${CYAN}${BOLD}  MetroPT-3 · Teste de Inferência Real via FastAPI${RESET}`);
  console.log(`${CYAN}${BOLD}══════════════════════════════════════════════════════${RESET}`);
  console.log(`${DIM}  Endpoint: ${API_URL}${RESET}`);
  console.log();

  await runTest(CENARIO_NORMAL);
  await runTest(CENARIO_AIR_LEAK);

  console.log();
  console.log(`${CYAN}${BOLD}══════════════════════════════════════════════════════${RESET}`);
  console.log();
})();
