/* Capturas do sistema REAL (docker compose + Ollama) — figuras do TCC. */
const path = require("path");
const fs = require("fs");
const { chromium } = require(
  "C:/Users/Raphael Okuyama/Desktop/projeto-tcc/apps/frontend/node_modules/@playwright/test",
);

const OUT = "C:/Users/Raphael Okuyama/Desktop/projeto-tcc/docs/figuras-tcc/app";
fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });
const BASE = "http://localhost";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function setMode(mode) {
  const r = await fetch(`${BASE}/api/simulator/mode`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  console.log("mode", mode, r.status);
}

async function latestProb() {
  const r = await fetch(`${BASE}/api/v1/predictions?page=1&size=1`);
  const j = await r.json();
  return j.items?.[0]?.failure_probability ?? null;
}

async function waitProb(pred, label, maxS = 150) {
  for (let t = 0; t < maxS; t += 3) {
    const p = await latestProb();
    if (p !== null && pred(p)) {
      console.log(label, "prob", p);
      return p;
    }
    await sleep(3000);
  }
  console.log(label, "TIMEOUT prob", await latestProb());
  return null;
}

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
    locale: "pt-BR",
  });
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    const s = document.createElement("style");
    s.textContent = "nextjs-portal,[data-nextjs-toast]{display:none!important}";
    document.addEventListener("DOMContentLoaded", () => document.head.appendChild(s));
  });

  const snap = async (name, full = true) => {
    await page.screenshot({ path: path.join(OUT, name), fullPage: full });
    console.log("ok", name);
  };
  const go = async (url, wait = 5000) => {
    await page.goto(`${BASE}${url}`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(wait);
  };

  // ── FALHA ────────────────────────────────────────────────────────────
  await setMode("FAILURE");
  await waitProb((p) => p > 0.85, "falha");
  await go("/", 7000);
  await snap("01-dashboard-frota-falha.png");
  await go("/sensors/APU-Trem-042", 8000);
  await snap("02-monitor-sensores-falha.png");

  // ── NORMAL ───────────────────────────────────────────────────────────
  await setMode("NORMAL");
  await waitProb((p) => p < 0.2, "normal");
  await go("/", 7000);
  await snap("01-dashboard-frota-normal.png");
  await go("/sensors/APU-Trem-042", 8000);
  await snap("02-monitor-sensores-normal.png");

  // ── DEGRADAÇÃO ───────────────────────────────────────────────────────
  await setMode("DEGRADATION");
  await waitProb((p) => p > 0.3 && p < 0.85, "degradacao", 120);
  await go("/", 7000);
  await snap("01-dashboard-frota-degradacao.png");
  await go("/sensors/APU-Trem-042", 8000);
  await snap("02-monitor-sensores-degradacao.png");

  // ── FALHA de novo: histórico, config, simulação, assistente ──────────
  await setMode("FAILURE");
  await waitProb((p) => p > 0.85, "falha2");

  await go("/history", 5000);
  await snap("03-historico-eventos.png");

  await go("/settings/alerts", 4000);
  await snap("04-configuracao-alertas.png");

  await go("/", 6000);
  await page.getByRole("button", { name: /simulação/i }).first().click();
  await page.waitForTimeout(2000);
  await snap("05-painel-simulacao.png", false);
  await page.keyboard.press("Escape");
  await page.waitForTimeout(600);

  await page.getByRole("button", { name: /assistente de ia/i }).first().click();
  await page.getByRole("heading", { name: "Assistente de Manutenção" }).waitFor();
  await page.waitForTimeout(800);
  await snap("06-assistente-manutencao-formulario.png", false);

  await page.getByLabel("Equipamento").fill("Compressor CX-500");
  await page
    .getByLabel("Sintoma observado")
    .fill("queda de pressão TP2/TP3, aumento da corrente do motor e da temperatura do óleo");
  await page.getByLabel("Probabilidade de falha").fill("0.93");
  await page.getByRole("button", { name: /gerar sugestão/i }).click();
  await page.waitForTimeout(6000);
  await snap("07-assistente-manutencao-gerando.png", false);
  try {
    await page.locator('[data-status="done"]').waitFor({ timeout: 180000 });
  } catch (e) {
    console.log("assistente nao terminou:", e.message.split("\n")[0]);
  }
  await page.waitForTimeout(1500);
  await snap("08-assistente-manutencao-plano.png", false);

  await setMode("NORMAL");
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
