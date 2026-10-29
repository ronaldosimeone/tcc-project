/* Capturas do sistema REAL (docker compose + Ollama) — figuras do TCC.
 *
 * Uso (na raiz do repositório, com `docker compose up` e o Ollama no ar):
 *   node docs/figuras-tcc/scripts/capturar_telas.js
 *
 * Falha = replay da falha real inédita do MetroPT2 (vazamento de ar, jun/2022)
 * até o modelo ativo entrar em CRÍTICO; normal = modo NORMAL. Recria
 * docs/figuras-tcc/app/ (exceto email/, que vem dos templates de e-mail). */
const path = require("path");
const fs = require("fs");
const { chromium } = require(
  path.resolve(__dirname, "../../../apps/frontend/node_modules/@playwright/test"),
);

const OUT = path.resolve(__dirname, "../app");
const BASE = "http://localhost";
const SCENARIO = "air_leak_mt2_06";

for (const entry of fs.existsSync(OUT) ? fs.readdirSync(OUT) : []) {
  if (entry !== "email") fs.rmSync(path.join(OUT, entry), { recursive: true });
}
for (const dir of ["", "tema", "celular"]) {
  fs.mkdirSync(path.join(OUT, dir), { recursive: true });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const api = async (url, method = "GET", body) => {
  const r = await fetch(`${BASE}/api${url}`, {
    method,
    headers: { "content-type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  return r.json();
};

const latest = async () => {
  const j = await api("/v1/predictions?page=1&size=1");
  return { p: j.items?.[0]?.failure_probability ?? null, thr: j.active_threshold };
};

const waitFor = async (pred, label, maxS = 400) => {
  for (let t = 0; t < maxS; t += 3) {
    const { p, thr } = await latest();
    if (p !== null && thr && pred(p, thr)) {
      console.log(label, "prob", p, "thr", thr);
      return;
    }
    await sleep(3000);
  }
  console.log(label, "TIMEOUT", await latest());
};

const normal = async () => {
  await api("/simulator/mode", "PUT", { mode: "NORMAL" });
  await waitFor((p, thr) => p < thr * 0.3, "normal");
  await sleep(35000); // a mediana de 6 leituras exibida acompanha
};

const failure = async () => {
  await api("/simulator/scenario", "PUT", { id: SCENARIO, lead_minutes: 0 });
  await waitFor((p, thr) => p >= thr, "falha");
  await sleep(35000);
};

// The displayed risk is the median of the last 6 predictions: wait until
// the 6 newest are all >= threshold, so a "failure" shot really shows CRÍTICO.
const critical = async () => {
  for (let t = 0; t < 200; t += 2) {
    const j = await api("/v1/predictions?page=1&size=6");
    const thr = j.active_threshold;
    if (thr && j.items?.length === 6 && j.items.every((i) => i.failure_probability >= thr)) {
      return;
    }
    await sleep(2000);
  }
  console.log("critical TIMEOUT");
};

(async () => {
  const browser = await chromium.launch();
  const context = async (theme, viewport = { width: 1440, height: 900 }, mobile = false) => {
    const ctx = await browser.newContext({
      viewport,
      deviceScaleFactor: 2,
      locale: "pt-BR",
      timezoneId: "America/Sao_Paulo",
      colorScheme: theme,
      isMobile: mobile,
      hasTouch: mobile,
    });
    await ctx.addInitScript((t) => {
      try {
        localStorage.setItem("predictiq-theme", t);
      } catch {}
    }, theme);
    const page = await ctx.newPage();
    const go = async (url, wait = 6000) => {
      await page.goto(`${BASE}${url}`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(wait);
    };
    // Waits until the risk shown on screen is CRÍTICO (>= 65%): the replay
    // is deterministic, so a fixed delay would always land on the same dip.
    const shownCritical = async () => {
      for (let t = 0; t < 40; t++) {
        const pct = await page
          .locator('[title^="Probabilidade bruta"]')
          .first()
          .textContent({ timeout: 2000 })
          .catch(() => null);
        if (pct && parseFloat(pct) >= 65) return;
        await page.waitForTimeout(3000);
      }
      console.log("shownCritical TIMEOUT");
    };
    const snap = async (name, full = false) => {
      await page.screenshot({ path: path.join(OUT, name), fullPage: full });
      console.log("ok", name);
    };
    return { ctx, page, go, snap, shownCritical };
  };

  const light = await context("light");
  const dark = await context("dark");
  const phone = await context("light", { width: 390, height: 844 }, true);

  // ── NORMAL ───────────────────────────────────────────────────────────
  await normal();
  await light.go("/");
  await light.snap("01-dashboard-normal.png");
  await light.go("/sensors/APU-Trem-042", 8000);
  await light.snap("02-sensores-normal.png", true);
  await light.go("/history");
  await light.snap("03-historico.png", true);
  await light.page.getByRole("row", { name: /APU-Trem/ }).first().click();
  await light.page.waitForTimeout(2500);
  await light.snap("03-historico-causa-raiz.png");
  await light.go("/settings/alerts", 4000);
  await light.snap("04-configuracao-alertas.png", true);
  await light.go("/", 5000);
  await light.page.getByRole("button", { name: /simulação/i }).first().click();
  await light.page.waitForTimeout(2000);
  await light.snap("05-painel-simulacao.png");
  await light.page.keyboard.press("Escape");
  await light.go("/login", 3000);
  await light.snap("09-login.png");

  await dark.go("/");
  await dark.snap("tema/dashboard-escuro.png");
  await dark.go("/history");
  await dark.snap("tema/historico-escuro.png");
  await dark.go("/settings/alerts", 4000);
  await dark.snap("tema/configuracoes-escuro.png");

  await phone.go("/");
  await phone.snap("celular/dashboard.png");
  await phone.go("/history");
  await phone.snap("celular/historico.png");

  // ── FALHA REAL (vazamento de ar inédito, MetroPT2) ───────────────────
  await failure();
  await critical();
  await light.go("/");
  await light.shownCritical();
  await light.snap("01-dashboard-falha.png");
  await critical();
  await light.go("/sensors/APU-Trem-042", 8000);
  await light.shownCritical();
  await light.snap("02-sensores-falha.png", true);
  await light.go("/", 5000);
  await light.page.getByRole("button", { name: /simulação/i }).first().click();
  await light.page.waitForTimeout(2500);
  await light.snap("05-painel-simulacao-falha-real.png");
  await light.page.keyboard.press("Escape");

  await critical();
  await dark.go("/");
  await dark.shownCritical();
  await dark.snap("tema/dashboard-falha-escuro.png");
  await critical();
  await dark.go("/sensors/APU-Trem-042", 8000);
  await dark.shownCritical();
  await dark.snap("tema/sensores-falha-escuro.png");
  await critical();
  await phone.go("/sensors/APU-Trem-042", 8000);
  await phone.shownCritical();
  await phone.snap("celular/sensores-falha.png");

  // ── Assistente de manutenção (RAG + Ollama) ──────────────────────────
  const p = light.page;
  await light.go("/", 4000);
  await p.getByRole("button", { name: /assistente de ia/i }).first().click();
  await p.getByRole("heading", { name: "Assistente de Manutenção" }).waitFor();
  await p.waitForTimeout(800);
  await light.snap("06-assistente-formulario.png");
  await p.getByLabel("Sintoma observado").fill("vazamento de ar");
  await p.getByRole("button", { name: /gerar sugestão/i }).click();
  await p.waitForTimeout(5000);
  await light.snap("07-assistente-gerando.png");
  try {
    await p.locator('[data-status="done"]').waitFor({ timeout: 300000 });
  } catch (e) {
    console.log("assistente nao terminou:", e.message.split("\n")[0]);
  }
  await p.waitForTimeout(1500);
  await light.snap("08-assistente-plano.png");
  const panel = p.getByRole("dialog");
  const full = await panel.evaluate((el) => {
    const scroller = [...el.querySelectorAll("*")].find(
      (n) => n.scrollHeight > n.clientHeight + 20 && getComputedStyle(n).overflowY !== "visible",
    );
    return scroller ? scroller.scrollHeight : 0;
  });
  if (full > 0) {
    await p.setViewportSize({ width: 1440, height: Math.min(full + 200, 4000) });
    await p.waitForTimeout(800);
    await light.snap("08-assistente-plano-completo.png");
    await p.setViewportSize({ width: 1440, height: 900 });
  }

  await api("/simulator/mode", "PUT", { mode: "NORMAL" });
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
