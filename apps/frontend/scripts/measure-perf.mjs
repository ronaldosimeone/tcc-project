/**
 * measure-perf.mjs — medição real de FPS/long tasks/render do Dashboard
 * durante o stream SSE (RNF-39). Usa Playwright (já é devDependency do
 * projeto) em vez do Browser pane do agente: o pane reporta
 * `document.visibilityState === "hidden"` mesmo em foco, o que suspende
 * `requestAnimationFrame` por spec do browser e zera qualquer medição de
 * FPS. Uma página Playwright não sofre esse throttling.
 *
 * Uso:
 *   node scripts/measure-perf.mjs [--url http://localhost/] [--seconds 20] [--label baseline]
 *
 * Saída: JSON no stdout com render stats (via window.__renderStats, ver
 * lib/dev-profiler.tsx), FPS real e long tasks, também salvo em
 * ../../frontend_perf_<label>.json na raiz do repo para o relatório.
 */
import { chromium } from "@playwright/test";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const out = { url: "http://localhost/", seconds: 20, label: "run" };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--url") out.url = argv[++i];
    else if (argv[i] === "--seconds") out.seconds = Number(argv[++i]);
    else if (argv[i] === "--label") out.label = argv[++i];
  }
  return out;
}

const { url, seconds, label } = parseArgs(process.argv.slice(2));

const INSTRUMENT = () => {
  window.__renderStats = [];
  window.__fpsFrames = [];
  window.__longTasks = [];
  let last = performance.now();
  function tick(t) {
    window.__fpsFrames.push(t - last);
    last = t;
    window.__rafId = requestAnimationFrame(tick);
  }
  window.__rafId = requestAnimationFrame(tick);
  window.__ltObserver = new PerformanceObserver((list) => {
    for (const e of list.getEntries()) {
      window.__longTasks.push({ start: e.startTime, duration: e.duration });
    }
  });
  window.__ltObserver.observe({ entryTypes: ["longtask"] });
  return { visibilityState: document.visibilityState };
};

const COLLECT = () => {
  cancelAnimationFrame(window.__rafId);
  window.__ltObserver.disconnect();

  const stats = window.__renderStats || [];
  const byId = {};
  for (const s of stats) {
    byId[s.id] ??= { renders: 0, totalMs: 0 };
    byId[s.id].renders += 1;
    byId[s.id].totalMs += s.actualDuration;
  }
  for (const k in byId) {
    byId[k].totalMs = +byId[k].totalMs.toFixed(2);
    byId[k].avgMs = +(byId[k].totalMs / byId[k].renders).toFixed(3);
  }

  const frames = window.__fpsFrames || [];
  // Descarta o primeiro frame (delta desde antes do observer armar — outlier).
  const deltas = frames.slice(1);
  const totalTimeMs = deltas.reduce((a, b) => a + b, 0);
  const fpsAvg = deltas.length > 0 ? deltas.length / (totalTimeMs / 1000) : 0;
  const sorted = [...deltas].sort((a, b) => a - b);
  const p95FrameMs = sorted.length
    ? sorted[Math.floor(sorted.length * 0.95)]
    : 0;
  // "Acima do orçamento" (>16.7ms, 1 vsync perdido a 60Hz) — informativo,
  // esperado que aconteça ocasionalmente mesmo em apps saudáveis (GC, timers).
  const framesOverBudget = deltas.filter((d) => d > 1000 / 60).length;
  // "Dropped frame" real (jank perceptível) = perdeu >=2 vsyncs a 60Hz
  // (~33ms) — heurística padrão (Chrome DevTools usa limiar equivalente).
  const droppedFrames = deltas.filter((d) => d > 2 * (1000 / 60)).length;

  const longTasks = window.__longTasks || [];

  return {
    windowSeconds: +(totalTimeMs / 1000).toFixed(1),
    renderStats: byId,
    totalRenderEvents: stats.length,
    fps: {
      avg: +fpsAvg.toFixed(1),
      frameCount: deltas.length,
      p95FrameMs: +p95FrameMs.toFixed(2),
      framesOverBudget,
      droppedFrames,
    },
    longTasks: {
      count: longTasks.length,
      totalMs: +longTasks.reduce((a, t) => a + t.duration, 0).toFixed(1),
      maxMs: longTasks.length
        ? +Math.max(...longTasks.map((t) => t.duration)).toFixed(1)
        : 0,
    },
  };
};

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "load" });
  // Aquece: deixa o seed inicial (fetch histórico) e a 1ª conexão SSE
  // estabilizarem antes de armar os observers.
  await page.waitForTimeout(2000);

  const armed = await page.evaluate(INSTRUMENT);
  console.error(`[measure-perf] observers armed | ${JSON.stringify(armed)}`);

  await page.waitForTimeout(seconds * 1000);

  const result = await page.evaluate(COLLECT);
  await browser.close();

  const payload = { label, url, requestedSeconds: seconds, ...result };
  const outPath = resolve(
    __dirname,
    "..",
    "..",
    "..",
    `frontend_perf_${label}.json`,
  );
  writeFileSync(outPath, JSON.stringify(payload, null, 2), "utf-8");

  console.log(JSON.stringify(payload, null, 2));
  console.error(`[measure-perf] saved -> ${outPath}`);
}

main().catch((e) => {
  console.error("[measure-perf] FAILED:", e);
  process.exit(1);
});
