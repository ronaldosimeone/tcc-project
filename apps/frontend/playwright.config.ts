import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright E2E configuration — RNF-16 / RNF-17.
 *
 * Strategy
 * --------
 * The webServer block starts `pnpm dev` automatically before each test run.
 * NEXT_PUBLIC_MSW_ENABLED=true tells MswProvider to activate the MSW Service
 * Worker so fetch calls are intercepted in the browser without a real backend.
 *
 * Run modes:
 *   pnpm e2e           → headless, fastest (CI)
 *   pnpm e2e:headed    → visible browser (debugging)
 *   pnpm e2e:ui        → Playwright UI mode (visual test explorer)
 */
export default defineConfig({
  testDir: "./e2e",
  outputDir: "./e2e-results",

  // Individual test timeout (includes any page.clock.fastForward calls)
  timeout: 30_000,
  // Assertion timeout — how long expect(...).toBeVisible() waits
  expect: { timeout: 10_000 },

  // Tests are not parallelised: each test creates its own browser context
  // with a fresh MSW service worker, so sequential is safer.
  fullyParallel: false,
  workers: 1,

  // Fail fast on CI; allow local retries for flakiness debugging.
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,

  reporter: [
    ["list"],
    ["html", { open: "never", outputFolder: "playwright-report" }],
  ],

  use: {
    baseURL: "http://localhost:3000",
    // Capture artefacts only on failure to keep the workspace lean.
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "off",
    // Locale matching the app (pt-BR date formatting)
    locale: "pt-BR",
  },

  projects: [
    {
      // Chrome is the only target for CI speed; add Firefox/WebKit locally if needed.
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  // ── Auto-start Next.js dev server ────────────────────────────────────────
  webServer: {
    command: "pnpm dev",
    url: "http://localhost:3000",
    // Reuse an already-running dev server locally; always start fresh on CI.
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      // Required by api-client.ts (RNF-13); MSW intercepts this origin.
      NEXT_PUBLIC_API_URL: "http://localhost:8000",
      // Activates MswProvider in the browser — keeps MSW out of production.
      NEXT_PUBLIC_MSW_ENABLED: "true",
    },
  },
});
