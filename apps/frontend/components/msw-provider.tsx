"use client";

/**
 * MswProvider — RNF-17.
 *
 * Activates the MSW Service Worker in the browser when
 * NEXT_PUBLIC_MSW_ENABLED='true'.  In production this is a no-op passthrough.
 *
 * Signal for Playwright
 * ---------------------
 * Once the worker is registered and ready, a hidden span with
 * data-testid="msw-ready" is injected into the DOM.  E2E tests wait for this
 * element before interacting with the page, guaranteeing that every fetch
 * call is already intercepted before the first assertion.
 *
 * Why a 'ready' signal?
 * ---------------------
 * Service Worker registration is asynchronous.  Without this signal the first
 * useSensorData tick (fired immediately via void tick() in useEffect) could
 * race against SW registration and hit the real network — which fails because
 * no backend is running during E2E tests.
 */

import { useEffect, useState } from "react";

interface MswProviderProps {
  children: React.ReactNode;
}

const MSW_ENABLED = process.env.NEXT_PUBLIC_MSW_ENABLED === "true";

export function MswProvider({ children }: MswProviderProps) {
  // Starts as `true` in production (MSW never enabled → always ready).
  // Starts as `false` in E2E so children aren't rendered until SW is up.
  const [ready, setReady] = useState(!MSW_ENABLED);

  useEffect(() => {
    if (!MSW_ENABLED) return;

    // Dynamic import keeps the MSW bundle out of the production chunk.
    import("@/mocks/browser")
      .then(({ worker }) =>
        worker.start({
          serviceWorker: { url: "/mockServiceWorker.js" },
          // Silently bypass any request not covered by a handler
          // (e.g. Next.js internal HMR, font requests).
          onUnhandledRequest: "bypass",
        }),
      )
      .then(() => setReady(true))
      .catch((err) => {
        console.error("[MSW] Failed to start service worker:", err);
        // Even on failure, unblock the app so tests can observe the error state.
        setReady(true);
      });
  }, []);

  return (
    <>
      {/* Hidden readiness signal consumed by Playwright's waitForSelector. */}
      {MSW_ENABLED && ready && (
        <span
          data-testid="msw-ready"
          aria-hidden="true"
          style={{ display: "none" }}
        />
      )}
      {/* Only render children once MSW is ready to avoid racing the first fetch. */}
      {ready && children}
    </>
  );
}
