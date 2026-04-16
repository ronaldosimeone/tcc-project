/**
 * MSW browser worker — RNF-17.
 *
 * Exported as a lazy-loaded module from MswProvider so the Service Worker
 * setup code is only included in the client bundle when
 * NEXT_PUBLIC_MSW_ENABLED is 'true'.  Production builds never import this.
 */

import { setupWorker } from "msw/browser";
import { handlers } from "./handlers";

export const worker = setupWorker(...handlers);
