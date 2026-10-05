import { defineConfig, devices } from "@playwright/test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const PORT = 4173;
const API_PORT = 8790;
const API_URL = `http://127.0.0.1:${API_PORT}/api/v1`;

/*
  Browser flows. The UI dev server proxies /api to a real product API (`python -m api.product`)
  started here on a fresh data directory, so live-mode flows read and write real backend state;
  mock-mode flows opt in with ?api=mock and never reach it.

  Uses the installed Chrome by default so no browser download is needed; set PW_CHANNEL=chromium
  to use Playwright's bundled build instead. WYNK_PYTHON picks the interpreter (default: python),
  which must have the project installed (`pip install -e .`), run from the repository root.
*/
// set once in the runner process; workers inherit it instead of creating their own
process.env.WYNK_E2E_DATA ??= mkdtempSync(join(tmpdir(), "wynk-e2e-"));
const PYTHON = process.env.WYNK_PYTHON ?? "python";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    viewport: { width: 1366, height: 820 },
  },
  projects: [{ name: "chrome", use: { ...devices["Desktop Chrome"], channel: process.env.PW_CHANNEL ?? "chrome" } }],
  webServer: [
    {
      // 1 MiB upload limit, so the size-limit error is reachable from a browser test
      command: `"${PYTHON}" -m api.product --data-dir "${process.env.WYNK_E2E_DATA}" --port ${API_PORT} --max-upload-mb 1`,
      cwd: "..",
      url: `${API_URL}/projects`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: `npx vite --port ${PORT} --strictPort --host 127.0.0.1`,
      url: `http://127.0.0.1:${PORT}`,
      env: { WYNK_API_PROXY: `http://127.0.0.1:${API_PORT}` },
      reuseExistingServer: false,
      timeout: 180_000,
    },
  ],
});
