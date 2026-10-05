import { defineConfig, devices } from "@playwright/test";

const PORT = 4173;

/*
  Browser flows against the mock adapter. Uses the installed Chrome by default so no browser
  download is needed; set PW_CHANNEL=chromium to use Playwright's bundled build instead.
*/
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
  webServer: {
    command: `npx vite --port ${PORT} --strictPort --host 127.0.0.1`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
