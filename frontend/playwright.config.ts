import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  timeout: 90000,
  reporter: "list",
  use: {
    // HONEST-FALLBACK: Default Playwright base URL when not provided in env
    baseURL: process.env.PLAYWRIGHT_TEST_BASE_URL || "http://127.0.0.1:3000",
    trace: "on-first-retry",
  },
  webServer: [
    {
      command: "node e2e/mockServer.js",
      url: "http://127.0.0.1:8000/health",
      reuseExistingServer: !process.env.CI,
      timeout: 15000,
    },
    {
      command: "node .next/standalone/server.js",
      url: "http://127.0.0.1:3000",
      reuseExistingServer: !process.env.CI,
      timeout: 60000,
      env: {
        PORT: "3000",
        INTERNAL_API_URL: "http://127.0.0.1:8000",
        NEXT_PUBLIC_API_URL: "http://127.0.0.1:8000",
      },
    },
  ],
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
