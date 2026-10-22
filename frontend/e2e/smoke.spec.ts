import { test, expect } from "@playwright/test";

test.describe("SteamIQ Navigation & Mode Smoke Tests", () => {
  test.describe("Player Mode", () => {
    test.beforeEach(async ({ page }) => {
      // Set Player mode in localStorage before page scripts execute
      await page.addInitScript(() => {
        window.localStorage.setItem("steamiq_user_mode", "player");
      });
    });

    test("loads player routes: home, catalog, compare, and ask", async ({ page }) => {
      // Home / Recommendations
      await page.goto("/");
      await expect(page).toHaveTitle(/SteamIQ/i);

      // Catalog / Search
      await page.goto("/dashboard");
      await expect(page.locator("body")).toBeVisible();

      // Compare
      await page.goto("/compare");
      await expect(page.locator("body")).toBeVisible();

      // Ask SteamIQ
      await page.goto("/ask");
      await expect(page.locator("body")).toBeVisible();

      // Game Details
      await page.goto("/games/1145360?tab=overview");
      await expect(page.locator("body")).toBeVisible();
    });

    test("enforces DevOnlyGate on developer routes in Player mode", async ({ page }) => {
      const devRoutes = [
        "/market-explorer",
        "/model-lab",
        "/embedding-map",
        "/monitoring",
      ];

      for (const route of devRoutes) {
        await page.goto(route);
        await expect(
          page.getByText("Developer Mode Required")
        ).toBeVisible();
      }
    });
  });

  test.describe("Developer Mode", () => {
    test.beforeEach(async ({ page }) => {
      // Set Developer mode in localStorage before page scripts execute
      await page.addInitScript(() => {
        window.localStorage.setItem("steamiq_user_mode", "developer");
      });
    });

    test("loads developer routes without gating", async ({ page }) => {
      // Market Explorer
      await page.goto("/market-explorer");
      await expect(page.getByText("Developer Mode Required")).not.toBeVisible();

      // Model Lab
      await page.goto("/model-lab");
      await expect(page.getByText("Developer Mode Required")).not.toBeVisible();
      await expect(page.getByText("Model Lab")).toBeVisible();

      // Embedding Map
      await page.goto("/embedding-map");
      await expect(page.getByText("Developer Mode Required")).not.toBeVisible();
      await expect(page.getByText("Item Embedding Map")).toBeVisible();

      // Monitoring
      await page.goto("/monitoring");
      await expect(page.getByText("Developer Mode Required")).not.toBeVisible();
      await expect(page.getByText("Operations & Model Monitoring")).toBeVisible();
    });

    test("monitoring supports ?section= deep linking", async ({ page }) => {
      await page.goto("/monitoring?section=retrain_gate");
      await expect(page.getByText("Retrain Gate Verdict")).toBeVisible();

      await page.goto("/monitoring?section=power");
      await expect(page.getByText("Detector Power")).toBeVisible();
    });
  });
});
