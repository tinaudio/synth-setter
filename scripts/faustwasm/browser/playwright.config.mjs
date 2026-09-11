import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./test",
  testMatch: "*.e2e.mjs",
  timeout: 60_000,
  use: {
    browserName: "chromium",
    headless: true,
  },
  workers: 1,
});
