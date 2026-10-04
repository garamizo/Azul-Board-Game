import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e',
  timeout: 300_000,
  workers: 1,
  retries: 0,
  use: { baseURL: 'http://127.0.0.1:5081', trace: 'retain-on-failure', actionTimeout: 10_000 },
  globalSetup: './e2e/global-setup.ts',
  globalTeardown: './e2e/global-teardown.ts',
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
