import { defineConfig } from '@playwright/test';

/**
 * Configurazione per lo smoke test con modello reale su WebGPU.
 * Richiede Chrome installato e una scheda grafica; non usata in CI.
 */
export default defineConfig({
  testDir: 'tests/e2e-real',
  timeout: 45 * 60_000,
  retries: 0,
  workers: 1,
  reporter: [['list']],
  webServer: {
    command: 'npm run serve:e2e',
    url: 'http://localhost:4173',
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
