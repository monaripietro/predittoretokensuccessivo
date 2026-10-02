import { defineConfig } from 'vitest/config';

// `vitest run --mode real` attiva i test con libreria e modelli reali (rete necessaria).
export default defineConfig(({ mode }) => ({
  test: {
    include: ['tests/unit/**/*.test.js'],
    environment: 'node',
    env: mode === 'real' ? { RUN_REAL_TESTS: '1' } : {},
  },
}));
