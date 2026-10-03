/**
 * Client della modalità simulata (`?mock`): stessa interfaccia del client
 * del Web Worker, stesso motore e stesso codice di scelta del token, ma con
 * un modello finto. Usato dai test dell'interfaccia; in pagina è sempre
 * segnalato come simulazione.
 */

import { createEngine } from './engine.js';
import { createFakeTf, MOCK_SPEC } from './fakeTf.js';

export function createMockClient({ delayMs = 40, modelOptions = {} } = {}) {
  const tf = createFakeTf({ delayMs, ...modelOptions });
  const engine = createEngine(tf, { device: 'simulazione' });
  const calls = [];
  const handlers = {
    probe: async () => ({ ok: true, shaderF16: true, limits: {}, adapter: { vendor: 'simulazione' }, simulated: true }),
    load: () => engine.load({ spec: MOCK_SPEC, dtype: 'mock' }),
    warmup: () => engine.warmup(),
    tokenize: (p) => engine.tokenize(p),
    begin: (p) => engine.begin(p),
    updateSettings: (p) => engine.updateSettings(p),
    step: (p) => engine.step(p),
    override: (p) => engine.override(p),
    end: () => engine.end(),
    unload: () => engine.unload(),
  };
  return {
    simulated: true,
    calls,
    tf,
    async call(type, payload) {
      calls.push(type);
      const handler = handlers[type];
      if (!handler) throw new Error(`Richiesta sconosciuta: ${type}`);
      return handler(payload ?? {});
    },
    terminate() {},
  };
}
