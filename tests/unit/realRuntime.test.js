import { describe, it, expect } from 'vitest';
import { RealRuntime } from '../../src/realRuntime.js';

/**
 * Test di integrazione con il modello reale (Xenova/distilgpt2, ~35MB).
 * Skippato se la rete non è disponibile: la CI usa il mock.
 */
const maybe = process.env.RUN_REAL_TESTS === '1' ? it : it.skip;

describe('RealRuntime (distilgpt2) — opzionale', () => {
  maybe('naive e KV cache producono lo stesso risultato', { timeout: 120_000 }, async () => {
    const rt = new RealRuntime();
    await rt.init();
    const ids = rt.encode('Il cielo è blu');
    expect(ids.length).toBeGreaterThan(0);
    const greedy = (logits) => logits.reduce((b, v, i) => (v > logits[b] ? i : b), 0);

    let ctxN = ids.slice();
    const naive = [];
    for (let i = 0; i < 3; i++) {
      const r = await rt.nextLogits({ ids: ctxN, cache: null });
      const b = greedy(r.logits);
      naive.push(b);
      ctxN = ctxN.concat([b]);
    }

    let ctxC = ids.slice();
    let cache = null;
    const cached = [];
    for (let i = 0; i < 3; i++) {
      const r = await rt.nextLogits({ ids: ctxC, cache });
      const b = greedy(r.logits);
      cached.push(b);
      cache = r.cache;
      ctxC = ctxC.concat([b]);
    }

    expect(cached).toEqual(naive);
  });

  maybe('con la cache il passo successivo calcola 1 solo token', { timeout: 120_000 }, async () => {
    const rt = new RealRuntime();
    await rt.init();
    const ids = rt.encode('a b c');
    const r1 = await rt.nextLogits({ ids });
    expect(r1.newTokens).toBe(ids.length);
    const r2 = await rt.nextLogits({ ids: ids.concat([123]), cache: r1.cache });
    expect(r2.newTokens).toBe(1);
    expect(r2.contextLength).toBe(ids.length + 1);
  });
});
