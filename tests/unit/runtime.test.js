import { describe, it, expect, beforeEach } from 'vitest';
import { MockRuntime, computeStep } from '../../src/mockRuntime.js';

const baseParams = {
  mode: 'greedy', temperature: 1.0, topK: 0, topP: 1.0,
  minP: 0, repeatPenalty: 1.0, topN: 10, seed: 42,
};

describe('MockRuntime — naive vs KV cache', () => {
  let rt;
  beforeEach(() => {
    rt = new MockRuntime();
    return rt.init();
  });

  it('prefill inizializza la cache con l\'intero prompt', async () => {
    const ids = [10, 20, 30];
    const r = await rt.nextLogits({ ids, cache: null });
    expect(r.cache.ids).toEqual(ids);
    expect(r.newTokens).toBe(3);
    expect(r.contextLength).toBe(3);
  });

  it('contesto logico identico in entrambe le modalità', async () => {
    const ids = [10, 20, 30, 40];
    const naive = await rt.nextLogits({ ids, cache: null });
    const pre = await rt.nextLogits({ ids: ids.slice(0, 3), cache: null });
    const cached = await rt.nextLogits({ ids, cache: pre.cache });
    expect(cached.contextLength).toBe(naive.contextLength);
    expect(cached.logits).toEqual(naive.logits);
  });

  it('con la cache calcola solo i token nuovi', async () => {
    const pre = await rt.nextLogits({ ids: [10, 20, 30], cache: null });
    const step = await rt.nextLogits({ ids: [10, 20, 30, 40], cache: pre.cache });
    expect(step.newTokens).toBe(1);
  });

  it('un contesto diverso invalida la cache', async () => {
    const pre = await rt.nextLogits({ ids: [10, 20], cache: null });
    const step = await rt.nextLogits({ ids: [99, 20, 40], cache: pre.cache });
    expect(step.newTokens).toBe(3);
  });

  it('computeStep produce ranking valido e token scelto coerente', async () => {
    const r = await computeStep(rt, { contextIds: [10, 20, 30], params: baseParams, cache: null });
    expect(r.ranking.length).toBeLessThanOrEqual(10);
    expect(r.ranking[0].prob).toBeGreaterThan(0);
    expect(r.chosen.tokenId).toBe(r.ranking[0].tokenId);
    for (let i = 1; i < r.ranking.length; i++) {
      expect(r.ranking[i - 1].prob).toBeGreaterThanOrEqual(r.ranking[i].prob);
    }
  });

  it('reset azzera la cache di prefill', async () => {
    await rt.nextLogits({ ids: [1, 2, 3], cache: null });
    rt.reset();
    expect(rt.prefillCount).toBe(0);
  });

  it('rifiuta sequenze vuote', async () => {
    await expect(rt.nextLogits({ ids: [], cache: null })).rejects.toThrow();
  });
});
