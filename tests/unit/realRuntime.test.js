import { describe, it, expect } from 'vitest';
import { RealRuntime, MODEL_CONFIG, isMobileLike } from '../../src/realRuntime.js';

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

describe('selezione modello per dispositivo', () => {
  it('isMobileLike è false in Node (nessun window)', () => {
    expect(isMobileLike()).toBe(false);
  });

  it('MODEL_CONFIG dichiara il profilo tiny per mobile', () => {
    expect(MODEL_CONFIG.tiny.id).toBe('onnx-community/SmolLM2-135M-Instruct-ONNX');
    expect(MODEL_CONFIG.tiny.vocabSize).toBeGreaterThan(0);
    expect([MODEL_CONFIG.id, MODEL_CONFIG.fallback.id, MODEL_CONFIG.tiny.id]).toContain(MODEL_CONFIG.fallback.id);
  });

  it('describe() risolve il profilo del modello caricato', async () => {
    const rt = new RealRuntime('onnx-community/SmolLM2-135M-Instruct-ONNX');
    rt.modelId = MODEL_CONFIG.tiny.id;
    rt.backend = 'WASM';
    rt.modelInfo = `${MODEL_CONFIG.tiny.id} (q4)`;
    rt.dtype = 'q4';
    const d = rt.describe();
    expect(d.modelId).toBe(MODEL_CONFIG.tiny.id);
    expect(d.tokenizerName).toBe(MODEL_CONFIG.tiny.tokenizerName);
    expect(d.contextLimit).toBe(MODEL_CONFIG.tiny.contextLimit);
  });

  it('con il 135M il runtime produce logits coerenti (rete richiesta)', { timeout: 180_000 }, async () => {
    const rt = new RealRuntime('onnx-community/SmolLM2-135M-Instruct-ONNX');
    const info = await rt.init();
    expect(info.modelId).toBe(MODEL_CONFIG.tiny.id);
    const ids = rt.encode('Il cielo è');
    expect(ids.length).toBeGreaterThan(0);
    const r1 = await rt.nextLogits({ ids, cache: null });
    expect(r1.logits.length).toBe(MODEL_CONFIG.tiny.vocabSize);
    const r2 = await rt.nextLogits({ ids: [r1.logits.indexOf(Math.max(...r1.logits))], cache: r1.cache });
    expect(r2.newTokens).toBe(1);
  });
});
