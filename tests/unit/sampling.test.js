import { describe, it, expect } from 'vitest';
import {
  softmax, rankTokens, applyTopK, applyTopP, applyMinP,
  applyRepeatPenalty, selectToken, validateParams, mulberry32,
} from '../../src/sampling.js';

describe('softmax', () => {
  it('somma a 1 e produce probabilità finite in [0,1]', () => {
    const p = softmax([2, 1, 0, -1]);
    const sum = p.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(1e-9);
    for (const v of p) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('è monotona rispetto ai logits', () => {
    const p = softmax([3, 2, 1]);
    expect(p[0]).toBeGreaterThan(p[1]);
    expect(p[1]).toBeGreaterThan(p[2]);
  });

  it('la temperatura alta appiattisce la distribuzione', () => {
    const cold = softmax([3, 0], 0.5);
    const hot = softmax([3, 0], 5);
    expect(hot[0] - hot[1]).toBeLessThan(cold[0] - cold[1]);
  });
});

describe('ranking', () => {
  it('ordina in modo decrescente per probabilità', () => {
    const logits = [0, 3, 1, 2];
    const probs = softmax(logits);
    const r = rankTokens(logits, probs);
    expect(r.map((c) => c.tokenId)).toEqual([1, 3, 2, 0]);
    expect(r[0].prob).toBeGreaterThanOrEqual(r[1].prob);
  });
});

describe('top-k', () => {
  it('tiene solo i k migliori', () => {
    const r = [0, 1, 2, 3, 4].map((id) => ({ tokenId: id, prob: 1 - id * 0.1 }));
    expect(applyTopK(r, 2).length).toBe(2);
    expect(applyTopK(r, 2)[0].tokenId).toBe(0);
  });
  it('con 0 o k >= lunghezza non filtra nulla', () => {
    const r = [1, 2, 3].map((id) => ({ tokenId: id, prob: 0.1 }));
    expect(applyTopK(r, 0).length).toBe(3);
    expect(applyTopK(r, 10).length).toBe(3);
  });
});

describe('top-p', () => {
  it('ferma la massa cumulativa a p', () => {
    const r = [
      { tokenId: 0, prob: 0.5 },
      { tokenId: 1, prob: 0.3 },
      { tokenId: 2, prob: 0.2 },
    ];
    const out = applyTopP(r, 0.8);
    expect(out.length).toBe(2);
  });
  it('con 1.0 non filtra', () => {
    const r = [{ tokenId: 0, prob: 0.5 }, { tokenId: 1, prob: 0.5 }];
    expect(applyTopP(r, 1.0).length).toBe(2);
  });
});

describe('min-p', () => {
  it('rimuove i candidati sotto la soglia relativa', () => {
    const probs = { 0: 0.8, 1: 0.15, 2: 0.05 };
    const r = [0, 1, 2].map((id) => ({ tokenId: id, prob: probs[id] }));
    expect(applyMinP(r, probs, 0.1).map((c) => c.tokenId)).toEqual([0, 1]);
  });
  it('con 0 non filtra', () => {
    const probs = { 0: 0.6, 1: 0.4 };
    const r = [0, 1].map((id) => ({ tokenId: id, prob: probs[id] }));
    expect(applyMinP(r, probs, 0).length).toBe(2);
  });
});

describe('repeat penalty', () => {
  it('penalizza i token già presenti nel contesto', () => {
    const logits = [5, 5, 5];
    const out = applyRepeatPenalty(logits, [0], 2.0);
    expect(out[0]).toBe(2.5);
    expect(out[1]).toBe(5);
  });
  it('con 1.0 non cambia nulla', () => {
    expect(applyRepeatPenalty([1, 2], [0], 1.0)).toEqual([1, 2]);
  });
});

describe('greedy e sampling', () => {
  const probs = { 0: 0.7, 1: 0.2, 2: 0.1 };
  const ranking = [0, 1, 2].map((id) => ({ tokenId: id, prob: probs[id] }));

  it('greedy sceglie il massimo', () => {
    expect(selectToken(ranking, probs, { mode: 'greedy' }).tokenId).toBe(0);
  });

  it('sampling è deterministico con lo stesso seed', () => {
    const a = selectToken(ranking, probs, { mode: 'sample', seed: 7 });
    const b = selectToken(ranking, probs, { mode: 'sample', seed: 7 });
    expect(a).toEqual(b);
  });

  it('sampling con seed diversi può cambiare risultato', () => {
    const seeds = new Set();
    for (let s = 0; s < 50; s++) {
      seeds.add(selectToken(ranking, probs, { mode: 'sample', seed: s }).tokenId);
    }
    expect(seeds.size).toBeGreaterThan(1);
  });

  it('il PRNG è riproducibile', () => {
    const r1 = mulberry32(3);
    const r2 = mulberry32(3);
    expect(r1()).toBe(r2());
  });
});

describe('validazione parametri', () => {
  it('accetta i valori consigliati', () => {
    const { errors } = validateParams({
      mode: 'greedy', temperature: 1.0, topK: 0, topP: 1.0,
      minP: 0, repeatPenalty: 1.0, topN: 10, seed: 42,
    });
    expect(errors).toEqual([]);
  });

  it('rifiuta valori fuori range', () => {
    expect(validateParams({ temperature: 0 }).errors.length).toBeGreaterThan(0);
    expect(validateParams({ temperature: -1 }).errors.length).toBeGreaterThan(0);
    expect(validateParams({ topP: 1.5 }).errors.length).toBeGreaterThan(0);
    expect(validateParams({ minP: -0.1 }).errors.length).toBeGreaterThan(0);
    expect(validateParams({ topK: 2.5 }).errors.length).toBeGreaterThan(0);
    expect(validateParams({ repeatPenalty: 0.5 }).errors.length).toBeGreaterThan(0);
    expect(validateParams({ topN: 0 }).errors.length).toBeGreaterThan(0);
    expect(validateParams({ mode: 'xyz' }).errors.length).toBeGreaterThan(0);
    expect(validateParams({ seed: 1.5 }).errors.length).toBeGreaterThan(0);
  });
});

describe('vocabolari grandi (regressione call stack)', () => {
  it('softmax gestisce 151.936 logits senza Maximum call stack size exceeded', () => {
    const n = 151_936;
    const logits = new Array(n);
    for (let i = 0; i < n; i++) logits[i] = Math.sin(i) * 10;
    const probs = softmax(logits, 1.0);
    expect(probs.length).toBe(n);
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 6);
  });

  it('applyMinP gestisce un ranking della dimensione del vocabolario', () => {
    const n = 151_936;
    const ranking = Array.from({ length: n }, (_, id) => ({ tokenId: id, prob: 1 / n }));
    const probs = new Array(n).fill(1 / n);
    const out = applyMinP(ranking, probs, 0.5);
    expect(out.length).toBe(n);
  });
});
