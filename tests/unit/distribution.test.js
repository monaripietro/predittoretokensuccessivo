import { describe, it, expect } from 'vitest';
import {
  scanLogits, modelProbability, topKIndices, rankOf, samplingPool, drawFromPool,
  decideNextToken, formatPercent, validatePolicy, mulberry32,
} from '../../src/core/distribution.js';

/** Softmax di riferimento in doppia precisione, ingenua ma chiara. */
function referenceSoftmax(logits, temperature = 1) {
  const finite = logits.filter(Number.isFinite);
  const m = Math.max(...finite);
  const e = logits.map((x) => (Number.isFinite(x) ? Math.exp((x - m) / temperature) : 0));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

describe('probabilità numericamente stabili', () => {
  it('coincidono con una softmax di riferimento', () => {
    const logits = Float32Array.from([2.5, -1, 0, 7.25, 3, 3]);
    const stats = scanLogits(logits);
    const ref = referenceSoftmax(Array.from(logits));
    for (let i = 0; i < logits.length; i++) {
      expect(modelProbability(logits, i, stats)).toBeCloseTo(ref[i], 12);
    }
  });

  it('non va in overflow con logit enormi né in underflow con logit molto negativi', () => {
    const logits = Float32Array.from([1e30, 1e30 - 1e24, -1e30]);
    const stats = scanLogits(logits);
    const p0 = modelProbability(logits, 0, stats);
    expect(Number.isFinite(p0)).toBe(true);
    expect(p0).toBeGreaterThan(0);
    expect(modelProbability(logits, 2, stats)).toBe(0);
    const big = Float32Array.from([1000, 999, 998]);
    const s2 = scanLogits(big);
    const sum = [0, 1, 2].reduce((a, i) => a + modelProbability(big, i, s2), 0);
    expect(sum).toBeCloseTo(1, 12);
  });

  it('NaN e -∞ sono esclusi e contati; +∞ prende tutta la massa', () => {
    const logits = Float32Array.from([1, NaN, -Infinity, 1]);
    const stats = scanLogits(logits);
    expect(stats.nonFinite).toBe(2);
    expect(modelProbability(logits, 1, stats)).toBe(0);
    expect(modelProbability(logits, 0, stats)).toBeCloseTo(0.5, 12);
    const inf = Float32Array.from([1, Infinity, 3, Infinity]);
    const s2 = scanLogits(inf);
    expect(modelProbability(inf, 1, s2)).toBe(0.5);
    expect(modelProbability(inf, 2, s2)).toBe(0);
  });

  it('rifiuta un vettore senza valori validi', () => {
    expect(() => scanLogits(Float32Array.from([NaN, -Infinity]))).toThrow();
    expect(() => scanLogits(new Float32Array(0))).toThrow();
  });

  it('gestisce un vocabolario reale (262.144 voci) senza errori di stack', () => {
    const n = 262_144;
    const logits = new Float32Array(n);
    for (let i = 0; i < n; i++) logits[i] = Math.sin(i) * 12;
    const d = decideNextToken(logits, { displayK: 8 });
    const total = d.candidates.reduce((s, c) => s + c.modelProb, 0) + d.other.modelProb;
    expect(total).toBeCloseTo(1, 9);
    expect(d.other.count).toBe(n - 8);
  });
});

describe('classifica e parità', () => {
  it('ordina per logit decrescente; a parità vince l\'ID più basso', () => {
    const logits = Float32Array.from([1, 5, 5, 3, NaN, 5]);
    expect(topKIndices(logits, 4)).toEqual([1, 2, 5, 3]);
    expect(rankOf(logits, 5)).toBe(3);
    expect(rankOf(logits, 0)).toBe(5);
    expect(rankOf(logits, 4)).toBe(6);
  });

  it('greedy sceglie il primo della classifica anche con parità', () => {
    const d = decideNextToken(Float32Array.from([0, 4, 4, 1]), { displayK: 3 });
    expect(d.selected.id).toBe(1);
    expect(d.selected.rank).toBe(1);
    expect(d.selected.modelProb).toBeCloseTo(d.candidates[1].modelProb, 15);
  });
});

describe('top-k + "altri token"', () => {
  it('i candidati visibili più "altri" sommano a 1 e non sono rinormalizzati', () => {
    const logits = Float32Array.from([3, 2, 1, 0, -1, -2, -3]);
    const ref = referenceSoftmax(Array.from(logits));
    const d = decideNextToken(logits, { displayK: 3 });
    expect(d.candidates.map((c) => c.id)).toEqual([0, 1, 2]);
    d.candidates.forEach((c) => expect(c.modelProb).toBeCloseTo(ref[c.id], 12));
    const shown = d.candidates.reduce((s, c) => s + c.modelProb, 0);
    expect(shown).toBeLessThan(1);
    expect(d.other.modelProb).toBeCloseTo(1 - shown, 12);
    expect(d.other.count).toBe(4);
  });
});

describe('campionamento', () => {
  it('temperatura → top-k → top-p → rinormalizzazione', () => {
    const logits = Float32Array.from([4, 3, 2, 1, 0]);
    const policy = validatePolicy({ kind: 'sample', temperature: 0.5, topP: 0.9, topK: 3 });
    const { pool, massBeforeRenormalization } = samplingPool(logits, policy);
    const ref = referenceSoftmax(Array.from(logits), 0.5);
    const topKMass = ref[0] + ref[1] + ref[2];
    // top-k=3 limita i candidati; top-p si ferma sulla massa già limitata
    // quando la cumulativa raggiunge 0,9.
    expect(pool.map((c) => c.id)).toEqual([0, 1]);
    expect(massBeforeRenormalization).toBeCloseTo((ref[0] + ref[1]) / topKMass, 12);
    expect(pool[0].prob).toBeCloseTo(ref[0] / (ref[0] + ref[1]), 12);
    expect(pool.reduce((s, c) => s + c.prob, 0)).toBeCloseTo(1, 12);
  });

  it('applica top-p dopo top-k anche quando la temperatura cambia la soglia', () => {
    const logits = Float32Array.from([4, 3, 0]);
    const policy = validatePolicy({ kind: 'sample', temperature: 1, topP: 0.73, topK: 2 });
    const { pool } = samplingPool(logits, policy);
    // Il primo token vale circa 0,731 nella distribuzione top-k, ma solo
    // circa 0,727 prima del filtro: top-p deve fermarsi al primo.
    expect(pool.map((c) => c.id)).toEqual([0]);
  });

  it('l\'estrazione usa le probabilità dell\'insieme ammesso', () => {
    const pool = [{ id: 7, prob: 0.6 }, { id: 3, prob: 0.4 }];
    expect(drawFromPool(pool, 0).id).toBe(7);
    expect(drawFromPool(pool, 0.5999).id).toBe(7);
    expect(drawFromPool(pool, 0.6).id).toBe(3);
    expect(() => drawFromPool(pool, 1)).toThrow();
  });

  it('può scegliere un token che non è il più probabile, e lo dichiara', () => {
    const logits = Float32Array.from([2, 1.9, 0]);
    const d = decideNextToken(logits, {
      policy: { kind: 'sample', temperature: 1, topP: 1, topK: 64 },
      random: () => 0.99,
    });
    expect(d.selected.id).toBe(2);
    expect(d.selected.rank).toBe(3);
    expect(d.selected.draw).toBe(0.99);
    expect(d.selected.policyProb).toBeCloseTo(d.candidates[2].policyProb, 15);
    expect(d.candidates.every((c) => typeof c.policyProb === 'number')).toBe(true);
  });

  it('con lo stesso seme le estrazioni sono riproducibili', () => {
    const logits = Float32Array.from([1, 1, 1, 1]);
    const run = (seed) => {
      const r = mulberry32(seed);
      return Array.from({ length: 10 }, () => decideNextToken(logits, {
        policy: { kind: 'sample', temperature: 1, topP: 1, topK: 4 }, random: r,
      }).selected.id);
    };
    expect(run(42)).toEqual(run(42));
  });

  it('valida i parametri', () => {
    expect(() => validatePolicy({ kind: 'beam' })).toThrow();
    expect(() => validatePolicy({ kind: 'sample', temperature: 0 })).toThrow();
    expect(() => validatePolicy({ kind: 'sample', topP: 0 })).toThrow();
    expect(validatePolicy({ kind: 'greedy', temperature: 9 })).toEqual({ kind: 'greedy' });
  });
});

describe('formattazione', () => {
  it('non mostra mai 0% per valori positivi né 100% per valori < 1', () => {
    expect(formatPercent(0.1234)).toBe('12,3%');
    expect(formatPercent(0.00001)).toBe('<0,1%');
    expect(formatPercent(0.99999)).toBe('>99,9%');
    expect(formatPercent(1)).toBe('100,0%');
    expect(formatPercent(0)).toBe('0%');
    expect(formatPercent(null)).toBe('—');
  });
});
