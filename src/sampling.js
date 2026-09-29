/**
 * Filtri e trasformazioni sulla distribuzione dei logits,
 * identici per mock runtime e runtime reale.
 */

export function validateParams(p = {}) {
  const errors = [];
  const mode = p.mode ?? 'greedy';
  if (mode !== 'greedy' && mode !== 'sample') errors.push('mode deve essere "greedy" o "sample"');
  const temperature = p.temperature ?? 1.0;
  if (!Number.isFinite(temperature) || temperature <= 0 || temperature > 5) errors.push('temperatura deve essere in (0, 5]');
  const topK = p.topK ?? 0;
  if (!Number.isFinite(topK) || topK < 0 || topK > 1000 || !Number.isInteger(topK)) errors.push('top-k deve essere un intero in [0, 1000]');
  const topP = p.topP ?? 1.0;
  if (!Number.isFinite(topP) || topP < 0 || topP > 1) errors.push('top-p deve essere in [0, 1]');
  const minP = p.minP ?? 0;
  if (!Number.isFinite(minP) || minP < 0 || minP > 1) errors.push('min-p deve essere in [0, 1]');
  const repeatPenalty = p.repeatPenalty ?? 1.0;
  if (!Number.isFinite(repeatPenalty) || repeatPenalty < 1 || repeatPenalty > 2) errors.push('repeat penalty deve essere in [1, 2]');
  const topN = p.topN ?? 10;
  if (!Number.isFinite(topN) || topN < 1 || topN > 20 || !Number.isInteger(topN)) errors.push('top-N deve essere un intero in [1, 20]');
  const seed = p.seed ?? 42;
  if (!Number.isFinite(seed) || !Number.isInteger(seed)) errors.push('seed deve essere un intero');
  return { errors, values: { mode, temperature, topK, topP, minP, repeatPenalty, topN, seed } };
}

export function softmax(logits, temperature = 1.0) {
  if (!Number.isFinite(temperature) || temperature <= 0) temperature = 1.0;
  const scaled = logits.map((x) => (Number.isFinite(x) ? x : -Infinity) / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => (x === -Infinity ? 0 : Math.exp(x - max)));
  const sum = exps.reduce((a, b) => a + b, 0);
  if (sum === 0) return logits.map(() => 1 / logits.length);
  return exps.map((e) => e / sum);
}

export function rankTokens(logits, probs) {
  return Array.from(logits.keys())
    .map((id) => ({ tokenId: id, prob: probs[id] }))
    .sort((a, b) => b.prob - a.prob || a.tokenId - b.tokenId);
}

export function applyRepeatPenalty(logits, contextIds, penalty = 1.0) {
  if (penalty === 1.0 || contextIds.length === 0) return logits.slice();
  const seen = new Set(contextIds);
  return logits.map((v, id) => (seen.has(id) ? v / penalty : v));
}

export function applyTopK(ranking, topK = 0) {
  if (!topK || topK <= 0 || topK >= ranking.length) return ranking.slice();
  return ranking.slice(0, topK);
}

export function applyTopP(ranking, topP = 1.0) {
  if (topP >= 1.0) return ranking.slice();
  const out = [];
  let cum = 0;
  for (const c of ranking) {
    out.push(c);
    cum += c.prob;
    if (cum >= topP) break;
  }
  return out;
}

export function applyMinP(ranking, probs, minP = 0) {
  if (!minP || minP <= 0) return ranking.slice();
  const maxProb = Math.max(...ranking.map((c) => probs[c.tokenId]));
  const threshold = minP * maxProb;
  return ranking.filter((c) => probs[c.tokenId] >= threshold);
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function selectToken(ranking, probs, { mode = 'greedy', seed = 42 } = {}) {
  if (ranking.length === 0) return null;
  if (mode === 'greedy') return ranking[0];
  const rand = mulberry32(seed);
  const total = ranking.reduce((a, c) => a + probs[c.tokenId], 0);
  let r = rand() * total;
  for (const c of ranking) {
    r -= probs[c.tokenId];
    if (r <= 0) return c;
  }
  return ranking[ranking.length - 1];
}
