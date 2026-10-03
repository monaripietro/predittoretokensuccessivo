/**
 * Distribuzione del prossimo token a partire dai logit reali del modello.
 *
 * Tutte le funzioni sono pure e lavorano su un array di punteggi (logit)
 * grande quanto il vocabolario. Nessun array di probabilità per l'intero
 * vocabolario viene conservato: per ogni passo resta solo un riepilogo
 * compatto (candidati principali + massa residua), verificabile.
 *
 * Definizioni usate nell'interfaccia:
 * - probabilità del modello: softmax dei logit grezzi (temperatura 1) su
 *   tutto il vocabolario, prima di qualunque regola di scelta;
 * - probabilità di estrazione (solo in modalità campionamento): la
 *   distribuzione effettivamente usata per estrarre il token, dopo
 *   temperatura, top-k e top-p e la rinormalizzazione sull'insieme ammesso.
 */

export const DEFAULT_POLICY = Object.freeze({ kind: 'greedy' });

/** Limite fisso dei candidati ammessi all'estrazione in modalità campionamento. */
export const SAMPLING_TOP_K = 64;

export function validatePolicy(policy = DEFAULT_POLICY) {
  if (!policy || (policy.kind !== 'greedy' && policy.kind !== 'sample')) {
    throw new Error('Regola di scelta non valida: usa "greedy" oppure "sample".');
  }
  if (policy.kind === 'greedy') return { kind: 'greedy' };
  const temperature = Number(policy.temperature ?? 1);
  const topP = Number(policy.topP ?? 1);
  const topK = Number(policy.topK ?? SAMPLING_TOP_K);
  if (!Number.isFinite(temperature) || temperature < 0.1 || temperature > 2) {
    throw new Error('La temperatura deve essere tra 0,1 e 2.');
  }
  if (!Number.isFinite(topP) || topP < 0.1 || topP > 1) {
    throw new Error('Top-p deve essere tra 0,1 e 1.');
  }
  if (!Number.isInteger(topK) || topK < 1 || topK > 1000) {
    throw new Error('Top-k deve essere un intero tra 1 e 1000.');
  }
  return { kind: 'sample', temperature, topP, topK };
}

/**
 * Generatore pseudo-casuale riproducibile (mulberry32). Il seme viene
 * registrato nella traccia così che ogni estrazione sia verificabile.
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Ordine totale e deterministico: logit più alto prima, a parità l'ID più basso. */
function before(scoreA, idA, scoreB, idB) {
  return scoreA > scoreB || (scoreA === scoreB && idA < idB);
}

/**
 * Statistiche numericamente stabili sull'intero vocabolario.
 * NaN e -Infinity sono trattati come token esclusi (probabilità 0).
 * Se compaiono valori +Infinity, la massa si divide in parti uguali tra loro.
 */
export function scanLogits(logits) {
  const n = logits.length;
  if (n === 0) throw new Error('Il modello ha restituito un vettore di punteggi vuoto.');
  let max = -Infinity;
  let nonFinite = 0;
  let posInf = 0;
  for (let i = 0; i < n; i++) {
    const v = logits[i];
    if (v === Infinity) posInf++;
    else if (!Number.isFinite(v)) nonFinite++;
    else if (v > max) max = v;
  }
  if (posInf > 0) {
    return { n, max: Infinity, sumExp: posInf, nonFinite, posInf };
  }
  if (max === -Infinity) {
    throw new Error('Tutti i punteggi del modello sono non validi (NaN o -∞).');
  }
  let sumExp = 0;
  for (let i = 0; i < n; i++) {
    const v = logits[i];
    if (Number.isFinite(v)) sumExp += Math.exp(v - max);
  }
  return { n, max, sumExp, nonFinite, posInf: 0 };
}

/** Probabilità del modello (temperatura 1, vocabolario intero) per un singolo ID. */
export function modelProbability(logits, id, stats) {
  const v = logits[id];
  if (stats.posInf > 0) return v === Infinity ? 1 / stats.posInf : 0;
  if (!Number.isFinite(v)) return 0;
  return Math.exp(v - stats.max) / stats.sumExp;
}

/**
 * I k token con punteggio più alto, in ordine (logit decrescente, ID crescente).
 * Selezione parziale O(V·log k) senza ordinare tutto il vocabolario.
 */
export function topKIndices(logits, k) {
  const n = logits.length;
  k = Math.max(0, Math.min(k, n));
  const ids = [];
  const scores = [];
  for (let i = 0; i < n; i++) {
    let v = logits[i];
    if (Number.isNaN(v)) v = -Infinity;
    if (ids.length === k) {
      const lastScore = scores[k - 1];
      if (!before(v, i, lastScore, ids[k - 1])) continue;
      ids.pop();
      scores.pop();
    }
    // inserimento binario
    let lo = 0;
    let hi = ids.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (before(scores[mid], ids[mid], v, i)) lo = mid + 1;
      else hi = mid;
    }
    ids.splice(lo, 0, i);
    scores.splice(lo, 0, v);
  }
  return ids;
}

/** Posizione (1 = primo) di un token nella classifica completa del vocabolario. */
export function rankOf(logits, id) {
  let v = logits[id];
  if (Number.isNaN(v)) v = -Infinity;
  let rank = 1;
  for (let i = 0; i < logits.length; i++) {
    if (i === id) continue;
    let w = logits[i];
    if (Number.isNaN(w)) w = -Infinity;
    if (before(w, i, v, id)) rank++;
  }
  return rank;
}

/**
 * Insieme ammesso all'estrazione e relative probabilità rinormalizzate.
 * Ordine applicato: temperatura → top-k → top-p → rinormalizzazione.
 */
export function samplingPool(logits, policy, stats = scanLogits(logits)) {
  const { temperature, topP, topK } = policy;
  const ordered = topKIndices(logits, topK).filter((id) => logits[id] === Infinity || Number.isFinite(logits[id]));
  if (ordered.length === 0) throw new Error('Nessun token ammesso all\'estrazione.');
  const maxScaled = stats.posInf > 0 ? 0 : stats.max / temperature;
  const scaled = ordered.map((id) => {
    if (stats.posInf > 0) return logits[id] === Infinity ? 1 : 0;
    return Math.exp(logits[id] / temperature - maxScaled);
  });
  // Massa dopo la temperatura sull'intero vocabolario (serve per top-p).
  let totalScaled = 0;
  if (stats.posInf > 0) totalScaled = stats.posInf;
  else {
    for (let i = 0; i < logits.length; i++) {
      const v = logits[i];
      if (Number.isFinite(v)) totalScaled += Math.exp(v / temperature - maxScaled);
    }
  }
  const kept = [];
  let cumulative = 0;
  for (let j = 0; j < ordered.length; j++) {
    kept.push(j);
    cumulative += scaled[j] / totalScaled;
    if (cumulative >= topP) break;
  }
  const keptMass = kept.reduce((s, j) => s + scaled[j], 0);
  const pool = kept.map((j) => ({ id: ordered[j], prob: scaled[j] / keptMass }));
  return { pool, massBeforeRenormalization: keptMass / totalScaled };
}

/** Estrazione pesata riproducibile da un insieme già ordinato. */
export function drawFromPool(pool, u) {
  if (!(u >= 0 && u < 1)) throw new Error('Numero casuale fuori intervallo [0, 1).');
  let acc = 0;
  for (const c of pool) {
    acc += c.prob;
    if (u < acc) return c;
  }
  return pool[pool.length - 1];
}

/**
 * Riepilogo compatto di un passo di generazione e scelta del token.
 *
 * @param {Float32Array|number[]} logits punteggi reali dell'ultima posizione
 * @param {object} options
 * @param {number} options.displayK quanti candidati mostrare
 * @param {object} options.policy regola di scelta (greedy | sample)
 * @param {() => number} [options.random] generatore in [0,1) (solo sample)
 */
export function decideNextToken(logits, { displayK = 10, policy = DEFAULT_POLICY, random } = {}) {
  policy = validatePolicy(policy);
  const stats = scanLogits(logits);
  const topIds = topKIndices(logits, displayK);
  const candidates = topIds.map((id, i) => ({
    id,
    logit: Number.isFinite(logits[id]) ? logits[id] : null,
    modelProb: modelProbability(logits, id, stats),
    rank: i + 1,
  }));

  let selectedId;
  let draw = null;
  let poolInfo = null;
  if (policy.kind === 'greedy') {
    selectedId = topIds[0];
  } else {
    if (typeof random !== 'function') throw new Error('Il campionamento richiede un generatore casuale.');
    const { pool, massBeforeRenormalization } = samplingPool(logits, policy, stats);
    const u = random();
    const picked = drawFromPool(pool, u);
    selectedId = picked.id;
    draw = u;
    const byId = new Map(pool.map((c) => [c.id, c.prob]));
    for (const c of candidates) c.policyProb = byId.get(c.id) ?? 0;
    poolInfo = { size: pool.length, massBeforeRenormalization, selectedPolicyProb: picked.prob };
  }

  const shownMass = candidates.reduce((s, c) => s + c.modelProb, 0);
  const selectedRank = topIds.indexOf(selectedId) >= 0 ? topIds.indexOf(selectedId) + 1 : rankOf(logits, selectedId);
  return {
    vocabSize: logits.length,
    invalidScores: stats.nonFinite,
    candidates,
    other: {
      count: logits.length - candidates.length,
      modelProb: Math.max(0, 1 - shownMass),
    },
    policy,
    pool: poolInfo,
    selected: {
      id: selectedId,
      modelProb: modelProbability(logits, selectedId, stats),
      policyProb: poolInfo ? poolInfo.selectedPolicyProb : null,
      rank: selectedRank,
      draw,
    },
  };
}

/**
 * Formattazione percentuali in italiano. Mai "0%" per un valore positivo,
 * mai "100%" per un valore minore di 1.
 */
export function formatPercent(p, digits = 1) {
  if (p === null || p === undefined || Number.isNaN(p)) return '—';
  if (p <= 0) return '0%';
  const pct = p * 100;
  const min = 10 ** -digits;
  if (pct < min) return `<${min.toFixed(digits).replace('.', ',')}%`;
  if (p < 1 && pct > 100 - min) return `>${(100 - min).toFixed(digits).replace('.', ',')}%`;
  return `${pct.toFixed(digits).replace('.', ',')}%`;
}
