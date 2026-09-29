/**
 * RuntimeInterface (documentazione del contratto richiesto dalla specifica):
 *
 * async init()                -> { backend, modelInfo }
 * encode(text)                -> [tokenIds]
 * decode(ids)                 -> testo
 * async nextLogits({ ids, cache }) -> { logits, cache, newTokens }
 *   - cache: opaco; se supportato permette il riuso degli stati precedenti,
 *     altrimenti il runtime ricalcola l'intera sequenza (modalità naive).
 * eosId                       -> token id del terminatore
 */

import { MockTokenizer, VOCAB_SIZE } from './mockTokenizer.js';
import {
  applyRepeatPenalty, softmax, rankTokens, applyTopK,
  applyTopP, applyMinP, selectToken,
} from './sampling.js';

/**
 * Logits deterministici: dipendono dall'intero contesto (per questo la
 * naive e la cache producono lo stesso risultato) ma sono calcolati con
 * costo O(1) per token, così la "cache" del mock salva davvero i passi
 * già calcolati e i test possono verificare prefill/invalidazione.
 */
function contextLogits(contextIds) {
  const logits = new Array(VOCAB_SIZE).fill(0);
  const h1 = contextIds.length;
  const last = contextIds[contextIds.length - 1] ?? 0;
  logits[(last + h1) % VOCAB_SIZE] = 6.0;
  logits[(last * 3 + h1 + 1) % VOCAB_SIZE] = 4.0;
  logits[(last * 7 + h1 + 2) % VOCAB_SIZE] = 3.0;
  logits[(last * 11 + h1 + 3) % VOCAB_SIZE] = 2.0;
  logits[(last * 13 + h1 + 4) % VOCAB_SIZE] = 1.5;
  return logits;
}

export class MockRuntime {
  constructor() {
    this.tokenizer = new MockTokenizer();
    this.backend = 'mock';
    this.eosId = 1;
    this.modelInfo = 'Mock deterministico (vocab 512)';
  }

  async init() {
    this.prefillCount = 0;
    return { backend: this.backend, modelInfo: this.modelInfo };
  }

  encode(text) {
    return this.tokenizer.encode(text);
  }

  decode(ids) {
    return this.tokenizer.decode(ids);
  }

  /**
   * Calcola i logits dell'ultima posizione.
   * Se `cache` è fornita e coerente, calcola solo i token nuovi;
   * altrimenti ricalcola l'intera sequenza (naive).
   */
  async nextLogits({ ids, cache = null }) {
    if (!ids || ids.length === 0) throw new Error('Sequenza vuota');
    let newTokens = ids.length;
    let context = ids;
    if (cache && cache.ids && arraysEqual(cache.ids, ids.slice(0, cache.ids.length))) {
      context = ids;
      newTokens = ids.length - cache.ids.length;
      this.prefillCount += newTokens;
    } else {
      this.prefillCount += ids.length;
    }
    const logits = contextLogits(ids);
    return {
      logits,
      cache: { ids: ids.slice() },
      newTokens,
      contextLength: ids.length,
    };
  }

  probsFromLogits(logits, temperature) {
    return softmax(logits, temperature);
  }

  idToToken(id) {
    return this.tokenizer.idToToken(id);
  }

  reset() {
    this.prefillCount = 0;
  }
}

function arraysEqual(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

/**
 * Flusso completo di un passo, condiviso da mock e runtime reale:
 * logits → penalità → softmax → filtri → ranking → selezione.
 */
export async function computeStep(runtime, { contextIds, params, cache = null }) {
  const { logits, cache: newCache, newTokens, contextLength } = await runtime.nextLogits({ ids: contextIds, cache });
  const penalized = applyRepeatPenalty(logits, contextIds, params.repeatPenalty);
  const probs = softmax(penalized, params.temperature);
  let ranking = rankTokens(penalized, probs);
  ranking = applyTopK(ranking, params.topK);
  ranking = applyTopP(ranking, params.topP);
  ranking = applyMinP(ranking, probs, params.minP);
  const chosen = selectToken(ranking, probs, { mode: params.mode, seed: params.seed + contextIds.length });
  const topN = ranking.slice(0, params.topN).map((c) => ({ ...c, prob: probs[c.tokenId] }));
  return {
    chosen,
    ranking: topN,
    probs,
    cache: newCache,
    newTokens,
    contextLength,
    isEos: chosen && chosen.tokenId === runtime.eosId,
  };
}
