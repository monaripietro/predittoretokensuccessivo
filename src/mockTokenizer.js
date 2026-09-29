/**
 * Tokenizer didattico deterministico usato dal mock runtime.
 * Segmenta il testo in "parole" e mappa ogni vocabolo a un token id stabile
 * tramite un hash deterministico. Non è un vero BPE: serve solo a rendere
 * verificabile il ciclo tokenizzazione → id → decodifica nei test.
 */

const SPECIAL_TOKENS = {
  '<pad>': 0,
  '<eos>': 1,
};

function stableHash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export const VOCAB_SIZE = 512;

export class MockTokenizer {
  constructor() {
    this.vocab = new Map(Object.entries(SPECIAL_TOKENS));
    this.invVocab = new Map(Object.entries(SPECIAL_TOKENS).map(([t, id]) => [id, t]));
  }

  tokenToId(token) {
    if (this.vocab.has(token)) return this.vocab.get(token);
    const id = 2 + (stableHash(token) % (VOCAB_SIZE - 2));
    this.vocab.set(token, id);
    this.invVocab.set(id, token);
    return id;
  }

  idToToken(id) {
    if (this.invVocab.has(id)) return this.invVocab.get(id);
    const token = `<tok-${id}>`;
    this.invVocab.set(id, token);
    return token;
  }

  encode(text) {
    const normalized = String(text).trim();
    if (!normalized) return [];
    const words = normalized.split(/\s+/);
    return words.map((w) => this.tokenToId(w));
  }

  decode(ids) {
    return ids.map((id) => this.idToToken(id)).join(' ');
  }
}
