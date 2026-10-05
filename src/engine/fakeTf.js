/**
 * Sostituto minimo di Transformers.js per test e per la modalità
 * dimostrativa `?mock` (sempre segnalata in pagina come simulazione).
 *
 * Imita la parte di API usata dal motore: tokenizer con modello di chat,
 * token speciali e frammenti di byte; modello con `generate()` che chiama i
 * LogitsProcessor e poi sceglie il massimo (come il campionatore greedy
 * della libreria), restituendo sequenze e una cache con lunghezza.
 * I logit sono deterministici e dipendono dal contesto: non sono
 * probabilità scritte a mano, passano dallo stesso codice di softmax.
 */

export class Tensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

export class LogitsProcessor {
  _call() {
    throw new Error('not implemented');
  }
}

const SPECIALS = ['<pad>', '<eos>', '<bos>', '<|turn>', '<turn|>'];
// Pezzi di vocabolario (con e senza spazio iniziale), in ordine di ID.
const WORDS = [
  '\n', ' ', '.', ',', '?', '!', "'", ':', '«', '»',
  'Il', ' il', ' cielo', ' è', ' blu', ' perché', ' la', ' luce', ' del', ' sole', ' si', ' diffonde',
  ' nell', 'aria', ' Roma', ' capitale', " d'", 'Italia', 'La', ' gatto', ' dorme', ' sul', ' divano',
  ' ragno', ' ha', ' otto', ' zampe', 'Un', ' un', ' una', ' frase', ' Per', 'ché', ' Ciao', 'Ciao',
  ' mattino', " l'", 'oro', ' in', ' bocca', 'user', 'model', 'system', ' Rispondi', ' italiano',
  ' modo', ' breve', ' e', ' semplice', ' al', ' massimo', ' due', ' frasi', ' a', ' di', ' che',
  ' cos', 'è', ' intelligenza', ' artificiale', ' programma', ' impara', ' dai', ' dati', ' Qual',
  ' Perché', 'Perché', 'Qual', 'Quante', ' Scrivi', 'Scrivi', ' Completa', 'Completa', ' C',
  "'era", ' volta', 'L', 'I', ' ragni', ' hanno',
];
const N_BYTES = 256;
export const FAKE_VOCAB_SIZE = SPECIALS.length + WORDS.length + N_BYTES + 37; // ID senza testo in coda
const BYTE_BASE = SPECIALS.length + WORDS.length;

function utf8(s) {
  return new TextEncoder().encode(s);
}

export class FakeTokenizer {
  constructor() {
    this.pieces = [...SPECIALS, ...WORDS];
    this.bos_token_id = 2;
    this.eos_token_id = 1;
    this.all_special_ids = SPECIALS.map((_, i) => i);
    this.byPiece = new Map(this.pieces.map((p, i) => [p, i]));
    this.sortedWords = WORDS.map((w, i) => [w, SPECIALS.length + i]).sort((a, b) => b[0].length - a[0].length);
    this.lastChatTemplate = null;
  }

  encode(text) {
    const ids = [];
    let i = 0;
    outer: while (i < text.length) {
      for (const [sp, id] of SPECIALS.map((s, k) => [s, k])) {
        if (text.startsWith(sp, i)) {
          ids.push(id);
          i += sp.length;
          continue outer;
        }
      }
      for (const [w, id] of this.sortedWords) {
        if (text.startsWith(w, i)) {
          ids.push(id);
          i += w.length;
          continue outer;
        }
      }
      // carattere sconosciuto → byte UTF-8 (un token per byte)
      const cp = text.codePointAt(i);
      const ch = String.fromCodePoint(cp);
      for (const b of utf8(ch)) ids.push(BYTE_BASE + b);
      i += ch.length;
    }
    return ids;
  }

  decode(ids, { skip_special_tokens = false } = {}) {
    if (!Array.isArray(ids) && !ArrayBuffer.isView(ids)) throw new Error('ids must be an array');
    const bytes = [];
    for (const raw of ids) {
      const id = Number(raw);
      if (id < SPECIALS.length) {
        if (!skip_special_tokens) bytes.push(...utf8(SPECIALS[id]));
      } else if (id < BYTE_BASE) {
        bytes.push(...utf8(this.pieces[id]));
      } else if (id < BYTE_BASE + N_BYTES) {
        bytes.push(id - BYTE_BASE);
      }
      // ID oltre il vocabolario: nessun testo (come i token di riempimento reali)
    }
    return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
  }

  apply_chat_template(messages, { add_generation_prompt = false } = {}) {
    this.lastChatTemplate = {
      messages: messages.map((message) => ({ ...message })),
      add_generation_prompt,
      options: arguments[1] ? { ...arguments[1] } : {},
    };
    let out = '<bos>';
    for (const m of messages) out += `<|turn>${m.role}\n${m.content.trim()}<turn|>\n`;
    if (add_generation_prompt) out += '<|turn>model\n';
    return out;
  }
}

/** Risposte di esempio usate dalla simulazione (solo per guidare i logit). */
const SCRIPTS = [
  [/cielo/i, 'Il cielo è blu perché la luce del sole si diffonde nell\'aria.'],
  [/capitale/i, "La capitale d'Italia è Roma."],
  [/gatto/i, 'Il gatto dorme sul divano.'],
  [/ragno/i, 'I ragni hanno otto zampe.'],
  [/mattino/i, "Il mattino ha l'oro in bocca."],
];

function hash(n) {
  let h = n | 0;
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
  return (h ^ (h >>> 16)) >>> 0;
}

export class FakeCache {
  constructor(length) {
    this.length = length;
    this.disposed = false;
  }

  get_seq_length() {
    return this.length;
  }

  async dispose() {
    this.disposed = true;
  }
}

export class FakeModel {
  constructor(tokenizer, options = {}) {
    this.tokenizer = tokenizer;
    this.generation_config = { eos_token_id: [1, 4] };
    this.options = options;
    this.forwardCalls = 0;
    this.caches = [];
  }

  /** Logit deterministici per l'ultima posizione del contesto. */
  logitsFor(ids) {
    const v = new Float32Array(FAKE_VOCAB_SIZE).fill(-12);
    const text = this.tokenizer.decode(ids);
    const userPart = text.slice(text.lastIndexOf('<|turn>user'));
    const answerStart = text.lastIndexOf('<|turn>model\n');
    const answerSoFar = answerStart >= 0 ? text.slice(answerStart + '<|turn>model\n'.length) : '';
    const script = (SCRIPTS.find(([re]) => re.test(userPart)) ?? [null, 'Ciao.'])[1];
    let target;
    if (answerSoFar.length >= script.length) {
      target = 4; // <turn|>: fine del turno
    } else {
      const rest = script.slice(answerSoFar.length);
      target = this.tokenizer.encode(rest)[0];
    }
    const h = hash(ids.length * 7919 + (ids[ids.length - 1] ?? 0));
    // Alternative plausibili con punteggi variabili ma riproducibili.
    for (let k = 0; k < 12; k++) {
      const id = 5 + (hash(h + k) % (WORDS.length));
      v[id] = Math.max(v[id], 2 + ((hash(h + 100 + k) % 1000) / 1000) * 3);
    }
    v[target] = 6 + ((h % 1000) / 1000) * 2;
    return v;
  }

  async generate({ input_ids, past_key_values = null, logits_processor = [], max_new_tokens = 1 }) {
    if (this.options.failOnCall && this.forwardCalls + 1 === this.options.failOnCall) {
      this.forwardCalls += 1;
      throw new Error('Errore simulato del modello');
    }
    if (this.options.delayMs) await new Promise((r) => setTimeout(r, this.options.delayMs));
    const ids = Array.from(input_ids.data, (x) => Number(x));
    const all = ids.slice();
    let cache = past_key_values;
    for (let t = 0; t < max_new_tokens; t++) {
      this.forwardCalls += 1;
      const raw = this.logitsFor(all);
      const original = Float32Array.from(raw);
      const logits = new Tensor('float32', raw, [1, raw.length]);
      let scores = logits;
      for (const p of logits_processor) scores = p._call([all.map(BigInt)], scores);
      const source = this.options.ignoreProcessors ? original : scores.data;
      let best = 0;
      for (let i = 1; i < source.length; i++) if (source[i] > source[best]) best = i;
      // la cache copre i token elaborati (non quello appena scelto)
      if (!cache) {
        cache = new FakeCache(all.length);
        this.caches.push(cache);
      } else cache.length = all.length;
      all.push(best);
      if (this.generation_config.eos_token_id.includes(best)) break;
    }
    return {
      sequences: new Tensor('int64', BigInt64Array.from(all, (x) => BigInt(x)), [1, all.length]),
      past_key_values: cache,
    };
  }

  async dispose() {
    this.disposed = true;
  }
}

/** Crea un modulo con la stessa forma di Transformers.js. */
export function createFakeTf(modelOptions = {}) {
  const created = { tokenizer: null, model: null };
  return {
    created,
    Tensor,
    LogitsProcessor,
    AutoTokenizer: {
      async from_pretrained() {
        created.tokenizer = new FakeTokenizer();
        return created.tokenizer;
      },
    },
    AutoModelForCausalLM: {
      async from_pretrained(id, { progress_callback } = {}) {
        progress_callback?.({ status: 'done', file: 'fake.onnx' });
        created.model = new FakeModel(created.tokenizer ?? new FakeTokenizer(), modelOptions);
        return created.model;
      },
    },
  };
}

export const MOCK_SPEC = Object.freeze({
  key: 'mock',
  name: 'Simulazione',
  label: 'Modello simulato (solo per test)',
  id: 'mock/simulated',
  revision: 'mock',
  baseFiles: {},
  variants: { mock: { requiresF16: false, files: {} } },
  chatTemplateOptions: {},
});
