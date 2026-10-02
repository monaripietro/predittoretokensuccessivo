/**
 * Motore di inferenza: carica tokenizer e modello, prepara l'input della
 * chat, esegue un passo di generazione alla volta e produce eventi di
 * traccia compatti. Gira dentro un Web Worker nel browser; nei test Node
 * riceve una libreria finta o Transformers.js reale.
 *
 * Nessun prompt esce dal dispositivo: l'unica rete usata è il download dei
 * file del modello (Hugging Face) e del runtime ONNX (servito dal sito).
 */

import { mulberry32, validatePolicy } from '../core/distribution.js';
import {
  createPieceDecoder, makeSpecialTokenTest, piecesOf, labelOrigins, candidatePiece,
} from '../core/tokens.js';
import {
  MODELS, SYSTEM_PROMPT, LIMITS, pickVariant,
} from '../core/models.js';
import { makeDecisionProcessorClass, NEUTRAL_GENERATION_OPTIONS } from './decision.js';

export class EngineError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

function toIdArray(tensorOrArray) {
  const data = tensorOrArray?.data ?? tensorOrArray;
  return Array.from(data, (x) => Number(x));
}

/**
 * @param {object} tf modulo Transformers.js (o un sostituto con la stessa API)
 * @param {object} [options]
 * @param {string} [options.device] 'webgpu' nel browser, 'cpu' in Node
 * @param {(e: object) => void} [options.onProgress]
 */
export function createEngine(tf, { device = 'webgpu', onProgress = () => {} } = {}) {
  const { AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, Tensor } = tf;
  const DecisionProcessor = makeDecisionProcessorClass(LogitsProcessor);

  let tokenizer = null;
  let model = null;
  let isSpecial = null;
  let info = null;
  let session = null;
  let busy = Promise.resolve();

  /** Serializza le operazioni sul modello (un passo alla volta). */
  function serial(fn) {
    const run = busy.then(fn, fn);
    busy = run.catch(() => {});
    return run;
  }

  function requireModel() {
    if (!model || !tokenizer) throw new EngineError('not-loaded', 'Il modello non è ancora caricato.');
  }

  async function load({ modelKey, dtype: forcedDtype = null, shaderF16 = false, spec: specOverride = null }) {
    const spec = specOverride ?? MODELS[modelKey];
    if (!spec) throw new EngineError('unknown-model', `Modello sconosciuto: ${modelKey}`);
    const dtype = forcedDtype ?? pickVariant(spec, { shaderF16 });
    const rev = spec.revision;
    const t0 = performance.now();
    // Avanzamento calcolato sui file realmente necessari (dimensioni fissate
    // per revisione). Il totale aggregato della libreria include anche file
    // che il caricamento solo-testo non scarica.
    const expected = spec.variants?.[dtype]
      ? { ...spec.baseFiles, ...spec.variants[dtype].files }
      : null;
    const totalBytes = expected ? Object.values(expected).reduce((a, b) => a + b, 0) : 0;
    const loadedByFile = new Map();
    let lastEmit = 0;
    const progress_callback = (e) => {
      if (!expected || !e?.file || !(e.file in expected)) return;
      if (e.status === 'progress' && Number.isFinite(e.loaded)) loadedByFile.set(e.file, e.loaded);
      else if (e.status === 'done') loadedByFile.set(e.file, expected[e.file]);
      else return;
      const now = performance.now();
      if (now - lastEmit < 200 && e.status !== 'done') return;
      lastEmit = now;
      let loaded = 0;
      for (const [f, n] of loadedByFile) loaded += Math.min(n, expected[f]);
      onProgress({ phase: 'files', loaded, total: totalBytes, file: e.file });
    };
    tokenizer = await AutoTokenizer.from_pretrained(spec.id, { revision: rev, progress_callback });
    const tTok = performance.now();
    model = await AutoModelForCausalLM.from_pretrained(spec.id, {
      revision: rev,
      dtype,
      device,
      progress_callback,
    });
    const tModel = performance.now();
    const genConfig = model.generation_config ?? {};
    const eos = genConfig.eos_token_id ?? tokenizer.eos_token_id;
    const eosIds = (Array.isArray(eos) ? eos : [eos]).filter((x) => Number.isInteger(x));
    isSpecial = makeSpecialTokenTest(tokenizer, eosIds);
    info = {
      modelKey: spec.key ?? modelKey,
      modelId: spec.id,
      revision: rev,
      label: spec.label,
      dtype,
      device,
      chatTemplateOptions: spec.chatTemplateOptions ?? {},
      eosIds,
      eosTokens: eosIds.map((id) => tokenizer.decode([id], { skip_special_tokens: false })),
      tokenizerMs: Math.round(tTok - t0),
      modelMs: Math.round(tModel - tTok),
      downloadBytes: totalBytes || null,
      systemPrompt: SYSTEM_PROMPT,
      vocabSize: null,
    };
    return info;
  }

  /**
   * Prima esecuzione di prova su un testo fisso: compila i programmi della
   * scheda grafica, così la prima domanda vera non paga questo costo.
   * Non produce eventi visibili nella demo; il tempo viene riportato.
   */
  async function warmup() {
    const t0 = performance.now();
    const b = await begin({ text: 'Ciao', policy: { kind: 'greedy' }, maxNewTokens: 2 });
    await step({ sessionId: b.sessionId });
    await step({ sessionId: b.sessionId });
    await end();
    return { warmupMs: Math.round(performance.now() - t0) };
  }

  /** Costruisce l'input esatto inviato al modello (modello di chat incluso). */
  function buildInput(text) {
    requireModel();
    const userText = String(text ?? '').trim();
    if (!userText) throw new EngineError('empty-prompt', 'Scrivi una domanda o una frase.');
    if (userText.length > LIMITS.maxPromptChars) {
      throw new EngineError('prompt-too-long', `Il testo supera ${LIMITS.maxPromptChars} caratteri.`);
    }
    const messages = [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: userText },
    ];
    const fullText = tokenizer.apply_chat_template(messages, {
      tokenize: false,
      add_generation_prompt: true,
      ...info.chatTemplateOptions,
    });
    const ids = toIdArray(tokenizer.encode(fullText, { add_special_tokens: false }));
    const pieces = piecesOf(tokenizer, ids, isSpecial);
    const sysStart = fullText.indexOf(SYSTEM_PROMPT);
    const userStart = fullText.indexOf(userText, sysStart >= 0 ? sysStart + SYSTEM_PROMPT.length : 0);
    const spans = [];
    if (sysStart >= 0) spans.push({ start: sysStart, end: sysStart + SYSTEM_PROMPT.length, origin: 'system' });
    if (userStart >= 0) spans.push({ start: userStart, end: userStart + userText.length, origin: 'user' });
    const labelled = labelOrigins(pieces, spans);
    const reconstructed = pieces.map((p) => p.text).join('');
    return {
      userText,
      fullText,
      ids,
      tokens: labelled.map((p) => ({
        id: p.id, text: p.text, groupText: p.groupText, groupSize: p.groupSize,
        groupIndex: p.groupIndex, special: p.special, fragment: p.fragment, origin: p.origin,
      })),
      piecesExact: reconstructed === fullText,
      tooLong: ids.length > LIMITS.maxInputTokens,
    };
  }

  function tokenize({ text }) {
    return buildInput(text);
  }

  async function disposeSession() {
    if (session?.cache) {
      try {
        await session.cache.dispose();
      } catch {
        // la cache potrebbe essere già stata liberata
      }
    }
    session = null;
  }

  /**
   * @param {'cache'|'naive'} [cacheMode] 'cache' = KV cache (si elaborano solo i
   *   token nuovi); 'naive' = ricalcolo completo dell'intera sequenza a ogni passo.
   */
  function begin({
    text, policy, seed, maxNewTokens = LIMITS.defaultNewTokens, cacheMode = 'cache',
  }) {
    return serial(async () => {
      requireModel();
      await disposeSession();
      const input = buildInput(text);
      if (input.tooLong) {
        throw new EngineError('input-too-long', `L'input supera ${LIMITS.maxInputTokens} token: accorcia il testo.`);
      }
      const p = validatePolicy(policy);
      const n = Math.max(1, Math.min(LIMITS.maxNewTokens, Math.floor(maxNewTokens)));
      const s = Number.isInteger(seed) ? seed >>> 0 : (Math.random() * 2 ** 32) >>> 0;
      session = {
        id: `${Date.now().toString(36)}-${s.toString(36)}`,
        ids: input.ids.slice(),
        promptLength: input.ids.length,
        generated: [],
        cache: null,
        cacheMode: cacheMode === 'naive' ? 'naive' : 'cache',
        policy: p,
        seed: s,
        random: p.kind === 'sample' ? mulberry32(s) : null,
        maxNewTokens: n,
        decoder: createPieceDecoder(tokenizer, isSpecial),
        finished: null,
      };
      return {
        sessionId: session.id, input, policy: p, seed: p.kind === 'sample' ? s : null, maxNewTokens: n, cacheMode: session.cacheMode,
      };
    });
  }

  function step({ sessionId, keepRawCopy = false } = {}) {
    return serial(async () => {
      requireModel();
      if (!session || session.id !== sessionId) {
        throw new EngineError('no-session', 'Nessuna generazione attiva.');
      }
      if (session.finished) throw new EngineError('finished', 'La generazione è già terminata.');
      const s = session;
      const contextLength = s.ids.length;
      if (s.cacheMode === 'naive' && s.cache) {
        // Ricalcolo completo: la cache del passo precedente viene buttata via.
        await s.cache.dispose();
        s.cache = null;
      }
      const cachedBefore = s.cache ? s.cache.get_seq_length() : 0;
      const processor = new DecisionProcessor({
        policy: s.policy,
        random: s.random,
        displayK: LIMITS.displayCandidates,
        keepRawCopy,
      });
      const ids = s.ids;
      const inputIds = new Tensor('int64', BigInt64Array.from(ids, (x) => BigInt(x)), [1, ids.length]);
      const attentionMask = new Tensor('int64', new BigInt64Array(ids.length).fill(1n), [1, ids.length]);
      const t0 = performance.now();
      const out = await model.generate({
        input_ids: inputIds,
        attention_mask: attentionMask,
        past_key_values: s.cache,
        max_new_tokens: 1,
        return_dict_in_generate: true,
        logits_processor: [processor],
        ...NEUTRAL_GENERATION_OPTIONS,
      });
      const elapsedMs = performance.now() - t0;
      s.cache = out.past_key_values ?? null;
      if (s.cacheMode === 'naive' && s.cache) {
        await s.cache.dispose();
        s.cache = null;
      }
      const sequence = toIdArray(out.sequences);
      const emitted = sequence[sequence.length - 1];
      const decision = processor.decision;
      // Verifiche di allineamento: un solo passaggio del modello, stesso token.
      if (processor.calls !== 1 || !decision) {
        throw new EngineError('alignment', `Atteso 1 calcolo dei punteggi, ricevuti ${processor.calls}.`);
      }
      if (sequence.length !== contextLength + 1) {
        throw new EngineError('alignment', 'La sequenza restituita non ha esattamente un token in più.');
      }
      if (emitted !== decision.selected.id) {
        throw new EngineError('alignment', `Token emesso ${emitted} diverso dal token scelto ${decision.selected.id}.`);
      }
      if (info.vocabSize === null) info.vocabSize = decision.vocabSize;

      const candidates = decision.candidates.map((c) => ({
        ...c,
        piece: candidatePiece(tokenizer, ids, c.id, isSpecial),
      }));
      const selectedId = decision.selected.id;
      const isEos = info.eosIds.includes(selectedId);
      s.ids.push(selectedId);
      s.generated.push(selectedId);
      const completed = s.decoder.push(selectedId);
      let finish = null;
      if (isEos) finish = 'eos';
      else if (s.generated.length >= s.maxNewTokens) finish = 'length';
      if (finish) {
        completed.push(...s.decoder.end());
        s.finished = finish;
      }
      const answerText = tokenizer.decode(s.generated, { skip_special_tokens: true });
      const event = {
        sessionId: s.id,
        index: s.generated.length - 1,
        contextLength,
        processedTokens: contextLength - cachedBefore,
        cachedTokens: cachedBefore,
        cacheMode: s.cacheMode,
        candidates,
        other: decision.other,
        vocabSize: decision.vocabSize,
        invalidScores: decision.invalidScores,
        policy: decision.policy,
        pool: decision.pool,
        selected: {
          ...decision.selected,
          // pezzo calcolato sul contesto PRIMA del token scelto
          piece: candidatePiece(tokenizer, ids.slice(0, contextLength), selectedId, isSpecial),
          special: isSpecial(selectedId),
          isEos,
        },
        completedPieces: completed,
        answerText,
        finish,
        timing: { stepMs: Math.round(elapsedMs * 10) / 10, decideMs: Math.round(processor.decideMs * 10) / 10 },
        rawLogits: keepRawCopy ? processor.rawCopy : undefined,
      };
      s.lastEvent = { ...event, rawLogits: undefined };
      return event;
    });
  }

  /**
   * Scelta del presentatore: sostituisce il token appena scelto con un altro
   * dei candidati mostrati per lo stesso passo. Quel token non è ancora
   * entrato nella KV cache (vi entra al passo successivo), quindi non si
   * ricalcola nulla: le probabilità restano quelle del modello per quel passo
   * e l'evento registra sia la scelta della regola sia quella del presentatore.
   */
  function override({ sessionId, id }) {
    return serial(async () => {
      requireModel();
      if (!session || session.id !== sessionId) {
        throw new EngineError('no-session', 'Nessuna generazione attiva.');
      }
      const s = session;
      const last = s.lastEvent;
      if (!last) throw new EngineError('no-step', "Non c'è ancora un token da sostituire.");
      if (s.finished) throw new EngineError('finished', 'La generazione è già terminata.');
      const cand = last.candidates.find((c) => c.id === id);
      if (!cand) throw new EngineError('not-candidate', 'Si può scegliere solo tra i candidati mostrati.');
      const original = last.override?.original ?? last.selected;
      s.ids[s.ids.length - 1] = id;
      s.generated[s.generated.length - 1] = id;
      // Il decodificatore dei pezzi viene ricostruito sulla nuova sequenza generata.
      s.decoder = createPieceDecoder(tokenizer, isSpecial);
      let completed = [];
      s.generated.forEach((g, i) => {
        const out = s.decoder.push(g);
        if (i === s.generated.length - 1) completed = out;
      });
      const isEos = info.eosIds.includes(id);
      let finish = null;
      if (isEos) finish = 'eos';
      else if (s.generated.length >= s.maxNewTokens) finish = 'length';
      if (finish) {
        completed.push(...s.decoder.end());
        s.finished = finish;
      }
      const event = {
        ...last,
        selected: {
          id,
          modelProb: cand.modelProb,
          policyProb: cand.policyProb ?? null,
          rank: cand.rank,
          draw: null,
          piece: cand.piece,
          special: isSpecial(id),
          isEos,
        },
        completedPieces: completed,
        answerText: tokenizer.decode(s.generated, { skip_special_tokens: true }),
        finish,
        override: { by: 'presenter', original },
      };
      s.lastEvent = event;
      return event;
    });
  }

  function end() {
    return serial(disposeSession);
  }

  async function unload() {
    await end();
    if (model?.dispose) await model.dispose();
    model = null;
    tokenizer = null;
    info = null;
  }

  return {
    load,
    warmup,
    tokenize,
    begin,
    step,
    override,
    end,
    unload,
    get info() {
      return info;
    },
    /** Solo diagnostica: accesso diretto per i test di verifica. */
    get _internals() {
      return { tokenizer, model, isSpecial, session };
    },
  };
}
