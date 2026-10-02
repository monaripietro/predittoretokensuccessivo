/**
 * Verifiche usate dalla pagina di diagnostica e dai test con modello reale.
 * Non fanno parte dell'esperienza didattica: servono a dimostrare che i
 * punteggi mostrati sono quelli grezzi del modello e che il ciclo passo
 * per passo coincide con la generazione nativa della libreria.
 */

import { NEUTRAL_GENERATION_OPTIONS } from './decision.js';

function ids2tensor(Tensor, ids) {
  return new Tensor('int64', BigInt64Array.from(ids, (x) => BigInt(x)), [1, ids.length]);
}

function ones(Tensor, n) {
  return new Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]);
}

/**
 * Confronta i logit ricevuti dal processore di scelta (primo passo) con
 * quelli di un forward diretto del modello sullo stesso input, senza
 * generate(). Se coincidono, nessun altro processore li ha modificati.
 */
export async function verifyRawScores(engine, tf, text) {
  const { Tensor } = tf;
  const begun = await engine.begin({ text, policy: { kind: 'greedy' } });
  const ev = await engine.step({ sessionId: begun.sessionId, keepRawCopy: true });
  await engine.end();
  const { model } = engine._internals;
  const ids = begun.input.ids;
  // generate() chiede al grafo solo i logit dell'ultima posizione quando il
  // modello lo supporta: si fa lo stesso, così il confronto usa gli stessi kernel.
  const session = model.sessions?.decoder_model_merged ?? model.sessions?.model;
  const keepsLast = Boolean(session?.inputNames?.includes('num_logits_to_keep'));
  const inputs = { input_ids: ids2tensor(Tensor, ids), attention_mask: ones(Tensor, ids.length) };
  if (keepsLast) inputs.num_logits_to_keep = new Tensor('int64', [1n], []);
  const out = await model(inputs);
  const logits = out.logits;
  const vocab = logits.dims.at(-1);
  const seq = logits.dims.at(-2);
  const asF32 = logits.to ? logits.to('float32') : logits;
  const last = asF32.data.slice((seq - 1) * vocab, seq * vocab);
  for (const t of Object.values(out)) {
    if (t?.location === 'gpu-buffer') t.dispose();
  }
  let maxAbsDiff = 0;
  let argmaxDirect = 0;
  for (let i = 0; i < vocab; i++) {
    const d = Math.abs(last[i] - ev.rawLogits[i]);
    if (d > maxAbsDiff) maxAbsDiff = d;
    if (last[i] > last[argmaxDirect]) argmaxDirect = i;
  }
  return {
    vocab,
    lastPositionOnly: keepsLast,
    maxAbsDiff,
    argmaxDirect,
    selectedByEngine: ev.selected.id,
    argmaxEqual: argmaxDirect === ev.selected.id,
  };
}

/**
 * Genera N token con generate() nativo (greedy, senza il nostro processore)
 * e con il ciclo passo-passo del motore; restituisce entrambe le sequenze.
 */
export async function verifyGreedyParity(engine, tf, text, n = 12) {
  const { Tensor } = tf;
  const begun = await engine.begin({ text, policy: { kind: 'greedy' }, maxNewTokens: n });
  const stepwise = [];
  const stepTimes = [];
  for (let i = 0; i < n; i++) {
    const ev = await engine.step({ sessionId: begun.sessionId });
    stepwise.push(ev.selected.id);
    stepTimes.push(ev.timing.stepMs);
    if (ev.finish) break;
  }
  await engine.end();
  const { model } = engine._internals;
  const ids = begun.input.ids;
  const t0 = performance.now();
  const seq = await model.generate({
    input_ids: ids2tensor(Tensor, ids),
    attention_mask: ones(Tensor, ids.length),
    max_new_tokens: n,
    ...NEUTRAL_GENERATION_OPTIONS,
  });
  const nativeMs = performance.now() - t0;
  const all = Array.from(seq.data ?? seq, (x) => Number(x));
  const native = all.slice(ids.length);
  const len = Math.min(native.length, stepwise.length);
  let firstMismatch = -1;
  for (let i = 0; i < len; i++) {
    if (native[i] !== stepwise[i]) {
      firstMismatch = i;
      break;
    }
  }
  return { native, stepwise, equal: firstMismatch === -1 && len > 0, firstMismatch, nativeMs, stepTimes };
}
