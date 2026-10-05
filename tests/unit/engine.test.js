import { describe, it, expect } from 'vitest';
import { createEngine } from '../../src/engine/engine.js';
import { createFakeTf, MOCK_SPEC } from '../../src/engine/fakeTf.js';
import { scanLogits, modelProbability, topKIndices } from '../../src/core/distribution.js';

const THINKING_SPEC = {
  ...MOCK_SPEC,
  thinking: { parameter: 'enable_thinking', default: false },
};

async function setup(modelOptions = {}, spec = MOCK_SPEC) {
  const tf = createFakeTf(modelOptions);
  const engine = createEngine(tf, { device: 'test' });
  await engine.load({ spec, dtype: 'mock' });
  return { tf, engine };
}

/** Ricalcola, in modo indipendente, i logit che il modello finto ha prodotto per un contesto. */
function recomputeLogits(tf, ids) {
  return tf.created.model.logitsFor(ids);
}

describe('motore: input della chat', () => {
  it('rende esplicito il testo completo inviato al modello', async () => {
    const { engine } = await setup();
    const input = engine.tokenize({ text: '  Perché il cielo è blu?  ' });
    expect(input.userText).toBe('Perché il cielo è blu?');
    expect(input.fullText.startsWith('<bos><|turn>system\n')).toBe(true);
    expect(input.fullText).toContain('Rispondi in italiano');
    expect(input.piecesExact).toBe(true);
    expect(input.tokens.map((t) => t.id)).toEqual(input.ids);
    const origins = new Set(input.tokens.map((t) => t.origin));
    expect(origins).toEqual(new Set(['control', 'template', 'system', 'user']));
  });

  it('rifiuta testo vuoto o troppo lungo', async () => {
    const { engine } = await setup();
    expect(() => engine.tokenize({ text: '   ' })).toThrow(/Scrivi/);
    expect(() => engine.tokenize({ text: 'a'.repeat(500) })).toThrow(/caratteri/);
  });

  it('usa il system prompt scelto e ne conserva l’origine dei token', async () => {
    const { tf, engine } = await setup();
    const input = engine.tokenize({ text: 'Ciao', systemPrompt: 'Rispondi con una parola.' });
    expect(input.systemPrompt).toBe('Rispondi con una parola.');
    expect(input.fullText).toContain('Rispondi con una parola.');
    expect(input.tokens.some((t) => t.origin === 'system')).toBe(true);
    expect(tf.created.tokenizer.lastChatTemplate.messages[0].content).toBe('Rispondi con una parola.');
  });

  it('rifiuta un system prompt oltre il limite', async () => {
    const { engine } = await setup();
    expect(() => engine.tokenize({ text: 'Ciao', systemPrompt: 'a'.repeat(241) }))
      .toThrow(/istruzione di sistema/);
  });

  it('passa al template il thinking solo per i modelli che lo supportano', async () => {
    const { tf, engine } = await setup({}, THINKING_SPEC);
    const off = engine.tokenize({ text: 'Ciao', thinking: false });
    expect(off.thinking).toBe(false);
    expect(tf.created.tokenizer.lastChatTemplate.options.enable_thinking).toBe(false);
    const on = engine.tokenize({ text: 'Ciao', thinking: true });
    expect(on.thinking).toBe(true);
    expect(tf.created.tokenizer.lastChatTemplate.options.enable_thinking).toBe(true);
    const begun = await engine.begin({ text: 'Ciao', policy: { kind: 'greedy' }, thinking: true });
    expect(begun.thinking).toBe(true);
    expect(begun.input.thinking).toBe(true);
  });
});

describe('motore: allineamento punteggi ↔ token scelto', () => {
  it('per più passi consecutivi, probabilità e scelta vengono dagli stessi logit del passo', async () => {
    const { tf, engine } = await setup();
    const b = await engine.begin({ text: 'Perché il cielo è blu?', policy: { kind: 'greedy' } });
    const context = b.input.ids.slice();
    for (let i = 0; i < 6; i++) {
      const ev = await engine.step({ sessionId: b.sessionId });
      const logits = recomputeLogits(tf, context);
      const stats = scanLogits(logits);
      expect(ev.index).toBe(i);
      expect(ev.contextLength).toBe(context.length);
      expect(ev.vocabSize).toBe(logits.length);
      // candidati = classifica reale di questo passo, con probabilità reali
      expect(ev.candidates.map((c) => c.id)).toEqual(topKIndices(logits, ev.candidates.length));
      for (const c of ev.candidates) expect(c.modelProb).toBeCloseTo(modelProbability(logits, c.id, stats), 12);
      expect(ev.selected.modelProb).toBeCloseTo(modelProbability(logits, ev.selected.id, stats), 12);
      // greedy: il token scelto è il primo della classifica
      expect(ev.selected.id).toBe(ev.candidates[0].id);
      context.push(ev.selected.id);
      if (ev.finish) break;
    }
  });

  it('la KV cache cresce: il primo passo elabora il prompt, poi un token per passo', async () => {
    const { engine } = await setup();
    const b = await engine.begin({ text: 'Ciao', policy: { kind: 'greedy' } });
    const e1 = await engine.step({ sessionId: b.sessionId });
    const e2 = await engine.step({ sessionId: b.sessionId });
    expect(e1.processedTokens).toBe(b.input.ids.length);
    expect(e1.cachedTokens).toBe(0);
    expect(e2.processedTokens).toBe(1);
    expect(e2.cachedTokens).toBe(b.input.ids.length);
  });

  it('se la libreria emettesse un token diverso da quello scelto, il passo fallisce', async () => {
    // modello difettoso: ignora il processore e prende sempre il massimo originale
    const { engine } = await setup({ ignoreProcessors: true });
    const b = await engine.begin({
      text: 'Perché il cielo è blu?',
      policy: { kind: 'sample', temperature: 2, topP: 1, topK: 64 },
      seed: 3,
    });
    let error = null;
    for (let i = 0; i < 20 && !error; i++) {
      try {
        const ev = await engine.step({ sessionId: b.sessionId });
        if (ev.finish) break;
      } catch (err) {
        error = err;
      }
    }
    expect(error).not.toBeNull();
    expect(error.code).toBe('alignment');
  });

  it('il campionamento è riproducibile con lo stesso seme e registra l\'estrazione', async () => {
    const run = async () => {
      const { engine } = await setup();
      const b = await engine.begin({
        text: 'Perché il cielo è blu?', policy: { kind: 'sample', temperature: 1.5, topP: 0.95, topK: 64 }, seed: 1234,
      });
      const out = [];
      for (let i = 0; i < 8; i++) {
        const ev = await engine.step({ sessionId: b.sessionId });
        expect(ev.selected.draw).toBeGreaterThanOrEqual(0);
        expect(ev.selected.policyProb).toBeGreaterThan(0);
        out.push(ev.selected.id);
        if (ev.finish) break;
      }
      return { out, seed: b.seed };
    };
    const a = await run();
    const c = await run();
    expect(a.seed).toBe(1234);
    expect(a.out).toEqual(c.out);
  });

  it('nessuna probabilità fissa: cambiando i logit cambiano i valori mostrati', async () => {
    const { tf, engine } = await setup();
    const b = await engine.begin({ text: 'Ciao', policy: { kind: 'greedy' } });
    const ev1 = await engine.step({ sessionId: b.sessionId });
    await engine.end();
    // altera i punteggi del modello finto: le probabilità devono seguire
    const original = tf.created.model.logitsFor.bind(tf.created.model);
    tf.created.model.logitsFor = (ids) => {
      const v = original(ids);
      for (let i = 0; i < v.length; i++) v[i] *= 0.5;
      return v;
    };
    const b2 = await engine.begin({ text: 'Ciao', policy: { kind: 'greedy' } });
    const ev2 = await engine.step({ sessionId: b2.sessionId });
    expect(ev2.selected.id).toBe(ev1.selected.id);
    expect(ev2.selected.modelProb).not.toBeCloseTo(ev1.selected.modelProb, 3);
  });
});

describe('motore: fine, stop ed errori', () => {
  it('rifiuta un limite di generazione non numerico', async () => {
    const { engine } = await setup();
    await expect(engine.begin({ text: 'Ciao', policy: { kind: 'greedy' }, maxNewTokens: Number.NaN }))
      .rejects.toMatchObject({ code: 'invalid-max-new-tokens' });
  });

  it('consente più di 128 token e conta anche quelli generati con il thinking attivo', async () => {
    const { tf, engine } = await setup({}, THINKING_SPEC);
    const newline = tf.created.tokenizer.byPiece.get('\n');
    const originalLogitsFor = tf.created.model.logitsFor.bind(tf.created.model);
    tf.created.model.logitsFor = (ids) => {
      const logits = originalLogitsFor(ids).fill(-100);
      logits[newline] = 10;
      return logits;
    };

    const begun = await engine.begin({
      text: 'Ciao', policy: { kind: 'greedy' }, maxNewTokens: 129, thinking: true,
    });
    expect(begun.maxNewTokens).toBe(129);
    expect(begun.requestedMaxNewTokens).toBe(129);
    expect(begun.limitReason).toBe('setting');
    expect(begun.thinking).toBe(true);
    expect(begun.input.generationBudget).toBe(129);
    expect(begun.input.fullText).toContain(begun.input.generationBudgetInstruction);
    let event;
    for (let i = 0; i < 129; i++) event = await engine.step({ sessionId: begun.sessionId });
    expect(event.index).toBe(128);
    expect(event.finish).toBe('length');
  });

  it('limita la sessione al contesto effettivo del modello senza cambiare i passi da un token', async () => {
    const maxContextTokens = 160;
    const { tf, engine } = await setup({
      config: { text_config: { max_position_embeddings: maxContextTokens } },
    });
    const newline = tf.created.tokenizer.byPiece.get('\n');
    const originalLogitsFor = tf.created.model.logitsFor.bind(tf.created.model);
    tf.created.model.logitsFor = (ids) => {
      const logits = originalLogitsFor(ids).fill(-100);
      logits[newline] = 10;
      return logits;
    };
    const begun = await engine.begin({
      text: 'Ciao', policy: { kind: 'greedy' }, maxNewTokens: 2048,
    });
    const available = begun.maxNewTokens;
    expect(begun.maxContextTokens).toBe(maxContextTokens);
    expect(begun.requestedMaxNewTokens).toBe(2048);
    expect(begun.input.generationBudget).toBe(available);
    expect(begun.input.ids.length + available).toBeLessThanOrEqual(maxContextTokens);
    expect(begun.limitReason).toBe('context');

    let event;
    for (let i = 0; i < available; i++) event = await engine.step({ sessionId: begun.sessionId });
    expect(event.contextLength).toBe(begun.input.ids.length + available - 1);
    expect(event.finish).toBe('length');
    expect(tf.created.model.forwardCalls).toBe(available);
  });

  it('rifiuta un input che lascia zero token di contesto per la risposta', async () => {
    const { engine } = await setup({ maxContextTokens: 8 });
    await expect(engine.begin({ text: 'Ciao', policy: { kind: 'greedy' }, maxNewTokens: 48 }))
      .rejects.toMatchObject({ code: 'input-too-long' });
  });

  it('si ferma al token di fine e lo dichiara', async () => {
    const { engine } = await setup();
    const b = await engine.begin({ text: "Qual è la capitale d'Italia?", policy: { kind: 'greedy' } });
    let ev;
    for (let i = 0; i < 40; i++) {
      ev = await engine.step({ sessionId: b.sessionId });
      if (ev.finish) break;
    }
    expect(ev.finish).toBe('eos');
    expect(ev.selected.isEos).toBe(true);
    expect(ev.selected.special).toBe(true);
    expect(ev.answerText).toBe("La capitale d'Italia è Roma.");
    await expect(engine.step({ sessionId: b.sessionId })).rejects.toThrow(/terminata/);
  });

  it('rispetta il limite di lunghezza', async () => {
    const { engine } = await setup();
    const b = await engine.begin({ text: 'Perché il cielo è blu?', policy: { kind: 'greedy' }, maxNewTokens: 3 });
    const evs = [];
    for (let i = 0; i < 3; i++) evs.push(await engine.step({ sessionId: b.sessionId }));
    expect(evs.map((e) => e.finish)).toEqual([null, null, 'length']);
  });

  it('end() libera la cache e invalida la sessione', async () => {
    const { tf, engine } = await setup();
    const b = await engine.begin({ text: 'Ciao', policy: { kind: 'greedy' } });
    await engine.step({ sessionId: b.sessionId });
    const cache = tf.created.model.caches[0];
    expect(cache.disposed).toBe(false);
    await engine.end();
    expect(cache.disposed).toBe(true);
    await expect(engine.step({ sessionId: b.sessionId })).rejects.toMatchObject({ code: 'no-session' });
  });

  it('un errore del modello viene propagato senza bloccare il motore', async () => {
    const { engine } = await setup({ failOnCall: 2 });
    const b = await engine.begin({ text: 'Ciao', policy: { kind: 'greedy' } });
    await engine.step({ sessionId: b.sessionId });
    await expect(engine.step({ sessionId: b.sessionId })).rejects.toThrow(/simulato/);
    await engine.end();
    const b2 = await engine.begin({ text: 'Ciao', policy: { kind: 'greedy' } });
    await expect(engine.step({ sessionId: b2.sessionId })).resolves.toBeTruthy();
  });

  it('senza modello caricato risponde con un errore chiaro', async () => {
    const engine = createEngine(createFakeTf(), { device: 'test' });
    expect(() => engine.tokenize({ text: 'Ciao' })).toThrow(/non è ancora caricato/);
  });
});

describe('motore: scelta del presentatore', () => {
  it('sostituisce il token scelto senza ricalcolare; il passo dopo parte dal nuovo token', async () => {
    const { tf, engine } = await setup();
    const b = await engine.begin({ text: 'Perché il cielo è blu?', policy: { kind: 'greedy' } });
    const ev1 = await engine.step({ sessionId: b.sessionId });
    const alt = ev1.candidates[2];
    const callsBefore = tf.created.model.forwardCalls;
    const ov = await engine.override({ sessionId: b.sessionId, id: alt.id });
    expect(tf.created.model.forwardCalls).toBe(callsBefore);
    expect(ov.selected.id).toBe(alt.id);
    expect(ov.selected.modelProb).toBe(alt.modelProb);
    expect(ov.selected.rank).toBe(3);
    expect(ov.override).toMatchObject({ by: 'presenter', original: { id: ev1.selected.id } });
    // probabilità del passo invariate: sono sempre quelle del modello
    expect(ov.candidates).toEqual(ev1.candidates);
    const ev2 = await engine.step({ sessionId: b.sessionId });
    const context = [...b.input.ids, alt.id];
    expect(ev2.contextLength).toBe(context.length);
    expect(ev2.processedTokens).toBe(1);
    const logits = recomputeLogits(tf, context);
    expect(ev2.candidates.map((c) => c.id)).toEqual(topKIndices(logits, ev2.candidates.length));
    expect(ev2.answerText.startsWith(ov.answerText)).toBe(true);
  });

  it('si può scegliere solo tra i candidati mostrati; scegliere la fine chiude la risposta', async () => {
    const { engine } = await setup();
    const b = await engine.begin({ text: 'Ciao', policy: { kind: 'greedy' } });
    const ev = await engine.step({ sessionId: b.sessionId });
    const notShown = 9999;
    await expect(engine.override({ sessionId: b.sessionId, id: notShown })).rejects.toMatchObject({ code: 'not-candidate' });
    const ov = await engine.override({ sessionId: b.sessionId, id: ev.candidates[1].id });
    // una seconda sostituzione conserva la scelta originale della regola
    const ov2 = await engine.override({ sessionId: b.sessionId, id: ev.candidates[0].id });
    expect(ov2.override.original.id).toBe(ev.selected.id);
    expect(ov.selected.id).toBe(ev.candidates[1].id);
  });
});

describe('motore: KV cache contro ricalcolo completo', () => {
  it('il ricalcolo completo elabora tutta la sequenza a ogni passo e dà gli stessi token', async () => {
    const run = async (cacheMode) => {
      const { tf, engine } = await setup();
      const b = await engine.begin({ text: 'Perché il cielo è blu?', policy: { kind: 'greedy' }, cacheMode });
      expect(b.cacheMode).toBe(cacheMode);
      const evs = [];
      for (let i = 0; i < 5; i++) evs.push(await engine.step({ sessionId: b.sessionId }));
      await engine.end();
      return { evs, b, caches: tf.created.model.caches };
    };
    const cached = await run('cache');
    const naive = await run('naive');
    expect(naive.evs.map((e) => e.selected.id)).toEqual(cached.evs.map((e) => e.selected.id));
    naive.evs.forEach((e, i) => {
      expect(e.cacheMode).toBe('naive');
      expect(e.cachedTokens).toBe(0);
      expect(e.processedTokens).toBe(naive.b.input.ids.length + i);
    });
    cached.evs.forEach((e, i) => expect(e.processedTokens).toBe(i === 0 ? cached.b.input.ids.length : 1));
    // in modalità ricalcolo ogni cache creata viene subito liberata
    expect(naive.caches.every((c) => c.disposed)).toBe(true);
  });
});
