import { describe, it, expect } from 'vitest';
import { createEngine } from '../../src/engine/engine.js';
import { createFakeTf, MOCK_SPEC } from '../../src/engine/fakeTf.js';
import { scanLogits, modelProbability, topKIndices } from '../../src/core/distribution.js';

async function setup(modelOptions = {}) {
  const tf = createFakeTf(modelOptions);
  const engine = createEngine(tf, { device: 'test' });
  await engine.load({ spec: MOCK_SPEC, dtype: 'mock' });
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

  it('applica campionamento e modalità di calcolo aggiornati tra i passi', async () => {
    const { engine } = await setup();
    const b = await engine.begin({ text: 'Perché il cielo è blu?', policy: { kind: 'greedy' } });
    const first = await engine.step({ sessionId: b.sessionId });
    expect(first.cacheMode).toBe('cache');

    await engine.updateSettings({
      sessionId: b.sessionId,
      policy: { kind: 'sample', temperature: 1.4, topP: 0.4, topK: 64 },
      cacheMode: 'naive',
    });
    const second = await engine.step({ sessionId: b.sessionId });
    expect(second.policy).toMatchObject({ kind: 'sample', temperature: 1.4, topP: 0.4 });
    expect(second.selected.draw).toBeGreaterThanOrEqual(0);
    expect(second.cacheMode).toBe('naive');
    expect(second.cachedTokens).toBe(0);
    expect(second.processedTokens).toBe(b.input.ids.length + 1);

    await engine.updateSettings({ sessionId: b.sessionId, cacheMode: 'cache' });
    const third = await engine.step({ sessionId: b.sessionId });
    expect(third.cacheMode).toBe('cache');
    expect(third.cachedTokens).toBe(0);
    expect(third.processedTokens).toBe(b.input.ids.length + 2);
    const fourth = await engine.step({ sessionId: b.sessionId });
    expect(fourth.cachedTokens).toBe(b.input.ids.length + 2);
    expect(fourth.processedTokens).toBe(1);
  });
});
