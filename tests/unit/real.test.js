/**
 * Test con la libreria reale (rete necessaria, non eseguiti di default).
 *
 *   RUN_TOKENIZER_TESTS=1 npm run test:unit   → tokenizer reali di Gemma 4 e Qwen3 (~30 MB)
 *   RUN_REAL_TESTS=1 npm run test:unit        → anche generate() reale con un modello piccolo
 *                                               (SmolLM2-135M q4, ~185 MB, CPU in Node)
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as tf from '@huggingface/transformers';
import { createEngine } from '../../src/engine/engine.js';
import { verifyRawScores, verifyGreedyParity } from '../../src/engine/verify.js';
import { piecesOf, makeSpecialTokenTest, labelOrigins } from '../../src/core/tokens.js';
import { scanLogits, modelProbability, topKIndices } from '../../src/core/distribution.js';
import { MODELS, SYSTEM_PROMPT } from '../../src/core/models.js';

const RUN_MODEL = process.env.RUN_REAL_TESTS === '1';
const RUN_TOKENIZERS = RUN_MODEL || process.env.RUN_TOKENIZER_TESTS === '1';

const TEXT = "Perché il cielo è blu? Ciao 👋🏽 «così»,  doppio spazio\ne l'àncora.";

describe.skipIf(!RUN_TOKENIZERS)('tokenizer reali', () => {
  for (const key of ['gemma-4-e2b', 'qwen3-0.6b']) {
    const spec = MODELS[key];
    describe(spec.name, () => {
      let tok;
      let isSpecial;
      beforeAll(async () => {
        tok = await tf.AutoTokenizer.from_pretrained(spec.id, { revision: spec.revision });
        isSpecial = makeSpecialTokenTest(tok);
      }, 300_000);

      it('i pezzi ricostruiscono esattamente il testo e gli ID coincidono', () => {
        const ids = Array.from(tok.encode(TEXT, { add_special_tokens: false }), Number);
        const pieces = piecesOf(tok, ids, isSpecial);
        expect(pieces.map((p) => p.id)).toEqual(ids);
        expect(pieces.map((p) => p.text).join('')).toBe(TEXT);
        expect(tok.decode(ids, { skip_special_tokens: false })).toBe(TEXT);
        // spazi e punteggiatura restano dentro i pezzi
        expect(pieces.some((p) => p.text.startsWith(' '))).toBe(true);
        expect(pieces.some((p) => p.text.includes('?'))).toBe(true);
        expect(pieces.some((p) => p.text.includes('\n'))).toBe(true);
      });

      it('modello di chat: token di controllo riconosciuti, testo dell\'utente isolato', () => {
        const user = 'Perché il cielo è blu?';
        const full = tok.apply_chat_template(
          [{ role: 'system', content: SYSTEM_PROMPT }, { role: 'user', content: user }],
          { tokenize: false, add_generation_prompt: true, ...spec.chatTemplateOptions },
        );
        const ids = Array.from(tok.encode(full, { add_special_tokens: false }), Number);
        const pieces = piecesOf(tok, ids, isSpecial);
        expect(pieces.map((p) => p.text).join('')).toBe(full);
        const u = full.indexOf(user);
        const s = full.indexOf(SYSTEM_PROMPT);
        const labelled = labelOrigins(pieces, [
          { start: s, end: s + SYSTEM_PROMPT.length, origin: 'system' },
          { start: u, end: u + user.length, origin: 'user' },
        ]);
        expect(labelled.filter((p) => p.origin === 'user').map((p) => p.text).join('')).toBe(user);
        const controls = labelled.filter((p) => p.origin === 'control');
        expect(controls.length).toBeGreaterThan(1);
        for (const c of controls) expect(tok.decode([c.id], { skip_special_tokens: false })).toBe(c.text);
      });
    });
  }

  it('Qwen3: un\'emoji divisa in token di byte viene dichiarata come frammento', async () => {
    const spec = MODELS['qwen3-0.6b'];
    const tok = await tf.AutoTokenizer.from_pretrained(spec.id, { revision: spec.revision });
    const ids = Array.from(tok.encode('Ciao 👋', { add_special_tokens: false }), Number);
    const pieces = piecesOf(tok, ids, makeSpecialTokenTest(tok));
    const frag = pieces.filter((p) => p.fragment);
    expect(frag.length).toBeGreaterThanOrEqual(2);
    expect(frag.at(-1).groupText).toContain('👋');
    expect(pieces.map((p) => p.text).join('')).toBe('Ciao 👋');
  }, 300_000);
});

const TINY = {
  key: 'smollm2-135m-test',
  name: 'SmolLM2 135M',
  label: 'SmolLM2 135M Instruct (solo test)',
  id: 'onnx-community/SmolLM2-135M-Instruct-ONNX',
  revision: 'b8a5c0f183b78c55955a5364f610c36668b5e681',
  baseFiles: {},
  variants: { q4: { requiresF16: false, files: {} } },
  chatTemplateOptions: {},
};

describe.skipIf(!RUN_MODEL)('generate() reale (Node, CPU)', () => {
  let engine;
  beforeAll(async () => {
    engine = createEngine(tf, { device: 'cpu' });
    await engine.load({ spec: TINY, dtype: 'q4' });
  }, 600_000);

  it('per più passi consecutivi le probabilità coincidono con un forward diretto sullo stesso contesto', async () => {
    const { Tensor } = tf;
    const b = await engine.begin({ text: 'What color is the sky?', policy: { kind: 'greedy' }, maxNewTokens: 6 });
    const events = [];
    for (let i = 0; i < 6; i++) {
      const ev = await engine.step({ sessionId: b.sessionId });
      events.push(ev);
      if (ev.finish) break;
    }
    await engine.end();
    const { model } = engine._internals;
    let context = b.input.ids.slice();
    for (const ev of events) {
      // forward completo senza cache: indipendente dal percorso della demo
      const out = await model({
        input_ids: new Tensor('int64', BigInt64Array.from(context, BigInt), [1, context.length]),
        attention_mask: new Tensor('int64', new BigInt64Array(context.length).fill(1n), [1, context.length]),
      });
      const v = out.logits.dims.at(-1);
      const n = out.logits.dims.at(-2);
      const logits = out.logits.data.slice((n - 1) * v, n * v);
      const stats = scanLogits(logits);
      expect(ev.candidates.map((c) => c.id)).toEqual(topKIndices(logits, ev.candidates.length));
      for (const c of ev.candidates) {
        expect(c.modelProb).toBeCloseTo(modelProbability(logits, c.id, stats), 4);
      }
      expect(ev.selected.id).toBe(ev.candidates[0].id);
      context = context.concat(ev.selected.id);
    }
  }, 300_000);

  it('i logit ricevuti dal processore sono quelli grezzi del modello', async () => {
    const r = await verifyRawScores(engine, tf, 'What color is the sky?');
    expect(r.maxAbsDiff).toBe(0);
    expect(r.argmaxEqual).toBe(true);
  }, 300_000);

  it('greedy passo per passo = generate() nativo della libreria', async () => {
    const r = await verifyGreedyParity(engine, tf, 'Name three fruits.', 12);
    expect(r.stepwise.length).toBeGreaterThan(3);
    expect(r.equal).toBe(true);
  }, 300_000);

  it('end() chiude la sessione; un nuovo avvio riparte pulito', async () => {
    const b = await engine.begin({ text: 'Hi', policy: { kind: 'greedy' } });
    const e1 = await engine.step({ sessionId: b.sessionId });
    await engine.end();
    await expect(engine.step({ sessionId: b.sessionId })).rejects.toMatchObject({ code: 'no-session' });
    const b2 = await engine.begin({ text: 'Hi', policy: { kind: 'greedy' } });
    const e2 = await engine.step({ sessionId: b2.sessionId });
    expect(e2.processedTokens).toBe(b2.input.ids.length);
    expect(e2.selected.id).toBe(e1.selected.id);
    await engine.end();
  }, 300_000);
});
