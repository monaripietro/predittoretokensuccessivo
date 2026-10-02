import { describe, it, expect } from 'vitest';
import {
  piecesOf, createPieceDecoder, makeSpecialTokenTest, candidatePiece, labelOrigins, visibleWhitespace,
} from '../../src/core/tokens.js';
import { FakeTokenizer, FAKE_VOCAB_SIZE } from '../../src/engine/fakeTf.js';

const tok = new FakeTokenizer();
const isSpecial = makeSpecialTokenTest(tok);

describe('pezzi di token (tokenizer simulato con byte)', () => {
  it('la concatenazione dei pezzi ricostruisce esattamente il testo', () => {
    const text = '<bos><|turn>user\nPerché il cielo è blu? Ciao 👋🏽 «così»  doppio\nà<turn|>\n';
    const ids = tok.encode(text);
    const pieces = piecesOf(tok, ids, isSpecial);
    expect(pieces.map((p) => p.id)).toEqual(ids);
    expect(pieces.map((p) => p.text).join('')).toBe(tok.decode(ids));
    expect(pieces.map((p) => p.text).join('')).toBe(text);
  });

  it('spazi iniziali, punteggiatura e a-capo restano nei pezzi', () => {
    const ids = tok.encode('Il cielo è blu?\n');
    const texts = piecesOf(tok, ids, isSpecial).map((p) => p.text);
    expect(texts).toEqual(['Il', ' cielo', ' è', ' blu', '?', '\n']);
  });

  it('un carattere diviso in più byte forma un gruppo dichiarato', () => {
    const ids = tok.encode('👋');
    expect(ids.length).toBe(4);
    const pieces = piecesOf(tok, ids, isSpecial);
    expect(pieces.every((p) => p.fragment && p.groupSize === 4 && p.groupText === '👋')).toBe(true);
    expect(pieces.map((p) => p.groupIndex)).toEqual([0, 1, 2, 3]);
    // solo l'ultimo frammento porta il testo, così la concatenazione resta esatta
    expect(pieces.map((p) => p.text)).toEqual(['', '', '', '👋']);
  });

  it('il decodificatore incrementale non inventa caratteri a metà', () => {
    const dec = createPieceDecoder(tok, isSpecial);
    const ids = tok.encode('aà');
    const out = [];
    const perStep = ids.map((id) => {
      const done = dec.push(id);
      out.push(...done);
      return done.length;
    });
    expect(perStep).toEqual([1, 0, 2]);
    expect(out.map((p) => p.text).join('')).toBe('aà');
  });

  it('i token speciali sono riconosciuti e non scompaiono', () => {
    const ids = tok.encode('<|turn>model\n');
    const pieces = piecesOf(tok, ids, isSpecial);
    expect(pieces[0]).toMatchObject({ id: 3, text: '<|turn>', special: true });
    expect(pieces.slice(1).every((p) => !p.special)).toBe(true);
  });

  it('candidato: mantiene lo spazio iniziale e segnala byte o ID senza testo', () => {
    const ctx = tok.encode('Il cielo');
    const blu = tok.encode(' blu')[0];
    expect(candidatePiece(tok, ctx, blu, isSpecial)).toMatchObject({ text: ' blu', fragment: false });
    const byte = tok.encode('à')[0];
    expect(candidatePiece(tok, ctx, byte, isSpecial)).toMatchObject({ text: '', fragment: true });
    // ID presente nei logit ma senza testo nel vocabolario (come i riempimenti reali)
    expect(candidatePiece(tok, ctx, FAKE_VOCAB_SIZE - 1, isSpecial)).toMatchObject({ text: '', empty: true });
    expect(candidatePiece(tok, ctx, 4, isSpecial)).toMatchObject({ text: '<turn|>', special: true });
  });
});

describe('origine dei token nell\'input', () => {
  it('distingue sistema, utente, modello di chat e controllo', () => {
    const user = 'Il cielo è blu?';
    const sys = 'Rispondi in italiano';
    const full = tok.apply_chat_template([
      { role: 'system', content: sys }, { role: 'user', content: user },
    ], { add_generation_prompt: true });
    const ids = tok.encode(full);
    const pieces = piecesOf(tok, ids, isSpecial);
    const sStart = full.indexOf(sys);
    const uStart = full.indexOf(user);
    const labelled = labelOrigins(pieces, [
      { start: sStart, end: sStart + sys.length, origin: 'system' },
      { start: uStart, end: uStart + user.length, origin: 'user' },
    ]);
    const userText = labelled.filter((p) => p.origin === 'user').map((p) => p.text).join('');
    expect(userText).toBe(user);
    expect(labelled.filter((p) => p.origin === 'system').map((p) => p.text).join('')).toBe(sys);
    expect(labelled.filter((p) => p.origin === 'control').map((p) => p.text)).toContain('<|turn>');
    expect(labelled.some((p) => p.origin === 'template' && p.text === 'user')).toBe(true);
  });

  it('un frammento di byte eredita l\'origine del suo gruppo', () => {
    const full = 'x👋y';
    const ids = tok.encode(full);
    const labelled = labelOrigins(piecesOf(tok, ids, isSpecial), [{ start: 1, end: 3, origin: 'user' }]);
    expect(labelled.filter((p) => p.fragment).every((p) => p.origin === 'user')).toBe(true);
  });
});

describe('spazi visibili', () => {
  it('mostra spazi, a-capo e tabulazioni senza perderli', () => {
    expect(visibleWhitespace(' a\n\tb')).toBe('·a↵⇥b');
  });
});
