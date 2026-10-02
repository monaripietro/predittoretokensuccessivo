/**
 * Rappresentazione dei token come "tessere": testo esatto del pezzo con
 * spazi e a-capo resi visibili, ID numerico su richiesta, stile diverso
 * per token di controllo, frammenti di byte e token generati.
 */

import { el } from './dom.js';
import { ORIGIN_LABELS } from './labels.js';

/** Testo del pezzo con spazi e a-capo visibili (in grigio) senza alterarlo. */
export function pieceContent(text) {
  const frag = document.createDocumentFragment();
  let buf = '';
  const flush = () => {
    if (buf) frag.append(document.createTextNode(buf));
    buf = '';
  };
  for (const ch of text) {
    if (ch === ' ') {
      flush();
      frag.append(el('span', { class: 'ws', 'aria-hidden': 'true' }, '·'));
    } else if (ch === '\n') {
      flush();
      frag.append(el('span', { class: 'ws', 'aria-hidden': 'true' }, '↵'));
    } else if (ch === '\t') {
      flush();
      frag.append(el('span', { class: 'ws', 'aria-hidden': 'true' }, '⇥'));
    } else {
      buf += ch;
    }
  }
  flush();
  return frag;
}

/** Descrizione testuale accessibile di un pezzo (gli spazi vengono nominati). */
export function spokenPiece(text) {
  if (text === '') return 'vuoto';
  return text
    .replace(/\n/g, ' a capo ')
    .replace(/^ /, 'spazio + ')
    .replace(/ {2,}/g, (m) => ` ${m.length} spazi `)
    .trim();
}

/**
 * Tessera di un token.
 * @param {object} t { id, text, special, fragment, groupText, groupSize, groupIndex, origin, empty }
 */
export function tokenChip(t, { extraClass = '' } = {}) {
  const classes = ['tok', `o-${t.origin ?? 'user'}`, extraClass];
  let body;
  let description;
  if (t.special) {
    classes.push('ctl');
    body = el('span', { class: 'tok-text' }, t.text || `#${t.id}`);
    description = `token di controllo ${t.text}`;
  } else if (t.fragment) {
    classes.push('frag');
    const label = t.groupText && !t.pending
      ? `${t.groupText}`
      : 'byte';
    body = el('span', { class: 'tok-text' }, label, el('sub', {}, `${(t.groupIndex ?? 0) + 1}/${t.groupSize ?? '?'}`));
    description = t.pending
      ? 'frammento di byte: da solo non forma un carattere'
      : `frammento ${(t.groupIndex ?? 0) + 1} di ${t.groupSize} del carattere ${t.groupText}`;
  } else if (t.empty) {
    classes.push('empty');
    body = el('span', { class: 'tok-text' }, '∅');
    description = 'token senza testo';
  } else {
    body = el('span', { class: 'tok-text' }, pieceContent(t.text));
    description = spokenPiece(t.text);
  }
  const origin = ORIGIN_LABELS[t.origin] ?? '';
  const chip = el('span', {
    class: classes.filter(Boolean).join(' '),
    dataset: { id: String(t.id), origin: t.origin ?? '' },
    title: `${description} · ID ${t.id}${origin ? ` · ${origin}` : ''}`,
  }, body);
  // L'ID è sempre nel DOM; il contenitore con classe .show-ids lo rende visibile.
  chip.append(el('span', { class: 'tok-id' }, String(t.id)));
  chip.append(el('span', { class: 'visually-hidden' }, ` (${description}, ID ${t.id})`));
  return chip;
}
