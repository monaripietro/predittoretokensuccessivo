/**
 * Grafico "Candidati per il prossimo token": barre orizzontali su scala
 * fissa 0–100% (una barra del 2% appare piccola, come deve), valori in
 * testo accanto alla barra, ultima riga = massa di tutti gli altri token.
 * In modalità estrazione una seconda barra sottile mostra la probabilità
 * di estrazione (con legenda). Nessun valore viene inventato: tutto viene
 * dall'evento del passo.
 */

import { el, clear } from './dom.js';
import { formatPercent } from '../core/distribution.js';
import { pieceContent, spokenPiece } from './tokenView.js';
import { formatNumber } from './labels.js';

function candidateLabel(c) {
  const p = c.piece ?? {};
  if (p.special) return el('span', { class: 'cand-text ctl' }, p.text || `#${c.id}`);
  if (p.fragment) return el('span', { class: 'cand-text frag' }, 'byte', el('sub', {}, 'parziale'));
  if (p.empty || p.text === '') return el('span', { class: 'cand-text empty' }, '∅');
  return el('span', { class: 'cand-text' }, pieceContent(p.text));
}

function describe(c) {
  const p = c.piece ?? {};
  if (p.special) return `token di controllo ${p.text}`;
  if (p.fragment) return 'frammento di byte di un carattere';
  if (p.empty || p.text === '') return 'token senza testo';
  return spokenPiece(p.text);
}

function bar(value, cls) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  // larghezza minima visibile solo per valori positivi (il numero resta esatto)
  const width = value > 0 ? `max(2px, ${pct}%)` : '0';
  return el('span', { class: `bar ${cls}`, style: `width: ${width}` });
}

/**
 * @param {HTMLElement} root <ol> del grafico
 * @param {object} step evento del passo
 * @param {{ phase: 'scores' | 'selected', showIds: boolean }} view
 */
export function renderChart(root, step, { phase = 'selected' } = {}) {
  clear(root);
  const sampling = step.policy?.kind === 'sample';
  root.classList.toggle('sampling', sampling);
  root.classList.toggle('decided', phase === 'selected');
  step.candidates.forEach((c, i) => {
    const chosen = c.id === step.selected.id;
    const row = el('li', {
      class: `row${chosen && phase === 'selected' ? ' chosen' : ''}${step.override && chosen && phase === 'selected' ? ' picked-by-user' : ''}${step.override?.original?.id === c.id ? ' policy-choice' : ''}`,
      dataset: { id: String(c.id), rank: String(i + 1), modelProb: String(c.modelProb) },
      title: `ID ${c.id} · logit ${c.logit === null ? 'non valido' : c.logit.toFixed(3)} · probabilità ${formatPercent(c.modelProb, 3)}`
        + (sampling ? ` · estrazione ${formatPercent(c.policyProb, 3)}` : ''),
    });
    row.append(
      el('span', { class: 'rank', 'aria-hidden': 'true' }, String(i + 1)),
      el('span', { class: 'cand' }, candidateLabel(c), el('span', { class: 'cand-id' }, `ID ${c.id}`)),
      el('span', { class: 'track' },
        bar(c.modelProb, 'model'),
        sampling ? bar(c.policyProb ?? 0, 'policy') : null),
      el('span', { class: 'val' },
        el('span', { class: 'v-model' }, formatPercent(c.modelProb)),
        sampling ? el('span', { class: 'v-policy' }, formatPercent(c.policyProb ?? 0)) : null),
      el('span', { class: 'mark', 'aria-hidden': 'true' }, chosen && phase === 'selected' ? '✓' : ''),
      el('span', { class: 'visually-hidden' },
        `${i + 1}: ${describe(c)}, ${formatPercent(c.modelProb)}${chosen && phase === 'selected' ? ', scelto' : ''}`),
    );
    root.append(row);
  });
  // Tutti gli altri token del vocabolario
  const shownPolicy = sampling ? step.candidates.reduce((s, c) => s + (c.policyProb ?? 0), 0) : 0;
  const otherPolicy = sampling ? Math.max(0, 1 - shownPolicy) : 0;
  const other = el('li', { class: 'row other', dataset: { modelProb: String(step.other.modelProb) } });
  other.append(
    el('span', { class: 'rank', 'aria-hidden': 'true' }, '…'),
    el('span', { class: 'cand' }, el('span', { class: 'cand-text muted' }, `altri ${formatNumber(step.other.count)} token`)),
    el('span', { class: 'track' }, bar(step.other.modelProb, 'model rest'), sampling ? bar(otherPolicy, 'policy') : null),
    el('span', { class: 'val' },
      el('span', { class: 'v-model' }, formatPercent(step.other.modelProb)),
      sampling ? el('span', { class: 'v-policy' }, formatPercent(otherPolicy)) : null),
    el('span', { class: 'mark' }, ''),
  );
  root.append(other);
  // Il token scelto può non essere tra i candidati mostrati (solo in estrazione).
  if (phase === 'selected' && !step.candidates.some((c) => c.id === step.selected.id)) {
    root.append(el('li', { class: `row chosen outside${step.override ? ' picked-by-user' : ''}` },
      el('span', { class: 'rank' }, String(step.selected.rank)),
      el('span', { class: 'cand' }, candidateLabel({ ...step.selected, piece: step.selected.piece }),
        el('span', { class: 'cand-id' }, `ID ${step.selected.id}`)),
      el('span', { class: 'track' }, bar(step.selected.modelProb, 'model'), bar(step.selected.policyProb ?? 0, 'policy')),
      el('span', { class: 'val' }, el('span', { class: 'v-model' }, formatPercent(step.selected.modelProb)),
        el('span', { class: 'v-policy' }, formatPercent(step.selected.policyProb ?? 0))),
      el('span', { class: 'mark' }, '✓')));
  }
}
