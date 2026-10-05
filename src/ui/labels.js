/**
 * Testi dell'interfaccia che dipendono dai dati: regola di scelta, token
 * scelto, motivo di fine. Funzioni pure, verificate dai test.
 */

import { formatPercent } from '../core/distribution.js';

/** Token tra virgolette, con spazi iniziali/finali e a-capo resi visibili come nelle tessere. */
export function quoteToken(text) {
  const visible = String(text)
    .replace(/^ +| +$/g, (m) => '·'.repeat(m.length))
    .replace(/\n/g, '↵');
  return `«${visible}»`;
}

export function formatNumber(n) {
  return Number(n).toLocaleString('it-IT');
}

function decimal(x, digits = 2) {
  return Number(x).toFixed(digits).replace(/\.?0+$/, '').replace('.', ',');
}

/** Descrizione breve della regola di scelta, sempre visibile sotto il grafico. */
export function policyLabel(policy) {
  if (!policy || policy.kind === 'greedy') {
    return 'Regola di scelta: sempre il token più probabile (decodifica greedy).';
  }
  return `Regola di scelta: estrazione casuale pesata — temperatura ${decimal(policy.temperature)}, `
    + `al massimo ${policy.topK} candidati, top-p ${decimal(policy.topP)}.`;
}

/** Che cosa rappresentano le barre, in base alla regola. */
export function chartCaption(policy, vocabSize) {
  const vocab = vocabSize ? formatNumber(vocabSize) : '…';
  const base = `Barre viola: probabilità calcolate dal modello per ciascun token, su tutto il vocabolario (${vocab} token), prima di qualunque regola di scelta.`;
  if (!policy || policy.kind === 'greedy') return base;
  return `${base} Barre arancioni: probabilità di estrazione effettivamente usate, dopo temperatura, limite dei candidati e top-p, rinormalizzate tra i token ammessi.`;
}

function ordinal(n) {
  return `${n}°`;
}

/** Frase sul token scelto in un passo. */
export function choiceSentence(step, tokenLabel, originalLabel = null) {
  const s = step.selected;
  const pModel = formatPercent(s.modelProb);
  if (step.override) {
    const o = step.override.original;
    const rule = originalLabel ? ` La regola avrebbe scelto ${originalLabel} (${formatPercent(o.modelProb)}).` : '';
    return `Scelto da te: ${tokenLabel} (ID ${s.id}, ${ordinal(s.rank)} in classifica, probabilità del modello ${pModel}).${rule}`;
  }
  if (s.isEos) {
    const verb = step.policy.kind === 'greedy' ? 'Scelto' : 'Estratto';
    return `${verb} il token di fine ${tokenLabel} (ID ${s.id}, ${pModel}): la risposta termina qui.`;
  }
  if (step.policy.kind === 'greedy') {
    return `Scelto ${tokenLabel} (ID ${s.id}): è il più probabile, ${pModel}.`;
  }
  const where = s.rank === 1 ? 'il più probabile' : `${ordinal(s.rank)} in classifica`;
  return `Estratto ${tokenLabel} (ID ${s.id}): ${where}; probabilità del modello ${pModel}, `
    + `probabilità di estrazione ${formatPercent(s.policyProb)}.`;
}

export function finishLabel(reason, { maxNewTokens, maxContextTokens, limitReason } = {}) {
  switch (reason) {
    case 'eos':
      return 'Il modello ha scelto il token di fine: la risposta è completa.';
    case 'length':
      if (limitReason === 'context') {
        return `Raggiunto il limite di contesto del modello (${formatNumber(maxContextTokens)} token complessivi): la risposta si è fermata dopo ${formatNumber(maxNewTokens)} token generati.`;
      }
      return maxNewTokens
        ? `Raggiunto il limite di ${formatNumber(maxNewTokens)} token impostato nell'app: la risposta è stata interrotta.`
        : "Raggiunto il limite di lunghezza impostato dall'app: la risposta è stata interrotta.";
    case 'stopped':
      return 'Generazione fermata.';
    case 'error':
      return 'La generazione si è interrotta per un errore.';
    default:
      return '';
  }
}

export const ORIGIN_LABELS = {
  user: 'il tuo testo',
  system: "istruzione aggiunta dall'app",
  template: "formato della chat aggiunto dall'app",
  control: 'token di controllo',
  mixed: "in parte tuo, in parte dell'app",
  generated: 'generato dal modello',
};
