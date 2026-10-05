import { describe, it, expect } from 'vitest';
import {
  policyLabel, chartCaption, choiceSentence, finishLabel, quoteToken,
} from '../../src/ui/labels.js';

const greedy = { kind: 'greedy' };
const sample = { kind: 'sample', temperature: 0.8, topP: 0.95, topK: 64 };

function step(policy, selected) {
  return { policy, selected: { id: 42, modelProb: 0.25, policyProb: 0.4, rank: 2, isEos: false, ...selected } };
}

describe('etichette: greedy contro estrazione', () => {
  it('la regola di scelta è dichiarata in modo diverso', () => {
    expect(policyLabel(greedy)).toMatch(/più probabile.*greedy/);
    expect(policyLabel(sample)).toMatch(/estrazione casuale pesata/);
    expect(policyLabel(sample)).toContain('temperatura 0,8');
    expect(policyLabel(sample)).toContain('top-p 0,95');
    expect(policyLabel(sample)).toContain('64 candidati');
  });

  it('la didascalia dice cosa rappresentano le barre, mai "probabilità della parola"', () => {
    const g = chartCaption(greedy, 262144);
    expect(g).toContain('262.144');
    expect(g).toMatch(/prima di qualunque regola di scelta/);
    expect(g).not.toMatch(/arancion/);
    const s = chartCaption(sample, 262144);
    expect(s).toMatch(/Barre arancioni: probabilità di estrazione/);
    expect(s).toMatch(/rinormalizzate/);
    expect(`${g} ${s}`).not.toMatch(/parol/i);
  });

  it('greedy: il token scelto è il più probabile', () => {
    const text = choiceSentence(step(greedy, { rank: 1 }), '«blu»');
    expect(text).toMatch(/^Scelto «blu» \(ID 42\): è il più probabile, 25,0%\.$/);
  });

  it('estrazione: può non essere il primo, e mostra entrambe le probabilità', () => {
    const text = choiceSentence(step(sample, {}), '«azzurro»');
    expect(text).toMatch(/^Estratto «azzurro»/);
    expect(text).toContain('2° in classifica');
    expect(text).toContain('probabilità del modello 25,0%');
    expect(text).toContain('probabilità di estrazione 40,0%');
  });

  it('il token di fine è dichiarato come tale', () => {
    expect(choiceSentence(step(greedy, { isEos: true }), '«<eos>»')).toMatch(/^Scelto il token di fine/);
    expect(choiceSentence(step(sample, { isEos: true }), '«<eos>»')).toMatch(/^Estratto il token di fine/);
  });
});

describe('etichette: fine e token', () => {
  it('distingue fine naturale, limite, stop ed errore', () => {
    expect(finishLabel('eos')).toMatch(/token di fine/);
    expect(finishLabel('length', { maxNewTokens: 48 })).toContain('48 token');
    expect(finishLabel('length', {
      maxNewTokens: 24, maxContextTokens: 40, limitReason: 'context',
    })).toMatch(/contesto del modello \(40 token complessivi\).*24 token generati/);
    expect(finishLabel('stopped')).toMatch(/fermata/);
    expect(finishLabel('error')).toMatch(/errore/);
  });

  it('rende visibili spazi iniziali e a-capo nelle citazioni', () => {
    expect(quoteToken(' blu')).toBe('«·blu»');
    expect(quoteToken('\n')).toBe('«↵»');
    expect(quoteToken('a b')).toBe('«a b»');
  });
});

describe('etichette: scelta del presentatore', () => {
  it('dichiara che la scelta è dell’utente e quale sarebbe stata quella della regola', () => {
    const step = {
      policy: { kind: 'greedy' },
      selected: { id: 7, modelProb: 0.05, rank: 3, isEos: false },
      override: { by: 'presenter', original: { id: 42, modelProb: 0.8 } },
    };
    const text = choiceSentence(step, '«azzurro»', '«blu»');
    expect(text).toMatch(/^Scelto da te: «azzurro» \(ID 7, 3° in classifica, probabilità del modello 5,0%\)\./);
    expect(text).toContain('La regola avrebbe scelto «blu» (80,0%).');
  });
});
