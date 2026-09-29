import { describe, it, expect } from 'vitest';
import { MockTokenizer } from '../../src/mockTokenizer.js';

describe('MockTokenizer', () => {
  it('tokenizza e decodifica in modo round-trip', () => {
    const t = new MockTokenizer();
    const ids = t.encode('Il cielo è blu');
    expect(ids.length).toBe(4);
    expect(t.decode(ids)).toBe('Il cielo è blu');
  });

  it('produce token id stabili per lo stesso testo', () => {
    const a = new MockTokenizer();
    const b = new MockTokenizer();
    expect(a.encode('ciao')).toEqual(b.encode('ciao'));
  });

  it('conta correttamente i token e ignora spazi extra', () => {
    const t = new MockTokenizer();
    expect(t.encode('   una   parola   ').length).toBe(2);
    expect(t.encode('')).toEqual([]);
  });

  it('idToToken è coerente con tokenToId', () => {
    const t = new MockTokenizer();
    const id = t.tokenToId('blu');
    expect(t.idToToken(id)).toBe('blu');
  });
});
