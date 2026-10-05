import { describe, it, expect } from 'vitest';
import {
  MODELS, pickVariant, modelsFor, suggestModelKey, downloadBytes, filesFor,
  DEFAULT_MODEL_KEY, LIGHT_MODEL_KEY, TINY_MODEL_KEY, CPU_MODEL_KEY,
} from '../../src/core/models.js';

describe('registro dei modelli', () => {
  it('ogni modello ha revisione fissata, file con dimensioni e almeno un dispositivo', () => {
    for (const m of Object.values(MODELS)) {
      expect(m.revision).toMatch(/^[0-9a-f]{40}$/);
      expect(m.devices?.length).toBeGreaterThan(0);
      expect(Number.isSafeInteger(m.maxContextTokens) && m.maxContextTokens > 0).toBe(true);
      for (const dtype of Object.keys(m.variants)) {
        const files = filesFor(m, dtype);
        expect(Object.values(files).every((n) => Number.isInteger(n) && n > 0)).toBe(true);
        expect(downloadBytes(m, dtype)).toBeGreaterThan(1e8);
      }
    }
  });

  it('senza WebGPU si offre solo il modello per processore; con WebGPU mai', () => {
    expect(modelsFor('wasm').map((m) => m.key)).toEqual([CPU_MODEL_KEY]);
    expect(modelsFor('webgpu').map((m) => m.key)).not.toContain(CPU_MODEL_KEY);
    expect(modelsFor('webgpu').map((m) => m.key)).toEqual(expect.arrayContaining([DEFAULT_MODEL_KEY, LIGHT_MODEL_KEY, TINY_MODEL_KEY]));
  });

  it('la precisione dipende dal dispositivo e dal supporto a shader-f16', () => {
    expect(pickVariant(MODELS[DEFAULT_MODEL_KEY], { shaderF16: true })).toBe('q4f16');
    expect(pickVariant(MODELS[DEFAULT_MODEL_KEY], { shaderF16: false })).toBe('q4');
    expect(pickVariant(MODELS[CPU_MODEL_KEY], { device: 'wasm' })).toBe('q8');
  });

  it('il suggerimento segue la memoria dichiarata dal browser', () => {
    expect(suggestModelKey({ deviceMemoryGB: 32 })).toBe(DEFAULT_MODEL_KEY);
    expect(suggestModelKey({ deviceMemoryGB: 16 })).toBe(DEFAULT_MODEL_KEY);
    expect(suggestModelKey({ deviceMemoryGB: 8 })).toBe(LIGHT_MODEL_KEY);
    expect(suggestModelKey({ deviceMemoryGB: 4 })).toBe(TINY_MODEL_KEY);
    expect(suggestModelKey({})).toBe(DEFAULT_MODEL_KEY);
  });
});
