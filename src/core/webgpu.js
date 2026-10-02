/**
 * Rilevamento WebGPU prima di qualunque download. Restituisce dati reali
 * dell'adattatore; non stima memoria o prestazioni.
 */

/** Soglie minime: sotto questi limiti i pesi quantizzati non entrano nei buffer GPU. */
export const MIN_LIMITS = Object.freeze({
  maxBufferSize: 512 * 1024 * 1024,
  maxStorageBufferBindingSize: 256 * 1024 * 1024,
});

export async function probeWebGPU(nav = globalThis.navigator) {
  if (!nav?.gpu) {
    return { ok: false, reason: 'no-webgpu' };
  }
  let adapter = null;
  try {
    adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch {
    adapter = null;
  }
  if (!adapter) return { ok: false, reason: 'no-adapter' };
  const limits = {
    maxBufferSize: Number(adapter.limits?.maxBufferSize ?? 0),
    maxStorageBufferBindingSize: Number(adapter.limits?.maxStorageBufferBindingSize ?? 0),
  };
  const info = adapter.info ?? {};
  const result = {
    ok: true,
    shaderF16: Boolean(adapter.features?.has?.('shader-f16')),
    limits,
    adapter: {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      description: info.description ?? '',
      isFallbackAdapter: Boolean(info.isFallbackAdapter ?? adapter.isFallbackAdapter),
    },
    deviceMemoryGB: nav.deviceMemory ?? null,
  };
  if (result.adapter.isFallbackAdapter) {
    return { ...result, ok: false, reason: 'software-adapter' };
  }
  if (limits.maxBufferSize < MIN_LIMITS.maxBufferSize
    || limits.maxStorageBufferBindingSize < MIN_LIMITS.maxStorageBufferBindingSize) {
    return { ...result, ok: false, reason: 'limits' };
  }
  return result;
}

export const PROBE_MESSAGES = {
  'no-webgpu': 'Questo browser non offre WebGPU, che serve per i modelli più capaci. Per usarli apri la pagina con Chrome o Edge aggiornati su computer.',
  'no-adapter': 'WebGPU è presente ma nessuna scheda grafica è disponibile (può essere disattivata o bloccata dal sistema). Prova un altro computer o aggiorna i driver.',
  'software-adapter': 'È disponibile solo un adattatore WebGPU software: il modello sarebbe troppo lento. Prova un computer con una scheda grafica supportata.',
  limits: 'La scheda grafica ha limiti di memoria troppo bassi per questo modello.',
};
