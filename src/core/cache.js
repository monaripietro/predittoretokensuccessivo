/**
 * Stato della cache del browser per i file del modello, senza richieste
 * di rete. Transformers.js salva ogni file nella Cache Storage
 * "transformers-cache" usando come chiave l'URL esatto (con la revisione).
 */

import { filesFor, fileUrl } from './models.js';

export const CACHE_NAME = 'transformers-cache';

async function openCache(storage = globalThis.caches) {
  if (!storage) return null;
  try {
    return await storage.open(CACHE_NAME);
  } catch {
    return null;
  }
}

/** true = tutti i file presenti, false = mancano file, null = cache non disponibile. */
export async function isModelCached(model, dtype, storage = globalThis.caches) {
  const cache = await openCache(storage);
  if (!cache) return null;
  for (const path of Object.keys(filesFor(model, dtype))) {
    if (!(await cache.match(fileUrl(model, path)))) return false;
  }
  return true;
}

/** Elimina dal browser i file di un modello (tutte le precisioni). */
export async function deleteModelFiles(model, storage = globalThis.caches) {
  const cache = await openCache(storage);
  if (!cache) return 0;
  let removed = 0;
  const prefix = `https://huggingface.co/${model.id}/`;
  for (const request of await cache.keys()) {
    if (request.url.startsWith(prefix) && (await cache.delete(request))) removed += 1;
  }
  return removed;
}

export async function storageEstimate(nav = globalThis.navigator) {
  try {
    const { usage, quota } = await nav.storage.estimate();
    return { usage, quota };
  } catch {
    return null;
  }
}

/** Chiede al browser di non cancellare automaticamente i file scaricati. */
export async function requestPersistentStorage(nav = globalThis.navigator) {
  try {
    if (await nav.storage.persisted()) return true;
    return await nav.storage.persist();
  } catch {
    return false;
  }
}
