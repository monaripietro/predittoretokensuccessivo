/**
 * Web Worker di inferenza. Tutto il lavoro pesante (tokenizer, modello,
 * calcolo delle probabilità) avviene qui, così l'interfaccia resta fluida.
 *
 * Protocollo: { id, type, payload } → { id, ok, result | error }
 * Eventi spontanei: { type: 'progress', payload }
 */

import * as tf from '@huggingface/transformers';
// Runtime ONNX servito dallo stesso sito (stessa versione usata da Transformers.js),
// non da una CDN esterna. Vite emette un solo file per ciascuno.
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import { createEngine } from './engine.js';
import { verifyRawScores, verifyGreedyParity } from './verify.js';
import { probeWebGPU } from '../core/webgpu.js';

let engine = null;

function configureRuntime() {
  tf.env.allowLocalModels = false;
  tf.env.useBrowserCache = true;
  if (tf.env.backends?.onnx?.wasm) {
    tf.env.backends.onnx.wasm.wasmPaths = {
      mjs: new URL(ortMjsUrl, self.location.href).href,
      wasm: new URL(ortWasmUrl, self.location.href).href,
    };
  }
}

const handlers = {
  async probe() {
    return probeWebGPU(globalThis.navigator);
  },
  async load(payload) {
    configureRuntime();
    engine = createEngine(tf, {
      device: 'webgpu',
      onProgress: (p) => self.postMessage({ type: 'progress', payload: p }),
    });
    return engine.load(payload);
  },
  async warmup() {
    return requireEngine().warmup();
  },
  async tokenize(payload) {
    return requireEngine().tokenize(payload);
  },
  async begin(payload) {
    return requireEngine().begin(payload);
  },
  async step(payload) {
    return requireEngine().step(payload);
  },
  async override(payload) {
    return requireEngine().override(payload);
  },
  async end() {
    return requireEngine().end();
  },
  async unload() {
    if (engine) await engine.unload();
    engine = null;
    return true;
  },
  async verifyRaw({ text }) {
    return verifyRawScores(requireEngine(), tf, text);
  },
  async verifyParity({ text, n }) {
    return verifyGreedyParity(requireEngine(), tf, text, n);
  },
};

function requireEngine() {
  if (!engine) {
    const e = new Error('Il modello non è ancora caricato.');
    e.code = 'not-loaded';
    throw e;
  }
  return engine;
}

self.onmessage = async (event) => {
  const { id, type, payload } = event.data ?? {};
  const handler = handlers[type];
  if (!handler) {
    self.postMessage({ id, ok: false, error: { code: 'unknown', message: `Richiesta sconosciuta: ${type}` } });
    return;
  }
  try {
    const result = await handler(payload ?? {});
    const transfer = result?.rawLogits?.buffer ? [result.rawLogits.buffer] : [];
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: { code: err?.code ?? 'error', message: String(err?.message ?? err) },
    });
  }
};
