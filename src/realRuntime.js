/**
 * Runtime reale basato su Transformers.js (ONNX in-browser).
 * Modello: onnx-community/Qwen2.5-0.5B-Instruct (Apache-2.0, ~400MB in q4).
 * Backend: WebGPU quando disponibile, fallback WASM.
 *
 * Nota tecnica: Transformers.js viene importato a runtime dall'ESM CDN
 * (jsdelivr) invece di essere bundlato da Vite: onnxruntime-web richiede
 * file .wasm/.mjs esterni che il bundling rompe. In Node (test) si usa
 * il pacchetto npm locale.
 *
 * Contratto (stesso del mock):
 *  - init() -> { backend, modelInfo }
 *  - encode / decode / idToToken / eosId
 *  - nextLogits({ ids, cache }) -> { logits, cache, newTokens, contextLength }
 * La cache è la past_key_values reale restituita dal forward pass.
 */

const CDN_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
const IS_NODE = typeof window === 'undefined' && typeof process !== 'undefined';

export const MODEL_CONFIG = {
  id: 'onnx-community/Qwen2.5-0.5B-Instruct',
  dtypeWebGPU: 'q4f16',
  dtypeWasm: 'q4',
  contextLimit: 32768,
  vocabSize: 151936,
  tokenizerName: 'Qwen2 BPE',
  fallback: {
    id: 'onnx-community/SmolLM2-360M-Instruct-ONNX',
    dtypeWasm: 'q4',
    contextLimit: 8192,
    vocabSize: 49152,
    tokenizerName: 'SmolLM2 BPE',
  },
  // Dispositivi mobile senza WebGPU: 360M in WASM supera spesso la memoria
  // disponibile su smartphone, quindi si usa direttamente il modello 135M.
  tiny: {
    id: 'onnx-community/SmolLM2-135M-Instruct-ONNX',
    dtypeWasm: 'q4',
    contextLimit: 8192,
    vocabSize: 49152,
    tokenizerName: 'SmolLM2 BPE',
  },
};

let transformersModule = null;

async function loadTransformers() {
  if (transformersModule) return transformersModule;
  if (!IS_NODE) {
    try {
      transformersModule = await import(/* @vite-ignore */ CDN_URL);
      return transformersModule;
    } catch {
      // rete o CDN non disponibili: prova il pacchetto npm (bundlato)
    }
  }
  transformersModule = await import('@huggingface/transformers');
  return transformersModule;
}

function isMobileLike() {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') return false;
  const coarse = window.matchMedia?.('(pointer: coarse)')?.matches ?? false;
  const touch = (navigator.maxTouchPoints ?? 0) > 0;
  const smallScreen = Math.min(window.screen?.width ?? 9999, window.screen?.height ?? 9999) < 820;
  return coarse && touch && smallScreen;
}

export function makeProgressTracker(onProgress) {
  const files = new Map();
  let lastEmit = 0;
  return (data) => {
    if (!data || onProgress == null) return;
    const { status, file, progress = 0, loaded = 0, total = 0 } = data;
    if (file) {
      if (status === 'done' || progress >= 100) {
        files.set(file, { loaded: 1, total: 1, done: true });
      } else if (total > 0) {
        files.set(file, { loaded, total, done: false });
      } else if (progress > 0) {
        const prev = files.get(file) ?? { loaded: 0, total: 100, done: false };
        files.set(file, { loaded: progress, total: 100, done: false });
      }
    }
    let sumLoaded = 0;
    let sumTotal = 0;
    let pending = 0;
    for (const f of files.values()) {
      sumLoaded += f.loaded;
      sumTotal += f.total;
      if (!f.done) pending += 1;
    }
    if (sumTotal <= 0) return;
    const percent = Math.min(100, Math.round((sumLoaded / sumTotal) * 100));
    const now = Date.now();
    if (percent < 100 && now - lastEmit < 250 && pending > 0) return;
    lastEmit = now;
    onProgress({ percent, pending, status: status ?? 'download' });
  };
}

export class RealRuntime {
  constructor(modelId = MODEL_CONFIG.id) {
    this.modelId = modelId;
    this.dtype = null;
    this.eosId = null;
    this.backend = 'wasm';
    this.modelInfo = modelId;
  }

  async init(onProgress) {
    const { AutoTokenizer, AutoModelForCausalLM, Tensor } = await loadTransformers();
    this.Tensor = Tensor;
    const tracker = makeProgressTracker(onProgress);
    this.tokenizer = await AutoTokenizer.from_pretrained(this.modelId, { progress_callback: tracker });
    this.eosId = this.tokenizer.eos_token_id;
    if (typeof this.eosId !== 'number') this.eosId = this.tokenizer.eos_token_id?.[0] ?? 50256;

    let webgpuOk = false;
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        webgpuOk = !!adapter;
      } catch {
        webgpuOk = false;
      }
    }

    if (webgpuOk) {
      try {
        this.dtype = MODEL_CONFIG.dtypeWebGPU;
        this.model = await AutoModelForCausalLM.from_pretrained(this.modelId, {
          device: 'webgpu',
          dtype: this.dtype,
          progress_callback: tracker,
        });
        this.backend = 'WebGPU';
        this.contextLimit = MODEL_CONFIG.contextLimit;
        this.modelInfo = `${this.modelId} (${this.dtype})`;
        return this.describe();
      } catch {
        webgpuOk = false;
      }
    }

    // WASM / Node: modello leggero. Qwen2.5-0.5B in WASM eccede la memoria
    // disponibile in molti browser e il crash del tab non è intercettabile,
    // quindi il backend senza WebGPU usa direttamente un modello più piccolo.
    // Su smartphone (nessun WebGPU, pointer coarse) anche il 360M è troppo grande:
    // si seleziona il 135M prima del caricamento.
    const profile = [MODEL_CONFIG, MODEL_CONFIG.fallback, MODEL_CONFIG.tiny]
      .find((c) => c.id === this.modelId)
      ?? MODEL_CONFIG.fallback;
    this.modelId = profile.id;
    this.tokenizer = await AutoTokenizer.from_pretrained(this.modelId);
    this.eosId = this.tokenizer.eos_token_id;
    if (typeof this.eosId !== 'number') this.eosId = this.tokenizer.eos_token_id?.[0] ?? 2;
    this.dtype = profile.dtypeWasm;
    this.contextLimit = profile.contextLimit;
    const device = IS_NODE ? 'cpu' : 'wasm';
    this.model = await AutoModelForCausalLM.from_pretrained(this.modelId, {
      device,
      dtype: this.dtype,
      progress_callback: tracker,
    });
    this.backend = IS_NODE ? 'WASM (cpu)' : 'WASM';
    this.modelInfo = `${this.modelId} (${this.dtype})`;
    return this.describe();
  }

  describe() {
    const config = [MODEL_CONFIG, MODEL_CONFIG.fallback, MODEL_CONFIG.tiny]
      .find((c) => c.id === this.modelId) ?? MODEL_CONFIG;
    return {
      backend: this.backend,
      modelInfo: this.modelInfo,
      dtype: this.dtype,
      modelId: this.modelId,
      tokenizerName: config.tokenizerName,
      vocabSize: config.vocabSize,
      contextLimit: config.contextLimit,
    };
  }

  encode(text) {
    const ids = this.tokenizer.encode(text);
    return Array.from(ids.data ?? ids);
  }

  decode(ids) {
    return this.tokenizer.decode(ids, { skip_special_tokens: false });
  }

  idToToken(id) {
    return this.tokenizer.decode([id], { skip_special_tokens: false });
  }

  /**
   * Forward pass. Con cache: passa solo i token nuovi.
   * Senza cache (naive): ricalcola l'intera sequenza.
   */
  async nextLogits({ ids, cache = null }) {
    if (!ids || ids.length === 0) throw new Error('Sequenza vuota');
    let inputIds = ids;
    let newTokens = ids.length;
    if (cache && cache.past && cache.length > 0 && cache.length < ids.length && cache.valid) {
      inputIds = ids.slice(cache.length);
      newTokens = inputIds.length;
    } else if (cache && cache.valid) {
      cache = null;
    }
    const { Tensor } = this;
    const tensor = new Tensor('int64', BigInt64Array.from(inputIds.map((x) => BigInt(x))), [1, inputIds.length]);
    const past = cache && cache.valid ? cache.past : null;
    const maskLen = inputIds.length + (past ? cache.length : 0);
    const attention = new Tensor(
      'int64',
      BigInt64Array.from({ length: maskLen }, () => 1n),
      [1, maskLen],
    );
    const output = await this.model.forward({
      input_ids: tensor,
      attention_mask: attention,
      past_key_values: past,
    });
    const logitsTensor = output.logits ?? output[0];
    const vocab = logitsTensor.dims.at(-1);
    const seqLen = logitsTensor.dims.at(-2);
    const logitsArr = Array.from(logitsTensor.data.slice((seqLen - 1) * vocab, seqLen * vocab));
    let pastOut = output.past_key_values ?? null;
    if (!pastOut) {
      const presentEntries = Object.entries(output).filter(([k]) => k.startsWith('present.'));
      pastOut = Object.fromEntries(presentEntries.map(([k, v]) => [
        k.replace(/^present\./, 'past_key_values.'),
        v,
      ]));
    }
    const hasPast = pastOut && Object.keys(pastOut).length > 0;
    return {
      logits: logitsArr,
      cache: { past: hasPast ? pastOut : null, length: ids.length, valid: hasPast },
      newTokens,
      contextLength: ids.length,
    };
  }
}
