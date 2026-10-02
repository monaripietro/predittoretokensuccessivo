/**
 * Modelli supportati. Revisioni fissate a un commit preciso del repository
 * Hugging Face, così che file, dimensioni e comportamento siano riproducibili.
 * Le dimensioni sono quelle dichiarate dall'API Hugging Face per quei file
 * a quella revisione (verificate il 2026-10-02), non stime. Sono elencati
 * solo i file che il browser scarica davvero (verificato sulla Cache Storage).
 */

export const SYSTEM_PROMPT = 'Rispondi in italiano, in modo breve e semplice: al massimo due frasi.';

const HUB = 'https://huggingface.co';

export const MODELS = {
  'gemma-4-e2b': {
    key: 'gemma-4-e2b',
    name: 'Gemma 4 E2B',
    label: 'Gemma 4 E2B (instruction-tuned, solo testo)',
    id: 'onnx-community/gemma-4-E2B-it-ONNX',
    revision: '9f4bef82ea6e296bc69f8a2f5939f73af81b07a6',
    license: 'Apache-2.0',
    // Caricato come ForCausalLM: Transformers.js scarica solo embed_tokens e
    // decoder_model_merged, non gli encoder di immagini e audio.
    baseFiles: {
      'tokenizer_config.json': 18807,
      'tokenizer.json': 19439251,
      'config.json': 5549,
      'generation_config.json': 238,
    },
    variants: {
      q4f16: {
        requiresF16: true,
        files: {
          'onnx/decoder_model_merged_q4f16.onnx': 673231,
          'onnx/decoder_model_merged_q4f16.onnx_data': 1519700992,
          'onnx/embed_tokens_q4f16.onnx': 5621,
          'onnx/embed_tokens_q4f16.onnx_data': 1590689792,
        },
      },
      q4: {
        requiresF16: false,
        files: {
          'onnx/decoder_model_merged_q4.onnx': 647599,
          'onnx/decoder_model_merged_q4.onnx_data': 1864102912,
          'onnx/embed_tokens_q4.onnx': 5142,
          'onnx/embed_tokens_q4.onnx_data': 1762656256,
        },
      },
    },
    chatTemplateOptions: { enable_thinking: false },
    tier: 'recommended',
    devices: ['webgpu'],
  },
  'qwen3-0.6b': {
    key: 'qwen3-0.6b',
    name: 'Qwen3 0.6B',
    label: 'Qwen3 0.6B (instruct)',
    id: 'onnx-community/Qwen3-0.6B-ONNX',
    revision: 'da1453100cf3ff33ef56d17983fc7a8648706db6',
    license: 'Apache-2.0',
    baseFiles: {
      'tokenizer_config.json': 9705,
      'tokenizer.json': 9117040,
      'config.json': 912,
      'generation_config.json': 219,
    },
    variants: {
      q4f16: { requiresF16: true, files: { 'onnx/model_q4f16.onnx': 569789750 } },
      q4: { requiresF16: false, files: { 'onnx/model_q4.onnx': 919096585 } },
    },
    chatTemplateOptions: { enable_thinking: false },
    tier: 'light',
    devices: ['webgpu'],
  },
  'gemma-3-270m': {
    key: 'gemma-3-270m',
    name: 'Gemma 3 270M',
    label: 'Gemma 3 270M (instruction-tuned)',
    id: 'onnx-community/gemma-3-270m-it-ONNX',
    revision: '2dbbfdb1b59bd034eb959428c6a7da9dd7ea27f0',
    license: 'Gemma Terms of Use',
    baseFiles: {
      'tokenizer_config.json': 2313,
      'tokenizer.json': 20323013,
      'config.json': 1652,
      'generation_config.json': 202,
    },
    variants: {
      q4f16: { requiresF16: true, files: { 'onnx/model_q4f16.onnx': 330717, 'onnx/model_q4f16.onnx_data': 272626176 } },
      q4: { requiresF16: false, files: { 'onnx/model_q4.onnx': 242522, 'onnx/model_q4.onnx_data': 322933248 } },
    },
    chatTemplateOptions: {},
    tier: 'tiny',
    // Sul processore (WASM) non parte: l'operatore GatherBlockQuantized non è disponibile.
    devices: ['webgpu'],
  },
  'smollm2-135m': {
    key: 'smollm2-135m',
    name: 'SmolLM2 135M',
    label: 'SmolLM2 135M (instruct)',
    id: 'onnx-community/SmolLM2-135M-Instruct-ONNX',
    revision: 'b8a5c0f183b78c55955a5364f610c36668b5e681',
    license: 'Apache-2.0',
    baseFiles: {
      'tokenizer_config.json': 3794,
      'tokenizer.json': 3522656,
      'config.json': 976,
      'generation_config.json': 132,
    },
    variants: {
      q8: { requiresF16: false, cpu: true, files: { 'onnx/model_quantized.onnx': 135658354 } },
    },
    chatTemplateOptions: {},
    tier: 'cpu',
    // Solo processore: su WebGPU (q4f16) nelle prove emetteva soltanto <|endoftext|>.
    devices: ['wasm'],
  },
};

export const DEFAULT_MODEL_KEY = 'gemma-4-e2b';
export const LIGHT_MODEL_KEY = 'qwen3-0.6b';
export const TINY_MODEL_KEY = 'gemma-3-270m';
export const CPU_MODEL_KEY = 'smollm2-135m';

/** Limiti didattici: risposte brevi e contesto contenuto. */
export const LIMITS = Object.freeze({
  maxPromptChars: 240,
  maxInputTokens: 160,
  defaultNewTokens: 48,
  minNewTokens: 8,
  maxNewTokens: 128,
  displayCandidates: 8,
});

export function pickVariant(model, { shaderF16, device = 'webgpu' } = {}) {
  if (device === 'wasm') {
    return Object.keys(model.variants).find((k) => model.variants[k].cpu) ?? 'q8';
  }
  if (shaderF16 && model.variants.q4f16) return 'q4f16';
  return model.variants.q4 ? 'q4' : Object.keys(model.variants)[0];
}

/** Modelli eseguibili su un dispositivo ('webgpu' | 'wasm'). */
export function modelsFor(device) {
  return Object.values(MODELS).filter((m) => (m.devices ?? ['webgpu']).includes(device));
}

/** File (percorso → byte) che il browser scarica per un modello e una precisione. */
export function filesFor(model, dtype) {
  return { ...model.baseFiles, ...model.variants[dtype].files };
}

export function downloadBytes(model, dtype) {
  return Object.values(filesFor(model, dtype)).reduce((a, b) => a + b, 0);
}

export function fileUrl(model, path) {
  return `${HUB}/${model.id}/resolve/${model.revision}/${path}`;
}

/**
 * Suggerimento iniziale: modello consigliato, salvo che il browser dichiari
 * poca memoria: meno di 16 GB → leggero, meno di 8 GB → minimo (il modello
 * consigliato ha richiesto ~6,6 GB di memoria del browser sul computer di prova).
 */
export function suggestModelKey({ deviceMemoryGB } = {}) {
  if (typeof deviceMemoryGB === 'number' && deviceMemoryGB > 0) {
    if (deviceMemoryGB < 8) return TINY_MODEL_KEY;
    if (deviceMemoryGB < 16) return LIGHT_MODEL_KEY;
  }
  return DEFAULT_MODEL_KEY;
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '—';
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1).replace('.', ',')} GB`;
  return `${Math.round(bytes / 1e6)} MB`;
}
