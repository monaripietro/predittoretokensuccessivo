/**
 * Misure tecniche su questo dispositivo: rilevamento WebGPU, download,
 * tempo al primo token, velocità, reattività della pagina e verifiche di
 * correttezza dei punteggi (grezzi e allineati alla generazione nativa).
 */

import { EngineClient } from './engine/client.js';
import { MODELS, pickVariant, downloadBytes } from './core/models.js';
import { isModelCached } from './core/cache.js';
import { probeWebGPU } from './core/webgpu.js';

const $ = (id) => document.getElementById(id);

/** Domande tipiche di una lezione introduttiva, uguali per tutti i modelli. */
const QUALITY_PROMPTS = [
  "Qual è la capitale d'Italia?",
  'Perché il cielo è blu?',
  'Scrivi una frase su un gatto che dorme.',
  "Che cos'è l'intelligenza artificiale?",
  'Quante zampe ha un ragno?',
  'Completa la frase: Il mattino ha',
];
const params = new URLSearchParams(location.search);
const report = { startedAt: new Date().toISOString(), userAgent: navigator.userAgent };

for (const m of Object.values(MODELS)) {
  const opt = document.createElement('option');
  opt.value = m.key;
  opt.textContent = m.label;
  $('model').appendChild(opt);
}
if (params.get('model')) $('model').value = params.get('model');
if (params.get('dtype')) $('dtype').value = params.get('dtype');
if (params.get('n')) $('n').value = params.get('n');
if (params.get('prompt')) $('prompt').value = params.get('prompt');

function show(status) {
  $('status').textContent = status;
  $('report').textContent = JSON.stringify(report, null, 2);
}

// Reattività: long task del thread principale durante il lavoro del modello.
const longTasks = [];
try {
  new PerformanceObserver((list) => {
    for (const e of list.getEntries()) longTasks.push(Math.round(e.duration));
  }).observe({ type: 'longtask', buffered: true });
} catch {
  // non supportato
}

let client = null;

async function run() {
  $('run').disabled = true;
  try {
    const modelKey = $('model').value;
    const spec = MODELS[modelKey];
    const n = Math.max(1, Math.min(128, Number($('n').value) || 32));
    const prompt = $('prompt').value;
    report.probe = await probeWebGPU();
    show('WebGPU rilevato, avvio del worker…');
    if (!report.probe.ok) throw new Error(`WebGPU non utilizzabile: ${report.probe.reason}`);
    const dtype = $('dtype').value || pickVariant(spec, { shaderF16: report.probe.shaderF16 });
    report.model = { key: modelKey, id: spec.id, revision: spec.revision, dtype, declaredBytes: downloadBytes(spec, dtype) };

    let lastProgress = 0;
    let maxTotal = 0;
    client?.terminate();
    client = new EngineClient({
      onProgress: (p) => {
        if (p.phase === 'files') {
          maxTotal = Math.max(maxTotal, p.total ?? 0);
          report.download = { loaded: p.loaded, total: p.total };
          const now = performance.now();
          if (now - lastProgress > 500) {
            lastProgress = now;
            show(`Download/caricamento: ${Math.round((p.loaded / p.total) * 100)}%`);
          }
        }
      },
    });
    report.cachedBeforeLoad = await isModelCached(spec, dtype);
    const tLoad = performance.now();
    const info = await client.call('load', { modelKey, dtype, shaderF16: report.probe.shaderF16 });
    report.load = { ...info, wallMs: Math.round(performance.now() - tLoad), progressTotalBytes: maxTotal };
    if (!params.has('nowarmup')) {
      show('Esecuzione di prova (compilazione GPU)…');
      report.load.warmup = await client.call('warmup');
    }
    show('Modello caricato. Generazione misurata…');

    // Prima generazione: tempo al primo token e velocità.
    longTasks.length = 0;
    const tBegin = performance.now();
    const begun = await client.call('begin', { text: prompt, policy: { kind: 'greedy' }, maxNewTokens: n });
    const steps = [];
    let first = null;
    for (let i = 0; i < n; i++) {
      const ev = await client.call('step', { sessionId: begun.sessionId });
      if (i === 0) first = performance.now() - tBegin;
      steps.push(ev);
      if (ev.finish) break;
    }
    const tEnd = performance.now();
    await client.call('end');
    const decodeSteps = steps.slice(1);
    const decodeMs = decodeSteps.reduce((s, e) => s + e.timing.stepMs, 0);
    report.generation = {
      inputTokens: begun.input.ids.length,
      piecesExact: begun.input.piecesExact,
      generatedTokens: steps.length,
      finish: steps.at(-1)?.finish ?? 'stopped',
      timeToFirstTokenMs: Math.round(first),
      prefillStepMs: steps[0]?.timing.stepMs,
      decodeTokensPerSecond: decodeSteps.length ? Math.round((decodeSteps.length / decodeMs) * 1000 * 10) / 10 : null,
      endToEndTokensPerSecond: Math.round((steps.length / (tEnd - tBegin)) * 1000 * 10) / 10,
      meanDecideMs: Math.round((steps.reduce((s, e) => s + e.timing.decideMs, 0) / steps.length) * 10) / 10,
      vocabSize: steps[0]?.vocabSize,
      answer: steps.at(-1)?.answerText,
      firstStep: steps[0] && {
        selected: steps[0].selected,
        top: steps[0].candidates.slice(0, 5).map((c) => ({ id: c.id, piece: c.piece.text, p: c.modelProb })),
        other: steps[0].other,
      },
      mainThreadLongTasksMs: longTasks.slice(),
    };
    show('Verifica dei punteggi grezzi…');
    report.verifyRaw = await client.call('verifyRaw', { text: prompt });
    show('Confronto con la generazione nativa…');
    report.verifyParity = await client.call('verifyParity', { text: prompt, n: Math.min(n, 24) });
    if (params.has('quality')) {
      report.quality = [];
      for (const q of QUALITY_PROMPTS) {
        show(`Prova di qualità: ${q}`);
        const b = await client.call('begin', { text: q, policy: { kind: 'greedy' }, maxNewTokens: 48 });
        let ev = null;
        const times = [];
        for (let i = 0; i < 48; i++) {
          ev = await client.call('step', { sessionId: b.sessionId });
          times.push(ev.timing.stepMs);
          if (ev.finish) break;
        }
        await client.call('end');
        report.quality.push({
          prompt: q,
          answer: ev?.answerText,
          tokens: times.length,
          finish: ev?.finish ?? 'stopped',
          meanDecodeMs: Math.round(times.slice(1).reduce((a, b) => a + b, 0) / Math.max(1, times.length - 1)),
        });
      }
    }
    report.finishedAt = new Date().toISOString();
    report.done = true;
    show('Completato.');
  } catch (err) {
    report.error = { code: err.code, message: err.message };
    report.done = true;
    show(`Errore: ${err.message}`);
  } finally {
    $('run').disabled = false;
  }
}

$('run').addEventListener('click', run);
$('unload').addEventListener('click', async () => {
  if (!client) return;
  await client.call('unload').catch(() => {});
  client.terminate();
  client = null;
  show('Modello liberato.');
});
if (params.has('auto')) run();
