/**
 * Smoke test con un modello reale su WebGPU (Chrome installato nel sistema).
 * Non fa parte della CI: scarica il modello (Gemma 4 E2B ~3,1 GB la prima
 * volta; Qwen3 0.6B ~0,6 GB) e richiede una scheda grafica.
 *
 *   npm run build
 *   npm run test:e2e:real                      # modello predefinito
 *   REAL_MODEL=qwen3-0.6b npm run test:e2e:real
 *
 * Il profilo del browser (con la cache del modello) resta in
 * .cache/e2e-chrome-profile per non riscaricare a ogni esecuzione.
 */

import { test, expect, chromium } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import { MODELS, DEFAULT_MODEL_KEY } from '../../src/core/models.js';

const MODEL_KEY = process.env.REAL_MODEL ?? DEFAULT_MODEL_KEY;
const SPEC = MODELS[MODEL_KEY];
const PROFILE = process.env.E2E_PROFILE
  ?? fileURLToPath(new URL('../../.cache/e2e-chrome-profile', import.meta.url));
const BASE = 'http://localhost:4173/';
const PROMPT = 'Perché il cielo è blu?';

function pct(p) {
  if (p <= 0) return '0%';
  const v = p * 100;
  if (v < 0.1) return '<0,1%';
  if (p < 1 && v > 99.9) return '>99,9%';
  return `${v.toFixed(1).replace('.', ',')}%`;
}

let context;
let page;
const requests = [];

test.describe.configure({ mode: 'serial' });

test.beforeAll(async () => {
  expect(SPEC, `modello sconosciuto: ${MODEL_KEY}`).toBeTruthy();
  context = await chromium.launchPersistentContext(PROFILE, {
    channel: 'chrome',
    headless: process.env.HEADED !== '1',
    viewport: { width: 1440, height: 900 },
  });
  page = context.pages()[0] ?? await context.newPage();
  page.on('request', (r) => requests.push({ url: r.url(), post: r.postData() ?? '', t: Date.now() }));
  await page.addInitScript(() => {
    window.__longTasks = [];
    try {
      new PerformanceObserver((list) => {
        for (const e of list.getEntries()) window.__longTasks.push(e.duration);
      }).observe({ type: 'longtask', buffered: true });
    } catch {
      // non supportato
    }
  });
});

test.afterAll(async () => {
  await context?.close();
});

test('il modello reale si carica su WebGPU dopo un\'azione esplicita', async () => {
  await page.goto(BASE);
  await expect(page.getByTestId('btn-load')).toBeEnabled({ timeout: 30_000 });
  // prima del clic: nessun download dei file del modello
  expect(requests.filter((r) => r.url.includes('/resolve/'))).toEqual([]);
  await page.getByTestId(`model-${MODEL_KEY}`).check();
  await page.getByTestId('btn-load').click();
  await expect(page.getByTestId('model-chip')).toHaveAttribute('data-state', 'ready', { timeout: 30 * 60_000 });
  const info = await page.evaluate(() => window.__nextTokenDemo.info);
  expect(info.modelId).toBe(SPEC.id);
  expect(info.revision).toBe(SPEC.revision);
  expect(info.device).toBe('webgpu');
  test.info().annotations.push({ type: 'load', description: JSON.stringify({ dtype: info.dtype, tokenizerMs: info.tokenizerMs, modelMs: info.modelMs, warmupMs: info.warmupMs, totalLoadMs: info.totalLoadMs }) });
});

test('una risposta reale: probabilità del modello, allineate e mostrate fedelmente', async () => {
  const before = requests.length;
  await page.evaluate(() => { window.__longTasks.length = 0; });
  await page.locator('input[name="speed"][value="fast"]').check();
  await page.getByTestId('prompt').fill(PROMPT);
  const t0 = Date.now();
  await page.getByTestId('btn-primary').click();
  // predefinito: un token alla volta; l'input è pronto e il modello aspetta
  await page.waitForFunction(() => window.__nextTokenDemo.state === 'paused');
  await expect(page.getByTestId('btn-primary')).toHaveText('Calcola il primo token');
  expect(await page.evaluate(() => window.__nextTokenDemo.engineCalls.filter((c) => c === 'step').length)).toBe(0);
  await page.getByTestId('btn-auto').click();
  await page.waitForFunction(() => window.__nextTokenDemo.state === 'done', null, { timeout: 10 * 60_000 });
  const elapsed = Date.now() - t0;

  // Inferenza nel browser: durante la generazione nessuna richiesta esce dal computer.
  const during = requests.slice(before).filter((r) => !r.url.startsWith(BASE));
  expect(during).toEqual([]);
  // Il testo della domanda non compare in nessuna richiesta di rete.
  expect(requests.filter((r) => r.url.includes(encodeURIComponent(PROMPT)) || r.post.includes(PROMPT))).toEqual([]);

  const trace = await page.evaluate(() => window.__nextTokenDemo.trace);
  expect(trace.meta.simulated).toBe(false);
  expect(trace.meta.model.modelId).toBe(SPEC.id);
  expect(['eos', 'length']).toContain(trace.finish);
  expect(trace.steps.length).toBeGreaterThan(2);
  const inputLen = trace.input.ids.length;
  trace.steps.forEach((s, i) => {
    expect(s.index).toBe(i);
    expect(s.contextLength).toBe(inputLen + i);
    expect(s.processedTokens).toBe(i === 0 ? inputLen : 1);
    expect(s.vocabSize).toBeGreaterThan(100_000);
    for (let k = 1; k < s.candidates.length; k++) {
      expect(s.candidates[k - 1].modelProb).toBeGreaterThanOrEqual(s.candidates[k].modelProb);
    }
    const total = s.candidates.reduce((a, c) => a + c.modelProb, 0) + s.other.modelProb;
    expect(Math.abs(total - 1)).toBeLessThan(1e-6);
    // greedy: il token emesso è il primo candidato dello stesso passo
    expect(s.selected.id).toBe(s.candidates[0].id);
    expect(s.selected.modelProb).toBe(s.candidates[0].modelProb);
  });
  // Valori reali: cambiano da un passo all'altro (nessuna percentuale fissa).
  expect(new Set(trace.steps.map((s) => s.selected.modelProb.toFixed(6))).size).toBeGreaterThan(1);

  // Quello che si vede è la traccia di questa stessa invocazione.
  const last = trace.steps.at(-1);
  const lastAnswer = await page.evaluate(() => window.__nextTokenDemo.trace.steps.at(-1).answerText);
  await expect(page.getByTestId('answer')).toContainText(lastAnswer.trim());
  const rows = page.locator('[data-testid="chart"] li.row:not(.other)');
  await expect(rows).toHaveCount(last.candidates.length);
  for (let i = 0; i < last.candidates.length; i++) {
    await expect(rows.nth(i)).toHaveAttribute('data-id', String(last.candidates[i].id));
    await expect(rows.nth(i).locator('.v-model')).toHaveText(pct(last.candidates[i].modelProb));
  }
  await expect(page.locator('[data-testid="tokens"] .tok.o-generated')).toHaveCount(trace.steps.length);

  // L'interfaccia resta reattiva: nessun blocco lungo del thread principale.
  const longTasks = await page.evaluate(() => window.__longTasks.slice());
  expect(Math.max(0, ...longTasks)).toBeLessThan(250);
  const stepMs = trace.steps.slice(1).map((s) => s.timing.stepMs);
  test.info().annotations.push({
    type: 'generation',
    description: JSON.stringify({
      tokens: trace.steps.length,
      finish: trace.finish,
      wallMs: elapsed,
      firstStepMs: trace.steps[0].timing.stepMs,
      meanDecodeStepMs: Math.round(stepMs.reduce((a, b) => a + b, 0) / Math.max(1, stepMs.length)),
      longTasksMs: longTasks.map(Math.round),
      answer: lastAnswer,
    }),
  });
});

test('replay: nessuna nuova chiamata al modello', async () => {
  const callsBefore = await page.evaluate(() => window.__nextTokenDemo.engineCalls.length);
  await page.locator('input[name="speed"][value="normal"]').check();
  await page.getByTestId('btn-replay').click();
  await expect(page.getByTestId('mode-badge')).toContainText('REPLAY');
  await page.locator('input[name="speed"][value="fast"]').check();
  await page.waitForFunction(() => window.__nextTokenDemo.state === 'replay-paused');
  await page.getByTestId('btn-auto').click();
  await page.waitForFunction(() => window.__nextTokenDemo.state === 'done', null, { timeout: 120_000 });
  expect(await page.evaluate(() => window.__nextTokenDemo.engineCalls.length)).toBe(callsBefore);
});

test('stop a metà: la generazione si ferma e la sessione viene chiusa', async () => {
  await page.locator('input[name="speed"][value="normal"]').check();
  const question = "Che cos'è l'intelligenza artificiale?";
  await page.getByTestId('prompt').fill(question);
  await page.getByTestId('btn-primary').click();
  await page.waitForFunction(() => window.__nextTokenDemo.state === 'paused');
  await page.getByTestId('btn-auto').click();
  // attende due passi della NUOVA generazione (non della traccia precedente)
  await page.waitForFunction((q) => {
    const t = window.__nextTokenDemo.trace;
    return t?.input?.userText === q && t.steps.length >= 2;
  }, question, { timeout: 120_000 });
  await page.getByTestId('btn-stop').click();
  await page.waitForFunction(() => window.__nextTokenDemo.state === 'done', null, { timeout: 120_000 });
  const trace = await page.evaluate(() => window.__nextTokenDemo.trace);
  expect(trace.input.userText).toBe(question);
  expect(trace.finish).toBe('stopped');
  expect(trace.steps.length).toBeGreaterThanOrEqual(2);
  expect(trace.steps.length).toBeLessThan(trace.maxNewTokens);
  expect(await page.evaluate(() => window.__nextTokenDemo.engineCalls.at(-1))).toBe('end');
  await expect(page.getByTestId('finish-note')).toContainText('fermata');
});

test('scelta del presentatore: un altro candidato, poi il modello continua da lì', async () => {
  await page.locator('input[name="speed"][value="fast"]').check();
  await page.getByTestId('prompt').fill(PROMPT);
  await page.getByTestId('btn-primary').click();
  await page.waitForFunction(() => window.__nextTokenDemo.state === 'paused');
  await expect(page.getByTestId('btn-primary')).toHaveText('Calcola il primo token', { timeout: 30_000 });
  await page.getByTestId('btn-primary').click();
  await page.waitForFunction(() => window.__nextTokenDemo.trace?.steps?.length === 1);
  await expect(page.getByTestId('btn-primary')).toHaveText('Prossimo token', { timeout: 30_000 });
  const before = await page.evaluate(() => window.__nextTokenDemo.trace.steps[0]);
  const steps = await page.evaluate(() => window.__nextTokenDemo.engineCalls.filter((c) => c === 'step').length);
  const alt = before.candidates[1];
  await page.locator(`[data-testid="chart"] li.row[data-id="${alt.id}"]`).click();
  await expect(page.getByTestId('choice')).toContainText('Scelto da te', { timeout: 30_000 });
  const after = await page.evaluate(() => window.__nextTokenDemo.trace.steps[0]);
  const afterAnswer = await page.evaluate(() => window.__nextTokenDemo.trace.steps[0].answerText);
  expect(after.selected.id).toBe(alt.id);
  expect(after.override.original.id).toBe(before.selected.id);
  expect(after.candidates).toEqual(before.candidates);
  expect(await page.evaluate(() => window.__nextTokenDemo.engineCalls.filter((c) => c === 'step').length)).toBe(steps);
  await page.getByTestId('btn-primary').click();
  await page.waitForFunction(() => window.__nextTokenDemo.trace.steps.length === 2, null, { timeout: 60_000 });
  const trace = await page.evaluate(() => window.__nextTokenDemo.trace);
  expect(trace.steps[1].contextLength).toBe(trace.input.ids.length + 1);
  expect(trace.steps[1].processedTokens).toBe(1);
  const nextAnswer = await page.evaluate(() => window.__nextTokenDemo.trace.steps[1].answerText);
  expect(nextAnswer.startsWith(afterAnswer.replace(/�+$/, ''))).toBe(true);
  test.info().annotations.push({ type: 'override', description: JSON.stringify({ original: before.selected.piece?.text, forced: alt.piece?.text, next: trace.steps[1].selected.piece?.text }) });
  await page.getByTestId('btn-stop').click();
  await page.waitForFunction(() => window.__nextTokenDemo.state === 'done');
});

test('cambio modello: il modello viene scaricato dalla memoria', async () => {
  await page.getByTestId('advanced').locator('summary').click();
  await page.getByTestId('btn-change-model').click();
  await expect(page.getByTestId('setup')).toBeVisible();
  await expect(page.getByTestId('model-chip')).toHaveAttribute('data-state', 'off');
  expect(await page.evaluate(() => window.__nextTokenDemo.engineCalls.length)).toBe(0);
});

test('diagnostica: punteggi grezzi e parità con generate() nativo', async () => {
  // eseguita dopo aver liberato il modello della pagina: una sola copia in memoria
  const diag = await context.newPage();
  await diag.goto(`${BASE}diagnostics.html?auto=1&model=${MODEL_KEY}&n=24`);
  await diag.waitForFunction(() => JSON.parse(document.getElementById('report').textContent || '{}').done, null, { timeout: 30 * 60_000 });
  const report = JSON.parse(await diag.locator('#report').textContent());
  expect(report.error).toBeUndefined();
  expect(report.verifyRaw.maxAbsDiff).toBe(0);
  expect(report.verifyRaw.argmaxEqual).toBe(true);
  expect(report.verifyParity.equal).toBe(true);
  test.info().annotations.push({
    type: 'diagnostics',
    description: JSON.stringify({
      adapter: report.probe.adapter,
      dtype: report.model.dtype,
      load: { tokenizerMs: report.load.tokenizerMs, modelMs: report.load.modelMs, warmup: report.load.warmup },
      timeToFirstTokenMs: report.generation.timeToFirstTokenMs,
      decodeTokensPerSecond: report.generation.decodeTokensPerSecond,
      parityTokens: report.verifyParity.native.length,
    }),
  });
  await diag.close();
});
