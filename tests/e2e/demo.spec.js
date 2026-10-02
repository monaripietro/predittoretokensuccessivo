import { test, expect } from '@playwright/test';

/**
 * Interfaccia in modalità simulata (`?mock`): stesso motore e stesso codice
 * di scelta del token, modello finto. Verifica la regia e la coerenza tra
 * ciò che si vede e la traccia registrata.
 */

const demo = (page, expr) => page.evaluate(expr);

async function openMock(page, query = '') {
  await page.goto(`./?mock&delay=5${query}`);
  await expect(page.getByTestId('model-chip')).toHaveAttribute('data-state', 'sim');
}

async function setSpeed(page, speed) {
  await page.locator(`input[name="speed"][value="${speed}"]`).check();
}

/** Avvia: l'app prepara l'input e si ferma in attesa del primo token. */
async function start(page, prompt) {
  await page.getByTestId('prompt').fill(prompt);
  await page.getByTestId('btn-primary').click();
  await page.waitForFunction(() => window.__nextTokenDemo.state === 'paused');
  // la preparazione dell'input (fasi 1-3) dura qualche secondo al ritmo lento/normale
  await expect(page.getByTestId('btn-primary')).toBeEnabled({ timeout: 15_000 });
}

const stepCalls = (page) => demo(page, () => window.__nextTokenDemo.engineCalls.filter((c) => c === 'step').length);

async function runToEnd(page, prompt) {
  await start(page, prompt);
  await page.getByTestId('btn-auto').click();
  await page.waitForFunction(() => window.__nextTokenDemo.state === 'done', null, { timeout: 20_000 });
}

/** Percentuale nel formato dell'interfaccia (stessa regola di formatPercent). */
function pct(p) {
  if (p <= 0) return '0%';
  const v = p * 100;
  if (v < 0.1) return '<0,1%';
  if (p < 1 && v > 99.9) return '>99,9%';
  return `${v.toFixed(1).replace('.', ',')}%`;
}

test.describe('simulazione: segnalata e locale', () => {
  test('la simulazione è dichiarata in pagina e non si fanno richieste esterne', async ({ page }) => {
    const external = [];
    page.on('request', (r) => {
      if (!r.url().startsWith('http://localhost')) external.push(r.url());
    });
    await openMock(page);
    await expect(page.getByTestId('sim-banner')).toBeVisible();
    await expect(page.getByTestId('sim-banner')).toContainText('modello finto');
    await setSpeed(page, 'fast');
    await runToEnd(page, 'Quante zampe ha un ragno?');
    await expect(page.getByTestId('mode-badge')).toHaveText('SIMULAZIONE');
    expect(external).toEqual([]);
  });
});

test.describe('generazione', () => {
  test.beforeEach(async ({ page }) => {
    await openMock(page);
  });

  test('un solo comando principale; senza domanda non si avvia', async ({ page }) => {
    await expect(page.getByTestId('btn-primary')).toHaveText('Avvia');
    await expect(page.getByTestId('btn-primary')).toBeDisabled();
    await expect(page.getByTestId('btn-auto')).toBeHidden();
    await expect(page.getByTestId('btn-stop')).toBeHidden();
    await page.getByTestId('prompt').fill('Ciao');
    await expect(page.getByTestId('btn-primary')).toBeEnabled();
  });

  test('ciò che si vede coincide con la traccia registrata', async ({ page }) => {
    await setSpeed(page, 'fast');
    await runToEnd(page, 'Quante zampe ha un ragno?');
    const trace = await demo(page, () => window.__nextTokenDemo.trace);
    expect(trace.finish).toBe('eos');
    const last = trace.steps.at(-1);
    // Risposta = decodifica degli ID generati
    await expect(page.getByTestId('answer')).toContainText(last.answerText);
    await expect(page.getByTestId('answer')).toContainText('fine');
    // Ogni token generato è tornato nell'input
    await expect(page.locator('[data-testid="tokens"] .tok.o-generated')).toHaveCount(trace.steps.length);
    // Grafico dell'ultimo passo: stessi ID, stesse percentuali, stesso token scelto
    const rows = page.locator('[data-testid="chart"] li.row:not(.other)');
    await expect(rows).toHaveCount(last.candidates.length);
    for (let i = 0; i < last.candidates.length; i++) {
      const c = last.candidates[i];
      await expect(rows.nth(i)).toHaveAttribute('data-id', String(c.id));
      await expect(rows.nth(i).locator('.v-model')).toHaveText(pct(c.modelProb));
    }
    await expect(page.locator('[data-testid="chart"] li.chosen')).toHaveAttribute('data-id', String(last.selected.id));
    await expect(page.locator('[data-testid="chart"] li.other .v-model')).toHaveText(pct(last.other.modelProb));
    // Candidati visibili + "altri" = 100% (non rinormalizzati)
    const total = last.candidates.reduce((s, c) => s + c.modelProb, 0) + last.other.modelProb;
    expect(Math.abs(total - 1)).toBeLessThan(1e-9);
    await expect(page.getByTestId('chart-caption')).toContainText('tutto il vocabolario');
  });

  test('ogni passo: scelta allineata ai punteggi dello stesso passo (greedy)', async ({ page }) => {
    await setSpeed(page, 'fast');
    await runToEnd(page, 'Perché il cielo è blu?');
    const trace = await demo(page, () => window.__nextTokenDemo.trace);
    expect(trace.steps.length).toBeGreaterThan(3);
    trace.steps.forEach((s, i) => {
      expect(s.index).toBe(i);
      expect(s.contextLength).toBe(trace.input.ids.length + i);
      expect(s.selected.id).toBe(s.candidates[0].id);
      expect(s.selected.modelProb).toBe(s.candidates[0].modelProb);
      expect(s.processedTokens).toBe(i === 0 ? trace.input.ids.length : 1);
    });
    // probabilità non costanti tra i passi: niente valori fissi
    const distinct = new Set(trace.steps.map((s) => s.selected.modelProb.toFixed(6)));
    expect(distinct.size).toBeGreaterThan(1);
  });

  test('predefinito: un token alla volta, solo quando lo chiedi', async ({ page }) => {
    await setSpeed(page, 'fast');
    await start(page, 'Perché il cielo è blu?');
    // l'input è pronto e tokenizzato, ma il modello non ha ancora calcolato nulla
    await expect(page.getByTestId('btn-primary')).toHaveText('Calcola il primo token');
    await expect(page.getByTestId('btn-auto')).toBeVisible();
    await expect(page.getByTestId('btn-stop')).toBeVisible();
    await expect(page.getByTestId('narration')).toContainText('Calcola il primo token');
    expect(await stepCalls(page)).toBe(0);
    for (let i = 1; i <= 3; i++) {
      await page.getByTestId('btn-primary').click();
      await page.waitForFunction((n) => window.__nextTokenDemo.trace.steps.length === n, i);
      await expect(page.getByTestId('btn-primary')).toHaveText('Prossimo token');
      await expect(page.getByTestId('btn-primary')).toBeEnabled();
      await expect(page.locator('[data-testid="tokens"] .tok.o-generated')).toHaveCount(i);
      await expect(page.getByTestId('narration')).toContainText('Prossimo token');
      // nessun calcolo in più finché non si preme di nuovo
      await page.waitForTimeout(400);
      expect(await stepCalls(page)).toBe(i);
      expect(await demo(page, () => window.__nextTokenDemo.state)).toBe('paused');
    }
  });

  test('ogni fase del token è raccontata, nell\'ordine reale', async ({ page }) => {
    await setSpeed(page, 'slow');
    await start(page, 'Perché il cielo è blu?');
    const seen = [];
    await page.exposeFunction('__seen', (t) => seen.push(t));
    await page.evaluate(() => {
      const list = document.querySelector('#narration .nar-list');
      new MutationObserver(() => {
        const lastLine = [...list.querySelectorAll(':scope > .nar-item')].at(-1);
        if (lastLine) window.__seen(lastLine.dataset.n);
      }).observe(list, { childList: true, characterData: true, subtree: true });
    });
    await page.getByTestId('btn-primary').click();
    await page.waitForFunction(() => [...document.querySelectorAll('#narration .nar-list > .nar-item')].at(-1)?.dataset.n === '↺', null, { timeout: 20_000 });
    const order = seen.filter((x, i) => x !== seen[i - 1]);
    expect(order).toEqual(['4', '5', '6', '↺']);
    // le frasi del token restano tutte visibili, una sotto l'altra, e la preparazione è riassunta
    const lines = await page.locator('#narration .nar-list > .nar-item').evaluateAll((els) => els.map((e) => e.dataset.n));
    expect(lines).toEqual(['4', '5', '6', '↺']);
    // la preparazione resta nella cronologia, richiudibile, con le sue 3 frasi
    const prep = page.locator('#narration .nar-list > .nar-group');
    await expect(prep).toHaveCount(1);
    await expect(prep).toContainText('Input pronto');
    await prep.locator('summary').click();
    await expect(prep.locator('.nar-sublist > .nar-item')).toHaveCount(3);
  });

  test('«Continua da solo» e poi «Pausa» riporta al passo per passo', async ({ page }) => {
    await setSpeed(page, 'normal');
    await start(page, 'Perché il cielo è blu?');
    await page.getByTestId('btn-auto').click();
    await page.waitForFunction(() => window.__nextTokenDemo.trace.steps.length >= 2);
    await expect(page.getByTestId('btn-primary')).toHaveText('Pausa');
    await page.getByTestId('btn-primary').click();
    await page.waitForFunction(() => window.__nextTokenDemo.state === 'paused');
    await expect(page.getByTestId('btn-primary')).toBeEnabled({ timeout: 10_000 });
    await expect(page.getByTestId('btn-primary')).toHaveText('Prossimo token');
    const steps = await demo(page, () => window.__nextTokenDemo.trace.steps.length);
    await page.waitForTimeout(600);
    expect(await demo(page, () => window.__nextTokenDemo.trace.steps.length)).toBe(steps);
  });

  test('stop: la generazione si ferma e lo dichiara', async ({ page }) => {
    await setSpeed(page, 'fast');
    await start(page, 'Perché il cielo è blu?');
    await page.getByTestId('btn-primary').click();
    await page.waitForFunction(() => window.__nextTokenDemo.trace.steps.length === 1);
    await page.getByTestId('btn-stop').click();
    await page.waitForFunction(() => window.__nextTokenDemo.state === 'done');
    expect(await demo(page, () => window.__nextTokenDemo.trace.finish)).toBe('stopped');
    await expect(page.getByTestId('finish-note')).toContainText('fermata');
    await expect(page.getByTestId('btn-primary')).toHaveText('Avvia');
    await expect(page.getByTestId('btn-stop')).toBeHidden();
    // la sessione del motore è stata chiusa (cache liberata)
    expect(await demo(page, () => window.__nextTokenDemo.engineCalls.at(-1))).toBe('end');
  });

  test('replay: rivede la registrazione, un token alla volta, senza nuovi calcoli', async ({ page }) => {
    await setSpeed(page, 'fast');
    await runToEnd(page, "Qual è la capitale d'Italia?");
    const answer = await page.getByTestId('answer').textContent();
    const callsBefore = await demo(page, () => window.__nextTokenDemo.engineCalls.length);
    await page.getByTestId('btn-replay').click();
    await expect(page.getByTestId('mode-badge')).toContainText('REPLAY');
    await expect(page.getByTestId('btn-primary')).toHaveText('Mostra il primo token');
    await page.getByTestId('btn-primary').click();
    await expect(page.getByTestId('step-label')).toHaveText('replay · token 1');
    await expect(page.getByTestId('btn-primary')).toHaveText('Mostra il prossimo token');
    await page.getByTestId('btn-auto').click();
    await page.waitForFunction(() => window.__nextTokenDemo.state === 'done', null, { timeout: 20_000 });
    await expect(page.getByTestId('finish-note')).toContainText('senza nuovi calcoli');
    expect(await page.getByTestId('answer').textContent()).toBe(answer);
    expect(await demo(page, () => window.__nextTokenDemo.engineCalls.length)).toBe(callsBefore);
  });

  test('tastiera: → e Spazio avanzano di un token, Esc ferma', async ({ page }) => {
    await setSpeed(page, 'fast');
    await start(page, 'Perché il cielo è blu?');
    await page.locator('h1').click();
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => window.__nextTokenDemo.trace.steps.length === 1);
    await expect(page.getByTestId('btn-primary')).toBeEnabled();
    await page.keyboard.press('Space');
    await page.waitForFunction(() => window.__nextTokenDemo.trace.steps.length === 2);
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => window.__nextTokenDemo.state === 'done');
    expect(await demo(page, () => window.__nextTokenDemo.trace.finish)).toBe('stopped');
  });

  test('il presentatore può scegliere un altro candidato e il modello continua da lì', async ({ page }) => {
    await setSpeed(page, 'fast');
    await start(page, 'Perché il cielo è blu?');
    await page.getByTestId('btn-primary').click();
    await page.waitForFunction(() => window.__nextTokenDemo.trace.steps.length === 1);
    await expect(page.getByTestId('btn-primary')).toBeEnabled();
    await expect(page.getByTestId('pick-hint')).toBeVisible();
    const originalId = await demo(page, () => window.__nextTokenDemo.trace.steps[0].selected.id);
    const stepsBefore = await stepCalls(page);
    const row = page.locator('[data-testid="chart"] li.row:not(.other):not(.outside)').nth(2);
    const pickedId = Number(await row.getAttribute('data-id'));
    expect(pickedId).not.toBe(originalId);
    await row.click();
    await page.waitForFunction((id) => window.__nextTokenDemo.trace.steps[0].selected.id === id, pickedId);
    expect(await demo(page, () => window.__nextTokenDemo.trace.steps[0].override.original.id)).toBe(originalId);
    expect(await stepCalls(page)).toBe(stepsBefore);
    await expect(page.getByTestId('choice')).toContainText('Scelto da te');
    await expect(page.getByTestId('btn-primary')).toBeEnabled();
    await page.getByTestId('btn-primary').click();
    await page.waitForFunction(() => window.__nextTokenDemo.trace.steps.length === 2);
    expect(await demo(page, () => window.__nextTokenDemo.trace.steps[1].contextLength))
      .toBe(await demo(page, () => window.__nextTokenDemo.trace.input.ids.length + 1));
  });
});

test.describe('regola di scelta', () => {
  test('greedy di default; l\'estrazione è dichiarata e mostra le due probabilità', async ({ page }) => {
    await openMock(page);
    await setSpeed(page, 'fast');
    await runToEnd(page, 'Perché il cielo è blu?');
    await expect(page.getByTestId('policy-line')).toContainText('sempre il token più probabile');
    await expect(page.getByTestId('chart-legend')).toBeHidden();
    await expect(page.locator('[data-testid="chart"] .v-policy')).toHaveCount(0);

    await page.locator('#settings-panel > summary').click();
    await page.getByTestId('policy-sample').check();
    await page.getByTestId('seed').fill('7');
    await runToEnd(page, 'Perché il cielo è blu?');
    await expect(page.getByTestId('policy-line')).toContainText('estrazione casuale pesata');
    await expect(page.getByTestId('chart-legend')).toBeVisible();
    await expect(page.getByTestId('chart-caption')).toContainText('probabilità di estrazione');
    const trace = await demo(page, () => window.__nextTokenDemo.trace);
    expect(trace.policy.kind).toBe('sample');
    expect(trace.seed).toBe(7);
    for (const s of trace.steps) {
      expect(s.selected.draw).toBeGreaterThanOrEqual(0);
      const poolSum = s.candidates.reduce((a, c) => a + c.policyProb, 0);
      expect(poolSum).toBeLessThanOrEqual(1 + 1e-9);
    }
    await expect(page.getByTestId('choice')).toContainText('Estratto');
  });
});

test.describe('input del modello', () => {
  test('le parti aggiunte dall\'app sono riassunte e consultabili, con gli ID su richiesta', async ({ page }) => {
    await openMock(page);
    await setSpeed(page, 'fast');
    await runToEnd(page, 'Perché il cielo è blu?');
    const tokens = page.getByTestId('tokens');
    await expect(tokens.locator('.tok.summary').first()).toContainText("token dell'app");
    // nell'input riassunto i token di controllo dell'app non compaiono (solo quelli generati)
    await expect(tokens.locator('.tok.ctl:not(.o-generated)')).toHaveCount(0);
    await page.getByTestId('btn-full-input').click();
    await expect(page.getByTestId('btn-full-input')).toHaveAttribute('aria-expanded', 'true');
    await expect(tokens.locator('.tok.ctl').first()).toBeVisible();
    await expect(page.getByTestId('full-input-note')).toContainText('<|turn>system');
    const ids = tokens.locator('.tok-id').first();
    await expect(ids).toBeHidden();
    await page.getByTestId('show-ids').check();
    await expect(ids).toBeVisible();
    const chip = tokens.locator('.tok').first();
    expect(await chip.locator('.tok-id').textContent()).toBe(await chip.getAttribute('data-id'));
  });
});

test.describe('preparazione del modello (senza download)', () => {
  test('senza WebGPU: messaggio chiaro, nessun download automatico, processore solo su scelta esplicita', async ({ page }) => {
    const hub = [];
    page.on('request', (r) => {
      if (r.url().includes('huggingface.co')) hub.push(r.url());
    });
    await page.addInitScript(() => {
      Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true });
    });
    await page.goto('./');
    await expect(page.getByTestId('setup-error')).toContainText('WebGPU');
    // si offre solo, come scelta esplicita, il modello piccolissimo per il processore
    await expect(page.getByTestId('model-choice').locator('input[name="model"]')).toHaveCount(1);
    await expect(page.getByTestId('model-smollm2-135m')).toBeChecked();
    await expect(page.getByTestId('model-choice')).toContainText('processore');
    await expect(page.getByTestId('btn-load')).toContainText('SmolLM2 135M');
    await expect(page.getByTestId('btn-primary')).toBeDisabled();
    await page.waitForTimeout(500);
    expect(hub).toEqual([]);
  });

  test('con WebGPU: dimensioni dichiarate prima, nessuna richiesta finché non si preme', async ({ page }) => {
    const hub = [];
    page.on('request', (r) => {
      if (!r.url().startsWith('http://localhost')) hub.push(r.url());
    });
    await page.addInitScript(() => {
      const adapter = {
        features: new Set(['shader-f16']),
        limits: { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 31 - 4 },
        info: { vendor: 'test', architecture: 'test' },
      };
      Object.defineProperty(Navigator.prototype, 'gpu', {
        get: () => ({ requestAdapter: async () => adapter }),
        configurable: true,
      });
      Object.defineProperty(Navigator.prototype, 'deviceMemory', { get: () => 32, configurable: true });
    });
    await page.goto('./');
    await expect(page.getByTestId('btn-load')).toContainText('Gemma 4 E2B');
    await expect(page.getByTestId('btn-load')).toContainText('3,1 GB');
    await expect(page.getByTestId('model-choice')).toContainText('579 MB');
    await page.getByTestId('model-qwen3-0.6b').check();
    await expect(page.getByTestId('btn-load')).toContainText('Qwen3 0.6B');
    await expect(page.getByTestId('btn-load')).toContainText('MB');
    await page.waitForTimeout(500);
    expect(hub).toEqual([]);
  });

  test('con poca memoria dichiarata viene suggerito il modello leggero', async ({ page }) => {
    await page.addInitScript(() => {
      const adapter = {
        features: new Set(['shader-f16']),
        limits: { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 31 - 4 },
        info: {},
      };
      Object.defineProperty(Navigator.prototype, 'gpu', { get: () => ({ requestAdapter: async () => adapter }), configurable: true });
      Object.defineProperty(Navigator.prototype, 'deviceMemory', { get: () => 8, configurable: true });
    });
    await page.goto('./');
    await expect(page.getByTestId('model-qwen3-0.6b')).toBeChecked();
    await expect(page.getByTestId('model-choice')).toContainText('suggerito per questo computer');
  });
});

test.describe('preparazione: hardware modesto', () => {
  test('con pochissima memoria dichiarata viene suggerito il modello minimo', async ({ page }) => {
    await page.addInitScript(() => {
      const adapter = {
        features: new Set(['shader-f16']),
        limits: { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 31 - 4 },
        info: {},
      };
      Object.defineProperty(Navigator.prototype, 'gpu', { get: () => ({ requestAdapter: async () => adapter }), configurable: true });
      Object.defineProperty(Navigator.prototype, 'deviceMemory', { get: () => 4, configurable: true });
    });
    await page.goto('./');
    await expect(page.getByTestId('model-gemma-3-270m')).toBeChecked();
    await expect(page.getByTestId('model-choice')).toContainText('Minimo');
    await expect(page.getByTestId('btn-load')).toContainText('Gemma 3 270M');
  });
});

test.describe('impostazioni della chatbot', () => {
  test.beforeEach(async ({ page }) => {
    await openMock(page);
  });

  test('riepilogo sempre visibile; temperatura e top-p inattive con la regola «il più probabile»', async ({ page }) => {
    await expect(page.getByTestId('settings-recap')).toHaveText('Scelta: sempre il più probabile · Calcolo: KV cache · Max 48 token');
    await page.locator('#settings-panel > summary').click();
    await expect(page.getByTestId('temperature')).toBeDisabled();
    await expect(page.getByTestId('temperature-off')).toBeVisible();
    await page.getByTestId('policy-sample').check();
    await expect(page.getByTestId('temperature')).toBeEnabled();
    await expect(page.getByTestId('settings-recap')).toContainText('estrazione (temperatura 1, top-p 0,95)');
    await expect(page.getByTestId('policy-explain')).toContainText('a sorte');
    // passando sopra un'impostazione si evidenzia la fase dell'anatomia
    await page.locator('#temperature-setting').hover();
    await expect(page.locator('#anatomy [data-stage="select"].setting-hint').first()).toBeVisible();
  });

  test('ricalcolo completo: a ogni passo il modello rielabora tutta la sequenza, e lo dice', async ({ page }) => {
    await page.locator('#settings-panel > summary').click();
    await page.getByTestId('cache-naive').check();
    await expect(page.getByTestId('settings-recap')).toContainText('ricalcolo completo');
    await setSpeed(page, 'fast');
    await start(page, 'Perché il cielo è blu?');
    for (let i = 0; i < 3; i++) {
      await page.getByTestId('btn-primary').click();
      await page.waitForFunction((n) => window.__nextTokenDemo.trace.steps.length === n, i + 1);
      await expect(page.getByTestId('btn-primary')).toBeEnabled();
    }
    const trace = await demo(page, () => window.__nextTokenDemo.trace);
    expect(trace.cacheMode).toBe('naive');
    trace.steps.forEach((st, i) => {
      expect(st.cachedTokens).toBe(0);
      expect(st.processedTokens).toBe(trace.input.ids.length + i);
    });
    await expect(page.locator('#narration .nar-list > .nar-item[data-n="4"]')).toContainText('ricalcolato da capo');
  });

  test('cronologia: una riga per token, la scelta del presentatore sostituisce la riga', async ({ page }) => {
    await setSpeed(page, 'fast');
    await start(page, 'Perché il cielo è blu?');
    for (let i = 0; i < 2; i++) {
      await page.getByTestId('btn-primary').click();
      await page.waitForFunction((n) => window.__nextTokenDemo.trace.steps.length === n, i + 1);
      await expect(page.getByTestId('btn-primary')).toBeEnabled();
    }
    await page.getByTestId('history').locator('summary').click();
    const rows = page.getByTestId('history-body').locator('tr');
    await expect(rows).toHaveCount(2);
    await expect(rows.nth(1)).toContainText('regola: il più probabile');
    await page.locator('[data-testid="chart"] li.row').nth(2).click();
    await expect(page.getByTestId('choice')).toContainText('Scelto da te');
    await expect(rows).toHaveCount(2);
    await expect(rows.nth(1)).toContainText('te (presentatore)');
    await expect(rows.nth(1)).toContainText('3°');
  });
});
