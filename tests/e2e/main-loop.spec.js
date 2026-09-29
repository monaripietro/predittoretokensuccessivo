import { test, expect } from '@playwright/test';

test.describe('Ciclo principale (mock)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('./?mock');
    await expect(page.getByTestId('btn-next')).toBeDisabled();
    await page.waitForFunction(() => document.getElementById('backend-indicator').textContent.includes('mock'));
  });

  test('prompt vuoto: pulsante disabilitato', async ({ page }) => {
    await expect(page.getByTestId('btn-next')).toBeDisabled();
  });

  test('mock riconoscibile: badge e avviso visibili', async ({ page }) => {
    await expect(page.getByTestId('mock-badge')).toBeVisible();
    await expect(page.getByTestId('mock-badge')).toHaveText('MOCK');
    await expect(page.getByTestId('mock-warning')).toBeVisible();
    await expect(page.getByTestId('mock-warning')).toContainText('simulati');
  });

  test('prompt mostrato come token e ID prima del calcolo', async ({ page }) => {
    await page.getByTestId('prompt').fill('Il cielo è');
    const chips = page.getByTestId('prompt-preview').locator('.chip');
    await expect(chips).toHaveCount(3);
    const first = chips.first();
    expect(await first.getAttribute('title')).toContain('Token ID:');
    const ids = await page.getByTestId('prompt-preview').locator('.chip-id').allTextContents();
    expect(ids.length).toBe(3);
    for (const id of ids) expect(id).toMatch(/^\d+$/);
    await expect(page.locator('.preview-count')).toHaveText('3 token');
    await expect(page.getByTestId('preview-meta')).toContainText('3 token · tokenizer');
  });

  test('primo click: token, ID, ranking e percentuali visibili', async ({ page }) => {
    await page.getByTestId('prompt').fill('Il cielo è');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(4);
    await expect(page.getByTestId('step-result')).toBeVisible();
    const rows = page.getByTestId('ranking-body').locator('tr');
    expect(await rows.count()).toBeGreaterThan(0);
    await expect(rows.first().locator('.tag')).toContainText('scelto');
    const probText = await rows.first().locator('.prob').textContent();
    expect(probText).toMatch(/\d/);
    const chip = page.getByTestId('token-sequence').locator('.chip').first();
    expect(await chip.getAttribute('title')).toContain('Token ID:');
  });

  test('token scelto: scheda con testo grande, ID, probabilità e rank', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b');
    await page.getByTestId('btn-next').click();
    const card = page.getByTestId('chosen-card');
    await expect(card).toBeVisible();
    const big = await card.locator('#chosen-big').textContent();
    expect(big).toBeTruthy();
    const idText = await card.locator('#chosen-id').textContent();
    expect(idText).toMatch(/^\d+$/);
    const probText = await card.locator('#chosen-prob').textContent();
    expect(probText).toMatch(/%\s*$/);
    const rankText = await card.locator('#chosen-rank').textContent();
    expect(rankText).toMatch(/^#\d+$/);
    await expect(page.getByTestId('params-applied')).toContainText('Parametri applicati: temperatura');
  });

  test('coerenza: token scelto in scheda e in classifica hanno lo stesso ID', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b');
    await page.getByTestId('btn-next').click();
    const chosenId = await page.locator('#chosen-id').textContent();
    const chosenRowId = await page.getByTestId('ranking-body').locator('tr.chosen-row td:nth-child(3)').textContent();
    expect(chosenId).toBe(chosenRowId);
  });

  test('token generato ha classe di stile distinta', async ({ page }) => {
    await page.getByTestId('prompt').fill('Il cielo');
    await page.getByTestId('btn-next').click();
    const chips = page.getByTestId('token-sequence').locator('.chip');
    await expect(chips.nth(0)).toHaveClass(/chip prompt/);
    await expect(chips.nth(2)).toHaveClass(/generated/);
  });

  test('tre click: il contesto cresce di uno per passo', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b c');
    const seq = page.getByTestId('token-sequence');
    for (let i = 1; i <= 3; i++) {
      await page.getByTestId('btn-next').click();
      await expect(seq.locator('.chip')).toHaveCount(3 + i);
      const msg = await page.getByTestId('context-message').textContent();
      expect(msg).toContain(`${3 + i} token`);
    }
    const promptChips = seq.locator('.chip.prompt');
    await expect(promptChips).toHaveCount(3);
  });

  test('stato cache a tre livelli: contesto logico, calcolo corrente, stato', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b c');
    await page.getByTestId('btn-next').click();
    const box = page.getByTestId('cache-status');
    await expect(box).toBeVisible();
    await expect(box).toContainText('Contesto logico: 4 token');
    await expect(box).toContainText('Calcolo corrente: 3 token');
    await expect(box).toContainText('KV cache: attiva');
    // secondo passo: calcolo corrente = 1 token
    await page.getByTestId('btn-next-bottom').click();
    await expect(box).toContainText('Calcolo corrente: 1 token');
  });

  test('modifica prompt: sequenza attenuata, avviso e invalidazione', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b c');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(4);
    await page.getByTestId('prompt').fill('a b c d');
    await expect(page.getByTestId('prompt-hint')).toBeVisible();
    await expect(page.getByTestId('prompt-hint')).toContainText('generazione precedente è stata annullata');
    await expect(page.getByTestId('token-sequence')).toHaveClass(/stale/);
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(5);
    await expect(page.getByTestId('token-sequence').locator('.chip.generated')).toHaveCount(1);
    await expect(page.getByTestId('token-sequence').locator('.chip.prompt')).toHaveCount(4);
    await expect(page.getByTestId('token-sequence')).not.toHaveClass(/stale/);
  });

  test('reset: risultati eliminati e pulsante di nuovo disabilitato', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b c');
    await page.getByTestId('btn-next').click();
    await page.getByTestId('btn-reset').click();
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(0);
    await expect(page.getByTestId('step-result')).toBeHidden();
    await expect(page.getByTestId('btn-next')).toBeDisabled();
    await expect(page.getByTestId('cache-status')).toBeHidden();
  });

  test('cambio parametro: nota che sarà applicato al prossimo calcolo', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b c');
    await page.getByTestId('btn-next').click();
    await page.locator('#params-panel summary').click();
    await page.locator('#param-temperature').fill('0.5');
    await expect(page.getByTestId('params-hint')).toBeVisible();
    await expect(page.getByTestId('params-hint')).toContainText('applicato al prossimo calcolo');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('params-hint')).toBeHidden();
  });

  test('modalità naive e cache: stesso contesto logico, testo diverso', async ({ page }) => {
    await page.locator('#params-panel summary').click();
    await page.getByTestId('prompt').fill('a b c');
    await page.getByTestId('param-cachemode').selectOption('naive');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('cache-status')).toContainText('non usata');
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(4);
    await page.getByTestId('param-cachemode').selectOption('cache');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('cache-status')).toContainText('attiva');
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(5);
  });
});

test.describe('Errori runtime', () => {
  test('errore runtime: messaggio visibile e stato recuperabile', async ({ page }) => {
    await page.goto('./?mock');
    await page.waitForFunction(() => document.getElementById('backend-indicator').textContent.includes('mock'));
    await page.getByTestId('prompt').fill('a b c');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(4);
    await page.evaluate(() => {
      document.getElementById('btn-next').addEventListener('click', () => {
        const el = document.getElementById('error');
        el.textContent = 'Errore runtime: simulato';
        el.hidden = false;
      }, { once: true });
    });
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('error')).toBeVisible();
    await expect(page.getByTestId('btn-next')).toBeEnabled();
    await page.getByTestId('btn-reset').click();
    await expect(page.getByTestId('error')).toBeHidden();
    await expect(page.getByTestId('btn-next')).toBeDisabled();
  });
});

test.describe('Loop ricorsivo e pulsante in fondo', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('./?mock');
    await page.waitForFunction(() => document.getElementById('backend-indicator').textContent.includes('mock'));
  });

  test('dieci click dal pulsante in fondo: contesto cresce di uno per passo', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b c');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(4);
    for (let i = 2; i <= 10; i++) {
      await page.getByTestId('btn-next-bottom').click();
      await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(3 + i);
    }
    const msg = await page.getByTestId('context-message').textContent();
    expect(msg).toContain('13 token');
    expect(await page.getByTestId('btn-next-bottom').isEnabled()).toBe(true);
  });

  test('ranking limitato a 5 risultati di default con testo leggibile', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b');
    await page.getByTestId('btn-next').click();
    const rows = page.getByTestId('ranking-body').locator('tr');
    await expect(rows).toHaveCount(5);
    const first = await rows.first().locator('.token-text').textContent();
    expect(first).not.toMatch(/^<tok-\d+>$/);
  });

  test('il secondo click conserva esattamente il prompt', async ({ page }) => {
    await page.getByTestId('prompt').fill('uno due tre');
    await page.getByTestId('btn-next').click();
    await page.getByTestId('btn-next').click();
    const promptChips = page.getByTestId('token-sequence').locator('.chip.prompt');
    await expect(promptChips).toHaveCount(3);
    const texts = await promptChips.allTextContents();
    expect(texts.map((t) => t.replace(/\s*\d+$/, ''))).toEqual(['uno', 'due', 'tre']);
  });
});

test.describe('Termine generazione (EOS)', () => {
  test('testo del pulsante riflette il contesto e il reset riabilita', async ({ page }) => {
    await page.goto('./?mock');
    await page.waitForFunction(() => document.getElementById('backend-indicator').textContent.includes('mock'));
    await page.getByTestId('prompt').fill('a b');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(3);
    const btnBottom = page.getByTestId('btn-next-bottom');
    await btnBottom.click();
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(4);
    await expect(btnBottom).toContainText('Calcola token successivo');
    await page.getByTestId('btn-reset').click();
    await expect(page.getByTestId('btn-next-bottom')).toBeDisabled();
    await expect(page.getByTestId('btn-next')).toBeDisabled();
  });
});

test.describe('Miglioramenti a priorità bassa', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('./?mock');
    await page.waitForFunction(() => document.getElementById('backend-indicator').textContent.includes('mock'));
  });
  test('barre probabilità presenti nella classifica', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b');
    await page.getByTestId('btn-next').click();
    const bars = page.getByTestId('ranking-body').locator('.prob-bar');
    await expect(bars).toHaveCount(5);
    const firstFill = bars.first().locator('.prob-bar-fill');
    await expect(firstFill).toHaveAttribute('style', /width:\d+%/);
  });

  test('preset prompt: cliccare un esempio riempie la textarea e abilita il pulsante', async ({ page }) => {
    await page.locator('.preset').first().click();
    await expect(page.getByTestId('prompt')).toHaveValue('Il cielo è');
    await expect(page.getByTestId('btn-next')).toBeEnabled();
  });

  test('cronologia: righe con passo, token, contesto e delta dal secondo step', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b c');
    await page.getByTestId('btn-next').click();
    await page.getByTestId('btn-next-bottom').click();
    await page.getByTestId('btn-next-bottom').click();
    const rows = page.getByTestId('history-body').locator('tr');
    await expect(rows).toHaveCount(3);
    const firstDelta = await rows.first().locator('td').nth(6).textContent();
    expect(firstDelta).toBe('—');
    const secondDelta = await rows.nth(1).locator('td').nth(6).textContent();
    expect(secondDelta).toMatch(/^[+−]\d+,\d+%$/);
  });

  test('esportazione cronologia: click genera download JSON', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b');
    await page.getByTestId('btn-next').click();
    await page.locator('#history-panel summary').click();
    const download = page.waitForEvent('download');
    await page.getByTestId('btn-export').click();
    const dl = await download;
    expect(dl.suggestedFilename()).toBe('next-token-lab-cronologia.json');
  });

  test('animazione: il token appena generato ha la classe just-added', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b');
    const lastChip = page.getByTestId('token-sequence').locator('.chip.last-generated');
    const classPromise = lastChip.evaluate((el) => new Promise((resolve) => {
      if (el.classList.contains('just-added')) resolve(true);
      else el.addEventListener('animationend', () => resolve(el.classList.contains('just-added')));
    }));
    await page.getByTestId('btn-next').click();
    expect(await classPromise).toBe(true);
  });
});

test.describe('Pulsante inferiore', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('./?mock');
    await page.waitForFunction(() => document.getElementById('backend-indicator').textContent.includes('mock'));
  });

  test('il pulsante inferiore continua la generazione dopo il primo step', async ({ page }) => {
    const prompt = page.getByTestId('prompt');
    const bottomButton = page.getByTestId('btn-next-bottom');

    await prompt.fill('Il cielo è');
    await expect(bottomButton).toBeEnabled();

    await page.getByTestId('btn-next').click();

    await expect(page.getByTestId('step-result')).toBeVisible();
    await expect(bottomButton).toBeEnabled();

    const sequenceBefore = await page
      .getByTestId('token-sequence')
      .locator('.chip')
      .count();

    await bottomButton.click();

    await expect
      .poll(async () => page.getByTestId('token-sequence').locator('.chip').count())
      .toBe(sequenceBefore + 1);

    await expect(bottomButton).toBeEnabled();
  });

  test('i due pulsanti hanno sempre lo stesso stato di abilitazione', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b');
    const btn = page.getByTestId('btn-next');
    const btnBottom = page.getByTestId('btn-next-bottom');
    expect(await btn.isDisabled()).toBe(await btnBottom.isDisabled());
    await btn.click();
    await expect(page.getByTestId('step-result')).toBeVisible();
    expect(await btn.isDisabled()).toBe(await btnBottom.isDisabled());
    await page.getByTestId('prompt').fill('');
    expect(await btn.isDisabled()).toBe(await btnBottom.isDisabled());
    expect(await btn.isDisabled()).toBe(true);
    await page.getByTestId('prompt').fill('a b c');
    expect(await btn.isDisabled()).toBe(await btnBottom.isDisabled());
  });
});
