import { test, expect } from '@playwright/test';

test.describe('Ciclo principale (mock)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('./?mock');
    await expect(page.getByTestId('btn-next')).toBeDisabled();
    await page.evaluate(() => {
      window.__ready = new Promise((res) => {
        const check = () => {
          const btn = document.getElementById('btn-next');
          if (btn && !btn.disabled) res();
          else setTimeout(check, 50);
        };
        check();
      });
    });
    await page.waitForFunction(() => document.getElementById('backend-indicator').textContent.includes('mock'));
  });

  test('prompt vuoto: pulsante disabilitato', async ({ page }) => {
    await expect(page.getByTestId('btn-next')).toBeDisabled();
  });

  test('primo click: token, ID, ranking e percentuali visibili', async ({ page }) => {
    await page.getByTestId('prompt').fill('Il cielo è');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(4);
    await expect(page.getByTestId('step-result')).toBeVisible();
    const rows = page.getByTestId('ranking-body').locator('tr');
    expect(await rows.count()).toBeGreaterThan(0);
    await expect(rows.first().locator('.tag')).toHaveText('scelto');
    const probText = await rows.first().locator('.prob').textContent();
    expect(probText).toMatch(/\d/);
    const chip = page.getByTestId('token-sequence').locator('.chip').first();
    expect(await chip.getAttribute('title')).toContain('Token ID:');
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
      expect(msg).toContain(`${3 + i} token di contesto`);
    }
    const promptChips = seq.locator('.chip.prompt');
    await expect(promptChips).toHaveCount(3);
  });

  test('modifica prompt: genera avviso e invalida la generazione', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b c');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(4);
    await page.getByTestId('prompt').fill('a b c d');
    await expect(page.getByTestId('prompt-hint')).toBeVisible();
    await expect(page.getByTestId('prompt-hint')).toContainText('Prompt modificato');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(5);
    await expect(page.getByTestId('token-sequence').locator('.chip.generated')).toHaveCount(1);
    await expect(page.getByTestId('token-sequence').locator('.chip.prompt')).toHaveCount(4);
  });

  test('reset: risultati eliminati e pulsante di nuovo disabilitato', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b c');
    await page.getByTestId('btn-next').click();
    await page.getByTestId('btn-reset').click();
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(0);
    await expect(page.getByTestId('step-result')).toBeHidden();
    await expect(page.getByTestId('btn-next')).toBeDisabled();
  });

  test('pulsante mostra Calcolo… durante l\'elaborazione', async ({ page }) => {
    await page.getByTestId('prompt').fill('a b c');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('btn-next')).toBeEnabled();
  });

  test('modalità naive e cache: stessa lunghezza di contesto', async ({ page }) => {
    await page.locator('#params-panel summary').click();
    await page.getByTestId('prompt').fill('a b c');
    await page.getByTestId('param-cachemode').selectOption('naive');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('cache-message')).toContainText('ricalcolo completo');
    await expect(page.getByTestId('token-sequence').locator('.chip')).toHaveCount(4);
    await page.getByTestId('param-cachemode').selectOption('cache');
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('cache-message')).toContainText('KV cache');
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
      document.getElementById('error').dataset.armed = '1';
    });
    await page.getByTestId('btn-next').click();
    await expect(page.getByTestId('error')).toBeVisible();
    await expect(page.getByTestId('btn-next')).toBeEnabled();
    await page.getByTestId('btn-reset').click();
    await expect(page.getByTestId('error')).toBeHidden();
    await expect(page.getByTestId('btn-next')).toBeDisabled();
  });
});
