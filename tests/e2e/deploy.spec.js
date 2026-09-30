import { test, expect } from '@playwright/test';

test.describe('Invariante di deploy (server locale)', () => {
  test('la pagina carica CSS e JS senza errori console né mixed content', async ({ page }) => {
    const errors = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', (err) => errors.push(String(err)));
    const failed = [];
    page.on('requestfailed', (req) => failed.push(req.url()));
    await page.goto('./?mock');
    await page.waitForFunction(() => document.getElementById('backend-indicator').textContent.includes('mock'));
    expect(errors).toEqual([]);
    expect(failed).toEqual([]);
    const sheet = await page.evaluate(() => document.styleSheets.length);
    expect(sheet).toBeGreaterThan(0);
  });

  test('nessun URL http:// né percorso assoluto /assets nel sorgente pubblicato', async ({ request }) => {
    const res = await request.get('./');
    const html = await res.text();
    expect(html).not.toMatch(/src="http:\/\//);
    expect(html).not.toMatch(/href="http:\/\//);
    const css = await (await request.get('./style.css')).text();
    expect(css).not.toMatch(/url\(["']?http:\/\//);
  });

  test('il loop principale funziona sul server di preview (base path relativo)', async ({ page }) => {
    await page.goto('./?mock');
    await page.waitForFunction(() => document.getElementById('backend-indicator').textContent.includes('mock'));
    await page.getByTestId('prompt').fill('test deploy');
    for (let i = 1; i <= 3; i++) {
      const top = page.getByTestId('btn-next');
      if (await top.isVisible()) await top.click();
      else await page.getByTestId('btn-next-bottom').click();
      await expect(page.getByTestId('next-step-context').locator('.chip')).toHaveCount(2 + i);
    }
    const msg = await page.getByTestId('next-step-explanation').textContent();
    expect(msg).toContain('5 token di contesto');
  });
});
