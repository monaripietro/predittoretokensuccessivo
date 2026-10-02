import { test, expect } from '@playwright/test';
import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const DIST_ASSETS = fileURLToPath(new URL('../../dist/assets/', import.meta.url));

test.describe('Invariante di deploy (build di produzione)', () => {
  test('la pagina carica senza errori in console né risorse mancanti', async ({ page }) => {
    const errors = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', (err) => errors.push(String(err)));
    const failed = [];
    page.on('requestfailed', (req) => failed.push(req.url()));
    page.on('response', (res) => {
      if (res.status() >= 400) failed.push(`${res.status()} ${res.url()}`);
    });
    await page.goto('./?mock');
    await expect(page.getByTestId('model-chip')).toHaveAttribute('data-state', 'sim');
    expect(errors).toEqual([]);
    expect(failed).toEqual([]);
  });

  test('percorsi relativi e nessun URL http:// nel sorgente pubblicato', async ({ request }) => {
    const html = await (await request.get('./')).text();
    expect(html).not.toMatch(/(src|href)="http:\/\//);
    expect(html).not.toMatch(/(src|href)="\/(?!\/)/);
  });

  test('il runtime ONNX è servito dallo stesso sito, in una sola copia', async ({ request }) => {
    const files = readdirSync(DIST_ASSETS);
    const wasm = files.filter((f) => f.endsWith('.wasm'));
    const mjs = files.filter((f) => f.startsWith('ort-wasm') && f.endsWith('.mjs'));
    expect(wasm).toHaveLength(1);
    expect(mjs).toHaveLength(1);
    expect((await request.get(`./assets/${wasm[0]}`)).status()).toBe(200);
    expect((await request.get(`./assets/${mjs[0]}`)).status()).toBe(200);
  });

  test('la pagina di diagnostica è pubblicata', async ({ request }) => {
    const res = await request.get('./diagnostics.html');
    expect(res.status()).toBe(200);
    expect(await res.text()).toContain('Diagnostica del modello locale');
  });
});
