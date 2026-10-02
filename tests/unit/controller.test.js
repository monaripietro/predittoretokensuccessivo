import { describe, it, expect } from 'vitest';
import { createController, STATES } from '../../src/core/controller.js';
import { createMockClient } from '../../src/engine/mockClient.js';

function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
}

async function waitFor(fn, ms = 2000) {
  const t0 = Date.now();
  while (!fn()) {
    if (Date.now() - t0 > ms) throw new Error('timeout');
    await new Promise((r) => setTimeout(r, 2));
  }
}

async function setup({ present, modelOptions } = {}) {
  const client = createMockClient({ delayMs: 1, modelOptions });
  await client.call('load');
  const events = [];
  const presented = [];
  const ctl = createController({
    client,
    present: present ?? (async (step, ctx) => { presented.push({ index: step.index, replay: ctx.replay }); }),
    onEvent: (e) => events.push(e),
  });
  return { client, ctl, events, presented };
}

const count = (client, type) => client.calls.filter((c) => c === type).length;

describe('regia: un token alla volta (predefinito)', () => {
  it("dopo l'avvio prepara l'input e aspetta: nessun passo del modello senza un comando", async () => {
    const { client, ctl, presented } = await setup();
    await ctl.start({ text: 'Perché il cielo è blu?', policy: { kind: 'greedy' } });
    expect(ctl.state).toBe(STATES.PAUSED);
    await new Promise((r) => setTimeout(r, 40));
    expect(count(client, 'step')).toBe(0);
    expect(presented).toEqual([]);
    expect(ctl.trace.input.ids.length).toBeGreaterThan(0);
  });

  it('ogni next() calcola e mostra esattamente un token, poi si ferma di nuovo', async () => {
    const { client, ctl, presented } = await setup();
    await ctl.start({ text: 'Perché il cielo è blu?', policy: { kind: 'greedy' } });
    for (let i = 1; i <= 3; i++) {
      await ctl.next();
      expect(count(client, 'step')).toBe(i);
      expect(presented.length).toBe(i);
      expect(ctl.state).toBe(STATES.PAUSED);
      await new Promise((r) => setTimeout(r, 20));
      expect(count(client, 'step')).toBe(i);
    }
  });

  it('«continua da solo» prosegue fino alla fine; si può tornare a un token alla volta', async () => {
    const { ctl } = await setup();
    await ctl.start({ text: "Qual è la capitale d'Italia?", policy: { kind: 'greedy' } });
    await ctl.next();
    ctl.resume();
    await waitFor(() => ctl.state === STATES.DONE);
    expect(ctl.trace.finish).toBe('eos');
  });

  it("il replay predefinito procede anch'esso un token per volta, senza chiamare il modello", async () => {
    const { client, ctl, presented } = await setup();
    await ctl.start({ text: 'Quante zampe ha un ragno?', policy: { kind: 'greedy' }, autoplay: true });
    await waitFor(() => ctl.state === STATES.DONE);
    const calls = client.calls.length;
    presented.length = 0;
    ctl.replay();
    await waitFor(() => ctl.state === STATES.REPLAY_PAUSED && !ctl.busy);
    expect(presented.length).toBe(0);
    await ctl.next();
    await ctl.next();
    expect(presented.map((p) => p.replay)).toEqual([true, true]);
    expect(client.calls.length).toBe(calls);
  });
});

describe('regia: scelta del presentatore', () => {
  it('in pausa sostituisce il token dell’ultimo passo senza chiamare il modello', async () => {
    const { client, ctl, presented } = await setup({
      present: async (step, ctx) => { presented.push({ index: step.index, override: Boolean(ctx.override), id: step.selected.id }); },
    });
    await ctl.start({ text: 'Perché il cielo è blu?', policy: { kind: 'greedy' } });
    await ctl.next();
    const first = ctl.trace.steps[0];
    const alt = first.candidates[1].id;
    const steps = count(client, 'step');
    expect(await ctl.choose(alt)).toBe(true);
    expect(count(client, 'step')).toBe(steps);
    expect(ctl.trace.steps).toHaveLength(1);
    expect(ctl.trace.steps[0].selected.id).toBe(alt);
    expect(ctl.trace.steps[0].override.original.id).toBe(first.selected.id);
    expect(presented.at(-1)).toEqual({ index: 0, override: true, id: alt });
    expect(ctl.state).toBe(STATES.PAUSED);
    await ctl.next();
    expect(ctl.trace.steps[1].contextLength).toBe(ctl.trace.input.ids.length + 1);
  });

  it('fuori dalla pausa la scelta non è permessa', async () => {
    const { ctl } = await setup();
    expect(await ctl.choose(5)).toBe(false);
  });
});

describe('regia: generazione', () => {
  it('genera fino al token di fine, un passo reale per ogni token mostrato', async () => {
    const { client, ctl, presented } = await setup();
    await ctl.start({ text: "Qual è la capitale d'Italia?", policy: { kind: 'greedy' }, autoplay: true });
    await waitFor(() => ctl.state === STATES.DONE);
    expect(ctl.trace.finish).toBe('eos');
    expect(presented.length).toBe(ctl.trace.steps.length);
    expect(count(client, 'step')).toBe(ctl.trace.steps.length);
    expect(count(client, 'end')).toBe(1);
  });

  it('pausa ferma i calcoli; "un token" esegue esattamente un passo; riprendi continua', async () => {
    const gate = deferred();
    let n = 0;
    const { client, ctl } = await setup({
      present: async () => {
        n += 1;
        if (n === 2) await gate.promise;
      },
    });
    await ctl.start({ text: 'Perché il cielo è blu?', policy: { kind: 'greedy' }, autoplay: true });
    await waitFor(() => n === 2);
    ctl.pause();
    gate.resolve();
    await waitFor(() => !ctl.busy);
    expect(ctl.state).toBe(STATES.PAUSED);
    const before = count(client, 'step');
    await new Promise((r) => setTimeout(r, 30));
    expect(count(client, 'step')).toBe(before);
    await ctl.next();
    expect(count(client, 'step')).toBe(before + 1);
    expect(ctl.state).toBe(STATES.PAUSED);
    ctl.resume();
    await waitFor(() => ctl.state === STATES.DONE);
    expect(ctl.trace.finish).toBe('eos');
  });

  it('stop durante un calcolo scarta quel risultato e libera la cache', async () => {
    const { client, ctl } = await setup({ modelOptions: { delayMs: 25 } });
    await ctl.start({ text: 'Perché il cielo è blu?', policy: { kind: 'greedy' }, autoplay: true });
    await waitFor(() => ctl.trace.steps.length >= 2);
    await waitFor(() => ctl.busy);
    const shown = ctl.trace.steps.length;
    await ctl.stop();
    await waitFor(() => ctl.state === STATES.DONE);
    expect(ctl.trace.finish).toBe('stopped');
    expect(ctl.trace.discardedInFlight).toBe(true);
    expect(ctl.trace.steps.length).toBe(shown);
    expect(count(client, 'end')).toBe(1);
    expect(client.tf.created.model.caches.every((c) => c.disposed)).toBe(true);
  });

  it('un errore del modello termina la generazione con stato di errore', async () => {
    const { ctl, events } = await setup({ modelOptions: { failOnCall: 3 } });
    await ctl.start({ text: 'Perché il cielo è blu?', policy: { kind: 'greedy' }, autoplay: true });
    await waitFor(() => ctl.state === STATES.DONE);
    expect(ctl.trace.finish).toBe('error');
    expect(ctl.trace.error).toMatch(/simulato/);
    expect(events.some((e) => e.type === 'finish' && e.reason === 'error')).toBe(true);
  });

  it('un input non valido non avvia nulla', async () => {
    const { ctl, events } = await setup();
    const ok = await ctl.start({ text: '   ', policy: { kind: 'greedy' } });
    expect(ok).toBe(false);
    expect(ctl.state).toBe(STATES.IDLE);
    expect(events.some((e) => e.type === 'error')).toBe(true);
  });
});

describe('regia: replay della registrazione', () => {
  it('riproduce tutti i passi registrati senza chiamare il modello', async () => {
    const { client, ctl, presented } = await setup();
    await ctl.start({ text: 'Quante zampe ha un ragno?', policy: { kind: 'greedy' }, autoplay: true });
    await waitFor(() => ctl.state === STATES.DONE);
    const callsBefore = client.calls.length;
    const recorded = ctl.trace.steps.map((s) => s.index);
    presented.length = 0;
    expect(ctl.replay({ autoplay: true })).toBe(true);
    await waitFor(() => ctl.state === STATES.DONE);
    expect(client.calls.length).toBe(callsBefore);
    expect(presented.map((p) => p.index)).toEqual(recorded);
    expect(presented.every((p) => p.replay)).toBe(true);
  });

  it('il replay si può mettere in pausa, avanzare di un token e fermare', async () => {
    const gate = deferred();
    let n = 0;
    const { client, ctl } = await setup({
      present: async (step, ctx) => {
        if (ctx.replay) {
          n += 1;
          if (n === 1) await gate.promise;
        }
      },
    });
    await ctl.start({ text: 'Quante zampe ha un ragno?', policy: { kind: 'greedy' }, autoplay: true });
    await waitFor(() => ctl.state === STATES.DONE);
    const callsBefore = client.calls.length;
    ctl.replay({ autoplay: true });
    await waitFor(() => n === 1);
    ctl.pause();
    gate.resolve();
    await waitFor(() => !ctl.busy);
    expect(ctl.state).toBe(STATES.REPLAY_PAUSED);
    await ctl.next();
    expect(n).toBe(2);
    await ctl.stop();
    expect(ctl.state).toBe(STATES.DONE);
    expect(client.calls.length).toBe(callsBefore);
  });

  it('non c\'è replay senza una registrazione', async () => {
    const { ctl } = await setup();
    expect(ctl.replay()).toBe(false);
  });
});
