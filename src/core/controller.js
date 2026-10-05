/**
 * Regia della dimostrazione: avvio, pausa, un token alla volta, ripresa,
 * stop e replay della registrazione.
 *
 * Regole che garantiscono l'onestà della visualizzazione:
 * - in generazione, ogni passo mostrato corrisponde a una chiamata reale al
 *   modello, in ordine; il passo successivo si calcola solo dopo che il
 *   precedente è stato mostrato (la pausa ferma davvero i calcoli);
 * - "un token" in pausa esegue esattamente un passo del modello, riusando
 *   la KV cache: nulla viene ricalcolato o cambiato;
 * - lo stop durante un calcolo scarta quel risultato (non viene mostrato
 *   né registrato) e libera la cache;
 * - il replay riproduce la traccia registrata senza alcuna chiamata al
 *   modello ed è segnalato come tale.
 */

export const STATES = Object.freeze({
  IDLE: 'idle',
  STARTING: 'starting',
  RUNNING: 'running',
  PAUSED: 'paused',
  STOPPING: 'stopping',
  DONE: 'done',
  REPLAYING: 'replaying',
  REPLAY_PAUSED: 'replay-paused',
});

/**
 * @param {object} deps
 * @param {{ call: (type: string, payload?: object) => Promise<any> }} deps.client
 * @param {(step: object, ctx: { replay: boolean, signal: { aborted: boolean } }) => Promise<void>} deps.present
 * @param {(trace: object, ctx: { replay: boolean, signal: { aborted: boolean } }) => Promise<void>} [deps.introduce]
 *   presentazione dell'input (testo assemblato e token) prima del primo passo
 * @param {(event: object) => void} [deps.onEvent]
 */
export function createController({
  client, present, introduce = async () => {}, onEvent = () => {},
}) {
  let state = STATES.IDLE;
  let trace = null;
  let sessionId = null;
  let busy = false;
  let stopRequested = false;
  let replayIndex = 0;
  const signal = { aborted: false };

  /** Segnala quando un calcolo o una presentazione inizia/finisce (es. per abilitare «Un token»). */
  function setBusy(value) {
    if (busy === value) return;
    busy = value;
    onEvent({ type: 'busy', busy });
  }

  function setState(next) {
    state = next;
    onEvent({ type: 'state', state });
  }

  async function finish(reason, error = null) {
    if (sessionId) {
      sessionId = null;
      // libera la KV cache del modello
      await client.call('end').catch(() => null);
    }
    if (trace) {
      trace.finish = reason;
      trace.finishedAt = new Date().toISOString();
      if (error) trace.error = String(error.message ?? error);
    }
    setState(STATES.DONE);
    onEvent({ type: 'finish', reason, error, trace });
  }

  /** Un passo reale del modello. Restituisce false se la generazione è finita. */
  async function doStep() {
    setBusy(true);
    const index = trace.steps.length;
    onEvent({ type: 'computing', index });
    let ev;
    try {
      ev = await client.call('step', { sessionId });
    } catch (err) {
      setBusy(false);
      if (stopRequested) {
        await finish('stopped');
        return false;
      }
      await finish('error', err);
      return false;
    }
    if (stopRequested) {
      trace.discardedInFlight = true;
      setBusy(false);
      await finish('stopped');
      return false;
    }
    trace.steps.push(ev);
    onEvent({ type: 'computed', index, step: ev });
    await present(ev, { replay: false, signal });
    setBusy(false);
    if (ev.finish) {
      await finish(ev.finish);
      return false;
    }
    if (stopRequested) {
      await finish('stopped');
      return false;
    }
    return true;
  }

  async function runLoop() {
    while (state === STATES.RUNNING && !busy) {
      const more = await doStep();
      if (!more) return;
    }
  }

  /**
   * Avvia una generazione. Per impostazione predefinita si procede un token
   * alla volta: dopo la preparazione dell'input la regia resta in pausa e
   * ogni next() calcola un solo token. Con autoplay i token si susseguono da soli.
   */
  async function start({
    text, policy, maxNewTokens, seed, meta = {}, autoplay = false, cacheMode = 'cache',
    systemPrompt, thinking,
  }) {
    if (busy || ![STATES.IDLE, STATES.DONE].includes(state)) return false;
    stopRequested = false;
    signal.aborted = false;
    setState(STATES.STARTING);
    let begun;
    try {
      begun = await client.call('begin', {
        text, policy, maxNewTokens, seed, cacheMode, systemPrompt, thinking,
      });
    } catch (err) {
      setState(STATES.IDLE);
      onEvent({ type: 'error', error: err });
      return false;
    }
    sessionId = begun.sessionId;
    trace = {
      version: 1,
      startedAt: new Date().toISOString(),
      meta,
      input: begun.input,
      thinking: begun.thinking ?? begun.input.thinking ?? null,
      policy: begun.policy,
      seed: begun.seed,
      maxNewTokens: begun.maxNewTokens,
      requestedMaxNewTokens: begun.requestedMaxNewTokens,
      maxContextTokens: begun.maxContextTokens,
      limitReason: begun.limitReason,
      cacheMode: begun.cacheMode ?? 'cache',
      steps: [],
      finish: null,
    };
    onEvent({ type: 'begin', trace, replay: false });
    setBusy(true);
    setState(autoplay ? STATES.RUNNING : STATES.PAUSED);
    await introduce(trace, { replay: false, signal });
    setBusy(false);
    if (stopRequested) {
      await finish('stopped');
      return true;
    }
    if (state === STATES.RUNNING) runLoop();
    return true;
  }

  function pause() {
    if (state === STATES.RUNNING) setState(STATES.PAUSED);
    else if (state === STATES.REPLAYING) setState(STATES.REPLAY_PAUSED);
  }

  function resume() {
    if (state === STATES.PAUSED) {
      setState(STATES.RUNNING);
      runLoop();
    } else if (state === STATES.REPLAY_PAUSED) {
      setState(STATES.REPLAYING);
      replayLoop();
    }
  }

  /**
   * In pausa, il presentatore sostituisce il token dell'ultimo passo con un
   * altro candidato. Nessun nuovo calcolo del modello: il passo successivo
   * partirà dal token scelto. La traccia conserva la scelta originale.
   */
  async function choose(id) {
    if (busy || state !== STATES.PAUSED || !trace || trace.steps.length === 0) return false;
    setBusy(true);
    let ev;
    try {
      ev = await client.call('override', { sessionId, id });
    } catch (err) {
      setBusy(false);
      onEvent({ type: 'error', error: err });
      return false;
    }
    trace.steps[trace.steps.length - 1] = ev;
    onEvent({ type: 'override', step: ev });
    await present(ev, { replay: false, signal, override: true });
    setBusy(false);
    if (ev.finish) await finish(ev.finish);
    return true;
  }

  /** Avanza di un solo token (solo in pausa). */
  async function next() {
    if (busy) return false;
    if (state === STATES.PAUSED) {
      const more = await doStep();
      // se nel frattempo è stata premuta "riprendi", si continua
      if (more && state === STATES.RUNNING) runLoop();
      return true;
    }
    if (state === STATES.REPLAY_PAUSED) {
      await replayOne();
      if (state === STATES.REPLAYING) replayLoop();
      return true;
    }
    return false;
  }

  async function stop() {
    if ([STATES.RUNNING, STATES.PAUSED].includes(state)) {
      stopRequested = true;
      signal.aborted = true;
      if (!busy) await finish('stopped');
      else setState(STATES.STOPPING);
      return true;
    }
    if ([STATES.REPLAYING, STATES.REPLAY_PAUSED].includes(state)) {
      signal.aborted = true;
      replayIndex = trace.steps.length;
      if (!busy) endReplay();
      return true;
    }
    return false;
  }

  function endReplay() {
    setState(STATES.DONE);
    onEvent({ type: 'replay-end', trace });
  }

  async function replayOne() {
    setBusy(true);
    const step = trace.steps[replayIndex];
    replayIndex += 1;
    await present(step, { replay: true, signal });
    setBusy(false);
    if (replayIndex >= trace.steps.length) endReplay();
  }

  async function replayLoop() {
    while (state === STATES.REPLAYING && !busy && replayIndex < trace.steps.length) {
      await replayOne();
    }
    if (state === STATES.DONE || state === STATES.REPLAY_PAUSED) return;
    if (!busy && replayIndex >= trace.steps.length && state === STATES.REPLAYING) endReplay();
  }

  /** Riproduce la traccia registrata: nessuna chiamata al modello. */
  function replay({ autoplay = false } = {}) {
    if (busy || state !== STATES.DONE || !trace || trace.steps.length === 0) return false;
    replayIndex = 0;
    signal.aborted = false;
    onEvent({ type: 'begin', trace, replay: true });
    setBusy(true);
    setState(autoplay ? STATES.REPLAYING : STATES.REPLAY_PAUSED);
    introduce(trace, { replay: true, signal }).then(() => {
      setBusy(false);
      if (replayIndex >= trace.steps.length) {
        if (state !== STATES.DONE) endReplay();
        return;
      }
      if (state === STATES.REPLAYING) replayLoop();
    });
    return true;
  }

  /** Riporta allo stato iniziale (es. cambio di modello). */
  async function reset() {
    if (sessionId) await client.call('end').catch(() => null);
    sessionId = null;
    trace = null;
    setBusy(false);
    setState(STATES.IDLE);
  }

  return {
    start,
    pause,
    resume,
    next,
    choose,
    stop,
    replay,
    reset,
    get state() {
      return state;
    },
    get trace() {
      return trace;
    },
    get busy() {
      return busy;
    },
  };
}
