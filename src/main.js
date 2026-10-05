/**
 * Pagina principale: preparazione del modello, regia della dimostrazione
 * e visualizzazione passo per passo. Tutti i numeri mostrati vengono dagli
 * eventi del motore (o dalla traccia registrata, nel replay).
 */

import './ui/fonts.js';
import { EngineClient } from './engine/client.js';
import { createMockClient } from './engine/mockClient.js';
import { createController, STATES } from './core/controller.js';
import {
  MODELS, SYSTEM_PROMPT, LIMITS, pickVariant, downloadBytes, formatBytes, suggestModelKey, modelsFor, CPU_MODEL_KEY,
} from './core/models.js';
import {
  isModelCached, deleteModelFiles, requestPersistentStorage, storageEstimate,
} from './core/cache.js';
import { probeWebGPU, PROBE_MESSAGES } from './core/webgpu.js';
import { formatPercent, SAMPLING_TOP_K } from './core/distribution.js';
import {
  $, el, clear,
} from './ui/dom.js';
import { createAnatomy } from './ui/anatomy.js';
import { tokenChip } from './ui/tokenView.js';
import { REPLACEMENT } from './core/tokens.js';
import { renderChart } from './ui/chart.js';
import { createNarration } from './ui/narration.js';
import {
  policyLabel, chartCaption, choiceSentence, finishLabel, formatNumber, quoteToken,
} from './ui/labels.js';

const params = new URLSearchParams(location.search);
const SIMULATED = params.has('mock');
const SYSTEM_PROMPT_STORAGE_KEY = 'system-prompt';

/** Ritmo della presentazione (ms per fase). Non influisce sui calcoli. */
const PACE = {
  slow: { intro: 2600, scores: 2200, select: 2200, decode: 1800, append: 1500 },
  normal: { intro: 1600, scores: 1300, select: 1400, decode: 1100, append: 900 },
  fast: { intro: 0, scores: 0, select: 0, decode: 0, append: 0 },
};

const app = {
  client: null,
  probe: null,
  modelKey: null,
  dtype: null,
  info: null,
  status: 'checking',
  speed: 'normal',
  fullInput: false,
  device: 'webgpu',
  view: null,
  lastAnnounce: 0,
};

function storageGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function storageSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // archiviazione non disponibile: si usa il valore predefinito
  }
}

/* ------------------------------------------------------------------ */
/* Regia                                                                */
/* ------------------------------------------------------------------ */

const controller = createController({
  client: {
    call: (type, payload) => {
      if (!app.client) return Promise.reject(new Error('Modello non caricato.'));
      return app.client.call(type, payload);
    },
  },
  introduce,
  present,
  onEvent: handleEvent,
});

// Accesso in sola lettura per i test end-to-end e la verifica manuale.
window.__nextTokenDemo = {
  get trace() {
    return controller.trace;
  },
  get state() {
    return controller.state;
  },
  get info() {
    return app.info;
  },
  get simulated() {
    return SIMULATED;
  },
  /** Elenco delle richieste inviate al motore (sola lettura). */
  get engineCalls() {
    return app.client?.calls ? app.client.calls.slice() : [];
  },
};

/* ------------------------------------------------------------------ */
/* Percorso (fasi)                                                      */
/* ------------------------------------------------------------------ */

let anatomy = null;

function setStage(name) {
  anatomy?.setActive(name);
  document.querySelectorAll('.stage').forEach((s) => {
    s.classList.toggle('active', s.dataset.stage === name);
  });
}

/* ------------------------------------------------------------------ */
/* Vista dei token                                                      */
/* ------------------------------------------------------------------ */

function resetView(trace) {
  app.view = {
    historyRows: [],
    input: trace?.input ?? null,
    generated: [],
    answerParts: [],
    answerText: '',
    answerSnaps: [],
    replay: false,
  };
}

function tokenLabel(selected) {
  const p = selected.piece ?? {};
  if (p.special || selected.special) return `«${p.text}»`;
  if (p.fragment) return 'un frammento di byte';
  if (p.empty) return 'un token senza testo';
  return quoteToken(p.text);
}

function renderTokens({ animateLast = false } = {}) {
  const box = $('tokens');
  clear(box);
  const v = app.view;
  if (!v?.input) {
    box.append(el('span', { class: 'muted' }, 'I token compariranno qui quando avvii.'));
    $('ctx-count').textContent = '';
    $('btn-full-input').hidden = true;
    return;
  }
  const tokens = v.input.tokens;
  const appCount = tokens.filter((t) => t.origin !== 'user').length;
  if (app.fullInput) {
    tokens.forEach((t) => box.append(tokenChip(t)));
  } else {
    // Le parti aggiunte dall'app vengono riassunte, mai nascoste del tutto.
    let run = 0;
    const flushRun = () => {
      if (run > 0) {
        box.append(el('button', {
          type: 'button',
          class: 'tok summary',
          title: "Mostra i token aggiunti dall'app",
          onclick: () => toggleFullInput(true),
        }, `+${run} token dell'app`));
      }
      run = 0;
    };
    for (const t of tokens) {
      if (t.origin === 'user' || t.origin === 'mixed') {
        flushRun();
        box.append(tokenChip(t));
      } else {
        run += 1;
      }
    }
    flushRun();
  }
  v.generated.forEach((g, i) => {
    box.append(tokenChip(g, { extraClass: animateLast && i === v.generated.length - 1 ? 'just-added' : '' }));
  });
  box.scrollTop = box.scrollHeight;
  $('ctx-count').textContent = `${formatNumber(tokens.length)} token di input · ${formatNumber(v.generated.length)} generati`;
  const btn = $('btn-full-input');
  btn.hidden = false;
  btn.setAttribute('aria-expanded', String(app.fullInput));
  btn.textContent = app.fullInput
    ? "Mostra solo il tuo testo"
    : `Mostra l'input completo (+${appCount} token aggiunti dall'app)`;
  const note = $('full-input-note');
  note.hidden = !app.fullInput;
  if (app.fullInput) {
    clear(note);
    note.append(
      'Testo completo inviato al modello, con i token di controllo del formato di chat: ',
      el('code', { class: 'full-text' }, v.input.fullText),
    );
  }
}

function toggleFullInput(force) {
  app.fullInput = typeof force === 'boolean' ? force : !app.fullInput;
  renderTokens();
}

function addGeneratedToken(step) {
  const s = step.selected;
  const p = s.piece ?? {};
  app.view.generated.push({
    id: s.id,
    text: p.text ?? '',
    special: Boolean(s.special),
    fragment: Boolean(p.fragment),
    pending: Boolean(p.fragment),
    empty: Boolean(p.empty),
    origin: 'generated',
  });
  // Un gruppo di frammenti di byte si è chiuso: si aggiornano le sue tessere.
  for (const piece of step.completedPieces ?? []) {
    if (piece.groupSize > 1) {
      const gen = app.view.generated;
      const idx = gen.length - piece.groupSize + piece.groupIndex;
      if (gen[idx] && gen[idx].id === piece.id) {
        Object.assign(gen[idx], {
          fragment: true, pending: false, groupText: piece.groupText, groupSize: piece.groupSize, groupIndex: piece.groupIndex,
        });
      }
    }
  }
}

function renderAnswer() {
  const box = $('answer');
  clear(box);
  const parts = app.view?.answerParts ?? [];
  if (parts.length === 0) {
    box.classList.add('placeholder');
    box.textContent = 'La risposta comparirà qui, un token alla volta.';
    return;
  }
  box.classList.remove('placeholder');
  parts.forEach((p, i) => {
    const last = i === parts.length - 1;
    if (p.control) {
      box.append(el('span', { class: 'end-token', title: p.eos ? 'token di fine: il modello ha chiuso la risposta' : 'token di controllo' },
        p.eos ? `fine · ${p.text}` : p.text));
    } else if (last) {
      box.append(el('span', { class: 'new' }, p.text));
    } else {
      box.append(p.text);
    }
  });
  if (app.view.pendingBytes) {
    box.append(el('span', { class: 'end-token', title: 'carattere incompleto: il token scelto contiene solo una parte dei suoi byte' }, 'byte…'));
  }
}

function appendAnswer(step) {
  const v = app.view;
  if (step.selected.special) {
    v.answerParts.push({ control: true, eos: step.selected.isEos, text: step.selected.piece?.text ?? '' });
  } else {
    // Un carattere di più byte non ancora completo produce U+FFFD in coda:
    // non lo si mostra come testo, si segnala che mancano dei byte.
    const decoded = step.answerText ?? '';
    let next = decoded;
    while (next.endsWith(REPLACEMENT)) next = next.slice(0, -1);
    v.pendingBytes = next.length !== decoded.length;
    if (next.startsWith(v.answerText)) {
      const delta = next.slice(v.answerText.length);
      if (delta) v.answerParts.push({ text: delta });
    } else {
      v.answerParts = v.answerParts.filter((p) => p.control);
      v.answerParts.push({ text: next });
    }
    v.answerText = next;
  }
  renderAnswer();
}

/* ------------------------------------------------------------------ */
/* Presentazione di un passo (eventi reali, nell'ordine reale)          */
/* ------------------------------------------------------------------ */

/** Riga di narrazione: che cosa sta succedendo adesso, in parole semplici. */
let narration = null;

function narrate(n, parts, kind = 'app') {
  // Le frasi si accumulano (riepilogo progressivo), non si sostituiscono.
  narration ??= createNarration($('narration'));
  narration.say(n, parts, kind);
}

const tq = (text) => el('span', { class: 'tokq' }, text);

function waitingForClick() {
  return controller.state === STATES.PAUSED || controller.state === STATES.REPLAY_PAUSED;
}

async function introduce(trace, { replay, signal }) {
  const t = PACE[app.speed];
  resetView(trace);
  app.view.replay = replay;
  $('mode-badge').hidden = !replay && !SIMULATED;
  $('mode-badge').textContent = replay ? 'REPLAY · dati registrati, nessun nuovo calcolo' : 'SIMULAZIONE';
  $('finish-note').textContent = '';
  anatomy.reset();
  $('btn-replay').hidden = true;
  $('btn-export').hidden = true;
  $('choice').textContent = '';
  clear($('chart'));
  $('step-label').textContent = '';
  $('chart-caption').textContent = chartCaption(trace.policy, app.info?.vocabSize);
  $('chart-legend').hidden = trace.policy.kind !== 'sample';
  $('policy-line').textContent = policyLabel(trace.policy);
  renderAnswer();
  renderTokens();
  renderHistory();
  const tokens = trace.input.tokens;
  const userCount = tokens.filter((x) => x.origin === 'user' || x.origin === 'mixed').length;
  setStage('write');
  narrate('1', [replay ? 'Replay della registrazione. Avevi scritto: ' : 'Hai scritto: ', tq(`«${trace.input.userText}»`)]);
  await anatomy.prompt(trace.input.userText, { duration: t.intro, signal });
  setStage('assemble');
  const systemPrompt = trace.input.systemPrompt ?? app.info?.systemPrompt ?? '';
  narrate('2', systemPrompt
    ? `L'app aggiunge un'istruzione («${systemPrompt}») e i marcatori della chat. Puoi vederli con «Mostra l'input completo».`
    : "L'app non aggiunge un'istruzione di sistema, ma solo i marcatori della chat. Puoi vederli con «Mostra l'input completo».");
  await anatomy.assemble({ appTokenCount: tokens.length - userCount, systemPrompt }, { duration: t.intro, signal });
  setStage('tokenize');
  renderTokens();
  narrate('3', [
    `Il tokenizer divide tutto in ${formatNumber(tokens.length)} token (${formatNumber(userCount)} vengono dal tuo testo). Ogni token è un numero: il suo ID nel vocabolario.`,
    waitingForClick() ? ` Premi «${replay ? 'Mostra il primo token' : 'Calcola il primo token'}».` : '',
  ]);
  await anatomy.tokenize({ tokens, total: tokens.length }, { duration: t.intro, signal });
}

async function present(step, { replay, signal, override = false }) {
  const t = PACE[app.speed];
  const s = step.selected;
  const label = tokenLabel(s);
  const v = app.view;
  $('cand-panel').classList.remove('computing');
  $('activity').hidden = true;
  $('step-label').textContent = `${replay ? 'replay · ' : ''}token ${step.index + 1}`;
  if (step.vocabSize) $('chart-caption').textContent = chartCaption(step.policy, step.vocabSize);
  if (override) {
    // Il presentatore ha cambiato la scelta: si annulla l'effetto dell'ultimo passo.
    const snap = v.answerSnaps.pop();
    if (snap) {
      v.answerParts = snap.parts.map((p) => ({ ...p }));
      v.answerText = snap.text;
      v.pendingBytes = snap.pendingBytes;
      renderAnswer();
    }
    v.generated.pop();
    renderTokens();
    anatomy.revertLast();
  } else {
    // 4 → punteggi trasformati in probabilità
    setStage('model');
    renderChart($('chart'), step, { phase: 'scores' });
    $('choice').textContent = '';
    let work;
    if (step.cacheMode === 'naive') work = `Ha ricalcolato da capo tutti i ${formatNumber(step.processedTokens)} token (ricalcolo completo, senza KV cache).`;
    else if (step.cachedTokens > 0) work = `Ha elaborato ${formatNumber(step.processedTokens)} token nuovo: gli altri ${formatNumber(step.cachedTokens)} li ricordava già (KV cache).`;
    else work = `Ha elaborato tutti i ${formatNumber(step.processedTokens)} token dell'input.`;
    narrate('4', `${work} Poi ${replay ? 'aveva dato' : 'ha dato'} un punteggio a ognuno dei ${formatNumber(step.vocabSize)} token del vocabolario: trasformati in probabilità sommano a 100%, qui vedi i più probabili.`, 'model');
    await anatomy.scored(step, { duration: t.scores, signal });
  }
  // 5 → scelta secondo la regola (o scelta del presentatore)
  setStage('select');
  renderChart($('chart'), step, { phase: 'selected' });
  if (override) {
    const original = tokenLabel(step.override.original);
    $('choice').textContent = choiceSentence(step, label, original);
    narrate('5', ['Hai scelto tu ', tq(label), ' al posto di ', tq(original), ' (la scelta della regola). Il modello continuerà da qui.']);
  } else {
    $('choice').textContent = choiceSentence(step, label);
    if (s.isEos) {
      narrate('5', ["L'app sceglie ", tq(label), ': è il segnale di fine della risposta.']);
    } else if (step.policy.kind === 'greedy') {
      narrate('5', ["L'app sceglie il token più probabile: ", tq(label), ` (${formatPercent(s.modelProb)}).`]);
    } else {
      narrate('5', ["L'app estrae a sorte, in proporzione alle probabilità: esce ", tq(label),
        s.rank === 1 ? ', il più probabile.' : `, che è solo ${s.rank}° in classifica.`]);
    }
  }
  await anatomy.selected(step, { duration: t.select, signal });
  // 6 → decodifica: dall'ID al testo, la risposta cresce
  setStage('decode');
  const p = s.piece ?? {};
  $('choice').append(' ', el('span', { class: 'decode' }, `ID ${s.id} → ${p.special ? p.text : JSON.stringify(p.text ?? '')}`));
  v.answerSnaps.push({ parts: v.answerParts.map((x) => ({ ...x })), text: v.answerText, pendingBytes: v.pendingBytes });
  appendAnswer(step);
  if (s.special) narrate('6', `Il token di controllo (ID ${s.id}) non diventa testo visibile: chiude la risposta.`);
  else if (p.fragment) narrate('6', `L'ID ${s.id} è solo un pezzo (byte) di un carattere: il testo comparirà quando arriveranno gli altri pezzi.`);
  else narrate('6', [`Il numero ${s.id} torna testo: `, tq(label), '. La risposta cresce.']);
  await anatomy.decoded(step, { answerText: v.answerText, duration: t.decode, signal });
  // cronologia: una riga per token (una scelta del presentatore sostituisce l'ultima)
  if (override) v.historyRows.pop();
  v.historyRows.push(historyRow(step));
  renderHistory();
  // ↺ il token scelto entra nell'input del passo successivo
  addGeneratedToken(step);
  renderTokens({ animateLast: app.speed !== 'fast' });
  renderLastStep(step);
  if (!step.finish) {
    narrate('↺', [`Il token scelto si aggiunge all'input, che ora ha ${formatNumber(step.contextLength + 1)} token.`,
      waitingForClick() ? ' Premi «Prossimo token».' : ' Si ricomincia dal punto 4.']);
  }
  setStage('loop');
  await anatomy.looped(step, { duration: t.append, signal });
}

/* ------------------------------------------------------------------ */
/* Eventi della regia                                                   */
/* ------------------------------------------------------------------ */

function handleEvent(e) {
  switch (e.type) {
    case 'state':
    case 'busy':
      updateControls();
      if (e.type === 'state') updateSettingsUi();
      break;
    case 'computing':
      setStage('model');
      anatomy.computing({ ids: [...(controller.trace?.input?.ids ?? []), ...(controller.trace?.steps ?? []).map((st) => st.selected.id)] });
      $('cand-panel').classList.add('computing');
      $('activity').hidden = false;
      narrate('4', `Il modello sta calcolando un punteggio per ogni token del suo vocabolario${app.info?.vocabSize ? ` (${formatNumber(app.info.vocabSize)})` : ''}…`, 'model');
      break;
    case 'finish': {
      $('cand-panel').classList.remove('computing');
      $('activity').hidden = true;
      setStage(null);
      anatomy.finished(e.reason);
      let text = finishLabel(e.reason, { maxNewTokens: e.trace?.maxNewTokens });
      if (e.reason === 'error') text += ` ${e.error?.message ?? ''}`;
      if (e.trace?.discardedInFlight) text += " L'ultimo calcolo in corso è stato scartato e non è mostrato.";
      $('finish-note').textContent = text;
      narrate('✓', e.trace?.steps?.length ? `${text} Puoi rivedere tutto con «Rivedi la registrazione».` : text);
      announce(text);
      updateControls();
      break;
    }
    case 'replay-end':
      setStage(null);
      anatomy.finished('replay');
      $('finish-note').textContent = 'Fine del replay: hai rivisto i passi registrati, senza nuovi calcoli.';
      narrate('✓', 'Fine del replay: hai rivisto i passi registrati, senza nuovi calcoli.');
      $('mode-badge').hidden = !SIMULATED;
      $('mode-badge').textContent = 'SIMULAZIONE';
      updateControls();
      break;
    case 'error':
      announce(e.error?.message ?? 'Errore.');
      updateControls();
      break;
    default:
      break;
  }
}

/** Messaggi brevi (errori, fine) nella riga di stato sotto la domanda. */
function announce(text) {
  $('status').textContent = text;
}

/* ------------------------------------------------------------------ */
/* Comandi                                                              */
/* ------------------------------------------------------------------ */

function readPolicy() {
  const kind = document.querySelector('input[name="policy"]:checked')?.value ?? 'greedy';
  if (kind === 'greedy') return { kind: 'greedy' };
  return {
    kind: 'sample',
    temperature: Number($('temperature').value),
    topP: Number($('top-p').value),
    topK: SAMPLING_TOP_K,
  };
}

function readSystemPrompt() {
  return $('system-prompt').value.trim();
}

function readThinking() {
  if (!app.info?.thinking?.supported) return undefined;
  return document.querySelector('input[name="thinking-mode"]:checked')?.value === 'on';
}

function readCacheMode() {
  return document.querySelector('input[name="cache-mode"]:checked')?.value === 'naive' ? 'naive' : 'cache';
}

/* ------------------------------------------------------------------ */
/* Impostazioni della chatbot                                          */
/* ------------------------------------------------------------------ */

function decimalIt(x) {
  return String(x).replace('.', ',');
}

/** Riepilogo, spiegazione della regola, impostazioni attive/inattive. */
function updateSettingsUi() {
  const policy = readPolicy();
  const sampling = policy.kind === 'sample';
  document.querySelectorAll('.setting.needs-sampling').forEach((f) => {
    f.classList.toggle('off', !sampling);
    f.querySelectorAll('input').forEach((i) => { i.disabled = !sampling; });
  });
  $('policy-explain').textContent = sampling
    ? "L'app estrae un token a sorte, in proporzione alle probabilità (dopo temperatura e top-p): di solito esce uno dei primi, ma può uscire anche un token meno probabile. Ripetendo la stessa domanda la risposta può cambiare."
    : "L'app prende sempre il token con la probabilità più alta (decodifica «greedy»): la stessa domanda dà sempre la stessa risposta. Temperatura e top-p qui non servono.";
  const thinkingSupported = Boolean(app.info?.thinking?.supported);
  const thinkingSetting = $('thinking-setting');
  thinkingSetting.hidden = !thinkingSupported;
  thinkingSetting.querySelectorAll('input').forEach((i) => { i.disabled = !thinkingSupported; });
  const parts = [
    sampling ? `Scelta: estrazione (temperatura ${decimalIt($('temperature').value)}, top-p ${decimalIt($('top-p').value)})` : 'Scelta: sempre il più probabile',
    readCacheMode() === 'naive' ? 'Calcolo: ricalcolo completo' : 'Calcolo: KV cache',
    `Max ${$('max-tokens').value} token`,
  ];
  if (thinkingSupported) parts.push(`Ragionamento: ${readThinking() ? 'attivo' : 'disattivato'}`);
  if (readSystemPrompt() !== SYSTEM_PROMPT) {
    parts.push(readSystemPrompt() ? 'Istruzione di sistema personalizzata' : 'Nessuna istruzione di sistema');
  }
  $('settings-recap').textContent = parts.join(' · ');
  const running = controller.state !== STATES.IDLE && controller.state !== STATES.DONE;
  $('settings-note').textContent = running ? 'Generazione in corso: le modifiche varranno dal prossimo «Avvia».' : '';
}

/** Passando sopra un'impostazione si evidenzia la fase dell'anatomia in cui agisce. */
function wireSettingHints() {
  document.querySelectorAll('.setting[data-stage-hint]').forEach((f) => {
    const stage = f.dataset.stageHint;
    const on = () => document.querySelectorAll(`#anatomy [data-stage="${stage}"]`).forEach((n) => n.classList.add('setting-hint'));
    const off = () => document.querySelectorAll('#anatomy .setting-hint').forEach((n) => n.classList.remove('setting-hint'));
    f.addEventListener('mouseenter', on);
    f.addEventListener('mouseleave', off);
    f.addEventListener('focusin', on);
    f.addEventListener('focusout', off);
  });
}

/* ------------------------------------------------------------------ */
/* Cronologia dei token                                                 */
/* ------------------------------------------------------------------ */

function historyRow(step) {
  const s = step.selected;
  const by = step.override ? 'te (presentatore)' : (step.policy.kind === 'greedy' ? 'regola: il più probabile' : 'regola: estrazione');
  const work = step.cacheMode === 'naive'
    ? `${formatNumber(step.processedTokens)} (ricalcolo completo)`
    : `${formatNumber(step.processedTokens)} + ${formatNumber(step.cachedTokens)} in cache`;
  const time = app.view?.replay ? `${formatNumber(Math.round(step.timing.stepMs))} ms (registrato)` : `${formatNumber(Math.round(step.timing.stepMs))} ms`;
  return [String(step.index + 1), tokenLabel(s), String(s.id), formatPercent(s.modelProb), `${s.rank}°`, by, work, time];
}

function renderHistory() {
  const body = $('history-body');
  clear(body);
  const rows = app.view?.historyRows ?? [];
  rows.forEach((cells, i) => {
    const tr = el('tr', { class: i === rows.length - 1 ? 'last' : '' });
    cells.forEach((c, k) => tr.append(el(k === 1 ? 'th' : 'td', k === 1 ? { scope: 'row', class: 'tok-cell' } : {}, c)));
    body.append(tr);
  });
  $('history-count').textContent = rows.length ? `(${rows.length})` : '';
}

function readSeed() {
  const raw = $('seed').value.trim();
  if (raw === '') return undefined;
  const n = Number(raw);
  return Number.isSafeInteger(n) && n >= 0 ? n : undefined;
}

async function onPrimary() {
  const st = controller.state;
  if (st === STATES.PAUSED || st === STATES.REPLAY_PAUSED) {
    await controller.next();
    return;
  }
  if (st === STATES.RUNNING || st === STATES.REPLAYING) {
    controller.pause();
    return;
  }
  if (st !== STATES.IDLE && st !== STATES.DONE) return;
  if (app.status !== 'ready') return;
  const text = $('prompt').value.trim();
  if (!text) {
    announce('Scrivi prima una domanda.');
    $('prompt').focus();
    return;
  }
  $('status').textContent = '';
  await controller.start({
    text,
    policy: readPolicy(),
    seed: readSeed(),
    maxNewTokens: Number($('max-tokens').value),
    cacheMode: readCacheMode(),
    systemPrompt: readSystemPrompt(),
    thinking: readThinking(),
    meta: {
      model: app.info,
      simulated: SIMULATED,
      app: 'come-risponde-una-chatbot',
    },
  });
}

/** Il presentatore può scegliere un altro candidato dell'ultimo passo, solo in pausa. */
function updatePickable() {
  const chart = $('chart');
  const ok = controller.state === STATES.PAUSED && !controller.busy && !app.view?.replay
    && (controller.trace?.steps?.length ?? 0) > 0 && Boolean(chart.querySelector('li.row'));
  chart.classList.toggle('pickable', ok);
  chart.querySelectorAll('li.row:not(.other):not(.outside)').forEach((r) => {
    if (ok) r.setAttribute('tabindex', '0');
    else r.removeAttribute('tabindex');
  });
  const hint = $('pick-hint');
  // visibilità e non display: la riga tiene il suo spazio e la pagina non si sposta
  if (hint) hint.classList.toggle('off', !ok);
}

function pickRow(row) {
  if (!row || !$('chart').classList.contains('pickable') || row.classList.contains('chosen')) return;
  controller.choose(Number(row.dataset.id));
}

function updateControls() {
  const st = controller.state;
  const busy = controller.busy;
  const primary = $('btn-primary');
  const auto = $('btn-auto');
  const stop = $('btn-stop');
  const ready = app.status === 'ready';
  const generating = [STATES.RUNNING, STATES.PAUSED, STATES.STOPPING].includes(st);
  const replaying = [STATES.REPLAYING, STATES.REPLAY_PAUSED].includes(st);
  const paused = st === STATES.PAUSED || st === STATES.REPLAY_PAUSED;
  const autoRunning = st === STATES.RUNNING || st === STATES.REPLAYING;
  const first = (app.view?.generated?.length ?? 0) === 0;
  let label = 'Avvia';
  let disabled = !ready || !$('prompt').value.trim();
  if (st === STATES.STARTING) {
    label = 'Preparazione…';
    disabled = true;
  } else if (st === STATES.STOPPING) {
    label = 'Arresto…';
    disabled = true;
  } else if (paused) {
    if (replaying) label = first ? 'Mostra il primo token' : 'Mostra il prossimo token';
    else label = first ? 'Calcola il primo token' : 'Prossimo token';
    disabled = busy;
  } else if (autoRunning) {
    label = 'Pausa';
    disabled = false;
  }
  primary.textContent = label;
  primary.disabled = disabled;
  auto.hidden = !paused;
  stop.hidden = !(generating || replaying) || st === STATES.STOPPING;
  stop.textContent = replaying ? 'Ferma il replay' : 'Ferma';
  $('prompt').readOnly = generating || replaying || st === STATES.STARTING;
  // In avanzamento automatico e veloce la narrazione cambia troppo spesso per un lettore di schermo.
  $('narration').setAttribute('aria-live', autoRunning && app.speed === 'fast' ? 'off' : 'polite');
  const trace = controller.trace;
  const done = st === STATES.DONE && trace && trace.steps.length > 0;
  $('btn-replay').hidden = !done;
  $('btn-export').hidden = !done;
  updatePickable();
}

function exportTrace() {
  const trace = controller.trace;
  if (!trace) return;
  const blob = new Blob([JSON.stringify(trace, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = el('a', { href: url, download: `registrazione-${trace.startedAt.replace(/[:.]/g, '-')}.json` });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ------------------------------------------------------------------ */
/* Dettagli tecnici                                                     */
/* ------------------------------------------------------------------ */

function fillDl(dl, rows) {
  clear(dl);
  for (const [k, v] of rows) {
    if (v === undefined || v === null || v === '') continue;
    dl.append(el('dt', {}, k), el('dd', {}, String(v)));
  }
}

function renderTech() {
  const i = app.info;
  const p = app.probe;
  if (!i) {
    fillDl($('tech'), [['Stato', 'modello non caricato']]);
    return;
  }
  const systemPrompt = app.view?.input?.systemPrompt ?? i.systemPrompt;
  fillDl($('tech'), [
    ['Modello', i.label],
    ['Repository', i.modelId],
    ['Revisione', i.revision?.slice(0, 12)],
    ['Precisione', i.dtype],
    ['Esecuzione', SIMULATED ? 'simulazione (nessun modello reale)'
      : (app.device === 'wasm' ? 'processore (WASM), senza scheda grafica' : `WebGPU${p?.adapter?.vendor ? ` · ${p.adapter.vendor} ${p.adapter.architecture}` : ''}`)],
    ['Vocabolario', i.vocabSize ? `${formatNumber(i.vocabSize)} token` : 'noto al primo passo'],
    ['Token di fine', i.eosTokens?.join('  ')],
    ['Download', i.downloadBytes ? formatBytes(i.downloadBytes) : null],
    ['Caricamento', `${formatNumber(i.tokenizerMs + i.modelMs)} ms`],
    ['Prova GPU', i.warmupMs ? `${formatNumber(i.warmupMs)} ms` : null],
    ['Istruzione di sistema', systemPrompt || '(nessuna)'],
  ]);
}

function renderLastStep(step) {
  if (app.info && step.index === 0) {
    app.info.vocabSize = step.vocabSize;
    renderTech();
  }
  const s = step.selected;
  fillDl($('last-step'), [
    ['Token n.', step.index + 1],
    ['Token elaborati', `${step.processedTokens} nuovi + ${step.cachedTokens} già in memoria (KV cache)`],
    ['Tempo del passo', app.view.replay ? 'registrato' : `${step.timing.stepMs} ms (di cui probabilità: ${step.timing.decideMs} ms)`],
    ['Scelto', `ID ${s.id}, posizione ${s.rank} in classifica`],
    ['Logit del primo candidato', step.candidates[0]?.logit?.toFixed(3)],
    ['Probabilità del modello', formatPercent(s.modelProb, 3)],
    ['Probabilità di estrazione', s.policyProb === null ? null : formatPercent(s.policyProb, 3)],
    ['Numero estratto', s.draw === null ? null : s.draw.toFixed(6)],
    ['Candidati ammessi', step.pool ? `${step.pool.size} (massa prima della rinormalizzazione ${formatPercent(step.pool.massBeforeRenormalization)})` : null],
    ['Punteggi non validi', step.invalidScores || null],
  ]);
}

/* ------------------------------------------------------------------ */
/* Preparazione del modello                                            */
/* ------------------------------------------------------------------ */

function setModelChip(state, text) {
  const chip = $('model-chip');
  chip.dataset.state = state;
  chip.textContent = text;
}

async function renderModelChoice() {
  const box = $('model-choice');
  clear(box);
  box.append(el('legend', { class: 'visually-hidden' }, 'Modello'));
  const suggested = app.device === 'wasm' ? CPU_MODEL_KEY : suggestModelKey({ deviceMemoryGB: navigator.deviceMemory });
  for (const m of modelsFor(app.device)) {
    const dtype = pickVariant(m, { shaderF16: app.probe?.shaderF16, device: app.device });
    const size = formatBytes(downloadBytes(m, dtype));
    const desc = {
      recommended: `Consigliato. Download ${size}. Risposte più accurate in italiano; serve un computer con molta memoria.`,
      light: `Leggero. Download ${size}. Più veloce e meno esigente, ma sbaglia spesso i fatti.`,
      tiny: `Minimo. Download ${size}. Per computer con poca memoria: scrive in italiano, ma sbaglia spesso e volentieri.`,
      cpu: `Senza scheda grafica. Download ${size}. Gira sul processore: è lento e quasi sempre ripete la domanda o risponde a caso. Serve solo a vedere il meccanismo.`,
    }[m.tier] ?? `Download ${size}.`;
    const input = el('input', {
      type: 'radio', name: 'model', value: m.key, checked: m.key === app.modelKey, 'data-testid': `model-${m.key}`,
    });
    input.addEventListener('change', () => {
      app.modelKey = m.key;
      storageSet('model', m.key);
      updateSetup();
    });
    box.append(el('label', { class: 'model-option' }, input,
      el('span', { class: 'name' }, `${m.name}${m.key === suggested ? ' — suggerito per questo computer' : ''}`),
      el('span', { class: 'desc' }, desc)));
  }
}

async function updateSetup() {
  const m = MODELS[app.modelKey];
  if (!m) return;
  app.dtype = pickVariant(m, { shaderF16: app.probe?.shaderF16, device: app.device });
  const bytes = downloadBytes(m, app.dtype);
  const cached = await isModelCached(m, app.dtype);
  const btn = $('btn-load');
  btn.disabled = app.status === 'loading';
  btn.textContent = cached
    ? `Prepara ${m.name} (già scaricato)`
    : `Scarica e prepara ${m.name} (${formatBytes(bytes)})`;
  const notes = [];
  if (cached) notes.push('I file sono già salvati in questo browser: non serve riscaricarli.');
  else notes.push(`Il download (${formatBytes(bytes)}) avviene solo quando premi il pulsante; poi i file restano salvati nel browser.`);
  if (app.device === 'wasm') notes.push('Senza WebGPU il modello gira sul processore: solo il modello più piccolo è utilizzabile.');
  else if (!app.probe?.shaderF16) notes.push('La scheda grafica non supporta i calcoli a 16 bit: si usa la versione a precisione mista più grande.');
  const est = await storageEstimate();
  if (!cached && est && est.quota - est.usage < bytes * 1.1) {
    notes.push('Attenzione: lo spazio disponibile per il browser sembra insufficiente per salvare il modello; potrebbe essere necessario riscaricarlo ogni volta.');
  }
  const suggested = suggestModelKey({ deviceMemoryGB: navigator.deviceMemory });
  if (app.device === 'webgpu' && suggested !== app.modelKey && m.tier === 'recommended') {
    notes.push('Il browser segnala poca memoria: il modello consigliato potrebbe non caricarsi. In quel caso usa il modello leggero o quello minimo.');
  }
  $('setup-note').textContent = notes.join(' ');
  $('privacy-sizes').textContent = Object.values(MODELS)
    .map((x) => `${x.name}: ${formatBytes(downloadBytes(x, pickVariant(x, { shaderF16: app.probe?.shaderF16, device: (x.devices ?? ['webgpu'])[0] })))} da scaricare`)
    .join(' · ');
}

function showSetupError(message) {
  const box = $('setup-error');
  box.textContent = message;
  box.hidden = !message;
}

async function loadModel() {
  const m = MODELS[app.modelKey];
  if (!m || app.status === 'loading') return;
  app.status = 'loading';
  showSetupError('');
  setModelChip('loading', `${m.name}: caricamento…`);
  $('btn-load').disabled = true;
  document.querySelectorAll('input[name="model"]').forEach((i) => { i.disabled = true; });
  const cached = await isModelCached(m, app.dtype);
  if (!cached) requestPersistentStorage();
  const bar = $('load-bar');
  const text = $('load-text');
  $('load-progress').hidden = false;
  bar.removeAttribute('value');
  text.textContent = cached ? 'Lettura dei file salvati nel browser…' : 'Avvio del download…';
  const t0 = performance.now();
  app.client?.terminate();
  app.client = new EngineClient({
    onProgress: (p) => {
      if (p.phase !== 'files' || !p.total) return;
      bar.max = p.total;
      bar.value = p.loaded;
      const pct = Math.floor((p.loaded / p.total) * 100);
      text.textContent = cached
        ? `Lettura dal browser: ${pct}%`
        : `Download: ${formatBytes(p.loaded)} di ${formatBytes(p.total)} (${pct}%)`;
    },
    onCrash: (err) => onEngineCrash(err),
  });
  try {
    const info = await app.client.call('load', {
      modelKey: m.key,
      dtype: app.dtype,
      device: app.device,
      shaderF16: app.probe?.shaderF16,
    });
    bar.removeAttribute('value');
    text.textContent = app.device === 'wasm'
      ? 'Prima esecuzione di prova sul processore…'
      : 'Prima esecuzione di prova: la scheda grafica prepara i suoi programmi…';
    const warm = await app.client.call('warmup');
    app.info = { ...info, warmupMs: warm.warmupMs, totalLoadMs: Math.round(performance.now() - t0) };
    app.status = 'ready';
    $('setup').hidden = true;
    $('load-progress').hidden = true;
    setModelChip('ready', `${m.name} · pronto${app.device === 'wasm' ? ' (processore)' : ''}`);
    updateSettingsUi();
    renderTech();
    announce(`Modello pronto: ${m.name}. Scrivi una domanda e premi Avvia.`);
  } catch (err) {
    app.status = 'error';
    app.client?.terminate();
    app.client = null;
    $('load-progress').hidden = true;
    setModelChip('error', `${m.name}: non caricato`);
    const light = { recommended: ' Puoi provare il modello leggero.', light: ' Puoi provare il modello minimo.' }[m.tier] ?? '';
    showSetupError(`Il modello non si è caricato: ${err.message}.${light}`);
  } finally {
    document.querySelectorAll('input[name="model"]').forEach((i) => { i.disabled = false; });
    if (app.status !== 'ready') {
      app.status = app.status === 'loading' ? 'needs-load' : app.status;
      updateSetup();
    }
    updateControls();
  }
}

function onEngineCrash(err) {
  app.status = 'error';
  app.client = null;
  setModelChip('error', 'Modello interrotto');
  $('setup').hidden = false;
  showSetupError(`Il processo del modello si è interrotto (${err.message}). Spesso succede per memoria insufficiente: prova il modello leggero o chiudi altre schede.`);
  controller.reset();
  updateSetup();
  updateControls();
}

async function changeModel() {
  await controller.reset();
  if (app.client) {
    await app.client.call('unload').catch(() => {});
    app.client.terminate();
  }
  app.client = null;
  app.info = null;
  app.status = 'needs-load';
  setModelChip('off', 'Modello non caricato');
  $('setup').hidden = false;
  renderTech();
  await renderModelChoice();
  await updateSetup();
  updateSettingsUi();
  updateControls();
  $('setup').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function deleteModel() {
  const m = MODELS[app.modelKey];
  if (!m) return;
  const removed = await deleteModelFiles(m);
  $('adv-note').textContent = removed
    ? `Eliminati ${removed} file di ${m.name} da questo browser. Il modello resta in uso fino al ricaricamento della pagina.`
    : `Nessun file di ${m.name} salvato in questo browser.`;
  updateSetup();
}

/* ------------------------------------------------------------------ */
/* Avvio della pagina                                                  */
/* ------------------------------------------------------------------ */

function wireUi() {
  const pickHint = el('p', { id: 'pick-hint', class: 'pick-hint muted off', 'data-testid': 'pick-hint' },
    'Vuoi vedere cosa succede con un altro token? Clicca un candidato: il modello continuerà da lì.');
  $('choice').after(pickHint);
  $('chart').addEventListener('click', (e) => pickRow(e.target.closest('li.row:not(.other):not(.outside)')));
  $('chart').addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const row = e.target.closest?.('li.row:not(.other):not(.outside)');
    if (!row) return;
    e.preventDefault();
    e.stopPropagation();
    pickRow(row);
  });
  $('btn-primary').addEventListener('click', onPrimary);
  $('btn-auto').addEventListener('click', () => controller.resume());
  $('btn-stop').addEventListener('click', () => controller.stop());
  $('btn-replay').addEventListener('click', () => controller.replay());
  $('btn-export').addEventListener('click', exportTrace);
  $('btn-load').addEventListener('click', loadModel);
  $('btn-change-model').addEventListener('click', changeModel);
  $('btn-delete-model').addEventListener('click', deleteModel);
  $('btn-full-input').addEventListener('click', () => toggleFullInput());
  $('prompt').addEventListener('input', updateControls);
  $('prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!$('btn-primary').disabled) onPrimary();
    }
  });
  document.querySelectorAll('.example').forEach((b) => b.addEventListener('click', () => {
    if ($('prompt').readOnly) return;
    $('prompt').value = b.textContent;
    updateControls();
    $('prompt').focus();
  }));
  document.querySelectorAll('input[name="speed"]').forEach((r) => r.addEventListener('change', () => {
    app.speed = r.value;
    storageSet('speed', r.value);
  }));
  $('show-ids').addEventListener('change', (e) => {
    document.querySelector('main').classList.toggle('show-ids', e.target.checked);
  });
  const projector = $('btn-projector');
  projector.addEventListener('click', () => {
    const on = !document.documentElement.classList.contains('projector');
    document.documentElement.classList.toggle('projector', on);
    projector.setAttribute('aria-pressed', String(on));
    storageSet('projector', on ? '1' : '0');
  });
  document.querySelectorAll('input[name="policy"], input[name="cache-mode"], #temperature, #top-p, #max-tokens')
    .forEach((i) => i.addEventListener('input', updateSettingsUi));
  document.querySelectorAll('input[name="policy"], input[name="cache-mode"]')
    .forEach((i) => i.addEventListener('change', updateSettingsUi));
  document.querySelectorAll('input[name="thinking-mode"]')
    .forEach((i) => i.addEventListener('change', updateSettingsUi));
  $('system-prompt').addEventListener('input', () => {
    storageSet(SYSTEM_PROMPT_STORAGE_KEY, $('system-prompt').value);
    updateSettingsUi();
  });
  const panel = $('settings-panel');
  if (storageGet('settings-open') === '1') panel.open = true;
  panel.addEventListener('toggle', () => storageSet('settings-open', panel.open ? '1' : '0'));
  wireSettingHints();
  const bindOutput = (id, outId) => {
    const update = () => { $(outId).textContent = String($(id).value).replace('.', ','); };
    $(id).addEventListener('input', update);
    update();
  };
  bindOutput('temperature', 'temperature-out');
  bindOutput('top-p', 'top-p-out');
  bindOutput('max-tokens', 'max-tokens-out');

  const savedSystemPrompt = storageGet(SYSTEM_PROMPT_STORAGE_KEY);
  if (savedSystemPrompt !== null) $('system-prompt').value = savedSystemPrompt;
  updateSettingsUi();

  document.addEventListener('keydown', (e) => {
    const tag = e.target?.tagName;
    if (['TEXTAREA', 'INPUT', 'SELECT'].includes(tag) || e.target?.isContentEditable) return;
    const st = controller.state;
    if ((e.key === ' ' && !['BUTTON', 'SUMMARY', 'A'].includes(tag)) || e.key === 'ArrowRight') {
      // Spazio o → : prossimo token (in pausa) oppure pausa (se procede da solo)
      if (st === STATES.PAUSED || st === STATES.REPLAY_PAUSED) {
        e.preventDefault();
        controller.next();
      } else if (st === STATES.RUNNING || st === STATES.REPLAYING) {
        e.preventDefault();
        controller.pause();
      }
    } else if (e.key === 'Escape') {
      controller.stop();
    }
  });

  // Preferenze dell'utente (solo comodità locali).
  const speed = storageGet('speed');
  if (speed && PACE[speed]) {
    app.speed = speed;
    const r = document.querySelector(`input[name="speed"][value="${speed}"]`);
    if (r) r.checked = true;
  }
  if (storageGet('projector') === '1') {
    document.documentElement.classList.add('projector');
    projector.setAttribute('aria-pressed', 'true');
  }
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches && !speed) {
    // con movimento ridotto si parte comunque al ritmo normale (nessuna animazione di movimento)
    app.speed = 'normal';
  }
}

async function boot() {
  anatomy = createAnatomy($('anatomy'));
  anatomy.reset();
  wireUi();
  resetView(null);
  renderTokens();
  renderAnswer();
  renderTech();
  updateControls();
  $('prompt').maxLength = LIMITS.maxPromptChars;

  if (SIMULATED) {
    $('sim-banner').hidden = false;
    $('setup').hidden = true;
    $('mode-badge').hidden = false;
    $('mode-badge').textContent = 'SIMULAZIONE';
    app.client = createMockClient({ delayMs: Number(params.get('delay') ?? 30) });
    app.info = await app.client.call('load');
    app.status = 'ready';
    setModelChip('sim', 'Simulazione · nessun modello reale');
    renderTech();
    updateControls();
    return;
  }

  setModelChip('off', 'Verifica del browser…');
  app.probe = await probeWebGPU();
  app.device = app.probe.ok ? 'webgpu' : 'wasm';
  if (!app.probe.ok) {
    // Nessun ripiego silenzioso: si spiega il problema e si offre, come scelta
    // esplicita, il solo modello abbastanza piccolo da girare sul processore.
    showSetupError(`${PROBE_MESSAGES[app.probe.reason] ?? 'WebGPU non disponibile.'} Puoi comunque usare un modello piccolissimo sul processore: è lento e poco capace, ma mostra il meccanismo.`);
    setModelChip('off', 'Senza WebGPU');
  }
  const saved = storageGet('model');
  const allowed = modelsFor(app.device).map((m) => m.key);
  app.modelKey = allowed.includes(saved) ? saved
    : (app.device === 'wasm' ? CPU_MODEL_KEY : suggestModelKey({ deviceMemoryGB: navigator.deviceMemory }));
  app.status = 'needs-load';
  if (app.probe.ok) setModelChip('off', 'Modello non caricato');
  await renderModelChoice();
  await updateSetup();
  updateControls();
}

boot();
