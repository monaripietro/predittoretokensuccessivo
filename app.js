import { MockRuntime, computeStep } from './src/mockRuntime.js';
import { RealRuntime } from './src/realRuntime.js';
import { validateParams } from './src/sampling.js';

const state = {
  promptText: '',
  promptTokens: [],
  generatedTokens: [],
  contextTokens: [],
  ranking: null,
  selectedToken: null,
  selectedProbability: null,
  stepIndex: 0,
  modelStatus: 'loading',
  backend: '…',
  isBusy: false,
  mode: 'cache',
  cacheState: null,
  history: [],
  error: null,
};

const ui = {
  prompt: () => document.getElementById('prompt'),
  btnNext: () => document.getElementById('btn-next'),
  btnNextBottom: () => document.getElementById('btn-next-bottom'),
  btnReset: () => document.getElementById('btn-reset'),
  modelIndicator: () => document.getElementById('model-indicator'),
  backendIndicator: () => document.getElementById('backend-indicator'),
  hint: () => document.getElementById('prompt-hint'),
  error: () => document.getElementById('error'),
  paramsError: () => document.getElementById('params-error'),
  sequence: () => document.getElementById('token-sequence'),
  contextMessage: () => document.getElementById('context-message'),
  stepResult: () => document.getElementById('step-result'),
  chosenToken: () => document.getElementById('chosen-token'),
  rankingBody: () => document.getElementById('ranking-body'),
  cacheMessage: () => document.getElementById('cache-message'),
};

const useMock = new URLSearchParams(location.search).has('mock');
let runtime = null;

async function initRuntime() {
  if (useMock) {
    runtime = new MockRuntime();
    const info = await runtime.init();
    return { ...info, fallback: false };
  }
  try {
    runtime = new RealRuntime();
    const info = await runtime.init();
    return { ...info, fallback: false };
  } catch (err) {
    runtime = new MockRuntime();
    const info = await runtime.init();
    return { ...info, backend: `${info.backend} (fallback)`, modelInfo: `${info.modelInfo} (fallback)`, fallback: true, cause: err };
  }
}
const committedPromptId = { value: null };

function readParams() {
  const get = (id) => document.getElementById(id).value;
  return {
    mode: get('param-mode'),
    temperature: parseFloat(get('param-temperature')),
    topK: parseInt(get('param-topk'), 10),
    topP: parseFloat(get('param-topp')),
    minP: parseFloat(get('param-minp')),
    repeatPenalty: parseFloat(get('param-repeat')),
    seed: parseInt(get('param-seed'), 10),
    topN: parseInt(get('param-topn'), 10),
  };
}

function setBusy(busy) {
  state.isBusy = busy;
  updateButton();
}

function updateButton() {
  const btn = ui.btnNext();
  const btnBottom = ui.btnNextBottom();
  const promptEmpty = !ui.prompt().value.trim();
  const modelReady = state.modelStatus === 'ready';
  const disabled = state.isBusy || promptEmpty || !modelReady;
  btn.disabled = disabled;
  btn.textContent = state.isBusy ? 'Calcolo…' : 'Calcola token successivo';
  if (btnBottom) {
    const started = state.stepIndex > 0;
    btnBottom.disabled = disabled || !started;
    btnBottom.textContent = state.isBusy ? 'Calcolo…' : started
      ? `Calcola token successivo (continua con ${state.contextTokens.length} token di contesto)`
      : 'Calcola token successivo';
  }
}

function invalidateGeneration() {
  state.generatedTokens = [];
  state.contextTokens = state.promptTokens.slice();
  state.ranking = null;
  state.selectedToken = null;
  state.selectedProbability = null;
  state.stepIndex = 0;
  state.cacheState = null;
  state.history = [];
  state.error = null;
  committedPromptId.value = ui.prompt().value;
}

function showError(message) {
  state.error = message;
  const el = ui.error();
  el.textContent = message;
  el.hidden = !message;
}

function clearError() {
  showError(null);
}

function makeChip(token, id, position, kind) {
  const chip = document.createElement('span');
  chip.className = `chip ${kind}`;
  chip.dataset.tokenId = id;
  chip.dataset.position = position;
  const isEos = id === runtime.eosId;
  if (isEos) chip.classList.add('eos');
  const displayText = isEos ? '⟨EOS⟩' : displayTokenText(id);
  chip.textContent = displayText;
  chip.title = `Testo: ${isEos ? '<eos>' : displayText} · Token ID: ${id} · Posizione: ${position}`;
  chip.tabIndex = 0;
  return chip;
}

/**
 * Mostra la tokenizzazione live del prompt corrente (token + ID)
 * anche prima del primo calcolo.
 */
function liveTokenizePrompt() {
  if (!runtime || state.modelStatus !== 'ready') return;
  const prompt = ui.prompt().value.trim();
  const committed = committedPromptId.value;
  if (committed !== null && committed !== ui.prompt().value) {
    const ids = runtime.encode(prompt);
    state.promptTokens = ids;
    state.contextTokens = ids;
    state.generatedTokens = [];
    renderPromptPreview(ids);
    return;
  }
  if (state.stepIndex === 0) {
    const ids = runtime.encode(prompt);
    state.promptTokens = ids;
    state.contextTokens = ids;
    renderPromptPreview(ids);
  }
}

function renderPromptPreview(ids) {
  const preview = document.getElementById('prompt-preview');
  if (!preview) return;
  preview.innerHTML = '';
  if (!ids || ids.length === 0) return;
  ids.forEach((id, i) => {
    const chip = makeChip(runtime.idToToken(id), id, i, 'prompt');
    const badge = document.createElement('span');
    badge.className = 'chip-id';
    badge.textContent = id;
    chip.appendChild(document.createTextNode(' '));
    chip.appendChild(badge);
    preview.appendChild(chip);
  });
  const count = document.createElement('span');
  count.className = 'preview-count';
  count.textContent = `${ids.length} token`;
  preview.appendChild(count);
}

function renderSequence() {
  const box = ui.sequence();
  box.innerHTML = '';
  box.classList.remove('empty');
  const context = state.contextTokens;
  if (!context || context.length === 0) {
    box.classList.add('empty');
    return;
  }
  const promptLen = state.promptTokens.length;
  const genLen = state.generatedTokens.length;
  context.forEach((id, i) => {
    let kind;
    if (i < promptLen) kind = 'prompt';
    else if (i === context.length - 1 && genLen > 0) kind = 'generated last-generated';
    else kind = 'generated';
    box.appendChild(makeChip(runtime.idToToken(id), id, i, kind));
  });
}

function renderRanking() {
  const body = ui.rankingBody();
  body.innerHTML = '';
  const ranking = state.ranking || [];
  ranking.forEach((c, i) => {
    const tr = document.createElement('tr');
    if (c.tokenId === state.selectedToken?.tokenId) tr.classList.add('chosen-row');
    const chosenTag = c.tokenId === state.selectedToken?.tokenId
      ? ' <span class="tag">scelto</span>' : '';
    const tokenText = escapeHtml(displayTokenText(c.tokenId));
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td><span class="token-text">${tokenText}</span>${chosenTag}</td>
      <td>${c.tokenId}</td>
      <td class="prob">${(c.prob * 100).toFixed(1).replace('.', ',')}%</td>`;
    body.appendChild(tr);
  });
  if (state.selectedToken) {
    const txt = displayTokenText(state.selectedToken.tokenId);
    ui.chosenToken().innerHTML =
      `<strong>${escapeHtml(txt)}</strong> · ID ${state.selectedToken.tokenId} · p=${(state.selectedProbability * 100).toFixed(1).replace('.', ',')}%`;
  }
}

/** Testo leggibile del token: il testo del vocabolario se disponibile. */
function displayTokenText(id) {
  if (id === runtime.eosId) return '⟨EOS⟩';
  const token = runtime.idToToken(id);
  if (/^<tok-\d+>$/.test(token)) return `<tok-${id}>`;
  return token;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function renderContextMessage() {
  const el = ui.contextMessage();
  const n = state.contextTokens.length;
  el.hidden = false;
  if (state.stepIndex === 0 && n > 0) {
    el.textContent = `Il prossimo token viene calcolato usando ${n} token di contesto.`;
  } else if (state.selectedToken) {
    el.textContent =
      `Il token «${runtime.idToToken(state.selectedToken.tokenId)}» è stato aggiunto. Il prossimo calcolo userà ${n} token di contesto.`;
  } else {
    el.hidden = true;
  }
}

function renderCacheMessage(newTokens) {
  const el = ui.cacheMessage();
  const n = state.contextTokens.length;
  if (state.mode === 'cache') {
    el.textContent = `Contesto: ${n} token · Modalità: KV cache · Nuovo calcolo: ${newTokens ?? 0} token`;
  } else {
    el.textContent = `Contesto: ${n} token · Modalità: ricalcolo completo`;
  }
  el.hidden = false;
}

function render() {
  renderSequence();
  renderRanking();
  renderContextMessage();
  const hasResult = state.selectedToken !== null;
  ui.stepResult().hidden = !hasResult;
  updateButton();
}

async function nextStep() {
  if (state.isBusy) return;
  clearError();
  const prompt = ui.prompt().value.trim();
  if (!prompt) return;
  const { errors, values } = validateParams(readParams());
  if (errors.length > 0) {
    const el = ui.paramsError();
    el.textContent = errors.join(' · ');
    el.hidden = false;
    return;
  }
  const el = ui.paramsError();
  el.hidden = true;

  setBusy(true);
  try {
    const promptChanged = committedPromptId.value !== prompt;
    if (promptChanged) {
      state.promptText = prompt;
      state.promptTokens = runtime.encode(prompt);
      invalidateGeneration();
    } else if (state.contextTokens.length === 0) {
      state.promptTokens = runtime.encode(prompt);
      state.contextTokens = state.promptTokens.slice();
    }

    if (state.contextTokens.length === 0) {
      throw new Error('Il prompt non produce token validi.');
    }

    const cache = state.mode === 'cache' ? state.cacheState : null;
    const step = await computeStep(runtime, {
      contextIds: state.contextTokens,
      params: { ...values },
      cache,
    });

    state.selectedToken = step.chosen;
    state.selectedProbability = step.chosen ? step.ranking.find((c) => c.tokenId === step.chosen.tokenId)?.prob ?? step.chosen.prob : null;
    state.ranking = step.ranking;
    state.generatedTokens.push(step.chosen.tokenId);
    state.contextTokens = state.promptTokens.concat(state.generatedTokens);
    state.stepIndex += 1;
    state.cacheState = step.cache;
    state.history.push({ step: state.stepIndex, tokenId: step.chosen.tokenId, prob: state.selectedProbability });
    committedPromptId.value = ui.prompt().value;
    ui.hint().hidden = true;

    if (step.isEos) {
      showError('Raggiunto il token EOS: la generazione è terminata.');
    }
    renderCacheMessage(step.newTokens);
    render();
  } catch (err) {
    showError(`Errore runtime: ${err.message}`);
  } finally {
    setBusy(false);
  }
}

function reset() {
  state.promptTokens = [];
  state.generatedTokens = [];
  state.contextTokens = [];
  state.ranking = null;
  state.selectedToken = null;
  state.selectedProbability = null;
  state.stepIndex = 0;
  state.cacheState = null;
  state.history = [];
  state.error = null;
  committedPromptId.value = null;
  ui.prompt().value = '';
  ui.hint().hidden = true;
  ui.stepResult().hidden = true;
  ui.contextMessage().hidden = true;
  ui.cacheMessage().hidden = true;
  ui.sequence().innerHTML = '';
  ui.sequence().classList.add('empty');
  const preview = document.getElementById('prompt-preview');
  if (preview) preview.innerHTML = '';
  clearError();
  updateButton();
}

function init() {
  ui.btnNext().addEventListener('click', nextStep);
  ui.btnNextBottom()?.addEventListener('click', nextStep);
  ui.btnReset().addEventListener('click', reset);
  ui.prompt().addEventListener('input', () => {
    const changed = committedPromptId.value !== null && committedPromptId.value !== ui.prompt().value;
    ui.hint().hidden = !changed;
    try {
      updateButton();
      liveTokenizePrompt();
    } catch {
      // il runtime non è ancora pronto: la preview partirà al termine dell'init
    }
  });
  document.getElementById('param-cachemode').addEventListener('change', (e) => {
    state.mode = e.target.value;
  });

  initRuntime().then(({ backend, modelInfo, fallback }) => {
    state.modelStatus = 'ready';
    state.backend = backend;
    ui.modelIndicator().textContent = `Modello: ${modelInfo}`;
    ui.backendIndicator().textContent = `Backend: ${backend}`;
    if (fallback) {
      showError('Modello reale non disponibile in questo ambiente: uso il mock deterministico.');
    }
    updateButton();
    liveTokenizePrompt();
  }).catch((err) => {
    state.modelStatus = 'error';
    showError(`Impossibile inizializzare il modello: ${err.message}`);
    updateButton();
  });
  updateButton();
  liveTokenizePrompt();
}

init();
