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
  selectedRank: null,
  stepIndex: 0,
  modelStatus: 'loading',
  backend: '…',
  isBusy: false,
  mode: 'cache',
  cacheState: null,
  history: [],
  eosReached: false,
  isStale: false,
  error: null,
  activeCalculationContext: [],
  calculationPhase: 'idle',
  lastCalculationInput: [],
};

const ui = (id) => document.getElementById(id);

const useMockParam = new URLSearchParams(location.search).has('mock');
let runtime = null;
let runtimeKind = 'real';

async function initRuntime() {
  if (useMockParam) {
    runtime = new MockRuntime();
    runtimeKind = 'mock';
    const info = await runtime.init();
    return { ...info, kind: 'mock' };
  }
  runtime = new RealRuntime();
  runtimeKind = 'real';
  try {
    const info = await runtime.init();
    return { ...info, kind: 'real' };
  } catch (err) {
    runtime = null;
    runtimeKind = 'error';
    return { backend: '—', modelInfo: 'non disponibile', kind: 'error', cause: err };
  }
}

function startMockDemo() {
  runtime = new MockRuntime();
  runtimeKind = 'mock';
  runtime.init().then((info) => {
    state.modelStatus = 'ready';
    state.backend = info.backend;
    ui('model-indicator').textContent = `Modello: ${info.modelInfo}`;
    ui('backend-indicator').textContent = `Backend: ${info.backend}`;
    showMockUi(true);
    if (state.contextTokens.length === 0 && state.promptTokens.length === 0) {
      invalidateGeneration();
    }
    liveTokenizePrompt();
    updateButton();
  });
}

function showMockUi(show) {
  ui('mock-badge').hidden = !show;
  ui('mock-warning').hidden = !show;
  ui('tech-runtime').textContent = show ? 'mock deterministico (simulato)' : 'reale (Transformers.js)';
}

const committedPromptId = { value: null };

function readParams() {
  const get = (id) => ui(id).value;
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
  const btn = ui('btn-next');
  const btnBottom = ui('btn-next-bottom');
  const promptEmpty = !ui('prompt').value.trim();
  const modelReady = state.modelStatus === 'ready';
  const disabled = state.isBusy || promptEmpty || !modelReady || state.eosReached;
  const label = state.eosReached
    ? 'Generazione terminata (EOS)'
    : state.isBusy ? 'Calcolo…' : 'Calcola token successivo';
  btn.disabled = disabled;
  btn.textContent = label;
  btn.classList.toggle('highlight', state.isStale && !disabled);
  if (btnBottom) {
    btnBottom.disabled = disabled || state.contextTokens.length === 0;
    btnBottom.textContent = nextButtonLabel();
    btnBottom.classList.toggle('highlight', state.isStale && !disabled);
  }
}

function formatTokenForUi(text) {
  const visible = String(text)
    .replace(/ /g, '␠')
    .replace(/\n/g, '\\n')
    .replace(/\t/g, '\\t');
  return `«${visible}»`;
}

function nextButtonLabel() {
  if (state.eosReached) return 'Generazione terminata';
  if (state.isBusy) return 'Calcolo…';
  if (state.generatedTokens.length === 0) return 'Calcola token successivo';
  const lastId = state.generatedTokens.at(-1);
  return `Calcola il token dopo ${formatTokenForUi(displayTokenText(lastId))}`;
}

function invalidateGeneration() {
  state.eosReached = false;
  state.isStale = false;
  state.generatedTokens = [];
  state.contextTokens = state.promptTokens.slice();
  state.ranking = null;
  state.selectedToken = null;
  state.selectedProbability = null;
  state.selectedRank = null;
  state.stepIndex = 0;
  state.cacheState = null;
  state.history = [];
  state.error = null;
}

function showError(message) {
  state.error = message;
  const el = ui('error');
  el.textContent = message;
  el.hidden = !message;
}

function clearError() {
  showError(null);
}

/** Rende leggibili spazi e caratteri speciali nei token (didattica GPT-2 BPE). */
function visualizeToken(token) {
  return String(token)
    .replace(/ /g, '␠')
    .replace(/\n/g, '⏎')
    .replace(/\t/g, '⇥')
    .replace(/\r/g, '␍');
}

function displayTokenText(id) {
  if (runtime && id === runtime.eosId) return '⟨EOS⟩';
  if (!runtime) return String(id);
  return visualizeToken(runtime.idToToken(id));
}

function makeChip(token, id, position, kind) {
  const chip = document.createElement('span');
  chip.className = `chip ${kind}`;
  chip.dataset.tokenId = id;
  chip.dataset.position = position;
  const isEos = runtime && id === runtime.eosId;
  if (isEos) chip.classList.add('eos');
  const displayText = isEos ? '⟨EOS⟩' : displayTokenText(id);
  chip.textContent = displayText;
  chip.title = `Testo: ${isEos ? '<eos>' : displayText} · Token ID: ${id} · Posizione: ${position}`;
  chip.tabIndex = 0;
  return chip;
}

function liveTokenizePrompt() {
  if (!runtime || state.modelStatus !== 'ready') return;
  const prompt = ui('prompt').value.trim();
  const committed = committedPromptId.value;
  const promptChanged = committed !== null && committed !== ui('prompt').value;
  if (promptChanged || state.stepIndex === 0) {
    const ids = runtime.encode(prompt);
    state.promptTokens = ids;
    if (promptChanged) {
      state.contextTokens = ids;
      state.generatedTokens = [];
      state.isStale = true;
      renderSequence();
    } else if (state.stepIndex === 0) {
      state.contextTokens = ids;
    }
    renderPromptPreview(ids);
    updateButton();
  }
}

function renderPromptPreview(ids) {
  const preview = ui('prompt-preview');
  if (!preview) return;
  preview.innerHTML = '';
  if (!ids || ids.length === 0) return;
  ids.forEach((id, i) => {
    const chip = makeChip('', id, i, 'prompt');
    chip.textContent = displayTokenText(id);
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
  const tokenizerName = runtimeKind === 'real' ? 'GPT-2 BPE' : 'didattico (parole)';
  ui('preview-meta').textContent = `${ids.length} token · tokenizer ${tokenizerName}`;
}

function renderSequence() {
  const box = ui('token-sequence');
  box.innerHTML = '';
  box.classList.remove('empty');
  box.classList.toggle('stale', state.isStale);
  const context = state.contextTokens;
  if (!context || context.length === 0) {
    box.classList.add('empty');
    return;
  }
  const promptLen = state.promptTokens.length;
  const genLen = state.generatedTokens.length;
  const animate = state.animateNext && genLen > 0;
  context.forEach((id, i) => {
    let kind;
    let justAdded = false;
    if (i < promptLen) kind = 'prompt';
    else if (i === context.length - 1 && genLen > 0) {
      kind = 'generated last-generated';
      justAdded = animate;
    } else kind = 'generated';
    const chip = makeChip('', id, i, kind);
    if (justAdded) {
      chip.classList.add('just-added');
      chip.addEventListener('animationend', () => chip.classList.remove('just-added'), { once: true });
    }
    box.appendChild(chip);
  });
  state.animateNext = false;
}

function renderRanking() {
  const body = ui('ranking-body');
  body.innerHTML = '';
  const ranking = state.ranking || [];
  const maxProb = ranking.length > 0 ? Math.max(...ranking.map((c) => c.prob)) : 1;
  ranking.forEach((c, i) => {
    const tr = document.createElement('tr');
    if (c.tokenId === state.selectedToken?.tokenId) tr.classList.add('chosen-row');
    const chosenTag = c.tokenId === state.selectedToken?.tokenId
      ? ' <span class="tag">✓ scelto</span>' : '';
    const tokenText = escapeHtml(displayTokenText(c.tokenId));
    const pct = (c.prob * 100).toFixed(1).replace('.', ',');
    const width = Math.max(2, Math.round((c.prob / maxProb) * 100));
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td><span class="token-text">${tokenText}</span>${chosenTag}</td>
      <td>${c.tokenId}</td>
      <td class="prob-cell">
        <span class="prob-bar" aria-hidden="true"><span class="prob-bar-fill" style="width:${width}%"></span></span>
        <span class="prob">${pct}%</span>
      </td>`;
    body.appendChild(tr);
  });
  if (state.selectedToken) {
    ui('chosen-big').textContent = displayTokenText(state.selectedToken.tokenId);
    ui('chosen-id').textContent = state.selectedToken.tokenId;
    ui('chosen-prob').textContent = `${(state.selectedProbability * 100).toFixed(1).replace('.', ',')}%`;
    ui('chosen-rank').textContent = state.selectedRank !== null
      ? `#${state.selectedRank}`
      : (runtimeKind === 'real' ? '#1' : '#1');
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function renderContextMessage() {
  const el = ui('context-message');
  const n = state.contextTokens.length;
  const promptLen = state.promptTokens.length;
  el.hidden = false;
  if (state.stepIndex === 0 && n > 0) {
    el.textContent = `Il prossimo token viene calcolato usando ${n} token di contesto (tutti dal prompt).`;
  } else if (state.stepIndex === 1) {
    el.textContent = `Il modello ha elaborato i ${promptLen} token del prompt e ha calcolato la distribuzione del token in posizione ${promptLen + 1}. Il prossimo calcolo userà ${n} token di contesto.`;
  } else if (state.selectedToken) {
    el.textContent = `Il modello usa ora i ${n} token di contesto per calcolare il token in posizione ${n + 1}.`;
  } else {
    el.hidden = true;
  }
}

function renderCacheStatus(newTokens) {
  const box = ui('cache-status');
  const n = state.contextTokens.length;
  ui('cache-logical').textContent = n;
  const cacheReallyValid = !!(state.cacheState && state.cacheState.valid !== false && (state.cacheState.past || state.cacheState.ids));
  if (state.mode === 'cache') {
    ui('cache-current').textContent = `${newTokens ?? n} token`;
    ui('cache-active').textContent = cacheReallyValid ? 'attiva' : 'non disponibile dal runtime';
  } else {
    ui('cache-current').textContent = `${n} token`;
    ui('cache-active').textContent = 'non usata (ricalcolo completo)';
  }
  box.hidden = false;
}

function renderNextStepContext() {
  const panel = ui('next-step-panel');
  const contextBox = ui('next-step-context');
  const explanation = ui('next-step-explanation');
  if (!panel || !contextBox) return;
  const context = state.contextTokens ?? [];
  if (context.length === 0) {
    panel.hidden = true;
    return;
  }
  contextBox.innerHTML = '';
  const promptLength = state.promptTokens.length;
  context.forEach((tokenId, index) => {
    const kind = index < promptLength ? 'prompt' : 'generated';
    const chip = makeChip('', tokenId, index, kind);
    if (index === context.length - 1 && index >= promptLength) {
      chip.classList.add('last-context-token');
    }
    contextBox.appendChild(chip);
  });
  const nextPosition = context.length + 1;
  explanation.textContent =
    `Il modello userà questi ${context.length} token di contesto per calcolare `
    + `il token in posizione ${nextPosition}.`;
  panel.hidden = false;
}

function render() {
  renderPromptPreview(state.promptTokens);
  renderSequence();
  renderNextStepContext();
  renderRanking();
  renderContextMessage();
  renderHistory();
  const hasResult = state.selectedToken !== null;
  ui('step-result').hidden = !hasResult;
  ui('eos-message').hidden = !state.eosReached;
  updateButton();
}

function prepareContextForCalculation() {
  const prompt = ui('prompt').value.trim();
  if (!prompt) {
    throw new Error('Inserisci un prompt.');
  }
  const currentPromptTokens = runtime.encode(prompt);
  const promptChanged = committedPromptId.value !== prompt;
  if (promptChanged) {
    state.promptText = prompt;
    state.promptTokens = currentPromptTokens;
    state.generatedTokens = [];
    state.contextTokens = currentPromptTokens.slice();
    state.cacheState = null;
    state.history = [];
    state.stepIndex = 0;
    state.eosReached = false;
  }
  if (state.contextTokens.length === 0) {
    state.promptTokens = currentPromptTokens;
    state.contextTokens = currentPromptTokens.slice();
  }
  if (state.contextTokens.length === 0) {
    throw new Error('Il prompt non produce token validi.');
  }
  renderPromptPreview(state.promptTokens);
  renderNextStepContext();
}

function applyStepResult(step, values) {
  if (!step?.chosen) {
    throw new Error('Il modello non ha prodotto un token valido.');
  }
  const chosen = step.chosen;
  state.ranking = step.ranking;
  state.selectedToken = chosen;
  state.selectedProbability =
    step.ranking.find((c) => c.tokenId === chosen.tokenId)?.prob ?? chosen.prob;
  state.selectedRank = step.ranking.findIndex((c) => c.tokenId === chosen.tokenId) + 1 || null;
  state.generatedTokens.push(chosen.tokenId);
  state.contextTokens = [...state.promptTokens, ...state.generatedTokens];
  state.stepIndex += 1;
  state.cacheState = state.mode === 'cache' ? step.cache : null;
  state.eosReached = Boolean(step.isEos);
  state.history.push({
    step: state.stepIndex,
    tokenId: chosen.tokenId,
    prob: state.selectedProbability,
    rank: state.selectedRank,
    inputIds: state.lastCalculationInput.slice(),
    outputId: chosen.tokenId,
    contextLength: state.contextTokens.length,
  });
  committedPromptId.value = ui('prompt').value.trim();
  state.isStale = false;
  ui('prompt-hint').hidden = true;
  ui('params-hint').hidden = true;
  const p = { ...values };
  ui('params-applied').textContent =
    `Parametri applicati: temperatura ${p.temperature} · top-k ${p.topK} · top-p ${p.topP} · min-p ${p.minP} · repeat penalty ${p.repeatPenalty}`
    + (p.mode === 'sample' ? ` · sample (seed ${p.seed})` : ' · greedy');
  state.animateNext = true;
  renderCacheStatus(step.newTokens);
}

async function nextStep() {
  if (state.isBusy || state.eosReached) return;
  clearError();
  const prompt = ui('prompt').value.trim();
  if (!prompt) return;
  const { errors, values } = validateParams(readParams());
  if (errors.length > 0) {
    const el = ui('params-error');
    el.textContent = errors.join(' · ');
    el.hidden = false;
    return;
  }
  const el = ui('params-error');
  el.hidden = true;

  setBusy(true);
  try {
    prepareContextForCalculation();

    state.lastCalculationInput = state.contextTokens.slice();
    state.activeCalculationContext = state.contextTokens.slice();
    state.calculationPhase = state.stepIndex === 0
      ? 'calculating-first-token'
      : 'calculating-next-token';

    const cache = state.mode === 'cache' ? state.cacheState : null;
    const step = await computeStep(runtime, {
      contextIds: state.activeCalculationContext,
      params: { ...values },
      cache,
    });

    applyStepResult(step, values);
    state.calculationPhase = state.eosReached
      ? 'eos'
      : state.stepIndex === 1
        ? 'first-token-ready'
        : 'next-token-ready';
    render();
  } catch (err) {
    state.calculationPhase = 'error';
    showError(`Errore runtime: ${err.message}`);
    render();
  } finally {
    setBusy(false);
  }
}

function renderHistory() {
  const body = ui('history-body');
  if (!body) return;
  body.innerHTML = '';
  state.history.forEach((h, i) => {
    const tr = document.createElement('tr');
    const prev = i > 0 ? state.history[i - 1] : null;
    let deltaText = '—';
    let deltaClass = '';
    if (prev) {
      const d = h.prob - prev.prob;
      const sign = d >= 0 ? '+' : '−';
      deltaText = `${sign}${(Math.abs(d) * 100).toFixed(1).replace('.', ',')}%`;
      deltaClass = d >= 0 ? 'delta-up' : 'delta-down';
    }
    tr.innerHTML = `
      <td>${h.step}</td>
      <td><span class="token-text">${escapeHtml(displayTokenText(h.tokenId))}</span></td>
      <td>${h.tokenId}</td>
      <td>${(h.prob * 100).toFixed(1).replace('.', ',')}%</td>
      <td>${h.rank !== null && h.rank !== undefined ? `#${h.rank}` : '—'}</td>
      <td>${h.contextLength}</td>
      <td class="${deltaClass}">${deltaText}</td>`;
    body.appendChild(tr);
  });
}

function exportHistory() {
  const data = {
    prompt: state.promptText,
    runtime: runtimeKind,
    backend: state.backend,
    steps: state.history.map((h, i) => ({
      step: h.step,
      tokenId: h.tokenId,
      tokenText: runtime ? runtime.idToToken(h.tokenId) : null,
      probability: h.prob,
      rank: h.rank,
      contextLength: h.contextLength,
      deltaProbability: i > 0 ? h.prob - state.history[i - 1].prob : null,
    })),
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'next-token-lab-cronologia.json';
  a.click();
  URL.revokeObjectURL(url);
}

function reset() {
  state.eosReached = false;
  state.isStale = false;
  state.promptTokens = [];
  state.generatedTokens = [];
  state.contextTokens = [];
  state.ranking = null;
  state.selectedToken = null;
  state.selectedProbability = null;
  state.selectedRank = null;
  state.stepIndex = 0;
  state.cacheState = null;
  state.history = [];
  state.error = null;
  state.activeCalculationContext = [];
  state.calculationPhase = 'idle';
  state.lastCalculationInput = [];
  committedPromptId.value = null;
  ui('prompt').value = '';
  ui('prompt-hint').hidden = true;
  ui('params-hint').hidden = true;
  ui('step-result').hidden = true;
  ui('context-message').hidden = true;
  ui('cache-status').hidden = true;
  ui('eos-message').hidden = true;
  ui('sequence-hidden')?.removeAttribute('hidden');
  const seq = ui('token-sequence');
  seq.innerHTML = '';
  seq.classList.add('empty');
  seq.classList.remove('stale');
  ui('prompt-preview').innerHTML = '';
  ui('preview-meta').textContent = '';
  ui('next-step-panel').hidden = true;
  renderHistory();
  clearError();
  updateButton();
}

function applyPreset(prompt) {
  ui('prompt').value = prompt;
  ui('prompt').dispatchEvent(new Event('input', { bubbles: true }));
}

function init() {
  document.querySelectorAll('.preset').forEach((btn) => {
    btn.addEventListener('click', () => applyPreset(btn.dataset.prompt));
  });
  ui('btn-export')?.addEventListener('click', exportHistory);
  ui('btn-next').addEventListener('click', nextStep);
  ui('btn-next-bottom')?.addEventListener('click', nextStep);
  ui('btn-reset').addEventListener('click', reset);
  ui('btn-mock').addEventListener('click', startMockDemo);
  ui('prompt').addEventListener('input', () => {
    const changed = committedPromptId.value !== null && committedPromptId.value !== ui('prompt').value;
    ui('prompt-hint').hidden = !changed;
    try {
      updateButton();
      liveTokenizePrompt();
    } catch {
      // runtime non ancora pronto
    }
  });
  document.querySelectorAll('.params input, .params select').forEach((el) => {
    const onParamChange = () => {
      if (state.stepIndex > 0) {
        ui('params-hint').hidden = false;
      }
    };
    el.addEventListener('change', onParamChange);
    el.addEventListener('input', onParamChange);
  });
  ui('param-cachemode').addEventListener('change', (e) => {
    const nextMode = e.target.value;
    if (state.mode !== nextMode) {
      state.mode = nextMode;
      state.cacheState = null;
    }
  });

  ui('model-download').hidden = false;
  initRuntime().then(({ backend, modelInfo, kind, cause }) => {
    if (kind === 'error') {
      state.modelStatus = 'error';
      ui('model-indicator').textContent = 'Modello: non disponibile';
      ui('backend-indicator').textContent = `Backend: errore (${cause && cause.message ? cause.message.slice(0, 80) : 'sconosciuto'})`;
      ui('model-download').hidden = true;
      ui('btn-mock').hidden = false;
      showError('Il modello reale non si è caricato. Puoi esplorare l\'interfaccia con la demo mock (risultati simulati).');
      updateButton();
      return;
    }
    state.modelStatus = 'ready';
    state.backend = backend;
    ui('model-indicator').textContent = `Modello: ${modelInfo}`;
    ui('backend-indicator').textContent = `Backend: ${backend}`;
    ui('tech-backend').textContent = backend;
    ui('model-download').hidden = true;
    if (kind === 'mock') {
      showMockUi(true);
    } else {
      showMockUi(false);
    }
    updateButton();
    liveTokenizePrompt();
  }).catch((err) => {
    state.modelStatus = 'error';
    ui('model-download').hidden = true;
    showError(`Impossibile inizializzare il modello: ${err.message}`);
    updateButton();
  });
  updateButton();
}

init();
