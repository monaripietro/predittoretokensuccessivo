/**
 * "Anatomia di una chatbot": diagramma animato. Ogni animazione corrisponde a
 * un evento reale dell'inferenza e mostra solo dati ricevuti dagli oggetti
 * passati (ID, testi dei token, probabilità, testo della risposta).
 * La "stanza" è un'analogia grafica, non lo schema interno del modello.
 */

import './anatomy.css';
import { formatPercent } from '../core/distribution.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const EASE = 'cubic-bezier(.2,.8,.2,1)';
const LANE_MAX = 4;
const ANSWER_TAIL = 80;
const STAGES = ['write', 'assemble', 'tokenize', 'model', 'select', 'decode', 'loop'];

const fmtInt = (n) => Number(n).toLocaleString('it-IT');

/** Testo di un token con spazi e a capo resi visibili. */
function visible(text) {
  return String(text ?? '').replace(/ /g, '·').replace(/\n/g, '↵');
}

/** Come mostrare un pezzo di testo (candidato o scelto). */
function pieceLabel(piece, special) {
  const p = piece ?? {};
  if (p.special || special) return { text: p.text ?? '', kind: 'special' };
  if (p.fragment && !p.text) return { text: 'pezzo di carattere', kind: 'fragment' };
  if (p.empty) return { text: 'senza testo', kind: 'fragment' };
  return { text: visible(p.text), kind: 'text' };
}



function arrowSvg(cls) {
  return `<svg class="${cls}" viewBox="0 0 48 14" aria-hidden="true" focusable="false"><path d="M2 7h38m-8-5.5L41 7l-9 5.5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}

const TEMPLATE = `
<div class="an" data-run="idle">
  <div class="an-grid">
    <div class="an-app an-app-left" aria-label="Applicazione chatbot: parte iniziale">
      <span class="an-app-label">Applicazione chatbot</span>
      <div class="an-bubble an-prompt" data-stage="write" data-role="prompt">
        <span class="an-bubble-tag">Prompt</span>
        <span class="an-bubble-text" data-role="prompt-text">La tua domanda</span>
        <span class="an-bubble-sub">richiesta in linguaggio naturale</span>
      </div>
      <span class="an-down" aria-hidden="true"></span>
      <div class="an-node an-node-small" data-stage="assemble" data-role="assemble">
        <h3 class="an-title">Preparazione</h3>
        <p class="an-sub">l'app aggiunge istruzioni e ruoli</p>
        <span class="an-chip an-chip-app" data-role="app-chip" hidden></span>
      </div>
      <span class="an-down" aria-hidden="true"></span>
      <div class="an-node" data-stage="tokenize" data-role="tokenizer">
        <h3 class="an-title">Tokenizzatore</h3>
        <p class="an-sub">testo → numeri (ID)</p>
        <p class="an-count" data-role="tok-count">&nbsp;</p>
      </div>
    </div>

    <div class="an-lane an-lane-in" data-role="lane-in">
      <div class="an-lane-head"><span class="an-lane-tag">IN</span><span class="an-lane-name">seq. numeri</span></div>
      <div class="an-lane-cards" data-role="in-cards"></div>
      ${arrowSvg('an-lane-arrow')}
    </div>

    <div class="an-room-cell">
      <div class="an-room" data-stage="model" data-role="room">
        <h3 class="an-room-title">Large Language Model</h3>
        <p class="an-room-sub">la «stanza»</p>
        <div class="an-room-view">
          <span class="an-slot an-slot-in" data-role="slot-in" aria-hidden="true"><b>IN</b></span>
          <p class="an-room-state"><i class="an-dot" aria-hidden="true"></i><span data-role="room-state">in attesa</span></p>
          <span class="an-slot an-slot-out" data-role="slot-out" aria-hidden="true"><b>OUT</b></span>
        </div>
      </div>
      <p class="an-room-note">La «stanza»: vediamo solo cosa entra e cosa esce. È un'analogia, non lo schema interno.</p>
    </div>

    <div class="an-lane an-lane-out" data-role="lane-out">
      <div class="an-lane-head"><span class="an-lane-tag">OUT</span><span class="an-lane-name">seq. numeri</span></div>
      <div class="an-lane-cards an-out-cards" data-role="out-cards"></div>
      <p class="an-lane-note" data-role="out-note"></p>
      ${arrowSvg('an-lane-arrow')}
    </div>

    <div class="an-app an-app-right" aria-label="Applicazione chatbot: parte finale">
      <span class="an-app-label">Applicazione chatbot</span>
      <div class="an-node an-node-small" data-stage="select" data-role="select">
        <h3 class="an-title">Scelta</h3>
        <p class="an-sub" data-role="select-sub">l'app sceglie UN token</p>
        <span class="an-chip an-chip-id" data-role="id-chip" hidden></span>
      </div>
      <span class="an-down" aria-hidden="true"></span>
      <div class="an-node" data-stage="decode" data-role="detok">
        <h3 class="an-title">Tokenizzatore a contrario</h3>
        <p class="an-sub">numeri → testo</p>
        <p class="an-decoded" data-role="decoded">&nbsp;</p>
      </div>
      <span class="an-down" aria-hidden="true"></span>
      <div class="an-bubble an-answer" data-stage="decode" data-role="answer">
        <span class="an-bubble-tag">Risposta</span>
        <span class="an-bubble-text" data-role="answer-text">La risposta</span>
        <span class="an-bubble-sub" data-role="answer-sub">sequenza di parole</span>
      </div>
    </div>

    <div class="an-loop" data-stage="loop" data-role="loop">
      <svg class="an-loop-svg" viewBox="0 0 100 24" preserveAspectRatio="none" aria-hidden="true" focusable="false">
        <path d="M98 0v14a8 8 0 0 1-8 8H10a8 8 0 0 1-8-8V3" fill="none" stroke="currentColor" stroke-width="2.4" stroke-dasharray="1 6" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      </svg>
      <span class="an-loop-label">↺ il token scelto torna in ingresso</span>
    </div>
  </div>
  <p class="an-foot">Dentro al modello non entrano parole: entrano numeri.</p>
</div>`;

/** Attesa interrompibile (durata in ms, `signal` = oggetto { aborted }). */
function pause(ms, signal) {
  if (!ms || ms <= 0 || signal?.aborted) return Promise.resolve();
  return new Promise((resolve) => {
    const t0 = performance.now();
    const tick = () => {
      if (signal?.aborted || performance.now() - t0 >= ms) resolve();
      else setTimeout(tick, Math.min(40, Math.max(1, ms - (performance.now() - t0))));
    };
    setTimeout(tick, Math.min(40, ms));
  });
}

export function createAnatomy(container, {
  reducedMotion = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches,
} = {}) {
  const noop = async () => {};
  if (!container) {
    return {
      reset() {}, setActive() {}, prompt: noop, assemble: noop, tokenize: noop, computing() {}, scored: noop, selected: noop, decoded: noop, looped: noop, finished() {}, revertLast() {},
    };
  }
  container.innerHTML = TEMPLATE;
  const root = container.querySelector('.an');
  const q = (role) => root.querySelector(`[data-role="${role}"]`);
  const refs = {};
  root.querySelectorAll('[data-role]').forEach((n) => { refs[n.dataset.role] = n; });
  if (getComputedStyle(container).position === 'static') container.style.position = 'relative';
  root.style.position = 'relative';

  const S = {
    inIds: [],
    prevAnswer: '',
    beforeLast: '',
    lastLooped: false,
    runToken: 0,
    flyers: new Set(),
    lastCard: null,
  };

  /* ---------------- utilità ---------------- */

  const reduced = () => reducedMotion;

  function rel(elem) {
    const a = root.getBoundingClientRect();
    const b = elem.getBoundingClientRect();
    return { x: b.left - a.left, y: b.top - a.top, w: b.width, h: b.height, cx: b.left - a.left + b.width / 2, cy: b.top - a.top + b.height / 2 };
  }

  /**
   * Fa volare una copia di `src` verso `dst` (o lungo i punti `via`).
   * Restituisce un'animazione già avviata (o null se il movimento è disattivato).
   */
  function fly(src, dst, { duration, via = [], scale = 1, fade = false, delay = 0, content } = {}) {
    if (!src || !dst || reduced() || !duration || !root.animate) return null;
    try {
      const a = rel(src);
      const b = rel(dst);
      const clone = src.cloneNode(true);
      clone.removeAttribute('hidden');
      clone.classList.add('an-flyer');
      if (content !== undefined) clone.textContent = content;
      Object.assign(clone.style, {
        position: 'absolute', left: `${a.x}px`, top: `${a.y}px`, width: `${a.w}px`, margin: '0', zIndex: '20', pointerEvents: 'none', boxSizing: 'border-box',
      });
      root.append(clone);
      const frames = [{ transform: 'translate(0,0) scale(1)', opacity: 1, offset: 0 }];
      const pts = [...via.map((p) => ({ dx: p.cx - a.cx, dy: p.cy - a.cy })), { dx: b.cx - a.cx, dy: b.cy - a.cy }];
      pts.forEach((p, i) => {
        const last = i === pts.length - 1;
        frames.push({
          transform: `translate(${p.dx}px,${p.dy}px) scale(${last ? scale : 1})`,
          opacity: last && fade ? 0 : 1,
          offset: (i + 1) / pts.length,
        });
      });
      const anim = clone.animate(frames, { duration, delay, easing: EASE, fill: 'forwards' });
      const entry = { anim, clone };
      S.flyers.add(entry);
      anim.finished.catch(() => {}).then(() => {
        clone.remove();
        S.flyers.delete(entry);
      });
      return anim;
    } catch {
      return null;
    }
  }

  function clearFlyers() {
    for (const f of S.flyers) {
      try { f.anim.cancel(); } catch { /* già finita */ }
      f.clone.remove();
    }
    S.flyers.clear();
  }

  const flyDuration = (duration, share = 0.7) => Math.max(180, Math.min(900, (duration || 0) * share));

  /** Un punto di passaggio (centro) per percorsi in più tratti. */
  const point = (x, y) => ({ cx: x, cy: y });

  function bump(node, cls = 'pop') {
    if (!node || reduced()) return;
    node.classList.remove(cls);
    void node.offsetWidth; // riavvia l'animazione CSS
    node.classList.add(cls);
  }

  function idCard(id, extra = '') {
    const c = document.createElement('span');
    c.className = `an-id ${extra}`.trim();
    c.textContent = String(id);
    c.dataset.id = String(id);
    return c;
  }

  /* ---------------- corsia IN ---------------- */

  function renderIn({ animateLast = false, animate = true } = {}) {
    const box = refs['in-cards'];
    box.textContent = '';
    if (S.inIds.length === 0) {
      for (let i = 0; i < 3; i += 1) {
        const ph = document.createElement('span');
        ph.className = 'an-id an-ph';
        ph.textContent = 'ID';
        box.append(ph);
      }
      return;
    }
    const shown = S.inIds.slice(-LANE_MAX);
    if (S.inIds.length > LANE_MAX) {
      const dots = document.createElement('span');
      dots.className = 'an-dots';
      dots.textContent = '…';
      box.append(dots);
    }
    shown.forEach((id, i) => {
      const c = idCard(id, animate && !reduced() ? 'grow' : '');
      if (animate) c.style.animationDelay = `${i * 45}ms`;
      if (animateLast && i === shown.length - 1) c.classList.add('grow', 'new');
      box.append(c);
    });
  }

  /* ---------------- computing ---------------- */

  function stopComputing() {
    S.runToken += 1;
    root.classList.remove('is-computing');
    refs.room.classList.remove('computing');
    refs['room-state'].textContent = 'in attesa';
    refs['in-cards'].querySelectorAll('.sending').forEach((c) => c.classList.remove('sending'));
  }

  async function computeLoop(token) {
    const slot = refs['slot-in'];
    let i = 0;
    while (S.runToken === token) {
      const cards = [...refs['in-cards'].querySelectorAll('.an-id:not(.an-ph)')];
      if (cards.length === 0) {
        await pause(400, null);
      } else {
        const card = cards[i % cards.length];
        i += 1;
        card.classList.add('sending');
        const dur = reduced() ? 260 : 520;
        if (!reduced()) fly(card, slot, { duration: dur, scale: 0.35, fade: true });
        await pause(dur + (reduced() ? 0 : 70), null);
        card.classList.remove('sending');
        if (S.runToken !== token) break;
        if (i % cards.length === 0) await pause(reduced() ? 200 : 260, null);
      }
    }
  }

  /* ---------------- API ---------------- */

  function reset() {
    stopComputing();
    clearFlyers();
    S.inIds = [];
    S.prevAnswer = '';
    S.beforeLast = '';
    S.lastLooped = false;
    root.dataset.run = 'idle';
    root.removeAttribute('data-finish');
    refs['prompt-text'].textContent = 'La tua domanda';
    refs.prompt.classList.remove('filled');
    refs['app-chip'].hidden = true;
    refs['app-chip'].textContent = '';
    refs.assemble.classList.remove('done');
    refs['tok-count'].innerHTML = '&nbsp;';
    refs.tokenizer.classList.remove('done');
    refs['out-cards'].textContent = '';
    refs['out-note'].textContent = '';
    refs['id-chip'].hidden = true;
    refs['select-sub'].textContent = "l'app sceglie UN token";
    refs.decoded.innerHTML = '&nbsp;';
    refs.decoded.className = 'an-decoded';
    refs['answer-text'].textContent = 'La risposta';
    refs['answer-sub'].textContent = 'sequenza di parole';
    refs.answer.classList.remove('filled', 'done');
    refs['room-state'].textContent = 'in attesa';
    renderIn({ animate: false });
    setActive(null);
  }

  function setActive(stage) {
    root.querySelectorAll('[data-stage]').forEach((n) => {
      n.classList.toggle('active', Boolean(stage) && n.dataset.stage === stage);
    });
    root.dataset.stage = STAGES.includes(stage) ? stage : '';
  }

  async function prompt(text, { duration = 0, signal } = {}) {
    root.dataset.run = 'running';
    root.removeAttribute('data-finish');
    const t = String(text ?? '').replace(/\s+/g, ' ').trim();
    refs['prompt-text'].textContent = t || 'La tua domanda';
    refs['prompt-text'].title = t;
    refs.prompt.classList.add('filled');
    bump(refs.prompt);
    await pause(duration, signal);
  }

  async function assemble({ appTokenCount = 0, systemPrompt = '' } = {}, { duration = 0, signal } = {}) {
    const chip = refs['app-chip'];
    chip.textContent = `+ ${fmtInt(appTokenCount)} token dell'app`;
    chip.title = systemPrompt ? `Istruzione dell'app: ${systemPrompt}` : '';
    chip.hidden = false;
    refs.assemble.classList.add('done');
    if (!reduced() && duration) {
      const a = chip.animate([
        { transform: 'translateY(-14px) scale(.9)', opacity: 0 },
        { transform: 'none', opacity: 1 },
      ], { duration: Math.min(500, duration * 0.6), easing: EASE, fill: 'both' });
      a.finished.catch(() => {});
    }
    await pause(duration, signal);
  }

  async function tokenize({ tokens = [], total } = {}, { duration = 0, signal } = {}) {
    const n = total ?? tokens.length;
    refs['tok-count'].textContent = `${fmtInt(n)} token`;
    refs.tokenizer.classList.add('done');
    S.inIds = tokens.map((t) => t.id);
    renderIn({ animate: true });
    bump(refs.tokenizer);
    await pause(duration, signal);
  }

  function computing({ ids } = {}) {
    stopComputing();
    if (Array.isArray(ids) && ids.length) {
      const same = ids.length === S.inIds.length && ids.every((v, i) => v === S.inIds[i]);
      if (!same) {
        S.inIds = ids.slice();
        renderIn({ animate: false });
      }
    }
    refs['out-cards'].textContent = '';
    refs['out-note'].textContent = '';
    refs['id-chip'].hidden = true;
    root.dataset.run = 'running';
    root.classList.add('is-computing');
    refs.room.classList.add('computing');
    refs['room-state'].textContent = 'sta calcolando…';
    const token = S.runToken;
    computeLoop(token);
  }

  function renderScoreCard(c, rank, { extra = false } = {}) {
    const info = pieceLabel(c.piece, c.special);
    const card = document.createElement('div');
    card.className = `an-score ${info.kind}${extra ? ' extra' : ''}`;
    card.dataset.id = String(c.id);
    const label = document.createElement('span');
    label.className = 'an-score-text';
    label.textContent = info.kind === 'special' ? info.text : (info.kind === 'text' ? `«${info.text}»` : info.text);
    const prob = document.createElement('span');
    prob.className = 'an-score-prob';
    prob.textContent = extra ? `n° ${rank} · ${formatPercent(c.modelProb)}` : formatPercent(c.modelProb);
    const bar = document.createElement('span');
    bar.className = 'an-score-bar';
    const fill = document.createElement('i');
    fill.style.setProperty('--w', `${Math.max(1, Math.min(100, (c.modelProb ?? 0) * 100))}%`);
    bar.append(fill);
    card.append(label, prob, bar);
    return card;
  }

  function tagCard(card, text, cls) {
    card.querySelectorAll('.an-score-tag').forEach((n) => n.remove());
    if (!card) return;
    const tag = document.createElement('span');
    tag.className = `an-score-tag ${cls}`;
    tag.textContent = text;
    card.prepend(tag);
  }

  async function scored(step, { duration = 0, signal } = {}) {
    stopComputing();
    root.dataset.run = 'running';
    const box = refs['out-cards'];
    box.textContent = '';
    refs['id-chip'].hidden = true;
    refs.decoded.className = 'an-decoded';
    refs.decoded.innerHTML = '&nbsp;';
    const cands = (step?.candidates ?? []).slice(0, 3);
    cands.forEach((c, i) => {
      const card = renderScoreCard(c);
      card.style.animationDelay = `${i * 90}ms`;
      box.append(card);
    });
    if (step?.vocabSize) {
      refs['out-note'].textContent = `un punteggio per ognuno dei ${fmtInt(step.vocabSize)} token`;
    }
    refs['select-sub'].textContent = step?.policy?.kind === 'sample' ? 'estrazione pesata' : 'il più probabile';
    refs['room-state'].textContent = 'punteggi pronti';
    if (!reduced() && duration) {
      fly(refs['slot-out'], box.firstElementChild ?? refs['slot-out'], { duration: flyDuration(duration, 0.5), scale: 0.6, fade: true });
    }
    await pause(duration, signal);
  }

  async function selected(step, { duration = 0, signal } = {}) {
    const sel = step?.selected;
    if (!sel) { await pause(duration, signal); return; }
    refs['select-sub'].textContent = step.policy?.kind === 'sample' ? 'estrazione pesata' : 'il più probabile';
    const box = refs['out-cards'];
    const orig = step.override?.original ?? null;
    const keep = new Set([String(sel.id), orig ? String(orig.id) : '']);
    box.querySelectorAll('.extra').forEach((c) => { if (!keep.has(c.dataset.id)) c.remove(); });
    box.querySelectorAll('.an-score-tag').forEach((n) => n.remove());
    box.querySelectorAll('.pick, .rule').forEach((c) => c.classList.remove('pick', 'rule'));
    const find = (id) => [...box.children].find((c) => c.dataset.id === String(id));
    let card = find(sel.id);
    if (!card) {
      card = renderScoreCard({ ...sel, id: sel.id, piece: sel.piece, special: sel.special }, sel.rank, { extra: true });
      box.append(card);
    }
    if (orig) {
      let ruleCard = find(orig.id);
      if (!ruleCard) {
        ruleCard = renderScoreCard({ ...orig, id: orig.id, piece: orig.piece, special: orig.special }, orig.rank, { extra: true });
        box.append(ruleCard);
      }
      ruleCard.classList.add('rule');
      tagCard(ruleCard, 'la regola', 'rule');
      tagCard(card, 'scelto da te', 'user');
    }
    card.classList.add('pick');
    bump(card, 'pop');
    const chip = refs['id-chip'];
    chip.textContent = `ID ${sel.id}`;
    chip.hidden = false;
    bump(chip, 'pop');
    if (!reduced() && duration) fly(card, chip, { duration: flyDuration(duration, 0.6), scale: 0.5, fade: true });
    await pause(duration, signal);
  }

  async function decoded(step, { answerText = '', duration = 0, signal } = {}) {
    const sel = step?.selected ?? {};
    const info = pieceLabel(sel.piece, sel.special);
    const out = refs.decoded;
    const chip = refs['id-chip'];
    out.className = `an-decoded ${info.kind}`;
    out.textContent = '';
    const idSpan = document.createElement('b');
    idSpan.textContent = String(sel.id ?? '');
    const arrow = document.createTextNode(' → ');
    const txt = document.createElement('span');
    txt.className = 'an-decoded-text';
    if (info.kind === 'special') {
      txt.textContent = info.text;
      out.append(idSpan, arrow, txt);
      if (sel.isEos) {
        const end = document.createElement('em');
        end.textContent = ' fine';
        out.append(end);
      }
    } else if (info.kind === 'fragment') {
      txt.textContent = info.text;
      out.append(idSpan, arrow, txt);
    } else {
      txt.textContent = `«${info.text}»`;
      out.append(idSpan, arrow, txt);
    }
    // testo della risposta (già decodificato)
    const full = String(answerText ?? '');
    const prev = S.prevAnswer;
    S.beforeLast = prev;
    const delta = full.startsWith(prev) ? full.slice(prev.length) : '';
    S.prevAnswer = full;
    const shown = full.length > ANSWER_TAIL ? full.slice(-ANSWER_TAIL) : full;
    const box = refs['answer-text'];
    box.textContent = '';
    if (full.length > ANSWER_TAIL) box.append('…');
    if (delta && shown.endsWith(delta)) {
      box.append(shown.slice(0, shown.length - delta.length));
      const m = document.createElement('mark');
      m.textContent = delta;
      box.append(m);
    } else {
      box.append(shown);
    }
    if (!full) box.textContent = 'La risposta';
    refs.answer.classList.toggle('filled', Boolean(full));
    refs['answer-sub'].textContent = sel.isEos ? 'fine della risposta' : 'sequenza di parole';

    if (!reduced() && duration) {
      const d1 = flyDuration(duration, 0.45);
      fly(chip.hidden ? refs.select : chip, refs.detok, { duration: d1, scale: 0.7, fade: true });
      bump(out, 'pop');
      if (delta) {
        const piece = document.createElement('span');
        piece.className = 'an-flyer-text';
        piece.textContent = visible(delta);
        txt.after(piece);
        const flyer = fly(piece, refs['answer-text'], { duration: flyDuration(duration, 0.45), delay: d1 * 0.9, scale: 0.9, fade: true });
        piece.remove();
        if (flyer) { /* la copia vola da sola */ }
      }
    }
    await pause(duration, signal);
  }

  async function looped(step, { duration = 0, signal } = {}) {
    const sel = step?.selected;
    const chip = refs['id-chip'];
    const start = chip.hidden ? refs.select : chip;
    if (sel) {
      S.inIds.push(sel.id);
      S.lastLooped = true;
    }
    if (!reduced() && duration && sel) {
      // Percorso: giù fino alla corsia di ritorno, a sinistra, su fino alla corsia IN.
      const loop = rel(refs.loop);
      const laneIn = rel(refs['in-cards']);
      const wide = laneIn.cy < loop.cy - 20 && laneIn.cx < rel(refs.select).cx - 120;
      const ghost = idCard(sel.id, 'an-id');
      refs['in-cards'].append(ghost);
      ghost.style.visibility = 'hidden';
      const dst = ghost;
      const via = wide
        ? [point(rel(start).cx, loop.cy), point(laneIn.cx, loop.cy)]
        : [point(rel(start).cx, loop.cy)];
      fly(start, dst, { duration: flyDuration(duration, 0.85), via, scale: 1 });
      ghost.remove();
      await pause(flyDuration(duration, 0.85) * 0.95, signal);
      renderIn({ animate: false, animateLast: true });
      await pause(Math.max(0, duration - flyDuration(duration, 0.85) * 0.95), signal);
    } else {
      renderIn({ animate: false, animateLast: true });
      await pause(duration, signal);
    }
  }

  /** Annulla gli effetti visivi dell'ultimo passo (per la scelta del presentatore). */
  function revertLast() {
    clearFlyers();
    if (S.lastLooped) {
      S.inIds.pop();
      S.lastLooped = false;
      renderIn({ animate: false });
    }
    S.prevAnswer = S.beforeLast;
    const box = refs['answer-text'];
    box.textContent = S.prevAnswer ? (S.prevAnswer.length > ANSWER_TAIL ? `…${S.prevAnswer.slice(-ANSWER_TAIL)}` : S.prevAnswer) : 'La risposta';
    refs.answer.classList.toggle('filled', Boolean(S.prevAnswer));
    refs.answer.classList.remove('done');
    refs['answer-sub'].textContent = 'sequenza di parole';
    refs['id-chip'].hidden = true;
    refs.decoded.className = 'an-decoded';
    refs.decoded.innerHTML = '&nbsp;';
    root.removeAttribute('data-finish');
  }

  function finished(reason) {
    stopComputing();
    clearFlyers();
    setActive(null);
    root.dataset.run = 'done';
    root.dataset.finish = reason ?? '';
    const sub = refs['answer-sub'];
    if (reason === 'eos') {
      refs.answer.classList.add('done');
      sub.textContent = '✓ risposta conclusa';
    } else if (reason === 'length') {
      sub.textContent = 'fermata al limite di token';
    } else if (reason === 'stopped') {
      sub.textContent = 'interrotta';
    } else if (reason === 'error') {
      sub.textContent = 'errore';
    }
  }

  reset();

  return {
    reset, setActive, prompt, assemble, tokenize, computing, scored, selected, decoded, looped, finished, revertLast,
  };
}
