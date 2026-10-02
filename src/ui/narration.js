/**
 * Narrazione con cronologia: le frasi che spiegano che cosa sta succedendo
 * non si sostituiscono, si accumulano una sotto l'altra, e nulla si perde.
 *
 * - Preparazione (1 → 2 → 3) e ogni token (4 → 5 → 6 → ↺) sono un «atto».
 * - Quando inizia l'atto successivo, le righe del precedente si chiudono in
 *   un gruppo richiudibile («Token 2 · «·cielo» 66,4%»): un clic lo riapre.
 * - Una stessa fase ripetuta (es. 4: «sta calcolando» → «ha calcolato»)
 *   aggiorna la propria riga invece di aggiungerne un'altra.
 * - Se il presentatore cambia token (5 dopo ↺), le righe della scelta
 *   precedente restano, segnate come superate.
 * - L'elenco ha altezza fissa e si scorre a mano: segue le frasi nuove solo
 *   se si è già in fondo; la pagina non scorre mai da sola.
 */

function makeNode(parts) {
  const frag = document.createDocumentFragment();
  for (const p of [].concat(parts)) frag.append(p instanceof Node ? p : document.createTextNode(String(p)));
  return frag;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function createNarration(root) {
  const list = root.querySelector('.nar-list');
  let act = 0; // 0 = preparazione dell'input, poi numero del token

  const lines = () => [...list.querySelectorAll(':scope > .nar-item')];
  const last = () => lines().at(-1) ?? null;
  const atBottom = () => list.scrollHeight - list.scrollTop - list.clientHeight < 32;

  function item(n, parts, kind) {
    const li = el('li', 'nar-item enter');
    li.dataset.n = n;
    li.dataset.kind = kind;
    const num = el('span', 'nar-n', n);
    num.setAttribute('aria-hidden', 'true');
    const text = el('span', 'nar-text');
    text.append(makeNode(parts));
    li.append(num, text);
    li.addEventListener('animationend', () => li.classList.remove('enter'), { once: true });
    return li;
  }

  /** Chiude l'atto corrente in un gruppo richiudibile che resta nella cronologia. */
  function collapse() {
    const current = lines();
    if (current.length === 0) return;
    let title;
    let detail = '';
    if (act === 0) {
      title = 'Input pronto';
      const three = current.find((l) => l.dataset.n === '3');
      const m = three?.textContent.match(/in ([\d.]+) token/);
      if (m) detail = `${m[1]} token`;
    } else {
      title = `Token ${act}`;
      const choice = current.filter((l) => l.dataset.n === '5' && !l.classList.contains('superseded')).at(-1);
      const tok = choice?.querySelector('.tokq')?.textContent ?? '';
      const pct = choice?.textContent.match(/\(([<>]?[\d,]+%)\)/)?.[1] ?? '';
      const byYou = choice?.textContent.startsWith('Hai scelto tu') ? ' · scelto da te' : '';
      detail = [tok, pct].filter(Boolean).join(' ') + byYou;
    }
    const group = el('li', 'nar-group summary');
    group.dataset.n = 'riepilogo';
    const details = el('details');
    const summary = el('summary');
    const num = el('span', 'nar-n', '✓');
    num.setAttribute('aria-hidden', 'true');
    const text = el('span', 'nar-text');
    text.append(el('strong', '', title), detail ? ` · ${detail}` : '');
    summary.append(num, text);
    const sub = el('ol', 'nar-sublist');
    for (const l of current) {
      l.classList.remove('current', 'enter');
      sub.append(l);
    }
    details.append(summary, sub);
    group.append(details);
    list.append(group);
  }

  function markCurrent(li) {
    list.querySelectorAll('.nar-item.current').forEach((x) => x.classList.remove('current'));
    li.classList.add('current');
  }

  return {
    /** Aggiunge (o aggiorna) la frase della fase `n`. */
    say(n, parts, kind = 'app') {
      const follow = atBottom();
      const prev = last();
      if (n === '1') {
        list.replaceChildren();
        act = 0;
      } else if (n === '4' && prev && ['3', '↺'].includes(prev.dataset.n)) {
        collapse();
        act += 1;
      } else if (n === '5' && prev && prev.dataset.n === '↺') {
        // il presentatore ha scelto un altro token per lo stesso passo
        for (const l of lines()) {
          if (['5', '6', '↺'].includes(l.dataset.n)) l.classList.add('superseded');
        }
      }
      const target = last();
      if (target && target.dataset.n === n && !target.classList.contains('superseded')) {
        target.querySelector('.nar-text').replaceChildren(makeNode(parts));
        target.dataset.kind = kind;
        markCurrent(target);
      } else {
        const li = item(n, parts, kind);
        list.append(li);
        markCurrent(li);
      }
      // segue le frasi nuove solo se si era già in fondo (scorre solo l'elenco, mai la pagina)
      if (follow || n === '1') list.scrollTop = list.scrollHeight;
    },
    reset(message) {
      list.replaceChildren();
      act = 0;
      if (message) this.say('1', message);
    },
  };
}
