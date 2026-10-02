/** Piccoli aiuti DOM, senza dipendenze. */

export const $ = (id) => document.getElementById(id);

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/** Attesa interrompibile, usata per il ritmo della presentazione. */
export function wait(ms, signal) {
  if (!ms || signal?.aborted) return Promise.resolve();
  return new Promise((resolve) => {
    const t0 = performance.now();
    const tick = () => {
      if (signal?.aborted || performance.now() - t0 >= ms) resolve();
      else setTimeout(tick, Math.min(50, ms));
    };
    setTimeout(tick, Math.min(50, ms));
  });
}
