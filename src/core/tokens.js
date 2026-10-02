/**
 * Ricostruzione fedele dei "pezzi" di testo dei token.
 *
 * Un token non è necessariamente una parola: può essere una parola intera,
 * una parte di parola, uno spazio, un segno, un token di controllo o perfino
 * un frammento di byte di un carattere (es. un'emoji divisa in due token).
 *
 * Il tokenizer reale fornisce solo `decode(ids)`. Per ottenere i pezzi
 * esatti si decodifica in modo incrementale a partire dall'ultimo confine
 * "pulito" (nessun carattere a metà). Se un token termina a metà di un
 * carattere UTF-8 il decoder produce U+FFFD: quel token resta in sospeso e
 * viene raggruppato con i successivi finché il carattere non è completo.
 * Così la concatenazione dei pezzi coincide con la decodifica dell'intera
 * sequenza (invariante verificata dai test).
 */

/** Carattere di sostituzione U+FFFD: il decoder lo produce per byte UTF-8 incompleti. */
export const REPLACEMENT = String.fromCharCode(0xfffd);

/**
 * @typedef {object} TokenPiece
 * @property {number} id
 * @property {string} text testo esatto prodotto dal token ('' per i frammenti non finali di un gruppo)
 * @property {string} groupText testo del gruppo di token che forma caratteri completi
 * @property {number} groupSize quanti token formano il gruppo
 * @property {number} groupIndex posizione del token nel gruppo (0-based)
 * @property {boolean} special token di controllo / speciale
 * @property {boolean} fragment il token da solo non forma un carattere completo
 */

/**
 * Crea un decodificatore incrementale legato a un tokenizer.
 * @param {{ decode: (ids: number[], opts?: object) => string }} tokenizer
 * @param {(id: number) => boolean} isSpecial
 */
export function createPieceDecoder(tokenizer, isSpecial) {
  let pending = [];
  const decode = (ids) => (ids.length ? tokenizer.decode(ids, { skip_special_tokens: false }) : '');

  function flush(groupText, ids) {
    return ids.map((id, i) => ({
      id,
      text: ids.length === 1 ? groupText : (i === ids.length - 1 ? groupText : ''),
      groupText,
      groupSize: ids.length,
      groupIndex: i,
      special: isSpecial(id),
      fragment: ids.length > 1,
    }));
  }

  return {
    /**
     * Aggiunge un token. Restituisce i pezzi completati (0, 1 o più se un
     * gruppo di frammenti si chiude) — mai pezzi inventati.
     */
    push(id) {
      if (isSpecial(id) && pending.length === 0) {
        return flush(decode([id]), [id]);
      }
      pending.push(id);
      const text = decode(pending);
      if (text.endsWith(REPLACEMENT) && pending.length < 8) {
        return [];
      }
      const out = flush(text, pending);
      pending = [];
      return out;
    },
    /** Chiude eventuali frammenti rimasti (es. fine della generazione). */
    end() {
      if (pending.length === 0) return [];
      const out = flush(decode(pending), pending);
      pending = [];
      return out;
    },
    get pendingIds() {
      return pending.slice();
    },
  };
}

/** Pezzi di un'intera sequenza (es. l'input completo inviato al modello). */
export function piecesOf(tokenizer, ids, isSpecial) {
  const dec = createPieceDecoder(tokenizer, isSpecial);
  const out = [];
  for (const id of ids) out.push(...dec.push(id));
  out.push(...dec.end());
  return out;
}

/**
 * Riconosce i token speciali/di controllo a partire dal tokenizer reale:
 * - ID elencati come speciali dal tokenizer;
 * - token "aggiunti" il cui testo è una marcatura tipo <...> (es. <think>);
 * - token che spariscono con skip_special_tokens.
 */
export function makeSpecialTokenTest(tokenizer, extraIds = []) {
  const ids = new Set([...(tokenizer.all_special_ids ?? []), ...extraIds]);
  const added = tokenizer._tokenizerJSON?.added_tokens;
  if (Array.isArray(added)) {
    for (const t of added) {
      if (t.special || /^<[^<>\s]+>$/.test(t.content)) ids.add(t.id);
    }
  }
  const cache = new Map();
  return (id) => {
    if (ids.has(id)) return true;
    if (cache.has(id)) return cache.get(id);
    let result = false;
    try {
      const withSpecial = tokenizer.decode([id], { skip_special_tokens: false });
      const without = tokenizer.decode([id], { skip_special_tokens: true });
      result = withSpecial !== '' && without === '';
    } catch {
      result = false;
    }
    cache.set(id, result);
    return result;
  };
}

/**
 * Pezzo leggibile di un candidato (non ancora scelto). Si decodifica il
 * candidato dopo un breve contesto reale e si prende la differenza, così
 * lo spazio iniziale non viene perso. Se il candidato è un frammento di
 * byte, lo si dichiara come tale invece di mostrare un carattere inventato.
 */
export function candidatePiece(tokenizer, contextIds, id, isSpecial) {
  if (isSpecial(id)) {
    return { text: tokenizer.decode([id], { skip_special_tokens: false }), special: true, fragment: false, empty: false };
  }
  const ctx = contextIds.slice(-4).filter((x) => !isSpecial(x));
  let text;
  try {
    const base = tokenizer.decode(ctx, { skip_special_tokens: false });
    const full = tokenizer.decode([...ctx, id], { skip_special_tokens: false });
    text = full.startsWith(base) && !base.endsWith(REPLACEMENT)
      ? full.slice(base.length)
      : tokenizer.decode([id], { skip_special_tokens: false });
  } catch {
    text = '';
  }
  const fragment = text.includes(REPLACEMENT);
  return { text: fragment ? '' : text, special: false, fragment, empty: !fragment && text === '' };
}

/** Rende visibili spazi e a-capo senza alterare il testo originale. */
export function visibleWhitespace(text) {
  return String(text)
    .replace(/\r/g, '␍')
    .replace(/\n/g, '↵')
    .replace(/\t/g, '⇥')
    .replace(/ /g, '·');
}

/**
 * Etichetta l'origine di ogni token dell'input inviato al modello.
 * @param {TokenPiece[]} pieces pezzi in ordine (la loro concatenazione è il testo completo)
 * @param {{start: number, end: number, origin: string}[]} spans intervalli di caratteri
 *   noti (es. testo dell'utente, istruzione di sistema); il resto è 'template'.
 * I token di controllo sono sempre 'control'; un token a cavallo di due
 * intervalli è 'mixed'.
 */
export function labelOrigins(pieces, spans) {
  let offset = 0;
  return pieces.map((p) => {
    const start = offset;
    // I frammenti non finali hanno testo vuoto: si classificano con l'intervallo del gruppo.
    const spanLength = p.groupSize > 1 ? p.groupText.length : p.text.length;
    const end = start + spanLength;
    offset += p.text.length;
    let origin = 'template';
    if (p.special) origin = 'control';
    else if (end > start) {
      for (const s of spans) {
        if (start >= s.start && end <= s.end) { origin = s.origin; break; }
        if (start < s.end && end > s.start) { origin = 'mixed'; break; }
      }
    }
    return { ...p, start, end, origin };
  });
}
