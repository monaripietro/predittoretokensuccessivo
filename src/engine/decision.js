/**
 * Collega la scelta del token al ciclo di generazione di Transformers.js.
 *
 * Transformers.js 4.3 non restituisce i punteggi per passo (`output_scores`
 * è dichiarato nella configurazione ma `generate()` non lo implementa) e non
 * applica `top_p`. Per questo la scelta del token è fatta qui, in modo
 * esplicito e verificabile:
 *
 *  1. `generate()` esegue un solo passaggio del modello (max_new_tokens = 1)
 *     riusando la KV cache del passo precedente;
 *  2. questo LogitsProcessor (API pubblica, ultimo della lista) riceve i
 *     logit dell'ultima posizione, calcola il riepilogo della distribuzione
 *     e sceglie il token con la regola dichiarata;
 *  3. tutti gli altri logit vengono portati a -∞, così il campionatore
 *     greedy della libreria emette esattamente il token scelto;
 *  4. il chiamante verifica che il token emesso coincida con quello scelto
 *     (stesso passo, stessa invocazione del modello).
 */

import { decideNextToken } from '../core/distribution.js';

/**
 * @param {typeof import('@huggingface/transformers').LogitsProcessor} LogitsProcessor
 */
export function makeDecisionProcessorClass(LogitsProcessor) {
  return class DecisionProcessor extends LogitsProcessor {
    constructor({ policy, random, displayK, keepRawCopy = false }) {
      super();
      this.options = { policy, random, displayK };
      this.keepRawCopy = keepRawCopy;
      this.calls = 0;
      this.decision = null;
      this.rawCopy = null;
      this.decideMs = 0;
    }

    _call(inputIds, logits) {
      this.calls += 1;
      if (this.calls > 1) {
        throw new Error('Il processore di scelta è stato chiamato più di una volta nello stesso passo.');
      }
      const batch = logits.dims.length === 2 ? logits.dims[0] : 1;
      if (batch !== 1) throw new Error(`Batch inatteso: ${batch}`);
      const data = logits.data;
      if (this.keepRawCopy) this.rawCopy = Float32Array.from(data);
      const t0 = performance.now();
      this.decision = decideNextToken(data, this.options);
      this.decideMs = performance.now() - t0;
      const chosen = this.decision.selected.id;
      data.fill(-Infinity);
      data[chosen] = 0;
      return logits;
    }
  };
}

/** Parametri che neutralizzano gli altri processori della configurazione del modello. */
export const NEUTRAL_GENERATION_OPTIONS = Object.freeze({
  do_sample: false,
  num_beams: 1,
  temperature: 1.0,
  top_k: 0,
  top_p: 1.0,
  repetition_penalty: 1.0,
  no_repeat_ngram_size: 0,
  min_length: 0,
  min_new_tokens: null,
  bad_words_ids: null,
  suppress_tokens: null,
  begin_suppress_tokens: null,
  forced_bos_token_id: null,
  forced_eos_token_id: null,
  guidance_scale: null,
});
