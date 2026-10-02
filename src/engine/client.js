/**
 * Client lato pagina per il Web Worker di inferenza.
 * Ogni richiesta restituisce una Promise; il progresso del download arriva
 * come evento separato.
 */

export class EngineClient {
  constructor({ onProgress = () => {}, onCrash = () => {} } = {}) {
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.nextId = 1;
    /** Registro dei tipi di richiesta inviati (usato dai test: il replay non ne aggiunge). */
    this.calls = [];
    this.onProgress = onProgress;
    this.worker.onmessage = (event) => {
      const msg = event.data;
      if (msg?.type === 'progress') {
        this.onProgress(msg.payload);
        return;
      }
      const entry = this.pending.get(msg?.id);
      if (!entry) return;
      this.pending.delete(msg.id);
      if (msg.ok) entry.resolve(msg.result);
      else {
        const err = new Error(msg.error?.message ?? 'Errore sconosciuto');
        err.code = msg.error?.code;
        entry.reject(err);
      }
    };
    this.worker.onerror = (event) => {
      const err = new Error(event?.message || 'Il processo del modello si è interrotto (memoria esaurita?).');
      err.code = 'worker-crash';
      for (const entry of this.pending.values()) entry.reject(err);
      this.pending.clear();
      onCrash(err);
    };
  }

  call(type, payload) {
    const id = this.nextId++;
    this.calls.push(type);
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type, payload });
    });
  }

  terminate() {
    this.worker.terminate();
    const err = new Error('Processo del modello chiuso.');
    err.code = 'terminated';
    for (const entry of this.pending.values()) entry.reject(err);
    this.pending.clear();
  }
}
