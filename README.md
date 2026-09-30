# Next Token Lab

Demo didattica che mostra nel browser il ciclo autoregressivo di un piccolo LLM locale:

```
Prompt → tokenizzazione → token successivo → classifica probabilità → aggiunta token → nuovo calcolo
```

**URL pubblico canonico:** <https://predittoretokensuccessivo.monaripietro.it>

Il modello gira interamente nel browser via [Transformers.js](https://huggingface.co/docs/transformers.js): nessun backend, nessuna API cloud. Con WebGPU disponibile usa **Qwen2.5-0.5B-Instruct** (q4f16); senza WebGPU usa direttamente il più leggero **SmolLM2-360M-Instruct** (q4), perché Qwen in WASM eccede la memoria di molti browser e causa il crash del tab (non intercettabile via JS). In ambienti senza GPU/rete è disponibile un runtime mock deterministico (`?mock`).

## Installazione

```bash
npm install
```

## Sviluppo locale

```bash
npm run dev        # server di sviluppo Vite su http://localhost:5173
```

## Server statico di produzione

```bash
npm run build      # output in dist/ con percorsi relativi (base './')
npm run preview    # serve dist/ su http://localhost:4173
```

## Test

```bash
npm run test:unit            # unit test (vitest) — sempre su mock deterministico
npm run test:e2e             # e2e Playwright (Chromium) su server locale con ?mock
RUN_REAL_TESTS=1 npx vitest run tests/unit/realRuntime.test.js   # opzionale: test con modello reale (scarica ~35MB)
```

I test CI usano il mock: nessun download del modello a ogni push.

## Scelta del backend

- **WebGPU**: usato automaticamente se `navigator.gpu` e un adapter sono disponibili (`device: 'webgpu'`).
- **WASM**: fallback per compatibilità.
- L'indicatore in alto mostra il backend effettivo. Con `?mock` si forza il mock deterministico.

## Trasparenza runtime reale/mock

Il mock **non è mai un fallback automatico silenzioso**. Quando è attivo è sempre riconoscibile:

- badge giallo `MOCK` nell'intestazione;
- avviso "I risultati sono simulati e servono solo per il test dell'interfaccia";
- pannello *Informazioni tecniche* mostra il runtime in uso.

Il mock si attiva solo in due modi:

1. parametro `?mock` nell'URL (usato da tutti i test CI/e2e);
2. pulsante "Avvia demo mock", che appare soltanto se il modello reale non riesce a caricarsi.

Alla prima visita la pagina mostra un avviso di download del modello (~35 MB) finché il runtime non è pronto; non vengono simulate percentuali di avanzamento.

I token sono visualizzati in forma leggibile: gli spazi sono mostrati come `␠`, gli a-capo come `⏎`, le tabulazioni come `⇥`, perché i token BPE spesso iniziano con uno spazio e contengono frammenti di parole.

Il pannello *KV cache* separa tre livelli: **contesto logico** (tutti i token, prompt + generati), **calcolo corrente** (token effettivamente inviati al modello in questo step: 1 con cache, tutti in naive) e **stato cache** (attiva / non usata). Nessuna metrica non misurata dal runtime viene mostrata.

## Altri strumenti didattici

- **Barre probabilità**: ogni riga della classifica ha una barra proporzionale al candidato più probabile.
- **Cronologia dei passi**: tabella richiudibile con passo, token, ID, probabilità, rank, contesto e delta di probabilità rispetto allo step precedente, esportabile in JSON con "Esporta cronologia".
- **Preset di prompt**: esempi cliccabili sotto la textarea per partire subito.
- **Animazione**: il token appena aggiunto alla sequenza compare con una breve animazione `pop-in`.
- **Input del prossimo calcolo**: pannello dedicato che mostra la sequenza esatta (prompt + token generati) che il modello userà per il calcolo successivo, con l'ultimo token evidenziato e il pulsante "Calcola il token dopo «...»" al suo interno. Rende esplicito il ciclo: contesto corrente → calcolo → classifica → token scelto → contesto esteso. La cronologia conserva l'`inputIds` di ogni passo, verificabile come `step N+1 = step N + token scelto`.

## Browser consigliati

- Chrome/Edge 113+ (WebGPU)
- Firefox 141+ / Safari 26+ (WebGPU in evoluzione)
- Qualsiasi browser recente per il fallback WASM (più lento)

## Deploy GitHub Pages

Il deploy è automatico tramite workflow (`.github/workflows/deploy.yml`) su push in `main`:

1. `npm ci && npm run build`
2. copia `CNAME` in `dist/`
3. pubblica `dist/` con GitHub Pages (source: GitHub Actions)

### Dominio personalizzato

Il file `CNAME` (root del repo) contiene `predittoretokensuccessivo.monaripietro.it` e viene copiato nella directory pubblicata. Il proprietario del dominio deve configurare presso il proprio provider DNS:

```
Nome/Host: predittoretokensuccessivo
Tipo:      CNAME
Destinazione: <utente-o-organizzazione>.github.io
```

Poi in `Settings → Pages`: inserire il custom domain e attivare `Enforce HTTPS` quando il certificato è disponibile. La propagazione DNS e il provisioning del certificato richiedono tempo.

## Modello

| | |
|---|---|
| Primario (WebGPU) | `onnx-community/Qwen2.5-0.5B-Instruct` — q4f16 (~400 MB) |
| Leggero (WASM desktop) | `onnx-community/SmolLM2-360M-Instruct-ONNX` — q4 (~250 MB) |
| Tiny (WASM mobile) | `onnx-community/SmolLM2-135M-Instruct-ONNX` — q4 (~90 MB) |
| Licenza | Apache-2.0 |
| Origine | Hugging Face Hub, CORS abilitato per uso browser |
| Vocabolario | Qwen2 151.936 token / SmolLM2 49.152 token (BPE) |
| Contesto | Qwen2 32.768 token / SmolLM2 8.192 token |

La scelta è automatica: WebGPU → Qwen2.5-0.5B; WASM desktop → SmolLM2-360M; WASM su smartphone/tablet (pointer coarse, touch, schermo piccolo) → SmolLM2-135M. Il modello è scaricato dal browser al primo avvio e messo in cache (Cache Storage). Nessun file di modello è nel repository. I valori mostrati nel pannello "Dettagli runtime" sono letti dal profilo del modello effettivamente caricato, non hardcoded.

## Limiti dichiarati

- **KV cache**: il runtime reale usa la `past_key_values` dell'export ONNX "merged" dei modelli sopra; la modalità naive ricalcola l'intera sequenza. L'equivalenza dei risultati tra le due modalità è verificata dai test (`tests/unit/realRuntime.test.js`). La cache è quella reale del runtime, non una metrica simulata.
- **Qualità linguistica**: i modelli da 0.5B/360M parametri restano piccoli; l'obiettivo è osservare token, ID e distribuzione, non produrre testo di qualità.
- **Memoria WASM**: Qwen2.5-0.5B in WASM può superare la memoria disponibile e far crashare il tab senza eccezione JS; per questo il backend senza WebGPU seleziona un modello più piccolo prima del caricamento.
- **Mobile**: su smartphone senza WebGPU (es. Chrome Android) anche SmolLM2-360M può superare la memoria disponibile; l'app seleziona direttamente SmolLM2-135M (~90 MB, ~270 MB RAM). Il primo download su rete mobile richiede tempo; in seguito il modello resta in cache del browser.
- **Rilevamento mobile**: `isMobileLike()` combina `pointer: coarse`, touch points e schermo piccolo. È un'euristica: un tablet grande con desktop mode caricherà il 360M, un dispositivo desktop con touch screen no.
- **Prompt lunghi**: nessun limite artificiale oltre il contesto del modello caricato; prestazioni WebGPU/WASM variano per browser e dispositivo.
- **Parametri**: temperatura, top-k, top-p, min-p, repeat penalty e seed sono applicati ai logits nel codice dell'app, quindi funzionanti anche col runtime reale.

## Checklist di rilascio (dominio pubblico)

- [ ] `https://predittoretokensuccessivo.monaripietro.it` risponde
- [ ] certificato HTTPS valido e redirect HTTP → HTTPS (Enforce HTTPS)
- [ ] `CNAME` presente nella sorgente pubblicata (`dist/CNAME`)
- [ ] DNS: `predittoretokensuccessivo` → `<account>.github.io`
- [ ] CSS, JS e modello caricati senza mixed content (tutti gli URL relativi o HTTPS)
- [ ] pulsante `Calcola token successivo` funzionante: token, ID, ranking, percentuali, contesto
- [ ] tre click consecutivi aggiungono esattamente un token ciascuno
- [ ] cambio parametro (es. temperatura) e naive/cache come previsto
- [ ] console senza errori

## Smoke test manuale su GitHub Pages

1. Aprire il sito in HTTPS e verificare `Modello:` e `Backend:` pronti.
2. Inserire un prompt breve (es. `Il cielo è`).
3. Premere `Calcola token successivo` almeno tre volte.
4. Verificare a ogni passo: token e ID visibili (hover/focus sui chip), ranking con percentuali, token scelto evidenziato, contatore di contesto incrementato.
5. Cambiare un parametro nel pannello Opzioni e ripetere un click.
6. Passare da KV cache a ricalcolo completo e ripetere un click.
7. Controllare la console del browser: nessun errore di mixed content o runtime.

Non verificare una frase esatta prodotta dal modello: verificare le invarianti (token validi, probabilità finite in [0,1], contesto coerente, UI recuperabile da errori).

## Struttura

```
index.html            UI minimale (un pulsante, sequenza chip, ranking)
style.css             stili (prompt neutro, generati in accento)
app.js                stato centrale + flusso del passo
src/mockTokenizer.js  tokenizer didattico deterministico
src/mockRuntime.js    runtime mock (prefill/cache verificabili)
src/realRuntime.js    runtime Transformers.js (WebGPU/WASM, past_key_values)
src/sampling.js       softmax, filtri, greedy/sampling deterministico
tests/unit/           vitest
tests/e2e/            Playwright
```

## Riferimenti

- [Transformers.js](https://huggingface.co/docs/transformers.js) — modelli ONNX nel browser
- [GitHub Pages — custom domains](https://docs.github.com/pages/configuring-a-github-pages-site/managing-a-custom-domain-for-your-github-pages-site)
- [Qwen2.5-0.5B-Instruct ONNX](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct) (Apache-2.0)
- [SmolLM2-360M-Instruct ONNX](https://huggingface.co/onnx-community/SmolLM2-360M-Instruct-ONNX) (Apache-2.0)
