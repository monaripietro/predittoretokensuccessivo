# Next Token Lab

Demo didattica che mostra nel browser il ciclo autoregressivo di un piccolo LLM locale:

```
Prompt → tokenizzazione → token successivo → classifica probabilità → aggiunta token → nuovo calcolo
```

**URL pubblico canonico:** <https://predittoretokensuccessivo.monaripietro.it>

Il modello (distilgpt2 quantizzato, ONNX) gira interamente nel browser via [Transformers.js](https://huggingface.co/docs/transformers.js): nessun backend, nessuna API cloud. WebGPU quando disponibile, fallback WASM. In ambienti senza GPU/rete è disponibile un runtime mock deterministico (`?mock`).

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
| Modello | `Xenova/distilgpt2` (distilgpt2) |
| Formato | ONNX quantizzato q8 (~35 MB) |
| Licenza | MIT |
| Origine | Hugging Face Hub, CORS abilitato per uso browser |
| Vocabolario | 50257 token (BPE GPT-2) |

Il modello viene scaricato dal browser al primo avvio e messo in cache (Cache Storage). Nessun file di modello è nel repository.

## Limiti dichiarati

- **KV cache**: il runtime reale usa la `past_key_values` dell'export ONNX "merged" di distilgpt2; la modalità naive ricalcola l'intera sequenza. L'equivalenza dei risultati tra le due modalità è verificata dai test (`tests/unit/realRuntime.test.js`). La cache è quella reale del runtime, non una metrica simulata.
- **Qualità linguistica**: distilgpt2 è volutamente minimale; l'obiettivo è osservare token, ID e distribuzione, non produrre testo di qualità.
- **Prompt lunghi**: nessun limite artificiale oltre il contesto del modello (1024 token); prestazioni WebGPU/WASM variano per browser e dispositivo.
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
- [distilgpt2](https://huggingface.co/distilbert/distilgpt2) (MIT)
