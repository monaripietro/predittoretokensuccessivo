# Come risponde una chatbot

Demo didattica che gira interamente nel browser: un vero modello linguistico (LLM) funziona in locale, con WebGPU, e la pagina mostra **un token alla volta** come nasce la risposta di una chatbot, con le probabilità reali calcolate dal modello. Serve a lezioni di alfabetizzazione sull'IA: nessun backend, nessuna API cloud.

**URL pubblico:** <https://predittoretokensuccessivo.monaripietro.it>

## Come si usa in aula

1. Aprire la pagina in Chrome (o Edge) da computer, con WebGPU disponibile. Prima della lezione premere il pulsante di caricamento: il modello (≈ 3,1 GB per Gemma 4 E2B) si scarica una volta e resta nella cache del browser.
2. Scrivere una domanda (al massimo 240 caratteri) e premere **Avvia**: l'app prepara l'input e mostra i token, poi si ferma. Premere **Calcola il primo token**.
3. Ogni clic su **Prossimo token** mostra un passo completo del ciclo. In alternativa **Continua da solo**, con pausa e stop.
4. Il presentatore può rivedere la risposta con il replay (senza nuova inferenza) ed esportare la traccia in JSON.
5. In modalità passo per passo il presentatore può cliccare un candidato diverso dal primo per forzarlo e vedere come il modello prosegue. Il passo è etichettato come "scelta del presentatore"; le probabilità mostrate restano quelle del modello.

Il ciclo mostrato a ogni passo:

```
prompt
 → l'app aggiunge un'istruzione di sistema e i token di controllo del formato chat
 → tokenizzatore (testo → pezzi di token → ID)
 → modello (un punteggio, "logit", per ogni token del vocabolario)
 → l'app converte i punteggi in probabilità (softmax) e sceglie un token con una regola dichiarata
 → ID decodificato in testo → la risposta cresce
 → il token scelto è aggiunto all'input e il ciclo si ripete
```

Il modello è disegnato come una "stanza" (analogia della stanza cinese). È un'analogia filosofica, non il meccanismo interno del modello: la pagina non mostra mai attivazioni interne.

## Principi di onestà

- Le probabilità sono quelle reali del modello, non simulate. I test verificano che i logit usati dall'app siano identici a quelli di un passaggio diretto del modello.
- La regola di scelta è sempre dichiarata (greedy o campionamento, con seme registrato).
- Nessun ripiego silenzioso: senza WebGPU non si passa a CPU/WASM e non si usa una simulazione.
- La modalità `?mock` (modello finto per i test dell'interfaccia) è sempre etichettata "Simulazione" e non è mai un ripiego automatico.
- Le probabilità sono per **token**, mai aggregate in probabilità di parole.
- Token di fine sequenza e di controllo sono mostrati come tali (es. "fine · `<turn|>`"), non come parole; i frammenti di caratteri multi-byte sono etichettati.
- Prima di scaricare si mostra la dimensione; il download parte solo dopo un clic esplicito.
- Un modello piccolo può dire cose false con tono sicuro: è un punto didattico, non un difetto nascosto.

## Architettura

Sito statico Vite, senza backend. Il modello gira in un Web Worker (Transformers.js + WebGPU), così la pagina resta reattiva.

| Parte | Contenuto |
|---|---|
| `index.html`, `src/main.js` | interfaccia |
| `src/core/` | `distribution.js` (softmax, top-k, campionamento), `tokens.js` (pezzi di token esatti, frammenti di byte, token di controllo), `controller.js` (macchina a stati: passo, auto, pausa, stop, replay), `models.js` (registro modelli con revisioni fissate e dimensioni esatte dei file), `cache.js` (controllo/cancellazione di Cache Storage senza rete), `webgpu.js` (verifica adapter e limiti) |
| `src/engine/` | `worker.js` (Web Worker), `engine.js` (input chat e un passo del modello per chiamata), `decision.js` (LogitsProcessor: registra i logit reali, sceglie il token e forza il campionatore greedy della libreria a emettere esattamente quello), `verify.js` (diagnostica), `client.js`, `mockClient.js` e `fakeTf.js` (modello simulato solo per i test UI) |
| `src/ui/` | grafico, chip dei token, etichette, diagramma dell'anatomia |
| `diagnostics.html`, `src/diagnostics.js` | pagina tecnica di misura |

## Modelli

| | Gemma 4 E2B (predefinito) | Qwen3 0,6B (leggero) |
|---|---|---|
| Repository | `onnx-community/gemma-4-E2B-it-ONNX` | `onnx-community/Qwen3-0.6B-ONNX` |
| Revisione fissata | `9f4bef82ea6e296bc69f8a2f5939f73af81b07a6` | `da1453100cf3ff33ef56d17983fc7a8648706db6` |
| Quantizzazione | q4f16 (q4 se la GPU non ha `shader-f16`) | q4f16 (q4 se manca `shader-f16`) |
| Download | 3.130.533.481 byte ≈ 3,1 GB (q4: ≈ 3,6 GB) | ≈ 579 MB (q4: ≈ 938 MB) |
| Vocabolario | 262.144 | 151.936 ID (alcuni senza testo) |
| Italiano nelle prove | 6 domande su 6 corrette | fatti sbagliati |
| Licenza | Apache-2.0 | Apache-2.0 |

Dettagli comuni:

- Gemma è caricato solo come testo (`AutoModelForCausalLM`): si scaricano `embed_tokens` e `decoder_model_merged`, non gli encoder di visione/audio.
- Il "pensiero" di Qwen3 è disattivato (`enable_thinking: false`).
- Istruzione di sistema aggiunta dall'app: "Rispondi in italiano, in modo breve e semplice: al massimo due frasi."
- La pagina suggerisce il modello leggero quando `navigator.deviceMemory` indica meno di 16 GB.

### Perché questa scelta

Risultati di una prova preliminare sul computer di test (vedi sotto):

- **Gemma 4 E2B** ha superato tutti i requisiti rigidi (tokenizzazione esatta, logit grezzi per passo, allineamento, parità con la generazione nativa, nessun blocco della UI, stop e rilascio) e ha risposto correttamente a 6 prompt di classe su 6, in italiano ("La capitale d'Italia è Roma.", "I ragni hanno otto zampe.", la diffusione di Rayleigh per il cielo blu).
- **Qwen3 0,6B** ha superato i controlli tecnici ma ha prodotto fatti sbagliati in italiano (es. "Un ragno ha al massimo due zampe"; il cielo è blu "perché la luna e il Sole brillano…"). Non è credibile come predefinito: resta come opzione leggera con etichetta onesta.
- **Qwen3 1,7B** (`onnx-community/Qwen3-1.7B-ONNX`, q4f16, singolo file .onnx da 1,43 GB) non si è caricato nel browser, in due prove: "Can't create a session … std::bad_alloc". Scartato.

## Requisiti del browser

- **WebGPU obbligatorio.** Nessun ripiego su CPU/WASM: se WebGPU, l'adapter o i limiti mancano, la pagina mostra un messaggio chiaro in italiano e non scarica nulla. Gli adapter software sono rifiutati.
- Testato solo su **Chrome da computer** (Windows, GPU integrata AMD). Edge dovrebbe comportarsi come Chrome (stesso motore) ma non è stato provato. Firefox, Safari e dispositivi mobili non sono stati provati e non sono target.
- Un dispositivo con poca memoria **non è ancora stato provato**: vedere [docs/checklist-manuale.md](docs/checklist-manuale.md).

## Download, memoria e cache

- Il download parte solo dopo un clic esplicito, con la dimensione mostrata prima. Il progresso è calcolato dalla lista di file fissata dall'app.
- I file sono salvati in Cache Storage (`transformers-cache`). La pagina chiede lo storage persistente e avvisa se la quota sembra troppo piccola. Da **Avanzate** si possono cancellare.
- La prima preparazione comprende un passaggio di riscaldamento (warm-up) dichiarato nella pagina: serve a compilare gli shader GPU prima del primo prompt.
- Memoria (contatori di Windows, approssimativi, Gemma q4f16): circa +1,9 GB di memoria GPU dedicata e +2,6 GB condivisa; il picco privato di Chrome in caricamento è stato 5,4-6,6 GB.
- Pianificare il primo caricamento prima della lezione.

## Privacy

I prompt e le risposte non lasciano mai il browser. Il solo traffico di rete è il sito stesso e i file del modello da huggingface.co (e dalla sua CDN), scaricati dopo un clic esplicito. Il runtime ONNX (WASM e loader) è servito dal sito stesso, senza CDN. Il test con il modello reale verifica che durante la generazione non esca alcuna richiesta e che il prompt non compaia in nessuna richiesta.

## Che cosa significano le percentuali

- Le barre sono la **probabilità del modello**: softmax dei logit grezzi (temperatura 1) sull'intero vocabolario, prima di qualsiasi regola di scelta. Per Gemma i logit includono già il soft-capping finale, che fa parte del grafo del modello.
- Il grafico mostra i primi 8 candidati più "altri N token" con la massa restante: le barre visibili non sommano al 100%. I valori sono arrotondati ("<0,1%" per i molto piccoli).
- **Regola predefinita: greedy.** Si sceglie sempre il token più probabile (a parità, l'ID più basso): la stessa domanda dà la stessa risposta.
- **Campionamento (opzione in Avanzate):** temperatura → al massimo 64 candidati → top-p → rinormalizzazione. Il generatore pseudocasuale è mulberry32 con seme registrato nella traccia. Il grafico mostra allora anche la probabilità di estrazione (seconda serie con legenda).

## Prestazioni misurate

Misure su **un solo computer di prova**, non valgono in generale: Windows 11 Pro, AMD Ryzen AI 7 PRO 350 con Radeon 860M integrata (RDNA 3, WebGPU con `shader-f16`, `maxBufferSize` 2 GiB), 32 GB di RAM, Google Chrome 154 (headless e con finestra), rete ≈ 50 MB/s.

| Misura | Gemma 4 E2B q4f16 | Qwen3 0,6B q4f16 |
|---|---|---|
| Prima visita, cache vuota | ≈ 95 s in tutto (download e sessione ≈ 85 s, warm-up ≈ 6,5 s) | ≈ 20 s (579 MB) |
| Da cache del browser | ≈ 25 s (tokenizzatore 1,1 s, modello 14,6 s, warm-up 8,9 s) | ≈ 5 s |
| Primo token | ≈ 0,85-1,45 s (senza warm-up il primo prompt pagava 5,7-13 s di compilazione shader) | 0,23-0,36 s |
| Per token | ≈ 115-245 ms (≈ 4-8,5 token/s; pagina diagnostica 6-6,9 token/s) | ≈ 40-50 ms (≈ 20-24 token/s) |
| Memoria GPU aggiuntiva | ≈ +1,9 GB dedicata, ≈ +2,6 GB condivisa | ≈ +1,3 GB |
| Picco memoria privata Chrome | 5,4-6,6 GB (in caricamento) | ≈ 4,3 GB |
| Task più lungo sul thread principale | 101 ms | nessun task lungo |

Su GPU integrata Gemma genera pochi token al secondo: adatto a una lezione passo per passo.

## Controlli di correttezza (tutti superati)

1. **Punteggi grezzi:** i logit ricevuti dal processore decisionale sono identici bit a bit a un passaggio diretto del modello sullo stesso input (differenza assoluta massima 0) per Gemma e Qwen nel browser e per SmolLM2-135M in Node.
2. **Parità greedy:** 22-24 token generati passo per passo sono uguali a `generate()` nativo della libreria (Gemma e Qwen nel browser; SmolLM2 in Node).
3. **Cache vs. ricalcolo completo:** passi consecutivi con cache e un passaggio completo senza cache sullo stesso contesto danno probabilità uguali a 4 decimali (SmolLM2 q4, CPU). Con pesi int8 quantizzati dinamicamente i due percorsi differiscono numericamente: per questo i test usano q4.
4. **Allineamento:** a ogni passo l'ID emesso è confrontato con l'ID scelto nella stessa chiamata del processore; se differiscono il passo fallisce con un errore di "allineamento".

## Che cosa abbiamo scoperto su Transformers.js 4.3.0

- `output_scores` esiste in `GenerationConfig` ma `generate()` non restituisce i punteggi (TODO nel sorgente).
- `top_p` non è implementato; `top_k` è applicato solo dentro il campionatore multinomiale, che usa un `Math.random` senza seme. Per questo l'app prende la decisione da sé in un LogitsProcessor (API pubblica) e forza il token emesso.
- Il progresso di download aggregato della libreria conta anche i file degli encoder di visione/audio che il percorso solo testo non scarica (invia richieste Range da 1 byte per conoscerne la dimensione): l'app calcola il progresso dalla propria lista di file fissata.
- `ModelRegistry.is_cached` fa richieste senza revisione fissata: l'app controlla direttamente le chiavi di Cache Storage.

## Limiti noti

- Prompt fino a 240 caratteri, input fino a 160 token, risposta da 8 a 128 token (predefinito 48). Gli strati locali di Gemma usano una finestra scorrevole di 512 token: l'app resta ben al di sotto.
- Solo Chrome desktop provato; nessuna prova su dispositivi con poca memoria, né su Firefox, Safari, mobile.
- La velocità di Gemma su GPU integrata è di pochi token al secondo.
- Il primo download è grande (≈ 3,1 GB; ≈ 3,6 GB con q4).
- I modelli piccoli possono affermare fatti falsi in modo fluente.

## Installazione e sviluppo

Richiede Node 22 o successivo (la CI usa 22; provato in locale con 24.17).

```bash
npm ci
npm run dev        # server di sviluppo Vite (http://localhost:5173)
npm run build      # output in dist/
npm run preview    # serve dist/ (http://localhost:4173)
```

### Test

| Comando | Cosa fa |
|---|---|
| `npm run test:unit` | Vitest: matematica delle distribuzioni, pezzi di token con tokenizzatore finto a byte, allineamento del motore con libreria finta (incluso un modello volutamente rotto che deve provocare l'errore di allineamento), un test che fallisce se le probabilità non seguono i logit, controller (passo/auto/pausa/stop/replay senza chiamate al motore), etichette greedy e campionamento |
| `npm run test:real` | `vitest --mode real`: tokenizzatori reali di Gemma 4 e Qwen3 su testo italiano, emoji e token di controllo; `generate()` reale con SmolLM2-135M q4 su CPU (allineamento, punteggi grezzi, parità greedy). Scarica ≈ 220 MB |
| `npm run build && npm run test:e2e` | Playwright (Chromium) con modello simulato: flusso passo per passo, DOM uguale alla traccia, stop, replay senza chiamate al motore, tastiera, etichette di campionamento, trasparenza dell'input, stato senza WebGPU senza download, dimensioni mostrate senza richieste di rete, invarianti di deploy (inclusa un'unica copia del runtime ONNX servita dal sito) |
| `npm run test:e2e:real` | Chrome di sistema con WebGPU e modello reale; profilo persistente in `.cache/e2e-chrome-profile`; `REAL_MODEL=qwen3-0.6b` per il modello leggero. Verifica: caricamento solo dopo clic, nessuna rete durante la generazione, prompt mai inviato, allineamento a ogni passo, DOM uguale alla traccia, task lunghi sotto 250 ms, replay senza chiamate al motore, stop, scaricamento del modello, uguaglianza dei punteggi grezzi e parità nella diagnostica |

Nella CI (`.github/workflows/test.yml`) girano build, unit test, i test dei tokenizzatori reali (con rete, `RUN_TOKENIZER_TESTS=1`) ed e2e con il modello simulato. I test con il modello reale non girano in CI. Il repository non ha script di lint o di controllo dei tipi.

Per la verifica su dispositivi reali: [docs/checklist-manuale.md](docs/checklist-manuale.md).

### Pagina di diagnostica

`diagnostics.html` è una pagina tecnica di misura (tempi, token al secondo, controlli sui logit). Esempio:

```
diagnostics.html?auto=1&model=gemma-4-e2b&n=32&quality=1
```

## Deploy su GitHub Pages

Il deploy è automatico con `.github/workflows/deploy.yml` a ogni push su `main`:

1. `npm ci && npm run build`
2. copia di `CNAME` in `dist/`
3. pubblicazione di `dist/` con GitHub Pages (source: GitHub Actions)

### Dominio personalizzato

Il file `CNAME` (radice del repository) contiene `predittoretokensuccessivo.monaripietro.it` e viene copiato nella directory pubblicata. Il proprietario del dominio deve configurare presso il proprio provider DNS:

```
Nome/Host: predittoretokensuccessivo
Tipo:      CNAME
Destinazione: <utente-o-organizzazione>.github.io
```

Poi in `Settings → Pages`: inserire il dominio personalizzato e attivare `Enforce HTTPS` quando il certificato è disponibile. La propagazione DNS e l'emissione del certificato richiedono tempo.

## Struttura del progetto

```
index.html              pagina principale
diagnostics.html        pagina tecnica di misura
style.css               stili
src/main.js             interfaccia
src/diagnostics.js      logica della pagina di diagnostica
src/core/               distribuzione, token, controller, modelli, cache, WebGPU
src/engine/             worker, motore, decisione, verifica, client, modello simulato
src/ui/                 grafico, chip dei token, etichette, diagramma
tests/unit/             Vitest
tests/e2e/              Playwright (modello simulato)
tests/e2e-real/         Playwright (Chrome di sistema, modello reale)
docs/                   checklist di verifica manuale
.github/workflows/      test.yml, deploy.yml
```

## Crediti e licenze

- I modelli sono distribuiti con licenza Apache-2.0.
- **Gemma 4** è di Google DeepMind; **Qwen3** è del team Qwen di Alibaba Cloud.
- Le conversioni ONNX sono di [onnx-community](https://huggingface.co/onnx-community) su Hugging Face.
- [Transformers.js](https://huggingface.co/docs/transformers.js) è di Hugging Face; il runtime è ONNX Runtime Web.
- [GitHub Pages: domini personalizzati](https://docs.github.com/pages/configuring-a-github-pages-site/managing-a-custom-domain-for-your-github-pages-site)
