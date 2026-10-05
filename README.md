# Come risponde una chatbot

Dimostrazione didattica nel browser: un modello linguistico vero (WebGPU) mostra token per token le probabilità reali da cui nasce una risposta.

```text
domanda → tokenizzazione → punteggi del vocabolario → classifica dei candidati
→ scelta del token → testo che cresce → nuovo calcolo
```

**URL pubblico canonico:** <https://predittoretokensuccessivo.monaripietro.it>

L'app mostra tutto il percorso: la domanda, l'istruzione di sistema che l'app aggiunge, i token di controllo del modello di chat, i punteggi trasformati in probabilità, la classifica dei candidati con percentuali, il token scelto e la risposta che cresce un pezzo alla volta. Ogni numero mostrato viene calcolato in quel momento dal modello: non ci sono esempi preparati.

## Che cosa fa

- **Modello reale nel browser**: Gemma 4 E2B (consigliato) o Qwen3 0.6B (leggero), eseguiti con WebGPU via [Transformers.js](https://huggingface.co/docs/transformers.js). Nessun backend, nessuna API cloud: i calcoli avvengono sul dispositivo.
- **Passo per passo**: ogni clic su «Prossimo token» esegue esattamente un calcolo del modello; «Continua da solo» procede senza bloccare la pagina (l'inferenza gira in un Web Worker).
- **Classifica dei candidati**: grafico con gli 8 candidati più probabili, le loro percentuali e la massa residua («altri N token»). Le barre visibili non sommano al 100% perché il vocabolario è molto più grande.
- **Sostituzione del token**: in pausa si può cliccare un candidato diverso per forzare la scelta del passo; il modello prosegue da lì. Le probabilità mostrate restano quelle originali del modello; la scelta è etichettata come «scelta del presentatore».
- **Regia onesta**: pausa, stop e replay. Lo stop durante un calcolo scarta quel risultato; il replay riproduce la traccia registrata senza chiamare il modello ed è segnalato come tale. La registrazione completa si esporta in JSON (seed e parametri inclusi, per verificare ogni estrazione).
- **Regola di scelta**: predefinita «sempre il più probabile» (greedy); nelle opzioni avanzate si passa all'estrazione casuale pesata con temperatura, top-p e seme.
- **Impostazioni del modello**: nelle opzioni avanzate si può personalizzare l'istruzione di sistema; Gemma 4 e Qwen3 espongono anche l'opzione di ragionamento del proprio template (`enable_thinking`).
- **Modalità simulata** (`?mock`): stesso codice di scelta e stesso flusso con un modello finto, sempre dichiarata in pagina come simulazione. Usata dai test automatici.

## Requisiti

- **WebGPU** (Chrome o Edge recenti). Il runtime ONNX non è caricato da una CDN esterna ma servito dallo stesso sito.
- Senza WebGPU la pagina lo spiega chiaramente e non scarica nulla: nessun ripiego silenzioso su CPU/WASM né simulazioni non dichiarate.
- Requisiti pratici: 16 GB di RAM consigliati per Gemma 4 E2B (download ~3,1 GB); con meno memoria dichiarata la pagina suggerisce il modello leggero Qwen3 0.6B (~579 MB).

## Installazione e sviluppo

```bash
npm install
npm run dev        # server di sviluppo Vite su http://localhost:5173
```

## Produzione

```bash
npm run build      # output in dist/ (index.html + diagnostics.html)
npm run preview    # serve dist/ su http://localhost:4173
```

## Test

```bash
npm run test:unit        # unit test (vitest), sempre offline e deterministici
npm run test:e2e         # Playwright su server locale con ?mock
npm run test:real        # opzionale: test con libreria e tokenizer reali (rete)
npm run test:e2e:real    # opzionale: smoke test con modello reale su WebGPU
npm test                 # unit + build + e2e
```

La CI esegue unit test, test con tokenizer reali, build e e2e su Chromium con la modalità simulata. I test con modello reale sono manuali (`tests/e2e-real/`).

## Architettura

```text
index.html            pagina principale (anatomia in 6 riquadri + narrazione)
diagnostics.html      pagina tecnica: download, tempi, verifica dei punteggi
src/main.js           preparazione modello, regia, presentazione
src/core/
  models.js           modelli, revisioni fisse, dimensioni file verificate
  distribution.js     softmax stabile, top-k, top-p, estrazione riproducibile
  controller.js       macchina a stati: avvio, pausa, un token, stop, replay
  tokens.js           pezzi di testo dei token, frammenti UTF-8, origini
  cache.js            cache del browser per i file del modello
  webgpu.js           rilevamento WebGPU prima di qualunque download
src/engine/
  worker.js           Web Worker di inferenza (UI sempre fluida)
  engine.js           caricamento, input di chat, un passo alla volta, override
  decision.js         scelta del token dentro generate() di Transformers.js
  verify.js           confronto punteggi grezzi e parità con la libreria
  fakeTf.js / mockClient.js   sostituto minimo per test e modalità simulata
src/ui/               anatomia della risposta, grafico, narrazione, etichette
tests/unit/           vitest (distribution, controller, engine, tokens, labels)
tests/e2e/            Playwright (interfaccia con ?mock)
tests/e2e-real/       smoke test con modello reale (manuale)
docs/checklist-manuale.md   verifica manuale su dispositivi reali
```

### Scelte notevoli

- **Onestà della visualizzazione**: ogni passo mostrato corrisponde a una chiamata reale del modello, in ordine; il passo successivo si calcola solo dopo che il precedente è stato mostrato. La pausa ferma davvero i calcoli.
- **Verificabilità**: Transformers.js 4.3 non espone i punteggi per passo né applica top-p, quindi la scelta del token avviene in un `LogitsProcessor` esplicito (`decision.js`) e il chiamante verifica l'allineamento (un passaggio, un token in più, token emesso = token scelto). `verify.js` confronta i punteggi con un forward diretto del modello e la sequenza passo-passo con la generazione nativa della libreria.
- **Token fedeli**: un token può essere una parola, un pezzo di parola, uno spazio o un frammento di byte di un carattere (es. un'emoji). I frammenti sono raggruppati e dichiarati come tali; la concatenazione dei pezzi coincide sempre con la decodifica dell'intera sequenza (invariante testata).
- **Percentuali corrette**: softmax numericamente stabile; mai «0%» per un valore positivo né «100%» per un valore minore di 1; l'ordine di applicazione dei filtri (temperatura → top-k → top-p → rinormalizzazione) è dichiarato sotto il grafico.
- **Sicurezza**: il testo dell'utente entra nel DOM solo via `textContent`; l'input inviato al modello è limitato a 240 caratteri e 160 token.

## Modelli

| | |
|---|---|
| Consigliato | `onnx-community/gemma-4-E2B-it-ONNX` — q4f16 (~3,1 GB) |
| Leggero | `onnx-community/Qwen3-0.6B-ONNX` — q4 (~579 MB) |
| Licenza | Apache-2.0 (Gemma: Google DeepMind; Qwen3: Alibaba Qwen) |
| Origine | Hugging Face Hub, revisioni fissate a commit preciso per riproducibilità |
| Runtime | Transformers.js 4.3 + onnxruntime-web servito dal sito |

La scelta tra i due è suggerita dalla memoria dichiarata dal browser (sotto 16 GB → modello leggero) ma resta dell'utente. I file si scaricano al primo uso e restano nella Cache Storage del browser; si possono eliminare dalle opzioni avanzate.

## Privacy

Le domande e le risposte restano sul dispositivo: durante la generazione non esce nessuna richiesta di rete (verificabile in DevTools). L'unica rete usata è il download dei file del modello da Hugging Face e del runtime ONNX dal sito stesso.

## Browser consigliati

- Chrome/Edge recenti con WebGPU (target principale; pensato anche per l'uso in aula: proiettore, tema chiaro/scuro, «Testo grande»)
- Accessibilità: navigazione completa da tastiera (Spazio/→ per avanzare, Esc per fermare), stati annunciati con `aria-live`, rispetto di «riduci animazioni»

## Deploy GitHub Pages

Deploy automatico tramite workflow su push in `main`: `npm ci && npm run build`, copia di `CNAME` in `dist/`, pubblicazione con GitHub Pages (source: GitHub Actions). Il dominio personalizzato (`CNAME`) è `predittoretokensuccessivo.monaripietro.it`.

## Limiti dichiarati

- **WebGPU obbligatorio**: senza scheda grafica il modello non viene eseguito né scaricato; non esiste fallback su CPU.
- **Gemma 4 E2B su macchine con poca memoria**: il caricamento può fallire o mandare in crash il tab su macchine con 8 GB; per questo esiste il suggerimento del modello leggero. Un crash di Gemma su questa classe di macchine è un esito atteso da documentare, non un difetto.
- **Qualità linguistica**: un modello da ~1B/0.6B parametri commette errori; l'obiettivo è osservare il processo, non produrre testo di qualità.
- **Contesto**: prompt fino a 240 caratteri, risposta fino a 2048 token generati (inclusi eventuali token di ragionamento), 8 candidati mostrati per passo.
- **Campionamento**: al massimo 64 candidati ammessi all'estrazione; l'ordine dei filtri è temperatura → 64 candidati → top-p → rinormalizzazione.

## Smoke test manuale

1. Aprire il sito e verificare il chip del modello («pronto» dopo il caricamento).
2. Inserire una domanda breve (es. «Qual è la capitale d'Italia?») e premere «Calcola il primo token».
3. Verificare a ogni passo: tokenizzazione (con origine system/user/control), classifica con percentuali, token scelto, risposta che cresce.
4. «Prossimo token» tre volte: esattamente un token per clic.
5. Provare «Continua da solo», «Pausa», «Ferma» e il replay della registrazione.
6. Cambiare l'istruzione di sistema nelle opzioni avanzate; sui modelli compatibili provare anche il ragionamento.
7. Attivare il campionamento nelle opzioni avanzate (temperatura, top-p, seme) e ripetere un avvio.
8. Controllare la console: nessun errore; in DevTools → Network nessuna richiesta durante la generazione.

Verificare le invarianti (token validi, percentuali in [0,1], contesto coerente), non una frase esatta. Per la verifica completa su dispositivi reali vedi `docs/checklist-manuale.md`.

## Diagnostica

`diagnostics.html` misura tempi di caricamento, primo token, token/s e verifica su questo dispositivo che i punteggi mostrati siano quelli grezzi del modello e che il ciclo passo-passo coincida con la generazione nativa della libreria.
