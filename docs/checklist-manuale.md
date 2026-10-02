# Checklist di verifica manuale

Questa checklist copre ciò che i test automatici non possono verificare: dispositivi diversi, leggibilità in aula, accessibilità, rete reale. Compilare la tabella dei risultati in fondo per ogni sessione.

Prima di iniziare: aprire il sito (locale con `npm run preview`, oppure l'URL pubblico). Per le misure di velocità usare la pagina tecnica `diagnostics.html?auto=1&model=gemma-4-e2b&n=32&quality=1` (per il modello leggero: `model=qwen3-0.6b`).

## A. Browser desktop con WebGPU (Chrome o Edge)

### Primo avvio e download
- [ ] Con cache vuota (finestra privata o dati del sito cancellati) la pagina NON scarica nulla finché non si preme il pulsante di caricamento.
- [ ] Prima del clic è mostrata la dimensione del download (Gemma 4 E2B ≈ 3,1 GB; Qwen3 0,6B ≈ 579 MB, con q4f16).
- [ ] Durante il download si vede un avanzamento coerente (byte scaricati / totale) che arriva al 100%.
- [ ] Al termine compare uno stato "pronto" e il pulsante "Calcola il primo token" si abilita.
- [ ] Se la dimensione dello storage disponibile appare insufficiente, la pagina avvisa.
- [ ] Dopo il ricaricamento (F5) il modello si prepara dalla cache senza nuovo download (DevTools, Network: nessuna richiesta di file del modello a huggingface.co).
- [ ] Da "Avanzate" i file in cache si possono cancellare; dopo la cancellazione la pagina torna a mostrare la dimensione da scaricare.

### Passo per passo
- [ ] Inserire un prompt (es. "Qual è la capitale d'Italia?") e premere "Calcola il primo token".
- [ ] Compaiono, in ordine: prompt, istruzione di sistema aggiunta dall'app e token di controllo, token e ID, grafico delle probabilità, token scelto, risposta che cresce.
- [ ] Il grafico mostra 8 candidati più "altri N token"; le barre visibili non sommano al 100%.
- [ ] Il pulsante diventa "Prossimo token" e ogni clic aggiunge esattamente un token.
- [ ] Il token di fine risposta e i token di controllo sono indicati come tali, non come parole.
- [ ] Il prompt e la risposta sono coerenti (modello Gemma: risposta corretta in italiano su prompt semplici).
- [ ] Prompt oltre 240 caratteri: l'app lo segnala/limita.

### Scegliere un token diverso
- [ ] In modalità passo per passo (in pausa, dopo un token) si può cliccare una riga del grafico diversa da quella scelta; funziona anche da tastiera (Tab sulle righe, Invio).
- [ ] Il passo è etichettato come scelta del presentatore; le probabilità mostrate restano quelle del modello.
- [ ] Il modello prosegue dal token forzato e la risposta cambia in modo coerente.
- [ ] La registrazione esportata (JSON) indica il passo forzato (campo `override`, con la scelta originale della regola).

### Continua, pausa, stop, replay
- [ ] "Continua da solo" genera token in sequenza senza bloccare la pagina (si può scorrere e cliccare).
- [ ] "Pausa" ferma dopo il token in corso; si può riprendere o tornare al passo per passo.
- [ ] "Ferma" interrompe la generazione; il modello può essere riutilizzato dopo.
- [ ] Il replay della traccia registrata riproduce i passi senza nuova inferenza (nessun aumento di uso GPU).
- [ ] "Scarica la registrazione (JSON)" scarica un file valido con i passi registrati.

### Campionamento (Avanzate)
- [ ] Di default la regola è "greedy": ripetendo lo stesso prompt si ottiene la stessa risposta.
- [ ] Attivando il campionamento compaiono temperatura, top-p e seme; il grafico mostra la seconda serie (probabilità di estrazione) con legenda.
- [ ] La riga sotto il grafico dichiara la regola usata (greedy / campionamento con i suoi parametri); il seme usato è salvato nella registrazione esportata (campo `seed`).
- [ ] Stesso seme e stessi parametri danno la stessa risposta.

### Trasparenza dell'input e ID
- [ ] L'input completo del passo (prompt + istruzione di sistema + token di controllo + token già generati) è visibile per intero.
- [ ] Gli ID dei token sono visibili a richiesta (non di default) e corrispondono ai pezzi di testo.
- [ ] I frammenti di byte (emoji, caratteri multi-byte) sono etichettati come tali.

### Proiettore, tema, accessibilità
- [ ] "Testo grande" ingrandisce testi e grafico senza sovrapposizioni o tagli.
- [ ] Tema scuro: tutti i testi, barre e legende restano leggibili e con contrasto adeguato.
- [ ] Solo tastiera: Tab raggiunge tutti i controlli nell'ordine logico, il focus è visibile, Invio/Spazio attivano i pulsanti, i candidati cliccabili sono raggiungibili.
- [ ] Screen reader (NVDA/Narrator): lo stato di caricamento, il token scelto e gli errori sono annunciati; il grafico ha un'alternativa testuale.
- [ ] Riduzione del movimento (impostazione di sistema "riduci animazioni"): le animazioni sono ridotte o assenti.

### Rete e privacy (DevTools, scheda Network)
- [ ] Aprire Network con cache già popolata, generare una risposta intera: nessuna richiesta esce durante la generazione.
- [ ] Cercare nel filtro Network il testo del prompt: non compare in nessun URL, header o payload.
- [ ] Le uniche richieste durante il caricamento sono il sito stesso e huggingface.co (e la sua CDN).
- [ ] La console non mostra errori.

## B. Dispositivo con poca memoria (es. portatile 8 GB di RAM)

- [ ] La pagina suggerisce il modello leggero (Qwen3 0,6B) quando `navigator.deviceMemory` indica meno di 16 GB. Nota: il valore è approssimato e il suggerimento non è un blocco.
- [ ] Il modello leggero si scarica (~579 MB) e funziona fino alla fine di una risposta.
- [ ] Il modello leggero è presentato con un'etichetta onesta (risposte meno affidabili).
- [ ] (Facoltativo) Provare anche Gemma 4 E2B e annotare se il caricamento riesce.

Da registrare:
- tempo di caricamento (prima volta e da cache);
- tempo al primo token;
- token al secondo (pagina diagnostica);
- memoria in Gestione attività (Chrome, GPU dedicata/condivisa) al picco;
- se il tab si blocca, si ricarica da solo o va in crash ("Aw, Snap");
- se la pagina resta reattiva durante la generazione.

Esito accettabile: caricamento del modello leggero senza crash, risposta completa generata, pagina reattiva, nessuna richiesta di rete durante la generazione. Un crash con Gemma su questa classe di macchine è un risultato atteso da documentare, non un difetto, a patto che il modello leggero funzioni. Misure di riferimento (un solo computer di prova, 32 GB di RAM): vedi `README.md`.

## C. Browser senza WebGPU

Esempi: Firefox senza WebGPU attiva, Chrome con accelerazione hardware disattivata, `chrome://flags` con WebGPU disabilitato.

- [ ] La pagina mostra un messaggio chiaro in italiano che spiega che serve WebGPU.
- [ ] Nessun download del modello parte (Network: nessuna richiesta a huggingface.co).
- [ ] Non c'è alcun ripiego silenzioso su CPU/WASM né su una simulazione.
- [ ] Se l'adapter è software (CPU emulata), viene rifiutato con un messaggio comprensibile.
- [ ] I controlli non utilizzabili sono disabilitati e la pagina non va in errore.

## D. Sessione con proiettore

- [ ] Da fondo aula (distanza reale), con "Testo grande" attivo, si leggono prompt, token scelto, risposta e percentuali.
- [ ] Il contrasto regge con la luce dell'aula (provare tema chiaro e scuro).
- [ ] Le barre e i colori sono distinguibili anche da chi ha difficoltà cromatiche.
- [ ] Il ritmo di un token per clic è adatto alla spiegazione; con "Continua da solo" la velocità dipende da "Animazione" (lento / normale / veloce): verificare che il gruppo riesca a seguire (altrimenti "Pausa").
- [ ] Cliccare una riga del grafico per scegliere un token diverso è comodo con mouse o telecomando.
- [ ] Il computer non va in stand-by e il caricamento è stato fatto prima della lezione (la prima preparazione può richiedere minuti).
- [ ] È disponibile un piano B: traccia esportata in JSON o replay già registrato.

## Tabella dei risultati

Copiare una riga per ogni prova.

| Data | Browser/versione | Sistema operativo | GPU | RAM | Modello | Tempo di caricamento (prima / cache) | Primo token | Token/s | Note |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
