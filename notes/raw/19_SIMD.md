> https://www.youtube.com/watch?v=gkMaO3yJMz0

- vector processors erano più comuni una volta perchè non c'era bisogno di replicare le FU più volte sul chip (problemi di spazio)
oggi questo problema di spazio si è ridotto e gli array processors sono diventati più comuni
- una gpu è un misto tra un array processor e un vector processor
- l'architettura SIMD usa dei registri di N elementi lunghi M-bit, appunti dei vettori. Non un unico valore scalare
- I dati in un vector processor prende dai registri vettoriali, uno per uno, i dati scalari e li elaborano in una FU. Ad ogni ciclo un elemento viene preso dal vettore e viene processato fino a che non sono finiti i valori del vettore
un array processor invece prende tutti i dati dei vettori e li processa insieme.
ad esempio, se devo fare la somma di due vettori di 32 elementi
-> array processor prende tutti i 32 elementi (32 LOAD allo stesso momento), 32 ADD allo stesso momento, 32 STORE allo stesso momento. richiede quindi 32 unità funzionali replicate
-> vector processor:
  fa la prima LOAD0
  al secondo ciclo fa LOAD1, ADD0
  al terzo LOAD2, ADD1, STORE0
  al quarto LOAD3, ADD2, STORE1
  al quindo LOAD4, ADD3, STORE2

quindi un array processor è più veloce, ma perchè ha più FU replicate
un vector processor è più frugale, ma a regime è in grado di processare ad ogni ciclo dati diversi
- vector stride: è la distanza tra un elemento e un'altro di un vettore.
stride = 1 elementi contigui, che è lo scenario migliore
- vector instructions possono essere messe in pipeline più lunghe:
    - non ci sono dipendenze tra i dati dei vettori, no dependecies to check
    - no interlock nella pipeline
    - no control flow tra elementi di un vettore
    - essendo lo stride conosciuto tra elementi di un vettore, caricare i dati è più semplice (prefetching, caching)
- gli svantaggi principali:
    - necessità di dati parallelizzabili (parallelismo regolare)
    - non funziona con le linked list (per come funzionano le liste, l'elemento n ha la reference all'elemento n+1, non si conoscono le posizioni a priori)
    - la velocità della memoria può essere un bottleneck. con una singola istruzione carichi un numero molto alto (potenzialmente) di dati
## vector registers
- ogni registro contiene N M-bit valori -> N elementi di M-bit
- vector control registers:
    - VLEN -> vector length
    - VSTR -> vector stride
    - VMASK -> per le operazioni condizionali. indica su quali elementi di un vettore si può operare
ad esempio: VMASK[i] = (V_k[i] == 0).
è un bit vector 0 1 0 0 0 1 1 0 -> le operazioni verranno fatte solo sugli elementi con valore 1 alla corrispondente posizione del vettore maschera
## vector functional units (FU)
- può essere pipelinizzata dato che gli elementi sono indipendenti tra di loro

un processore vettoriale tipicamente contiene anche unità scalari, è un mix tra le due architetture
