# 20_GPU architectures

> https://www.youtube.com/watch?v=UFD8K-lprbQ

- le gpu sono siano vector processors che array processors, sono un mix dei due. entrambe ricadono nella categoria SIMD (Single Instruction Multiple Data)
- **memory banking**, è la suddivisione della memoria in più unità per permettere un maggior accessi allo stesso momento ai dati
- ha dei limiti se i dati su cui opera non sono vettorizzabili:
    - ad esempio le linked list
   - o qualsiasi algoritmo che non è vettorizzabile
cosa significa che non è vettorizzabile? ad esempio se i dati tra di loro non sono indipendenti
- Laddove SIMD funziona, si ha un guadagno in termini di prestazioni notevole. si è comunque vincolati dalla legge di amdahl, la quale si applica a programmi che sono solo in parte parallelizzati, il guadagno in termini di prestazioni è limitato dal resto del programma non ottimizzato
- oggi i processori moderni, hanno delle estensioni SIMD e possono switchare dalla modalità 'seriale' a quella con istruzioni SIMD
- i primi processori intel hanno introdotto le operazioni MMX
## Intel Pentium MMX operations - 90s
idea: one instruction operates on multiple data elements simoultaneously -> SIMD!
- NO VLEN reg
- OPCODE determina il tipo di dato
    - 8 -> 8-bit bytes
    - 4 -> 16-bit words
    - 2 -> 32-bit doublewords
    - 1 -> 64-bit quadwords
- STRIDE = 1 sempre

Esempio
PCMPEQ (b,w,d) -> operazione di >=
                                                51    3     5     23
                                                >     >     >     >
                                                73    2     5      6
                                               0..0  1..1 0..0  1..1
ritorna una maschera di 1s if true or 0s if false

## GPUs
- le gpu sono SIMD engines underneath
- la pipeline funziona come una pipeline SIMD (array processor)
- la programmazione non è fatta tramite istruzioni SIMD ma usando **threads**
- programming model (sw) vs execution model (hw)
**programming model**: come il programmatore scrive il codice, può essere:
  - sequenziale (von neumann)
  - data parallel (SIMD)
  - dataflow
  - multi-threaded (MIMD, SPMD)
**execution model**: come l'hardware esegue il codice
    - out-of-order execution
    - vector processor
    - array processor
    - dataflow processor
    - multiprocessor
    - multithreaded processor
> l'execution model può essere molto diverso dal programming model, se coincidono è molto vantaggioso
> von neumann model sono implementati dai processori OoO (out of order processors)
> SPMD (single program multiple data) model implementati da un processore SIMD aka una GPU
quindi una GPU è una SIMD machine senza la necessità di usare le istruzioni SIMD -> è programmata usando i THREADs (SPMD programming model)
- ogni thread esegue lo stesso codice ma su dati diversi
- ogni thread ha il suo contesto (può essere usato, riavviato ed eseguito in modo indipendente)
- un set di thread che eseguono la stessa istruzione sono dinamicamente gruppati in **WARP (wavefront)** dall'hw
- possiamo vedere le GPU come macchine SIMD non esposte al programmatore (SIMT=single instruction multiple threads)
### SIMD vs SIMT execution model
- SIMD: uno stream di una singola istruzione sequenziale -> [VLD, VLD, VADD, VST], VLEN
- SIMT: uno stream di multiple istruzioni scalari -> threads gruppati dinamicamente in **warps**, praticamente rimuovono la necessità di un VLEN, VMASK necessari in SIMD
-> [LD, LD, ADD, ST], NumThreads
- quali sono i vantaggi quindi?
    - in SIMT il programmatore può continuare a programmare seguendo il modello di von neumann, l'hw pensa al resto, mentre per SIMD il programmatore deve conoscere meglio l'hw
    - ogni thread può essere trattato separatamente -> può eseguire più thread in modo indipendente aka MIMD processing
    - può gruppare i threads in warps in modo flessibile -> in modo da massimizzare i benefici di SIMD
#### fine grained multi-threading of warps
- warp di 32 threads
- se abbiamo 32k iterations & 1 iteration/thread -> 1k warps (ogni warp ha 32 threads)
- ogni warp può essere interleaved sulla stessa pipeline

- le gpu non fanno branch prediction, check sulle dipendenze sui dati
- una gpu essenzialmente schedula i warp nella pipeline
- la pipeline rimane molto semplice:
    - un'istruzione per thread alla volta (no interlocking)
    - interleaving warp execution per mascherare le latenze
- SIMD vs SIMT:
SIMD: VADD A,B -> C
SIMT: ADD A[tid], B[tid] -> C[tid]
tid = threadId
la struttura può rimanere uguale a quella di un processore SIMD, ma usando i **tid **
#### memory access
- la stessa istruzione in thread diversi usano il **tid** come index per accedere a diversi dati
  <img src="../images/Pasted image 20240421164115.png" />
  - quando si programma è necessario partizionare i dati su thread diversi
  - for maximum performance, la memoria deve avere sufficiente bandwidth

- i warp non sono esposti al programmatore
- cpu threads & gpu kernels:
	- sequenziali o poco parallelizzate sezioni sulla CPU
	- forte parallelismo in sezioni della GPU: **blocchi di threads**
	- il codice seriale ha senso eseguirlo sulla CPU -> perchè è più 'brava' a farlo
	  <img src="../images/Pasted image 20240421165556.png" />
- le gpu hanno avuto molto successo anche perchè il codice di una CPU è molto simile ad una di una GPU
  <img src="../images/Pasted image 20240421165810.png" />
  <img src="../images/Pasted image 20240421165900.png" />
  #### dai blocks ai warps
  - GPU core = una pipeline SIMD
	  - Streaming processor SP
	  - Many such SIMD processors
		  - Streaming multiprocessor SM
		    <img src="../images/Pasted image 20240421170114.png" />
- I blocchi sono divisi in WARPS
	- unità SIMD / SIMT (32 threads)
	  <img src="../images/Pasted image 20240421170146.png" />
#### warp-based SIMD vs SIMD tradizionale
- SIMD tradizionale è eseguito su un singolo thread
	- sequential instruction execution -> operazioni lock-step in un'istruzione SIMD
	- il programming model deve essere SIMD (no extra threads) -> il SW deve conoscere il vector length
- WARP based SIMD, più thread scalari in esecuzione in a SIMD-like manner (stessa istruzione eseguita da tutti i threads)
	- no lock step
	- ogni thread può essere trattato individualmente (warp differenti) -> il programming model NON È SIMD
		- il sw non deve conoscere il VLEN
		- è possibile il multithreading e il raggruppamento dinamico dei threads
	- ISA scalare
	=> è essenzialmente SPMD programming model implementato su un hardware SIMD
#### SPMD single program multiple data
- è un modello di programmazione, non una struttura architetturale
- ogni FU esegue la stessa prcedura, ma su dati differenti
	- le procedure possono essere sincronizzati in certi punti del programma (eg barriers)
- stream di esecuzione multipli eseguono lo stesso programma
	- ogni programma / procedura:
		  - lavora su dati differenti
		  - può eseguire un control-flow path diverso a runtime (!)
#### cosa significa che SIMT può gruppare i threads in warp in modo flessibile?
<img src="../images/Pasted image 20240421171233.png" />
<img src="../images/Pasted image 20240421171318.png" />
- se ci sono molti thread si possono
	- raggruppare più thread che sono allo stesso PC
	- e grupparli in un singolo warp dinamicamente
	- il risultato è che si riduce la "divergenza" -> SIMD utilization aumenta
		- SIMD utilization: fraction of SIMD lanes executing useful operation (eg. che esegue un thread attivo)
- l'idea è di mergiare i threads che stanno eseguendo la stessa istruzione (ovvero allo stesso PC) dopo aver eseguito un branch
- ovvero creare nuovi warp con i warp in attesa per migliorare la SIMD utilization
  <img src="../images/Pasted image 20240421172104.png" />
  in un esempio complesso..
  <img src="../images/Pasted image 20240421172335.png" />
  - si possono spostare i thread su linee diverse della pipeline? -> NO

#### un esempio di GPU - NVIDIA GeForce GTX 285
- 240 stream processors
- SIMT execution
- 30 core
- 8 SIMD FU per core
  per gli standard odierni è PICCOLA
  <img src="../images/Pasted image 20240421172837.png" />
  32 thread in a warp & 32 warps -> 1024 threads che possono essere usati e quindi necessità di 64 KB of storage per i threads (registri)
  <img src="../images/Pasted image 20240421173033.png" />
  cosa sono i TENSOR CORE -> essenzialmente sono core specializzati per fare operazioni matriciale in modo ottimizzato. sono dei core specializzati
  anche nei tensor core ci sono dei processori SIMD 
