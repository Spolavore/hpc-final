# Roteiro da apresentação (≈ 9min30 de fala + sobra para perguntas)

Cada slide tem: **tempo alvo**, **o que dizer** (fala sugerida, em linguagem
natural) e **para você entender** (o conceito por trás, explicado do zero —
leia antes, não durante).

---

## Slide 1 — Título (0:20)

**O que dizer:** "Meu trabalho é sobre otimização de multiplicação de matrizes
em CPU usando OpenMP. A ideia central não é chegar no menor tempo possível, e
sim mostrar o ciclo completo de otimização: medir, diagnosticar o gargalo,
aplicar técnicas e explicar por que cada uma funcionou ou não."

---

## Slide 2 — Problema, aplicação e ambiente (1:00)

**O que dizer:** "O problema é a multiplicação de matrizes densas: duas
matrizes 1024×1024 de números reais, que geram uma terceira. São cerca de 2
bilhões de operações de ponto flutuante por execução. Escolhi esse kernel
porque ele é a base de álgebra linear, simulações e machine learning, e porque
tem muito paralelismo: cada elemento da matriz resultado pode ser calculado de
forma independente. Executei tudo no PCAD, na máquina hype: 2 processadores
Xeon com 10 cores cada — 20 cores físicos, 2 nós NUMA — com nó exclusivo via
SLURM e governor de frequência fixo em performance para a medição ser estável."

**Para você entender:**
- *Multiplicação de matrizes*: cada célula `C[i][j]` do resultado é o "produto
  escalar" da **linha i de A** com a **coluna j de B**: você percorre os 1024
  pares, multiplica um a um e soma tudo. Isso repetido para cada uma das
  1024×1024 células dá N³ ≈ 1 bilhão de multiplicações+somas (2 GFLOP).
- *Densa* = a matriz é "cheia" (todos os valores importam). O contrário seria
  *esparsa* (quase tudo zero, com algoritmos especiais que pulam os zeros).
- *NUMA* (Non-Uniform Memory Access): a máquina tem 2 processadores físicos
  (sockets), cada um com sua memória "local". Um core acessa a memória do seu
  socket rápido, e a do socket vizinho mais devagar. Isso vai explicar
  resultados adiante.
- *Governor performance*: trava o processador na frequência máxima, senão o
  Linux varia o clock e contamina a medição.

---

## Slide 3 — Baseline e metodologia (1:00)

**O que dizer:** "A versão inicial, v0, é o algoritmo clássico de três laços:
para cada linha i e coluna j, o laço interno k percorre a linha de A e a
coluna de B multiplicando e somando. Mediu 2,3 segundos. Sobre a metodologia:
não usei nenhum profiler — o tempo é medido pelo relógio do próprio OpenMP, o
omp_get_wtime, em volta do kernel, então não há overhead de instrumentação
contaminando os números. Cada configuração roda uma vez de aquecimento que eu
descarto, e depois 5 vezes medidas — reporto a mediana, e o desvio ficou
abaixo de 1%. As threads
ficam presas nos cores físicos para não migrarem no meio da medição. E o ponto
mais importante: toda versão otimizada é validada elemento por elemento contra
a v0, e as 4 versões deram diferença zero — otimização que dá resultado errado
não vale nada."

**Para você entender:**
- *O código do slide*: `A[i*n+k]` é como se acessa `A[i][k]` quando a matriz
  está guardada como um vetor gigante, linha após linha (isso se chama
  *row-major*, e é como C guarda matrizes). A variável `sum` acumula o produto
  escalar e no final vira o `C[i][j]`.
- *Warm-up*: a primeira execução paga custos únicos (trazer os dados para a
  memória/cache pela primeira vez). Descartá-la evita medir esse ruído.
- *Mediana em vez de média*: se uma das 5 execuções der um pico por culpa do
  sistema operacional, a média se contamina; a mediana não.
- *Pinning* (`OMP_PLACES=cores`): cada thread fica fixa num core físico.

---

## Slide 4 — Diagnóstico (1:15) ⭐ slide mais importante da primeira metade

**O que dizer:** "Antes de otimizar, diagnostiquei. A máquina consegue
teoricamente 736 bilhões de operações por segundo; a v0 faz 0,94 bilhão —
0,1% do pico. Ou seja, o processador não está limitado por conta — ele passa
o tempo **esperando memória**. O culpado é o padrão de acesso: o laço interno
lê a coluna de B, e como a matriz está guardada por linhas, cada leitura pula
8 KiB na memória. Cada acesso traz da memória um bloco de 64 bytes com 8
números, e o código usa **um só** e joga os outros 7 fora. Disso saíram três
hipóteses: vetorizar o laço interno; paralelizar o laço externo; e trocar a
ordem dos laços para eliminar os pulos — atacar a causa raiz."

**Para você entender:**
- *Linha de cache*: a memória nunca entrega 1 número sozinho; entrega blocos
  de 64 bytes (= 8 doubles). Se você usa os 8, ótimo; se usa 1, desperdiçou
  7/8 do tráfego.
- *Stride*: o "tamanho do pulo" entre dois acessos consecutivos. Ler a linha
  de A tem stride 1 (endereços vizinhos, ótimo). Ler a coluna de B tem stride
  N = 1024 doubles = 8 KiB (péssimo: cada acesso cai numa linha de cache
  diferente).
- *GFLOP/s*: bilhões de operações de ponto flutuante por segundo. Comparar o
  medido com o pico teórico diz se o gargalo é cálculo ou memória.

---

## Slide 5 — v1: omp simd, sem ganho (1:00)

**O que dizer:** "A primeira tentativa foi vetorização: a diretiva `omp simd`
pede para o compilador usar as instruções vetoriais do processador, que operam
4 números de uma vez. A hipótese era ganhar até 4 vezes. O compilador
confirmou que vetorizou — verifiquei no relatório de compilação — mas o
resultado foi 0,96x: levemente pior. Por quê? Vetorizar multiplica a
capacidade de **conta**, mas o tempo era gasto esperando **memória**. Como a
coluna de B continua espalhada, o compilador monta os vetores carregando os
números um a um — o tráfego de memória não muda nada. É a lição central da
disciplina: uma técnica só é boa otimização se resolve o gargalo dominante."

**Para você entender:**
- *SIMD / vetorização* (Single Instruction, Multiple Data): instruções
  especiais (AVX2 nesta máquina) que fazem a mesma operação em 4 doubles ao
  mesmo tempo, num registrador largo de 256 bits. É paralelismo *dentro* de um
  core, sem threads.
- *`reduction(+:sum)`*: avisa que `sum` é um acumulador, para o compilador
  somar as parcelas em paralelo com segurança.
- Por que não ganhou: para encher um vetor de 4 valores da coluna de B, o
  hardware precisa buscar 4 linhas de cache diferentes do mesmo jeito que
  antes. A conta fica 4x mais rápida, mas a conta não era o gargalo.

---

## Slide 6 — v2 e v3: granularidade (0:50)

**O que dizer:** "As versões 2 e 3 usam a MESMA técnica — `parallel for` — em
laços diferentes. A v2 paraleliza o laço externo: o trabalho é dividido uma
única vez, cada uma das 20 threads pega um bloco de ~51 linhas da matriz e
trabalha independente, sem sincronizar com ninguém. A v3 paraleliza o laço
interno: as threads são coordenadas de novo a cada célula da matriz — 1
milhão de vezes — e a cada vez precisam combinar as somas parciais e esperar
todas na barreira, que aqui atravessa os 2 sockets. Mesma diretiva, mesmos 20
cores... e os resultados não poderiam ser mais opostos." *(transição para o
próximo slide)*

**Para você entender:**
- *`#pragma omp parallel for`*: divide as iterações do laço seguinte entre as
  threads. No laço externo, cada thread recebe um intervalo de linhas
  (`schedule(static)` = divisão fixa em blocos iguais).
- *Granularidade*: o "tamanho do pedaço" de trabalho que cada thread recebe
  entre sincronizações. Grossa = pedaços grandes (bom); fina = pedaços
  pequenos com coordenação frequente (overhead domina).
- *Barreira*: ponto onde todas as threads esperam todas chegarem. Entre dois
  sockets a comunicação é mais lenta, então cada barreira custa caro — e a v3
  faz 1 milhão delas.
- *Fork/join*: abrir a região paralela (distribuir trabalho) e fechá-la
  (juntar). Tem custo fixo; feito 10⁶ vezes, vira o dominante.

---

## Slide 7 — v2 e v3: resultados (0:30)

**O que dizer:** "A v2 fez o serviço em 0,13 segundo — 17 vezes e meia mais
rápido. A v3, com os MESMOS 20 cores, levou 10 segundos: 4 vezes MAIS LENTA
que rodar com uma única thread. É a lição de granularidade: pedaços grandes e
independentes escalam; pedaços minúsculos e sincronizados fazem o custo de
coordenação superar o próprio trabalho útil."

---

## Slide 8 — Escalabilidade da v2 (1:00)

**O que dizer:** "Esse é o gráfico que explica o resultado da v2, não só
mostra que melhorou. À esquerda, o speedup medido contra a linha ideal
conforme aumento as threads: 1, 2, 4, 8, 16, 20. Quase linear. À direita, a
eficiência: começa em 99% e desce suavemente até 87%. Essa queda tem causa
conhecida: a partir de 10 threads o trabalho passa a ocupar o segundo socket,
e parte dos acessos vira acesso remoto NUMA — as matrizes foram criadas na
memória do socket 0. É uma limitação mapeada, que afeta todas as versões
igualmente."

**Para você entender:**
- *Linha ideal*: speedup = nº de threads (20 threads → 20x). Nunca se atinge
  na prática; a distância até ela mostra o custo real do paralelismo.
- *First-touch*: o Linux coloca cada página de memória perto de quem a tocou
  primeiro. Como a thread 0 inicializa as matrizes, tudo nasce no socket 0; as
  threads do socket 1 pagam acesso remoto.

---

## Slide 9 — v4: troca de laços (1:00)

**O que dizer:** "A v4 ataca a causa raiz do diagnóstico. Troquei a ordem dos
laços de i-j-k para i-k-j: matematicamente calcula a mesma coisa, mas agora o
laço interno percorre a **linha** de B, não a coluna — endereços vizinhos,
stride 1. Três efeitos: o compilador vetoriza de verdade, o prefetcher enxerga
um padrão sequencial e busca os dados antes de precisarem, e cada linha de
cache é usada por inteiro, 8 números em vez de 1. E um detalhe importante:
para cada célula as somas acontecem na mesma ordem, então o resultado é bit a
bit idêntico ao da v0. Combinada com o paralelismo da v2, é a versão final."

**Para você entender:**
- *A troca*: em vez de "para cada célula C[i][j], varre k", a v4 faz "para
  cada linha i, para cada k, espalha `A[i][k] × B[k][j]` sobre a linha j
  inteira de C". Cada célula de C recebe as mesmas 1024 parcelas, só que
  intercaladas com as das células vizinhas — por isso o resultado é idêntico.
- *Prefetcher*: circuito do processador que detecta padrões de acesso
  (ex.: "está lendo endereços em sequência") e busca os próximos blocos
  antecipadamente, escondendo a latência da memória.
- Por que zera C antes: no i-k-j o acumulador não é mais uma variável local;
  é a própria linha de C, que precisa começar em zero.

---

## Slide 10 — Tabela de resultados (0:50)

**O que dizer:** "Resumo dos números: v1 empatou (0,96x), v3 piorou 4,4 vezes,
v2 deu 17,5x com 87% de eficiência. E reparem nas duas linhas da v4: com UMA
única thread, só de trocar a ordem dos laços, já são 7,9 vezes — maior que
qualquer outra versão; com 20 threads, 132 vezes — de 2,3 segundos para 17
milissegundos, com resultado idêntico validado. Duas observações: com
Hyper-Threading, 40 threads, a v4 ganha mais 38%, porque threads extras ajudam
a esconder latência de memória. E a v3 está aqui de propósito — uma otimização
que piora deve ser discutida, não escondida, como o material da disciplina
pede."

**Para você entender:**
- *Speedup* = tempo da v0 ÷ tempo da versão. 1x = igual; <1x = piorou.
- *Eficiência paralela* = speedup ÷ nº de threads. 20 threads com speedup
  17,5 → 87%: cada thread "entrega" 87% do que entregaria no mundo ideal.
- *Hyper-Threading*: cada core físico expõe 2 threads lógicas que compartilham
  as unidades do core. Quando uma thread trava esperando memória, a outra usa
  os recursos ociosos — por isso ajuda mais a v4 (+38%) que a v2 (+4%).

---

## Slide 11 — Gráfico de speedup (0:30, rápido)

**O que dizer:** "Visualmente: as três primeiras barras ficam no chão — o
empate da v1 e a piora da v3. A v2 dá o salto do paralelismo. A quinta barra
é a v4 com UMA única thread: 7,9 vezes só com a troca de laços — um core bem
alimentado de memória vale quase metade dos 20 cores da v2. E a última barra
é a combinação dos dois efeitos: 132 vezes."

---

## Slide 12 — Discussão (1:00)

**O que dizer:** "Amarrando os quatro desfechos: a v1 não ganhou porque
vetorizar multiplica conta e o gargalo era memória. A v3 piorou porque um
milhão de sincronizações entre dois sockets custa mais que o próprio trabalho.
A v2 escala porque a granularidade é grossa, e a eficiência cai suave por
causa do NUMA. E a v4 fecha a prova do diagnóstico: o par v1/v4 mostra as
MESMAS instruções vetoriais rendendo 0,96x quando o acesso é ruim e 7,9x — com
uma única thread — quando o acesso é corrigido. A diferença nunca foi a
técnica; foi o gargalo que ela resolve."

---

## Slide 13 — Agradecimento (0:40)

O slide mostra só "Obrigado" — a conclusão é falada por cima dele (a rubrica
cobra conclusão, então não pule esta fala).

**O que dizer:** "Concluindo: a melhor configuração foi a v4 com 20 threads —
17 milissegundos, 132 vezes o baseline, com resultado idêntico ao original. O
aprendizado que levo: o valor de uma técnica depende do gargalo, não da
técnica — quatro construções OpenMP sobre o mesmo kernel deram de 0,23x a
132x. E nada disso seria explicável sem medição controlada: warm-up, mediana,
threads fixas, frequência travada e validação de corretude. O código, o job
SLURM e os dados brutos estão no repositório do GitHub, reproduzíveis com dois
comandos. Obrigado!"

---

# Perguntas prováveis do professor (e respostas curtas)

**"De onde você tirou que o SIMD processa 4 doubles?"**
Da ficha técnica do processador: o Haswell usa AVX2, cujos registradores têm
256 bits. Um double ocupa 64 bits. 256 ÷ 64 = 4 doubles por instrução. E dá
para confirmar que o compilador realmente vetorizou assim: compilando com
`-fopt-info-vec`, o gcc reporta "loop vectorized using 32 byte vectors" para
o laço da v1 — 32 bytes = 4 doubles. (No relatório do trabalho, na máquina
local com AVX-512, o mesmo laço vetoriza com 64 bytes = 8 doubles — a largura
depende do hardware, não da diretiva.)

**"A v4 não tem `omp simd` — de onde vem a vetorização dela?"**
Da auto-vetorização do compilador: com `-O3`, o gcc vetoriza sozinho laços
seguros e lucrativos. Na v1 o pragma era necessário porque vetorizar uma soma
acumulada exige reordenar parcelas de ponto flutuante, e o gcc não faz isso
sem autorização explícita (`reduction`). Na v4 o laço interno não tem
acumulador — cada iteração escreve num elemento diferente de C — então nada
precisa ser reordenado e o compilador vetoriza por conta própria. Confirmado
com `-fopt-info-vec`: "loop vectorized using 32 byte vectors" no laço da v4.
Isso reforça a tese: o bloqueio nunca foi falta de diretiva, era o layout do
acesso.

**"Por que a v4 dá resultado bit a bit idêntico se mudou a ordem dos laços?"**
Porque para cada célula C[i][j] as parcelas continuam sendo somadas na mesma
ordem crescente de k — o que muda é a intercalação entre células diferentes,
que não interagem. Se a ordem das somas de uma célula mudasse, haveria
diferença de arredondamento (ponto flutuante não é associativo).

**"Por que usar mediana e não média?"**
Robustez a outliers: um pico de ruído do SO numa das 5 execuções desloca a
média, mas não a mediana. O desvio-padrão reportado mostra que a dispersão foi
< 1% de qualquer forma.

**"Por que a v3 não foi simplesmente igual à v0, em vez de 4x pior?"**
Porque ela paga overhead ativo: 10⁶ aberturas de região paralela, cada uma com
distribuição de trabalho, reduction e barreira envolvendo 20 threads em 2
sockets. Esse custo é maior que o trabalho útil de cada região (1024
multiplicações).

**"Como você sabe que o gargalo era memória sem usar perf?"**
Métricas derivadas: 0,1% do pico teórico de FLOP/s com desvio baixíssimo
descarta gargalo de processamento; a análise do padrão de acesso (stride de 8 KiB,
1/8 da linha de cache aproveitada) aponta a causa; e a v4 é o experimento que
confirma — mudando só o padrão de acesso, o mesmo hardware rendeu 46x mais
por thread (0,29 s → contra 2,3 s da v0 com 1 thread ≈ 7,9x… e 132x com 20).
Contadores de hardware exigiriam privilégio que o ambiente não dá.

**"Por que não comparar com BLAS/MKL?"**
Fora do escopo: o objetivo era o ciclo diagnóstico→técnica→explicação, não o
recorde. A v4 usa ~17% do pico; uma BLAS chega a ~90% com tiling para
registradores/L1 — citado como próximo passo nas limitações.

**"O que é tiling e por que seria o próximo passo?"**
Dividir as matrizes em blocos que cabem na cache e trabalhar bloco a bloco,
maximizando o reuso de cada dado carregado. Ataca o próximo gargalo da v4
(tráfego com a L3/memória em Ns maiores).

**"E se N fosse maior / não coubesse na cache?"**
Com N=1024 as 3 matrizes (24 MiB) cabem na L3 de um socket (25 MiB) com
folga apertada. N maiores estouram a L3, o custo por acesso sobe e o tiling
vira indispensável — limitação declarada no trabalho.

**"Threads em vez de 5 nós? Por que só 1 nó?"**
OpenMP é memória compartilhada: as threads precisam enxergar as mesmas
matrizes. Usar vários nós exigiria MPI (memória distribuída), outro paradigma.

---

# Dicas de apresentação

- Total: ~9min30. Se estourar, corte primeiro o slide 11 (o gráfico fala por
  si) e encurte o slide 2.
- Os slides 4 (diagnóstico) e 12 (discussão) são onde a rubrica pesa
  (20% + 10%) — não os apresse.
- Se travar em algum número: os quatro que importam são **0,96x / 0,23x /
  17,5x / 132x** — a história inteira está neles.
