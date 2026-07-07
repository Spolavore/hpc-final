# Material de apoio — as contas do trabalho, explicadas do zero

Todas as contas que aparecem nos slides, refeitas passo a passo. Nenhuma
exige mais que multiplicação e divisão.

---

## 1. O pico teórico da máquina: 736 GFLOP/s

**A pergunta que essa conta responde:** "se cada ciclo de clock fizesse
trabalho útil, quantas operações por segundo essa máquina entregaria?"

A ideia é multiplicar quatro números, do menor para o maior:

### Passo A — quantos números cabem numa instrução? → 4

O processador da hype (Haswell) tem registradores vetoriais AVX2 de 256 bits.
Cada número `double` ocupa 64 bits. Então:

    256 bits ÷ 64 bits = 4 doubles por instrução

Uma instrução vetorial opera esses 4 números **de uma vez**.

### Passo B — quantas operações cada instrução FMA faz? → 2

FMA (Fused Multiply-Add) é a instrução que faz `a × b + c` de uma vez:
uma multiplicação **e** uma soma. Por isso cada FMA conta como
**2 operações**. (E o laço do trabalho é exatamente isso:
`sum += A[...] * B[...]` — multiplica e acumula.)

### Passo C — quantas FMA o core dispara por ciclo? → 2

Cada core Haswell tem **2 unidades FMA** independentes, que trabalham em
paralelo no mesmo ciclo.

### Passo D — juntar tudo

    Por core, por ciclo:  4 doubles × 2 operações × 2 unidades = 16 operações
    Por core, por segundo: 16 × 2,3 GHz (2,3 bilhões de ciclos/s) = 36,8 GFLOP/s
    O nó inteiro:          36,8 × 20 cores = 736 GFLOP/s

> **Resumo em uma frase:** 4 números por instrução, vezes 2 operações por
> FMA, vezes 2 unidades, vezes 2,3 bilhões de ciclos por segundo, vezes 20
> cores = 736 bilhões de operações por segundo.

Importante: esse número é o **teto físico**. Nenhum programa real chega nele;
ele serve de régua para saber o quão longe o código está do máximo.

---

## 2. O que a v0 realmente entrega: 0,94 GFLOP/s

**A pergunta:** "quantas operações por segundo o baseline fez de verdade?"

Duas informações, uma divisão:

1. **Trabalho total** (fixo, sai da matemática do problema): a multiplicação
   de matrizes faz 2 × N³ operações. Com N = 1024:

       2 × 1024³ = 2,15 bilhões de operações (2,15 GFLOP)

   (De onde vem o 2×N³: são N×N células no resultado, e cada célula precisa
   de N multiplicações + N somas → N² × 2N = 2N³.)

2. **Tempo medido**: 2,298 segundos.

Então:

    2,15 GFLOP ÷ 2,298 s = 0,94 GFLOP/s

---

## 3. A comparação que fecha o diagnóstico: 0,1%

    0,94 ÷ 736 = 0,0013 ≈ 0,1% do pico

**Como ler esse número:** as unidades de cálculo do processador passam
99,9% do tempo paradas. Um processador parado com a frequência travada no
máximo só pode estar esperando uma coisa: **dados da memória**. É assim que
se conclui que o gargalo é memória, sem nenhuma ferramenta especial — só
com uma régua (o pico) e um cronômetro (o tempo medido).

---

## 4. As outras contas que aparecem nos slides

**Tamanho das matrizes (24 MiB):**

    1024 × 1024 = 1.048.576 elementos por matriz
    × 8 bytes por double = 8 MiB por matriz
    × 3 matrizes (A, B, C) = 24 MiB

**O pulo de 8 KiB (stride) na coluna de B:**

    Para descer uma linha na matriz B, o endereço avança uma linha inteira:
    1024 elementos × 8 bytes = 8192 bytes = 8 KiB

    A memória entrega blocos de 64 bytes (8 doubles). Pulando 8 KiB por
    leitura, cada acesso cai num bloco diferente e usa 1 dos 8 números
    → 7/8 do tráfego é jogado fora.

**Speedup:**

    speedup = tempo da v0 ÷ tempo da versão
    Ex.: v2 com 20 threads: 2,298 ÷ 0,132 = 17,5×
    Se der menos que 1, piorou. Ex.: v3: 2,298 ÷ 10,193 = 0,23×

**Eficiência paralela:**

    eficiência = speedup ÷ número de threads
    Ex.: v2 com 20 threads: 17,5 ÷ 20 = 87%
    Leia como: "cada thread entrega 87% do que entregaria no mundo ideal".

**GFLOP/s de qualquer versão:**

    2,15 GFLOP ÷ tempo da versão
    Ex.: v4 com 20 threads: 2,15 ÷ 0,017 = 123,5 GFLOP/s
    → 123,5 ÷ 736 ≈ 17% do pico (por isso "ainda há espaço: tiling").

---

## 5. Tabela-resumo para decorar

| Número | Conta | Resultado |
|---|---|---|
| Pico do nó | 4 × 2 × 2 × 2,3 GHz × 20 cores | 736 GFLOP/s |
| Trabalho por execução | 2 × 1024³ | 2,15 GFLOP |
| v0 medido | 2,15 ÷ 2,298 s | 0,94 GFLOP/s |
| Diagnóstico | 0,94 ÷ 736 | ≈ 0,1% do pico |
| Stride da coluna de B | 1024 × 8 bytes | 8 KiB |
| Dados totais | 3 × 8 MiB | 24 MiB |
| v4 final | 2,15 ÷ 0,017 s | 123,5 GFLOP/s (~17% do pico) |
