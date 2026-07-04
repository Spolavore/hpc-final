# hpc-final

Trabalho final da disciplina **Software de Alto Desempenho (INF01094 — UFRGS)**.

Investigação de otimização de multiplicação de matrizes densas (`C = A × B`,
`double`, N = 1024) em CPU com **OpenMP**: uma otimização sem ganho
(`omp simd` sobre acesso com stride), uma com benefício comprovado
(`omp parallel for` no laço externo, **12,3×** em 8 cores), um experimento
de granularidade (paralelismo no laço interno, 2,2×) e a correção da causa
raiz (troca de laços i-k-j + paralelismo, **319×**).

**O relatório completo está em [`relatorio.md`](relatorio.md).**

## Estrutura

```
src/matmul.c       kernels v0 (baseline), v1 (simd), v2 (paralelo externo),
                   v3 (paralelo interno), v4 (ikj + paralelo) + validação
                   e benchmark
bench/run.sh       roda todas as medições -> results/raw.csv
                   (parametrizável: N, REPS, THREADS_SWEEP, MAX_THREADS, OUT_DIR)
bench/pcad/        job SLURM para o cluster PCAD/UFRGS (partição hype)
analysis/plot.py   gera os gráficos a partir de um CSV (default results/raw.csv)
results/           medições (CSV), gráficos, validação e experimentos de controle
relatorio.md       relatório (problema, baseline, diagnóstico, otimizações,
                   resultados, discussão, conclusão)
```

## Como reproduzir

Requisitos: `gcc` com OpenMP, `python3` + `matplotlib`.

```bash
make            # compila
make check      # valida corretude das versões otimizadas contra a v0
make bench      # executa o benchmark completo (~3 min)
make plots      # regenera os gráficos a partir de results/raw.csv
```

Uso direto do binário:

```bash
./matmul <v0|v1|v2|v3|v4> <N> <reps> <threads>   # benchmark (CSV no stdout)
./matmul --check <v1|v2|v3|v4> <N> <threads>     # validação contra a v0
```

Ambiente dos resultados reportados: AMD Ryzen 7 7800X3D (8 cores, AVX-512,
L3 96 MiB), gcc 15.2, Linux 6.17, `OMP_PROC_BIND=close OMP_PLACES=cores`.

## Execução no PCAD (partição hype)

No front-end do PCAD (`gppd-hpc.inf.ufrgs.br`):

```bash
curl -O https://raw.githubusercontent.com/Spolavore/hpc-final/main/bench/pcad/hype.slurm
sbatch hype.slurm
squeue -u "$USER"          # acompanhar a fila
```

O job clona o repo no `$SCRATCH` do nó, valida a corretude, roda o benchmark
com sweep de 1–20 cores físicos (+ uma medição extra com 40 threads/HT em
`raw_ht.csv`) e copia tudo para `~/resultados-hpc-final/<jobid>/` no `$HOME`
(NFS — os arquivos ficam acessíveis no front-end após o fim da alocação).
Para gerar os gráficos localmente a partir desses dados:

```bash
python3 analysis/plot.py results-hype/raw.csv
```
