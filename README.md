# hpc-final

Trabalho final da disciplina **Software de Alto Desempenho (INF01094 — UFRGS)**.

Investigação de otimização de multiplicação de matrizes densas (`C = A × B`,
`double`, N = 1024) em CPU com **OpenMP**: uma otimização sem ganho
(`omp simd` sobre acesso com stride), uma com benefício comprovado
(`omp parallel for` no laço externo, speedup de **11,6×** em 8 cores) e um
experimento de granularidade (paralelismo no laço interno).

**O relatório completo está em [`relatorio.md`](relatorio.md).**

## Estrutura

```
src/matmul.c       kernels v0 (baseline), v1 (simd), v2 (paralelo externo),
                   v3 (paralelo interno) + validação e benchmark
bench/run.sh       roda todas as medições -> results/raw.csv
analysis/plot.py   gera os gráficos -> results/*.png
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
./matmul <v0|v1|v2|v3> <N> <reps> <threads>   # benchmark (CSV no stdout)
./matmul --check <v1|v2|v3> <N> <threads>     # validação contra a v0
```

Ambiente dos resultados reportados: AMD Ryzen 7 7800X3D (8 cores, AVX-512,
L3 96 MiB), gcc 15.2, Linux 6.17, `OMP_PROC_BIND=close OMP_PLACES=cores`.
