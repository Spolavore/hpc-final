#!/usr/bin/env bash
# Executa todas as medicoes do trabalho e gera results/raw.csv
# Metodologia: 1 warm-up nao medido (dentro do binario) + reps medidas.
# Threads fixadas nos cores fisicos (OMP_PROC_BIND/OMP_PLACES).
set -euo pipefail
cd "$(dirname "$0")/.."

make matmul >/dev/null

export OMP_PROC_BIND=close
export OMP_PLACES=cores

N=1024
REPS=5

OUT=results/raw.csv
VAL=results/validacao.txt
mkdir -p results

echo "versao,N,threads,rep,tempo_s" > "$OUT"

echo "== validacao de corretude ==" | tee "$VAL"
./matmul --check v1 "$N" 1 | tee -a "$VAL"
./matmul --check v2 "$N" 8 | tee -a "$VAL"
./matmul --check v3 "$N" 8 | tee -a "$VAL"

echo "== v0 baseline sequencial =="
./matmul v0 "$N" "$REPS" 1 | tee -a "$OUT"

echo "== v1 omp simd no laco interno (sequencial) =="
./matmul v1 "$N" "$REPS" 1 | tee -a "$OUT"

echo "== v3 omp parallel for no laco interno (8 threads) =="
./matmul v3 "$N" "$REPS" 8 | tee -a "$OUT"

echo "== v2 omp parallel for no laco externo: escalabilidade 1,2,4,8 threads =="
for t in 1 2 4 8; do
    ./matmul v2 "$N" "$REPS" "$t" | tee -a "$OUT"
done

echo "medicoes salvas em $OUT"
