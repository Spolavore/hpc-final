#!/usr/bin/env bash
# Executa todas as medicoes do trabalho e gera $OUT_DIR/raw.csv
# Metodologia: 1 warm-up nao medido (dentro do binario) + reps medidas.
# Threads fixadas nos cores fisicos (OMP_PROC_BIND/OMP_PLACES).
#
# Parametros (variaveis de ambiente):
#   N             tamanho da matriz            (default: 1024)
#   REPS          repeticoes medidas           (default: 5)
#   THREADS_SWEEP sweep de escalabilidade v2   (default: "1 2 4 8")
#   MAX_THREADS   threads das versoes v2/v3/v4 (default: 8)
#   OUT_DIR       diretorio de saida           (default: results)
#
# Ex. na maquina hype do PCAD (20 cores fisicos):
#   THREADS_SWEEP="1 2 4 8 16 20" MAX_THREADS=20 bash bench/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

make matmul >/dev/null

export OMP_PROC_BIND=close
export OMP_PLACES=cores

N=${N:-1024}
REPS=${REPS:-5}
THREADS_SWEEP=${THREADS_SWEEP:-"1 2 4 8"}
MAX_THREADS=${MAX_THREADS:-8}
OUT_DIR=${OUT_DIR:-results}

OUT="$OUT_DIR/raw.csv"
VAL="$OUT_DIR/validacao.txt"
mkdir -p "$OUT_DIR"

echo "[INFO] N=$N REPS=$REPS THREADS_SWEEP='$THREADS_SWEEP' MAX_THREADS=$MAX_THREADS OUT_DIR=$OUT_DIR"

echo "versao,N,threads,rep,tempo_s" > "$OUT"

echo "== validacao de corretude ==" | tee "$VAL"
./matmul --check v1 "$N" 1 | tee -a "$VAL"
./matmul --check v2 "$N" "$MAX_THREADS" | tee -a "$VAL"
./matmul --check v3 "$N" "$MAX_THREADS" | tee -a "$VAL"
./matmul --check v4 "$N" "$MAX_THREADS" | tee -a "$VAL"

echo "== v0 baseline sequencial =="
./matmul v0 "$N" "$REPS" 1 | tee -a "$OUT"

echo "== v1 omp simd no laco interno (sequencial) =="
./matmul v1 "$N" "$REPS" 1 | tee -a "$OUT"

echo "== v3 omp parallel for no laco interno ($MAX_THREADS threads) =="
./matmul v3 "$N" "$REPS" "$MAX_THREADS" | tee -a "$OUT"

echo "== v2 omp parallel for no laco externo: escalabilidade ($THREADS_SWEEP threads) =="
for t in $THREADS_SWEEP; do
    ./matmul v2 "$N" "$REPS" "$t" | tee -a "$OUT"
done

echo "== v4 troca de lacos i-k-j + parallel for externo (1 e $MAX_THREADS threads) =="
for t in 1 "$MAX_THREADS"; do
    ./matmul v4 "$N" "$REPS" "$t" | tee -a "$OUT"
done

echo "medicoes salvas em $OUT"
