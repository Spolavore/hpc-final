#!/bin/bash
#
# SLURM job para executar o benchmark do hpc-final na particao hype do PCAD
# (2x Intel Xeon E5-2650 v3: 20 cores fisicos / 40 threads HT, 128 GB, NUMA).
#
# Fluxo (padrao do projeto perf-analysis/stage3):
#   1. clona uma copia limpa do repo no $SCRATCH local do no;
#   2. seta o governor DVFS para performance;
#   3. valida corretude e executa todas as medicoes (bench/run.sh);
#   4. medicao extra com Hyper-Threading (40 threads) em CSV separado;
#   5. copia results/ para $HOME (NFS compartilhado com o front-end,
#      sobrevive ao fim da alocacao).
#
# Submissao (no front-end gppd-hpc.inf.ufrgs.br):
#   sbatch hype.slurm
#
# A alocacao de nos no PCAD e exclusiva: cpus-per-task=1 segue o padrao
# comprovado do projeto perf-analysis (stage3/run.slurm), que usa todos os
# cores do no com essa mesma configuracao.
#SBATCH --job-name=hpc-final-matmul
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
    echo "[ERRO] A variavel de ambiente SCRATCH nao esta definida. Abortando." >&2
    exit 1
fi

REPO_URL="https://github.com/Spolavore/hpc-final.git"
REPO_DIR="${SCRATCH}/hpc-final"
HOME_RESULTS_DIR="${HOME}/resultados-hpc-final/${SLURM_JOB_ID:-manual}"

N=1024
REPS=5
THREADS_SWEEP="1 2 4 8 16 20"   # cores fisicos da hype
MAX_THREADS=20

echo "[INFO] Hostname: $(hostname) | Job: ${SLURM_JOB_ID:-n/a} | $(date)"

# O $SCRATCH e local ao no e o acesso termina com a alocacao: copia os
# resultados para o $HOME (NFS) SEMPRE — inclusive em falha, cancelamento
# ou timeout (SIGTERM) — para nunca perder o que ja foi medido.
RESULTS_COPIED=0
copy_results() {
    (( RESULTS_COPIED )) && return 0
    RESULTS_COPIED=1
    if [[ -d "$REPO_DIR/results" ]]; then
        mkdir -p "$HOME_RESULTS_DIR"
        cp -r "$REPO_DIR/results/." "$HOME_RESULTS_DIR"/ || true
        echo "[INFO] Resultados copiados para $HOME_RESULTS_DIR"
    fi
}
trap copy_results EXIT
trap 'copy_results; exit 143' TERM

# 1. copia limpa do repositorio no scratch local do no
if [[ -d "$REPO_DIR" ]]; then
    echo "[INFO] Removendo copia anterior: $REPO_DIR"
    rm -rf "$REPO_DIR"
fi
cd "$SCRATCH"
git clone "$REPO_URL"
cd "$REPO_DIR"

# 2. governor performance em todos os cores (padrao perf-analysis;
#    tolera falha caso o no nao permita)
for i in $(seq 0 $(( $(nproc) - 1 ))); do
    cpufreq-set -c "$i" -g performance || true
done

# registro do ambiente de execucao
mkdir -p results
{
    echo "hostname: $(hostname)"
    echo "data: $(date -Is)"
    echo "job: ${SLURM_JOB_ID:-n/a}"
    gcc --version | head -1
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo indisponivel)"
    echo "nproc: $(nproc)"
    lscpu
} > results/ambiente.txt

# 3. validacao + medicoes (bench/run.sh valida v1..v4 contra v0 e gera raw.csv)
N="$N" REPS="$REPS" THREADS_SWEEP="$THREADS_SWEEP" MAX_THREADS="$MAX_THREADS" \
    bash bench/run.sh

# 4. extra: Hyper-Threading (1 thread por processador logico), CSV separado
#    para nao poluir os graficos principais (metodologia = cores fisicos).
#    So roda se o no expuser mais processadores logicos que cores fisicos.
HT_THREADS=$(nproc)
if (( HT_THREADS > MAX_THREADS )); then
    echo "[INFO] Medicao extra com HT: ${HT_THREADS} threads (OMP_PLACES=threads)"
    export OMP_PROC_BIND=close OMP_PLACES=threads
    HT_OUT=results/raw_ht.csv
    echo "versao,N,threads,rep,tempo_s" > "$HT_OUT"
    ./matmul v2 "$N" "$REPS" "$HT_THREADS" | tee -a "$HT_OUT"
    ./matmul v4 "$N" "$REPS" "$HT_THREADS" | tee -a "$HT_OUT"
else
    echo "[INFO] HT indisponivel (nproc=$HT_THREADS <= $MAX_THREADS cores); pulando medicao extra"
fi

# 5. a copia para o HOME e feita pelo trap copy_results (EXIT/TERM)
echo "[INFO] Job finalizado com sucesso."
