/*
 * Trabalho Final - Software de Alto Desempenho (INF01094)
 * Multiplicacao de matrizes densas C = A x B (dupla precisao)
 *
 * Versoes:
 *   v0 - sequencial, lacos ijk (baseline)
 *   v1 - omp simd no laco INTERNO k (vetorizacao explicita, sequencial)
 *   v2 - omp parallel for no laco EXTERNO i (granularidade grossa)
 *   v3 - omp parallel for no laco INTERNO k (granularidade fina, reduction)
 *   v4 - v2 + troca de lacos i-k-j (acesso unit-stride, vetorizavel)
 *
 * Uso:
 *   ./matmul <v0|v1|v2|v3|v4> <N> <reps> <threads>  # benchmark, saida CSV
 *   ./matmul --check <v1|v2|v3|v4> <N> <threads>    # valida contra v0
 *
 * Saida do benchmark (stdout), uma linha por repeticao:
 *   versao,N,threads,rep,tempo_s
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* LCG deterministico: mesmas matrizes em qualquer execucao/versao */
static void init_matrices(double *A, double *B, int n) {
    unsigned long long s = 42ULL;
    for (long i = 0; i < (long)n * n; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        A[i] = (double)(s >> 40) / (double)(1 << 24) - 0.5;
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        B[i] = (double)(s >> 40) / (double)(1 << 24) - 0.5;
    }
}

/* v0: baseline sequencial ijk */
static void matmul_v0(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[(long)i * n + k] * B[(long)k * n + j];
            C[(long)i * n + j] = sum;
        }
    }
}

/* v1: omp simd no laco interno k (sequencial).
 * Pede vetorizacao explicita do produto escalar. O acesso B[k*n+j] tem
 * stride n (uma linha de cache nova por iteracao), entao o vetor so pode
 * ser montado com gather -- que nao reduz o trafego nem a latencia. */
static void matmul_v1(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (int k = 0; k < n; k++)
                sum += A[(long)i * n + k] * B[(long)k * n + j];
            C[(long)i * n + j] = sum;
        }
    }
}

/* v2: OpenMP no laco externo i (granularidade grossa).
 * Uma unica regiao paralela; cada thread recebe um bloco de linhas de C
 * e executa N*N/p produtos internos sem qualquer sincronizacao. */
static void matmul_v2(const double *A, const double *B, double *C, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[(long)i * n + k] * B[(long)k * n + j];
            C[(long)i * n + j] = sum;
        }
    }
}

/* v3: OpenMP no laco interno k (granularidade fina).
 * A regiao paralela e criada N*N vezes; cada uma distribui apenas N
 * multiplicacoes entre as threads e termina com reduction + barreira. */
static void matmul_v3(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
#pragma omp parallel for reduction(+ : sum) schedule(static)
            for (int k = 0; k < n; k++)
                sum += A[(long)i * n + k] * B[(long)k * n + j];
            C[(long)i * n + j] = sum;
        }
    }
}

/* v4: v2 + troca de lacos i-k-j.
 * O laco interno passa a percorrer B e C com stride 1 (enderecos
 * consecutivos), permitindo vetorizacao efetiva e prefetch sequencial.
 * Para cada (i,j) as contribuicoes continuam sendo acumuladas em ordem
 * crescente de k, entao o resultado permanece identico ao da v0. */
static void matmul_v4(const double *A, const double *B, double *C, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            C[(long)i * n + j] = 0.0;
        for (int k = 0; k < n; k++) {
            const double a = A[(long)i * n + k];
            for (int j = 0; j < n; j++)
                C[(long)i * n + j] += a * B[(long)k * n + j];
        }
    }
}

typedef void (*kernel_fn)(const double *, const double *, double *, int);

static kernel_fn select_kernel(const char *name) {
    if (strcmp(name, "v0") == 0) return matmul_v0;
    if (strcmp(name, "v1") == 0) return matmul_v1;
    if (strcmp(name, "v2") == 0) return matmul_v2;
    if (strcmp(name, "v3") == 0) return matmul_v3;
    if (strcmp(name, "v4") == 0) return matmul_v4;
    return NULL;
}

static double *alloc_matrix(int n) {
    double *m = (double *)aligned_alloc(64, (size_t)n * n * sizeof(double));
    if (!m) { fprintf(stderr, "erro: sem memoria\n"); exit(1); }
    return m;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "uso: %s <v0|v1|v2> <N> <reps> <threads>   # benchmark (CSV)\n"
            "     %s --check <v1|v2> <N> <threads>     # validacao vs v0\n",
            prog, prog);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 2) usage(argv[0]);

    if (strcmp(argv[1], "--check") == 0) {
        if (argc != 5) usage(argv[0]);
        kernel_fn kernel = select_kernel(argv[2]);
        if (!kernel) usage(argv[0]);
        int n = atoi(argv[3]);
        omp_set_num_threads(atoi(argv[4]));

        double *A = alloc_matrix(n), *B = alloc_matrix(n);
        double *C_ref = alloc_matrix(n), *C = alloc_matrix(n);
        init_matrices(A, B, n);
        matmul_v0(A, B, C_ref, n);
        kernel(A, B, C, n);

        double max_abs = 0.0, max_rel = 0.0;
        for (long i = 0; i < (long)n * n; i++) {
            double diff = fabs(C[i] - C_ref[i]);
            if (diff > max_abs) max_abs = diff;
            double denom = fabs(C_ref[i]) > 1e-30 ? fabs(C_ref[i]) : 1.0;
            if (diff / denom > max_rel) max_rel = diff / denom;
        }
        int ok = max_rel < 1e-9;
        printf("validacao %s N=%d threads=%s: max_abs_diff=%.3e "
               "max_rel_diff=%.3e -> %s\n",
               argv[2], n, argv[4], max_abs, max_rel,
               ok ? "OK" : "FALHOU");
        return ok ? 0 : 2;
    }

    if (argc != 5) usage(argv[0]);
    kernel_fn kernel = select_kernel(argv[1]);
    if (!kernel) usage(argv[0]);
    int n = atoi(argv[2]);
    int reps = atoi(argv[3]);
    int threads = atoi(argv[4]);
    omp_set_num_threads(threads);

    double *A = alloc_matrix(n), *B = alloc_matrix(n), *C = alloc_matrix(n);
    init_matrices(A, B, n);

    kernel(A, B, C, n); /* warm-up: nao medido (paginas tocadas, caches quentes) */

    for (int r = 1; r <= reps; r++) {
        double t0 = omp_get_wtime();
        kernel(A, B, C, n);
        double t1 = omp_get_wtime();
        printf("%s,%d,%d,%d,%.6f\n", argv[1], n, threads, r, t1 - t0);
        fflush(stdout);
    }

    free(A); free(B); free(C);
    return 0;
}
