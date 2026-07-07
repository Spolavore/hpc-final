#!/usr/bin/env python3
"""Gera os graficos do relatorio a partir de um CSV de medicoes.

Uso: python3 analysis/plot.py [caminho/para/raw.csv]
     (default: results/raw.csv; os PNGs sao salvos ao lado do CSV)

O script deriva do proprio CSV: N, numero de repeticoes, o sweep de
threads da v2 e o numero maximo de threads de cada versao — o mesmo
codigo plota os resultados da maquina local e os do cluster (PCAD).
"""

import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

ROOT = Path(__file__).resolve().parent.parent

# paleta validada (skill dataviz): azul serie unica + tokens de tinta
BLUE = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

DPI = 200


def load(csv_path):
    data, sizes = {}, set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["versao"], int(row["threads"]))
            data.setdefault(key, []).append(float(row["tempo_s"]))
            sizes.add(int(row["N"]))
    if len(sizes) != 1:
        sys.exit(f"erro: esperava um unico N no CSV, encontrei {sorted(sizes)}")
    stats = {k: (statistics.median(v),
                 statistics.stdev(v) if len(v) > 1 else 0.0, len(v))
             for k, v in data.items()}
    return stats, sizes.pop()


def fmt(v, dec=2, suf=""):
    return f"{v:.{dec}f}".replace(".", ",") + suf


def new_fig():
    fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0, labelcolor=INK2)
    ax.tick_params(axis="y", labelcolor=MUTED)
    return fig, ax


def titles(fig, title, subtitle):
    fig.text(0.075, 0.945, title, fontsize=13, fontweight="bold", color=INK)
    fig.text(0.075, 0.885, subtitle, fontsize=9.5, color=INK2)
    fig.subplots_adjust(top=0.80, left=0.105, right=0.97, bottom=0.17)


def rounded_bar(ax, cx, h, w, color, r_px=8):
    """Barra com topo arredondado (raio em pixels) e base reta."""
    inv = ax.transData.inverted()
    (x0p, y0p), (x1p, y1p) = inv.transform([(0, 0), (r_px, r_px)])
    rx, ry = min(abs(x1p - x0p), w / 2), min(abs(y1p - y0p), h)
    xl, xr = cx - w / 2, cx + w / 2
    verts = [(xl, 0), (xl, h - ry), (xl, h), (xl + rx, h),
             (xr - rx, h), (xr, h), (xr, h - ry), (xr, 0), (xl, 0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                           linewidth=0, zorder=3))


def bar_chart(out, title, subtitle, labels, values, val_labels,
              ylabel, ymax, ref_line=None):
    fig, ax = new_fig()
    ax.set_xlim(-0.55, len(values) - 0.45)
    ax.set_ylim(0, ymax)
    for i, v in enumerate(values):
        rounded_bar(ax, i, v, 0.38, BLUE)
        ax.text(i, v + ymax * 0.025, val_labels[i], ha="center",
                fontsize=10.5, fontweight="semibold", color=INK)
    if ref_line is not None:
        ax.axhline(ref_line, color=MUTED, linewidth=1.2,
                   linestyle=(0, (4, 3)), zorder=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9 if len(labels) <= 5 else 8)
    ax.set_ylabel(ylabel, fontsize=10, color=INK2)
    titles(fig, title, subtitle)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "raw.csv"
    outdir = csv_path.parent
    d, n = load(csv_path)
    t = {k: v[0] for k, v in d.items()}  # medianas

    base = t[("v0", 1)]
    reps = d[("v0", 1)][2]
    pmax = {v: max(p for (vv, p) in t if vv == v)
            for v in {vv for (vv, _) in t}}

    def label_of(v, p):
        return {
            "v0": "v0\nbaseline ijk\nsequencial",
            "v1": "v1\n+ omp simd\n(laço interno)",
            "v3": f"v3\nparalelo interno\n{p} threads",
            "v2": f"v2\nparalelo externo\n{p} threads",
            "v4": ("v4\ntroca de laços\n1 thread" if p == 1
                   else f"v4\nikj + paralelo\n{p} threads"),
        }[v]

    versions = [(v, 1 if v in ("v0", "v1") else pmax[v])
                for v in ("v0", "v1", "v3", "v2", "v4") if v in pmax]
    labels = [label_of(v, p) for v, p in versions]

    # no grafico de speedup, a v4 com 1 thread entra como barra extra
    # (isola o efeito da troca de lacos sem paralelismo)
    versions_sp = list(versions)
    if ("v4", 1) in t and pmax.get("v4", 1) != 1:
        versions_sp.insert(-1, ("v4", 1))
    labels_sp = [label_of(v, p) for v, p in versions_sp]

    # 1. tempo por versao
    times = [t[k] for k in versions]
    bar_chart(
        outdir / "tempo_por_versao.png",
        "Tempo de execução por versão",
        f"Matmul N={n}, dupla precisão — mediana de {reps} repetições — menor é melhor",
        labels, times,
        [fmt(v, 3 if v < 0.1 else 2, " s") for v in times],
        "Tempo (s)", max(times) * 1.18,
    )

    # 2. speedup por versao
    sp = [base / t[k] for k in versions_sp]
    bar_chart(
        outdir / "speedup_por_versao.png",
        "Speedup por versão (vs. baseline v0)",
        f"Matmul N={n} — maior é melhor; linha tracejada marca o baseline (1×)",
        labels_sp, sp,
        [fmt(v, 0 if v >= 100 else (1 if v >= 10 else 2), "×") for v in sp],
        "Speedup", max(sp) * 1.18,
        ref_line=1.0,
    )

    # 3. escalabilidade v2: medido vs ideal
    threads = sorted(p for (v, p) in t if v == "v2")
    measured = [base / t[("v2", p)] for p in threads]
    top = max(max(measured), threads[-1]) * 1.1
    fig, ax = new_fig()
    ax.plot(threads, threads, color=MUTED, linewidth=1.6,
            linestyle=(0, (4, 3)), zorder=2, label="ideal (linear)")
    ax.plot(threads, measured, color=BLUE, linewidth=2.2, marker="o",
            markersize=7.5, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=3, label="v2 medido")
    for p, s in zip(threads, measured):
        dx, dy, align = ((10, -4, "left") if p == threads[0] else (-2, 9, "right"))
        ax.annotate(fmt(s, 1, "×"), (p, s), textcoords="offset points",
                    xytext=(dx, dy), ha=align, fontsize=9.5,
                    fontweight="semibold", color=INK)
    span = threads[-1] - threads[0]
    ax.set_xlim(threads[0] - span * 0.06, threads[-1] + span * 0.06)
    ax.set_ylim(0, top)
    ax.set_xticks(threads)
    ax.set_xlabel("Número de threads", fontsize=10, color=INK2)
    ax.set_ylabel("Speedup vs. v0", fontsize=10, color=INK2)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2)
    titles(fig, "Escalabilidade da v2: speedup vs. número de threads",
           f"Matmul N={n} — pontos acima da linha ideal indicam "
           "speedup superlinear (ver discussão)")
    fig.savefig(outdir / "escalabilidade_v2.png", facecolor=SURFACE)
    plt.close(fig)

    # 4. eficiencia paralela v2 (vs v2 com o menor numero de threads do sweep)
    v2_ref = t[("v2", threads[0])] * threads[0]
    eff = [(v2_ref / t[("v2", p)]) / p * 100 for p in threads]
    bar_chart(
        outdir / "eficiencia_v2.png",
        "Eficiência paralela da v2",
        f"eficiência(p) = speedup(p) / p, relativa à v2 com {threads[0]} thread"
        f"{'s' if threads[0] > 1 else ''} — acima de 100% = superlinear",
        [f"{p} thread{'s' if p > 1 else ''}" for p in threads],
        eff, [fmt(v, 0, "%") for v in eff],
        "Eficiência (%)", max(eff) * 1.22,
        ref_line=100,
    )

    # tabela-resumo no stdout (para o relatorio)
    print("versao,threads,mediana_s,desvio_s,speedup_vs_v0,gflops")
    flops = 2 * n**3
    for (v, p), (med, sd, _) in sorted(d.items()):
        print(f"{v},{p},{med:.4f},{sd:.4f},{base/med:.3f},{flops/med/1e9:.3f}")
    print(f"[INFO] graficos salvos em {outdir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
