#!/usr/bin/env python3
"""Gera os graficos do relatorio a partir de results/raw.csv."""

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# paleta validada (skill dataviz): azul serie unica + tokens de tinta
BLUE = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

DPI = 200


def load():
    data = {}
    with open(RESULTS / "raw.csv") as f:
        for row in csv.DictReader(f):
            key = (row["versao"], int(row["threads"]))
            data.setdefault(key, []).append(float(row["tempo_s"]))
    return {k: (statistics.median(v), statistics.stdev(v)) for k, v in data.items()}


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
    fig.subplots_adjust(top=0.80, left=0.075, right=0.97, bottom=0.17)


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


def bar_chart(fname, title, subtitle, labels, values, val_labels,
              ylabel, ymax, ref_line=None, ref_label=None):
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
        if ref_label:
            ax.text(len(values) - 0.48, ref_line + ymax * 0.015, ref_label,
                    ha="right", fontsize=8.5, color=MUTED)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10, color=INK2)
    titles(fig, title, subtitle)
    fig.savefig(RESULTS / fname, facecolor=SURFACE)
    plt.close(fig)


def main():
    d = load()
    t = {k: v[0] for k, v in d.items()}  # medianas
    base = t[("v0", 1)]

    versions = [("v0", 1), ("v1", 1), ("v3", 8), ("v2", 8)]
    labels = [
        "v0\nbaseline ijk\nsequencial",
        "v1\n+ omp simd\n(laço interno)",
        "v3\nparallel for interno\n8 threads",
        "v2\nparallel for externo\n8 threads",
    ]

    # 1. tempo por versao
    times = [t[v] for v in versions]
    bar_chart(
        "tempo_por_versao.png",
        "Tempo de execução por versão",
        "Matmul N=1024, dupla precisão — mediana de 5 repetições (desvio < 3%) — menor é melhor",
        labels, times, [fmt(v, 2, " s") for v in times],
        "Tempo (s)", max(times) * 1.18,
    )

    # 2. speedup por versao
    sp = [base / t[v] for v in versions]
    bar_chart(
        "speedup_por_versao.png",
        "Speedup por versão (vs. baseline v0)",
        "Matmul N=1024 — maior é melhor; linha tracejada marca o baseline (1×)",
        labels, sp, [fmt(v, 2, "×") for v in sp],
        "Speedup", max(sp) * 1.18,
        ref_line=1.0,
    )

    # 3. escalabilidade v2: medido vs ideal
    threads = [1, 2, 4, 8]
    measured = [base / t[("v2", p)] for p in threads]
    fig, ax = new_fig()
    ax.plot(threads, threads, color=MUTED, linewidth=1.6,
            linestyle=(0, (4, 3)), zorder=2, label="ideal (linear)")
    ax.plot(threads, measured, color=BLUE, linewidth=2.2, marker="o",
            markersize=7.5, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=3, label="v2 medido")
    for p, s in zip(threads, measured):
        dx, dy, align = ((10, -4, "left") if p == 1 else (-2, 9, "right"))
        ax.annotate(fmt(s, 1, "×"), (p, s), textcoords="offset points",
                    xytext=(dx, dy), ha=align, fontsize=9.5,
                    fontweight="semibold", color=INK)
    ax.set_xlim(0.6, 8.4)
    ax.set_ylim(0, 13)
    ax.set_xticks(threads)
    ax.set_xlabel("Número de threads", fontsize=10, color=INK2)
    ax.set_ylabel("Speedup vs. v0", fontsize=10, color=INK2)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5,
              labelcolor=INK2)
    titles(fig, "Escalabilidade da v2: speedup vs. número de threads",
           "Matmul N=1024 — pontos acima da linha ideal indicam "
           "speedup superlinear (ver discussão)")
    fig.savefig(RESULTS / "escalabilidade_v2.png", facecolor=SURFACE)
    plt.close(fig)

    # 4. eficiencia paralela v2 (vs v2 com 1 thread)
    v2_1 = t[("v2", 1)]
    eff = [(v2_1 / t[("v2", p)]) / p * 100 for p in threads]
    bar_chart(
        "eficiencia_v2.png",
        "Eficiência paralela da v2",
        "eficiência(p) = speedup(p) / p, relativa à v2 com 1 thread — "
        "acima de 100% = superlinear",
        [f"{p} thread{'s' if p > 1 else ''}" for p in threads],
        eff, [fmt(v, 0, "%") for v in eff],
        "Eficiência (%)", max(eff) * 1.22,
        ref_line=100,
    )

    # tabela-resumo no stdout (para o relatorio)
    print("versao,threads,mediana_s,desvio_s,speedup_vs_v0,gflops")
    flops = 2 * 1024**3
    for (v, p), (med, sd) in sorted(d.items()):
        print(f"{v},{p},{med:.4f},{sd:.4f},{base/med:.3f},{flops/med/1e9:.3f}")


if __name__ == "__main__":
    main()
