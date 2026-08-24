"""Рендерит results/plots/schematic.png."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PLOTS = Path(__file__).resolve().parents[1] / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

INK = "#0b0b0b"
BOX = "#f5f4f0"
EDGE = "#52514e"
ACCENT = {"point": "#eb6834", "concat": "#1baf7a", "chemotaxis": "#eda100"}


def box(ax, xy, w, h, text, fc=BOX, ec=EDGE, fontsize=10.5, weight="normal"):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            fc=fc,
            ec=ec,
            linewidth=1.3,
            mutation_aspect=1,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        weight=weight,
        linespacing=1.4,
    )


def arrow(ax, p0, p1, color=EDGE):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(10, 5.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 5.4)
    ax.axis("off")

    box(
        ax,
        (0.3, 3.9),
        2.3,
        1.0,
        "environment\ntopic + style\n$\\rightarrow$ text $s_t$",
        weight="bold",
    )
    box(ax, (3.1, 3.9), 2.3, 1.0, "memory\nFIFO, last $N$\n$(s_j, a_j, r_j)$")
    box(ax, (5.9, 3.9), 2.6, 1.0, "query at step $t$\n$s_t$, anchor $s_{t-k}$")

    arrow(ax, (2.6, 4.4), (3.1, 4.4))
    arrow(ax, (5.4, 4.4), (5.9, 4.4))

    labels = [
        (
            "point",
            "$\\mathrm{embed}(s_t)$\nvs. $\\mathrm{embed}(s_j)$",
            ACCENT["point"],
        ),
        (
            "concat",
            "$[\\mathrm{embed}(s_t);\\,\\mathrm{embed}(s_{t-k})]$\n"
            "vs. $[\\mathrm{embed}(s_j);\\,\\mathrm{embed}(s_{j-k})]$",
            ACCENT["concat"],
        ),
        (
            "chemotaxis",
            "$\\mathrm{embed}(s_t)-\\mathrm{embed}(s_{t-k})$\nvs. $\\mathrm{embed}(s_j)-\\mathrm{embed}(s_{j-k})$",
            ACCENT["chemotaxis"],
        ),
    ]
    x0 = 0.6
    for i, (name, formula, color) in enumerate(labels):
        x = x0 + i * 3.1
        box(ax, (x, 2.2), 2.7, 1.15, formula, fc="#fff", ec=color, fontsize=9.5)
        arrow(ax, (7.2, 3.9), (x + 1.35, 3.35), color=EDGE)
        arrow(ax, (x + 1.35, 2.2), (5.0, 1.5), color=color)

    box(ax, (3.4, 0.85), 3.2, 0.65, "top-$k$ by cosine similarity", fc="#fff")
    box(
        ax,
        (3.1, 0.0),
        3.8,
        0.65,
        "vote: $w_j = \\mathrm{sim}_j \\cdot (2r_j-1)$, sum by action, argmax",
        fc="#fff",
        fontsize=9,
    )

    arrow(ax, (5.0, 0.85), (5.0, 0.65))

    fig.tight_layout()
    fig.savefig(PLOTS / "schematic.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
