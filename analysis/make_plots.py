"""Рендерит графики report из results/tables/ в results/plots/ и сводные CSV."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

COLOR = {
    "recency": "#2a78d6",
    "point": "#eb6834",
    "concat": "#1baf7a",
    "chemotaxis": "#eda100",
}
METHOD_LABEL = {
    "recency": "Recency",
    "point": "Point cosine",
    "concat": "Concat cosine",
    "chemotaxis": "Chemotaxis",
}
METHOD_ORDER = ["recency", "point", "concat", "chemotaxis"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

N_ACTIONS = 6
CHANCE = 1.0 / N_ACTIONS


def ci95(s: pd.Series) -> float:
    n = s.count()
    return 1.96 * s.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0


def agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols)["accuracy"]
    out = g.agg(mean="mean", ci95=ci95, n="count").reset_index()
    return out


def plot_mode_and_stability(main: pd.DataFrame):
    d = main[main["N_mem"] == 40]

    g1 = agg(d, ["mode", "method"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(2)
    width = 0.19

    for i, method in enumerate(METHOD_ORDER):
        sub = g1[g1.method == method].set_index("mode").reindex([1, 2])
        ax.bar(
            x + (i - 1.5) * width,
            sub["mean"],
            width,
            yerr=sub["ci95"],
            color=COLOR[method],
            label=METHOD_LABEL[method],
            capsize=3,
            error_kw={"ecolor": INK2, "elinewidth": 1},
        )

    ax.axhline(CHANCE, color=MUTED, linestyle="--", linewidth=1)
    ax.text(1.55, CHANCE + 0.01, "chance (1/6)", color=MUTED, fontsize=9)
    ax.set_xticks(x, ["Mode 1\n(similarity = usefulness)", "Mode 2\n(similarity broken)"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Graph 1 — Accuracy by method and regime (N=40, averaged over p)")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(PLOTS / "graph1_mode_comparison.png", dpi=170)
    plt.close(fig)

    g2 = agg(d, ["mode", "p", "method"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)

    for ax, mode in zip(axes, (1, 2)):
        sub_mode = g2[g2["mode"] == mode]

        for method in METHOD_ORDER:
            sub = sub_mode[sub_mode.method == method].sort_values("p")
            ax.plot(
                sub.p,
                sub["mean"],
                color=COLOR[method],
                linewidth=2,
                marker="o",
                markersize=5,
                label=METHOD_LABEL[method],
            )
            ax.fill_between(
                sub.p,
                sub["mean"] - sub.ci95,
                sub["mean"] + sub.ci95,
                color=COLOR[method],
                alpha=0.15,
                linewidth=0,
            )

        ax.axhline(CHANCE, color=MUTED, linestyle="--", linewidth=1)
        ax.text(0.02, CHANCE + 0.02, "chance (1/6)", color=MUTED, fontsize=9)
        ax.set_xlabel("Smoothness parameter p (jump probability)")
        ax.set_title(f"Mode {mode}")
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel("Accuracy")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=4)
    fig.suptitle("Graph 2 (main) — Stability curve: accuracy vs p, by mode and method (N=40)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS / "graph2_stability_curve.png", dpi=170)
    plt.close(fig)

    return g1, g2


def plot_budget_curve(main: pd.DataFrame):
    d = main[np.isclose(main["p"], 0.5)]
    g3 = agg(d, ["mode", "N_mem", "method"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ylims = {1: (0, 1.0), 2: (0, 0.5)}

    for ax, mode in zip(axes, (1, 2)):
        sub_mode = g3[g3["mode"] == mode]

        for method in METHOD_ORDER:
            sub = sub_mode[sub_mode.method == method].sort_values("N_mem")
            ax.plot(
                sub.N_mem,
                sub["mean"],
                color=COLOR[method],
                linewidth=2,
                marker="o",
                markersize=5,
                label=METHOD_LABEL[method],
            )
            ax.fill_between(
                sub.N_mem,
                sub["mean"] - sub.ci95,
                sub["mean"] + sub.ci95,
                color=COLOR[method],
                alpha=0.15,
                linewidth=0,
            )

        ax.axhline(CHANCE, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(sorted(d.N_mem.unique()))
        ax.set_xlabel("Memory budget N")
        ax.set_title(f"Mode {mode}")
        ax.set_ylim(*ylims[mode])

    axes[0].set_ylabel("Accuracy")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=4)
    fig.suptitle("Graph 3 — Accuracy vs memory budget N (p=0.5)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS / "graph3_budget_curve.png", dpi=170)
    plt.close(fig)

    return g3


def plot_kp_heatmap(kp: pd.DataFrame):
    g4 = kp.groupby(["mode", "p", "k"])["accuracy"].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ks = sorted(g4.k.unique())
    ps = sorted(g4.p.unique())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)

    for ax, mode in zip(axes, (1, 2)):
        M = np.zeros((len(ks), len(ps)))

        for i, k in enumerate(ks):
            for j, p in enumerate(ps):
                row = g4[(g4["mode"] == mode) & (g4.k == k) & (np.isclose(g4.p, p))]
                M[i, j] = row.accuracy.values[0] if len(row) else np.nan

        vmin, vmax = np.nanmin(M), np.nanmax(M)
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(ps)), [f"{p:g}" for p in ps])
        ax.set_yticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlabel("p")
        ax.set_ylabel("window k")
        ax.set_title(f"Mode {mode}")
        ax.grid(False)

        for i in range(len(ks)):
            for j in range(len(ps)):
                frac = (M[i, j] - vmin) / (vmax - vmin + 1e-9)
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=INK if frac < 0.6 else "white",
                    fontsize=9,
                )

        for j in range(len(ps)):
            best_i = np.nanargmax(M[:, j])
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, best_i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=INK,
                    linewidth=2.2,
                    zorder=5,
                )
            )

        fig.colorbar(im, ax=ax, shrink=0.85, label="Chemotaxis accuracy")

    fig.suptitle("Graph 4 — k x p ablation (chemotaxis); outlined cell = best k per column")
    fig.savefig(PLOTS / "graph4_kp_heatmap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    return g4


def plot_sanity_checks(sanity: pd.DataFrame):
    g5 = agg(sanity, ["control", "method"])
    controls = ["none", "shuffled_r", "shuffled_d"]
    control_label = {
        "none": "No control\n(real data)",
        "shuffled_r": "Control 1\nshuffled r",
        "shuffled_d": "Control 2\nshuffled d",
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(controls))
    width = 0.19

    for i, method in enumerate(METHOD_ORDER):
        sub = g5[g5.method == method].set_index("control").reindex(controls)
        ax.bar(
            x + (i - 1.5) * width,
            sub["mean"],
            width,
            yerr=sub["ci95"],
            color=COLOR[method],
            label=METHOD_LABEL[method],
            capsize=3,
            error_kw={"ecolor": INK2, "elinewidth": 1},
        )

    ax.axhline(CHANCE, color=MUTED, linestyle="--", linewidth=1)
    ax.text(2.35, CHANCE + 0.005, "chance (1/6)", color=MUTED, fontsize=9)
    ax.set_xticks(x, [control_label[c] for c in controls])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.12, 0.28)
    ax.set_title("Graph 5 — Sanity checks (mode 2, N=40, averaged over p in {0.3, 0.6, 0.9})")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(PLOTS / "graph5_sanity_checks.png", dpi=170)
    plt.close(fig)

    return g5


def plot_overlap(overlap: pd.DataFrame):
    g = agg(overlap.rename(columns={"mean_overlap": "accuracy"}), ["mode", "p"])
    fig, ax = plt.subplots(figsize=(7, 4.6))

    for mode, color in ((1, "#4a3aa7"), (2, "#e34948")):
        sub = g[g["mode"] == mode].sort_values("p")
        ax.plot(
            sub.p,
            sub["mean"],
            color=color,
            linewidth=2,
            marker="o",
            label=f"Mode {mode}",
        )
        ax.fill_between(
            sub.p,
            sub["mean"] - sub.ci95,
            sub["mean"] + sub.ci95,
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel("p")
    ax.set_ylabel("Mean top-k overlap (chemotaxis vs concat cosine)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Retrieval-set overlap: chemotaxis vs concat cosine")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "overlap_chemotaxis_vs_concat.png", dpi=170)
    plt.close(fig)

    return g


def plot_tier2(tier2: pd.DataFrame, main: pd.DataFrame):
    g = agg(tier2, ["mode", "p", "method"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)

    for ax, mode in zip(axes, (1, 2)):
        sub_mode = g[g["mode"] == mode]

        for method in METHOD_ORDER:
            sub = sub_mode[sub_mode.method == method].sort_values("p")
            ax.plot(
                sub.p,
                sub["mean"],
                color=COLOR[method],
                linewidth=2,
                marker="o",
                markersize=5,
                label=METHOD_LABEL[method],
            )
            ax.fill_between(
                sub.p,
                sub["mean"] - sub.ci95,
                sub["mean"] + sub.ci95,
                color=COLOR[method],
                alpha=0.15,
                linewidth=0,
            )

        ax.axhline(CHANCE, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xlabel("p")
        ax.set_title(f"Mode {mode}")
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel("Accuracy")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=4)
    fig.suptitle("Tier 2 — LLM agent (Qwen2.5-0.5B-Instruct) stability curve, N=40")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS / "tier2_stability_curve.png", dpi=170)
    plt.close(fig)

    t1 = agg(main[main.N_mem == 40], ["mode", "p", "method"]).rename(columns={"mean": "tier1"})
    t2 = g.rename(columns={"mean": "tier2"})
    merged = pd.merge(
        t1[["mode", "p", "method", "tier1"]],
        t2[["mode", "p", "method", "tier2"]],
        on=["mode", "p", "method"],
        how="inner",
    )
    fig, ax = plt.subplots(figsize=(5.6, 5.6))

    for method in METHOD_ORDER:
        sub = merged[merged.method == method]
        ax.scatter(sub.tier1, sub.tier2, color=COLOR[method], label=METHOD_LABEL[method], s=45)

    lims = [0, 1]
    ax.plot(lims, lims, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Tier 1 (kNN-vote) accuracy")
    ax.set_ylabel("Tier 2 (LLM agent) accuracy")
    corr = np.corrcoef(merged.tier1, merged.tier2)[0, 1]
    ax.set_title(f"Tier 1 vs Tier 2 agreement (r={corr:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "tier1_vs_tier2_agreement.png", dpi=170)
    plt.close(fig)

    return g, merged, corr


def plot_real_data_recall(df: pd.DataFrame, title: str, filename: str, ylim: float, all_label: str):
    order = ["20", "50", "100", "all"]
    g = df.groupby(["N_mem", "method"])["recall"].agg(mean="mean", ci95=ci95, n="count").reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(order))

    for method in METHOD_ORDER:
        sub = g[g.method == method].set_index("N_mem").reindex(order)
        ax.plot(
            x,
            sub["mean"],
            color=COLOR[method],
            linewidth=2,
            marker="o",
            markersize=6,
            label=METHOD_LABEL[method],
        )
        ax.fill_between(
            x,
            sub["mean"] - sub.ci95,
            sub["mean"] + sub.ci95,
            color=COLOR[method],
            alpha=0.15,
            linewidth=0,
        )

    ax.set_xticks(x, ["20", "50", "100", all_label])
    ax.set_xlabel("Memory budget N (turns eligible for retrieval)")
    ax.set_ylabel("Recall@5 (fraction of gold evidence turns retrieved)")
    ax.set_ylim(0, ylim)
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / filename, dpi=170)
    plt.close(fig)

    return g


def plot_relational(df: pd.DataFrame):
    sub = df[~df.has_exact_pair]
    alphas = sorted(float(r.split("=")[1]) for r in sub.regime.unique() if r.startswith("alpha="))
    g = sub[sub.regime != "minilm"].copy()
    g["alpha"] = g.regime.str.split("=").str[1].astype(float)
    g = g.groupby(["alpha", "method"])["recall"].agg(mean="mean", ci95=ci95).reset_index()
    minilm = sub[sub.regime == "minilm"].groupby("method")["recall"].mean()

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for method in METHOD_ORDER:
        m = g[g.method == method].set_index("alpha").reindex(alphas)
        ax.plot(
            alphas,
            m["mean"],
            color=COLOR[method],
            linewidth=2,
            marker="o",
            markersize=6,
            label=METHOD_LABEL[method],
        )
        ax.fill_between(
            alphas,
            m["mean"] - m.ci95,
            m["mean"] + m.ci95,
            color=COLOR[method],
            alpha=0.15,
            linewidth=0,
        )
        ax.axhline(minilm[method], color=COLOR[method], linewidth=1, linestyle=":", alpha=0.6)

    ax.set_xlabel(r"$\alpha$ (gap-signal strength relative to topic-identity noise)")
    ax.set_ylabel("Recall@5 (same-gap, different-pair queries)")
    ax.set_title("Mode 3: retrieval recall vs. gap-signal strength\n(dotted = real MiniLM embeddings, no alpha)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "relational_recall.png", dpi=170)
    plt.close(fig)

    return g


def main():
    main_df = pd.read_csv(TABLES / "main_matrix.csv")
    kp_df = pd.read_csv(TABLES / "kp_ablation.csv")
    sanity_df = pd.read_csv(TABLES / "sanity_checks.csv")
    overlap_df = pd.read_csv(TABLES / "overlap.csv")

    g1, g2 = plot_mode_and_stability(main_df)
    g3 = plot_budget_curve(main_df)
    g4 = plot_kp_heatmap(kp_df)
    g5 = plot_sanity_checks(sanity_df)
    g_ov = plot_overlap(overlap_df)

    g1.to_csv(TABLES / "summary_graph1_mode.csv", index=False)
    g2.to_csv(TABLES / "summary_graph2_stability.csv", index=False)
    g3.to_csv(TABLES / "summary_graph3_budget.csv", index=False)
    g4.to_csv(TABLES / "summary_graph4_kp.csv", index=False)
    g5.to_csv(TABLES / "summary_graph5_sanity.csv", index=False)
    g_ov.to_csv(TABLES / "summary_overlap.csv", index=False)

    tier2_path = TABLES / "tier2_matrix.csv"

    if tier2_path.exists():
        tier2_df = pd.read_csv(tier2_path)
        g_t2, merged, corr = plot_tier2(tier2_df, main_df)
        g_t2.to_csv(TABLES / "summary_tier2.csv", index=False)
        merged.to_csv(TABLES / "summary_tier1_vs_tier2.csv", index=False)
        print(f"Tier1-Tier2 accuracy correlation: r={corr:.3f}")

    locomo_path = TABLES / "locomo_recall.csv"

    if locomo_path.exists():
        locomo_df = pd.read_csv(locomo_path)
        g_loc = plot_real_data_recall(
            locomo_df,
            "LoCoMo — real dialogue, avg. over k, all 10 conversations",
            "locomo_recall.png",
            ylim=0.45,
            all_label="all (~400+)",
        )
        g_loc.to_csv(TABLES / "summary_locomo.csv", index=False)

    lme_path = TABLES / "longmemeval_recall.csv"

    if lme_path.exists():
        lme_df = pd.read_csv(lme_path)
        g_lme = plot_real_data_recall(
            lme_df,
            "LongMemEval-S — real dialogue, avg. over k, 470 questions",
            "longmemeval_recall.png",
            ylim=0.45,
            all_label="all (~500)",
        )
        g_lme.to_csv(TABLES / "summary_longmemeval.csv", index=False)

    rel_path = TABLES / "relational_recall.csv"

    if rel_path.exists():
        rel_df = pd.read_csv(rel_path)
        g_rel = plot_relational(rel_df)
        g_rel.to_csv(TABLES / "summary_relational.csv", index=False)

    print("plots written to", PLOTS)


if __name__ == "__main__":
    main()
