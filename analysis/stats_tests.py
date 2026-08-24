"""Парные t-тесты H0/H1 с поправкой Holm-Bonferroni, results/tables/stats_tests*.csv."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def holm_bonferroni(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    reject_sorted = np.zeros(m, dtype=bool)

    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)

        if pvals[idx] < threshold:
            reject_sorted[idx] = True
        else:
            break

    return reject_sorted


def h1_kp_test():
    kp = pd.read_csv(TABLES / "kp_ablation.csv")
    rows = []

    for mode in (1, 2):
        dm = kp[kp["mode"] == mode]

        for p in sorted(dm.p.unique()):
            sub = dm[np.isclose(dm.p, p)]
            piv = sub.pivot(index="seed", columns="k", values="accuracy")
            t, pv = stats.ttest_rel(piv[1], piv[8])
            diff = (piv[1] - piv[8]).mean()
            rows.append(
                {
                    "mode": mode,
                    "p": p,
                    "n_seeds": len(piv),
                    "mean_diff_k1_minus_k8": diff,
                    "t": t,
                    "p_raw": pv,
                }
            )

    df = pd.DataFrame(rows)
    df["reject_holm_within_mode"] = False

    for mode in (1, 2):
        mask = df["mode"] == mode
        df.loc[mask, "reject_holm_within_mode"] = holm_bonferroni(df.loc[mask, "p_raw"].to_numpy())

    df.to_csv(TABLES / "stats_tests_h1_kp.csv", index=False)
    print("\n--- H1 (k=1 vs k=8, paired, within-mode Holm-Bonferroni) ---")
    print(df.to_string(index=False))

    return df


def main():
    main_df = pd.read_csv(TABLES / "main_matrix.csv")
    d = main_df[main_df.N_mem == 40]
    rows = []

    for mode in (1, 2):
        dm = d[d["mode"] == mode]

        for p in sorted(dm.p.unique()):
            sub = dm[np.isclose(dm.p, p)]
            piv = sub.pivot(index="seed", columns="method", values="accuracy")

            for other in ("point", "concat"):
                t, pv = stats.ttest_rel(piv["chemotaxis"], piv[other])
                diff = (piv["chemotaxis"] - piv[other]).mean()
                rows.append(
                    {
                        "mode": mode,
                        "p": p,
                        "comparison": f"chemotaxis_vs_{other}",
                        "n_seeds": len(piv),
                        "mean_diff": diff,
                        "t": t,
                        "p_raw": pv,
                    }
                )

    df = pd.DataFrame(rows)
    df["reject_holm_within_mode"] = False

    for mode in (1, 2):
        mask = df["mode"] == mode
        df.loc[mask, "reject_holm_within_mode"] = holm_bonferroni(df.loc[mask, "p_raw"].to_numpy())

    df["reject_uncorrected_p05"] = df["p_raw"] < 0.05
    df.to_csv(TABLES / "stats_tests.csv", index=False)

    print(df.to_string(index=False))
    print()

    n_uncorrected = df["reject_uncorrected_p05"].sum()
    n_corrected = df["reject_holm_within_mode"].sum()
    print(f"Uncorrected (raw p<0.05): {n_uncorrected}/{len(df)} tests reach nominal significance.")
    print(
        f"Holm-Bonferroni (within mode, family size = {int((df['mode'] == 2).sum())} for mode 2): "
        f"{n_corrected}/{len(df)} survive correction."
    )

    h1_kp_test()


if __name__ == "__main__":
    main()
