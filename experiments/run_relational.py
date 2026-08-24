"""Mode 3 / H2: recall@k по gap темы, results/tables/relational_recall.csv и relational_gap_structure.csv."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

import experiments.config as C
from env.synthetic_embed import synthetic_embed
from env.trajectory import TrajectoryConfig, generate_trajectory
from memory.embeddings import encode_texts
from memory.methods import METHODS, rank_candidates

RESULTS = Path(__file__).resolve().parents[1] / C.RESULTS_DIR / "tables"
RESULTS.mkdir(parents=True, exist_ok=True)

MODE = 3
P_FIXED = C.BUDGET_P
ALPHA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
SEEDS = tuple(range(20))
N_MEM = C.N_MEM_MAIN
N_EVAL = 400
N_STEPS = C.BURN_IN + N_EVAL


def score(retrieved: np.ndarray, gold: set[int]) -> tuple[float, int]:
    hit_set = set(retrieved.tolist()) & gold
    recall = len(hit_set) / min(len(gold), C.TOPK)
    return recall, int(len(hit_set) > 0)


def run_recall() -> pd.DataFrame:
    rows = []

    for seed in tqdm(SEEDS, desc="relational"):
        cfg = TrajectoryConfig(mode=MODE, p=P_FIXED, seed=seed, n_steps=N_STEPS, eps_explore=C.EPS_EXPLORE)
        traj = generate_trajectory(cfg)
        E_minilm = encode_texts(traj.texts, model_name=C.ENCODER_MODEL)
        regimes = [("minilm", E_minilm)] + [
            (
                f"alpha={a}",
                synthetic_embed(traj.topics, traj.styles, alpha=a, seed=seed),
            )
            for a in ALPHA_GRID
        ]

        for t in range(C.BURN_IN, N_STEPS):
            if not traj.is_entry[t]:
                continue

            src, tgt = traj.topics[t - 1], traj.topics[t]
            gap = tgt - src
            mem_idxs = np.arange(t - N_MEM, t)
            entries = mem_idxs[traj.is_entry[mem_idxs]]
            entry_gaps = traj.topics[entries] - traj.topics[entries - 1]
            gold_gap = set(entries[entry_gaps == gap].tolist())

            if not gold_gap:
                continue

            has_exact_pair = any((traj.topics[j - 1], traj.topics[j]) == (src, tgt) for j in gold_gap)

            for regime, E in regimes:
                for method in METHODS:
                    retrieved, _ = rank_candidates(method, t, E, mem_idxs, k=C.K_DEFAULT, topk=C.TOPK)
                    recall, hit = score(retrieved, gold_gap)
                    rows.append(
                        {
                            "p": P_FIXED,
                            "seed": seed,
                            "t": t,
                            "has_exact_pair": has_exact_pair,
                            "n_gold": len(gold_gap),
                            "regime": regime,
                            "method": method,
                            "recall": recall,
                            "hit": hit,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "relational_recall.csv", index=False)

    return df


def run_gap_structure() -> pd.DataFrame:
    rows = []

    for seed in SEEDS[:8]:
        cfg = TrajectoryConfig(mode=MODE, p=P_FIXED, seed=seed, n_steps=N_STEPS, eps_explore=C.EPS_EXPLORE)
        traj = generate_trajectory(cfg)
        E_minilm = encode_texts(traj.texts, model_name=C.ENCODER_MODEL)
        regimes = [("minilm", E_minilm)] + [
            (
                f"alpha={a}",
                synthetic_embed(traj.topics, traj.styles, alpha=a, seed=seed),
            )
            for a in ALPHA_GRID
        ]

        entries = np.array([t for t in range(1, N_STEPS) if traj.is_entry[t]])
        gaps = traj.topics[entries] - traj.topics[entries - 1]
        rng = np.random.default_rng(seed)
        sample = rng.choice(len(entries), size=min(300, len(entries)), replace=False)

        for regime, E in regimes:
            diffs = E[entries] - E[entries - 1]
            diffs = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-8)

            for a in sample:
                for b in sample:
                    if a >= b:
                        continue

                    cos = float(diffs[a] @ diffs[b])
                    rows.append(
                        {
                            "seed": seed,
                            "regime": regime,
                            "same_gap": bool(gaps[a] == gaps[b]),
                            "cos": cos,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "relational_gap_structure.csv", index=False)

    return df


if __name__ == "__main__":
    print("== relational (mode 3) recall@k against same-gap gold ==")
    rec_df = run_recall()
    print(rec_df.groupby(["regime", "has_exact_pair", "method"])[["recall", "hit"]].mean().round(3))

    print("== gap structure diagnostic ==")
    gap_df = run_gap_structure()
    print(gap_df.groupby(["regime", "same_gap"])["cos"].mean().round(3))
