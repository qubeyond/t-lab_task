"""LearnedScorer обучен на mode 3 при фиксированной alpha, results/tables/learned_scorer_relational.csv."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

import experiments.config as C
from env.synthetic_embed import synthetic_embed
from env.trajectory import TrajectoryConfig, generate_trajectory
from experiments.train_learned_scorer import holm_bonferroni, mcnemar, run_epoch
from memory.learned import LearnedScorer, extract_features

RESULTS = Path(__file__).resolve().parents[1] / "results" / "tables"
RESULTS.mkdir(parents=True, exist_ok=True)

MODE = 3
ALPHA = 1.0
SEEDS_TRAIN = tuple(range(40))
SEEDS_VAL = tuple(range(40, 50))
SEEDS_TEST = tuple(range(50, 60))
N_MEM = C.N_MEM_MAIN
K = C.K_DEFAULT
N_EPOCHS = 60
LR = 0.02


def build_dataset(seeds: tuple[int, ...]) -> dict[str, torch.Tensor]:
    feats_list, actions_list, targets = [], [], []

    for seed in seeds:
        cfg = TrajectoryConfig(
            mode=MODE,
            p=C.BUDGET_P,
            seed=seed,
            n_steps=C.N_STEPS_TIER1,
            eps_explore=C.EPS_EXPLORE,
        )
        traj = generate_trajectory(cfg)
        E = synthetic_embed(traj.topics, traj.styles, alpha=ALPHA, seed=seed)
        end = min(C.BURN_IN + C.N_EVAL_TIER1, len(traj.texts))
        checkpoints = [t for t in range(C.BURN_IN, end) if traj.is_entry[t]]

        for t in checkpoints:
            mem_idxs = np.arange(t - N_MEM, t)
            feats_list.append(extract_features(t, E, mem_idxs, K, traj.correct))
            actions_list.append(traj.a_behavior[mem_idxs])
            targets.append(traj.a_star[t])

    return {
        "features": torch.from_numpy(np.stack(feats_list)),
        "actions": torch.from_numpy(np.stack(actions_list).astype(np.int64)),
        "targets": torch.tensor(targets, dtype=torch.long),
    }


def train_variant(name, feature_mask, train_d, val_d, test_d):
    torch.manual_seed(0)
    model = LearnedScorer(feature_mask=feature_mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_acc, best_state = -1.0, None

    for epoch in range(N_EPOCHS):
        run_epoch(model, train_d, optimizer)
        _, val_acc = run_epoch(model, val_d)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, test_acc, test_preds = run_epoch(model, test_d, return_preds=True)
    print(f"[{name}] test accuracy = {test_acc:.3f}")

    return test_acc, test_preds


def main():
    train_d = build_dataset(SEEDS_TRAIN)
    val_d = build_dataset(SEEDS_VAL)
    test_d = build_dataset(SEEDS_TEST)
    print(f"mode=3 alpha={ALPHA}  train={len(train_d['targets'])}  test={len(test_d['targets'])}")

    variants = {
        "full": (True, True, True, True, True),
        "no_chemotaxis": (True, True, False, True, True),
        "chemotaxis_only": (False, False, True, True, True),
    }
    rows, preds = [], {}

    for name, mask in variants.items():
        acc, p = train_variant(name, mask, train_d, val_d, test_d)
        rows.append({"variant": name, "test_acc": acc})
        preds[name] = p

    targets = test_d["targets"]
    correct = {n: (p == targets) for n, p in preds.items()}
    pairs = [("full", "no_chemotaxis"), ("chemotaxis_only", "no_chemotaxis")]
    results = [mcnemar(correct[a], correct[b]) for a, b in pairs]
    sig = holm_bonferroni([p for _, _, p in results])

    for (a, b), (b01, b10, pval), reject in zip(pairs, results, sig):
        print(f"McNemar {a} vs {b}: {a}-only-right={b01}  {b}-only-right={b10}  p={pval:.2e}  significant={reject}")
        rows.append(
            {
                "variant": f"mcnemar_{a}_vs_{b}",
                "test_acc": np.nan,
                "b01": b01,
                "b10": b10,
                "p": pval,
                "holm_significant": reject,
            }
        )

    pd.DataFrame(rows).to_csv(RESULTS / "learned_scorer_relational.csv", index=False)


if __name__ == "__main__":
    main()
