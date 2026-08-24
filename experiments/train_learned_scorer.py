"""Обучает LearnedScorer на mode 2, три варианта feature_mask, results/tables/learned_scorer.csv."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn

import experiments.config as C
from env.trajectory import TrajectoryConfig, generate_trajectory
from memory.embeddings import encode_texts
from memory.learned import FEATURE_NAMES, LearnedScorer, extract_features
from memory.methods import METHODS

RESULTS = Path(__file__).resolve().parents[1] / "results" / "tables"
RESULTS.mkdir(parents=True, exist_ok=True)

MODE = 2
P_VALUES = (0.15, 0.3, 0.5, 0.65, 0.8, 0.9, 1.0)
SEEDS_TRAIN = tuple(range(40))
SEEDS_VAL = tuple(range(40, 50))
SEEDS_TEST = tuple(range(50, 60))
N_MEM = C.N_MEM_MAIN
K = C.K_DEFAULT
N_EPOCHS = 60
LR = 0.02
BATCH_SIZE = 1024


def build_dataset(seeds: tuple[int, ...]) -> dict[str, torch.Tensor]:
    feats_list, actions_list, targets = [], [], []

    for seed in seeds:
        for p in P_VALUES:
            cfg = TrajectoryConfig(
                mode=MODE,
                p=p,
                seed=seed,
                n_steps=C.N_STEPS_TIER1,
                eps_explore=C.EPS_EXPLORE,
            )
            traj = generate_trajectory(cfg)
            E = encode_texts(traj.texts, model_name=C.ENCODER_MODEL)
            checkpoints = np.arange(C.BURN_IN, min(C.BURN_IN + C.N_EVAL_TIER1, len(traj.texts)))

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


def run_epoch(
    model: LearnedScorer,
    data: dict,
    optimizer=None,
    batch_size: int = BATCH_SIZE,
    return_preds: bool = False,
):
    train = optimizer is not None
    model.train(train)
    n = len(data["targets"])
    idx = torch.randperm(n) if train else torch.arange(n)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct = 0.0, 0
    preds = torch.empty(n, dtype=torch.long) if return_preds else None
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            logits = model.action_logits(data["features"][b], data["actions"][b], C.N_ACTIONS)
            target = data["targets"][b]
            loss = loss_fn(logits, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(b)
            batch_pred = logits.argmax(dim=-1)
            correct += (batch_pred == target).sum().item()

            if return_preds:
                preds[b] = batch_pred

    if return_preds:
        return total_loss / n, correct / n, preds

    return total_loss / n, correct / n


def train_variant(
    name: str, feature_mask: tuple[bool, ...], train_d, val_d, test_d
) -> tuple[list[dict], float, torch.Tensor]:
    torch.manual_seed(0)
    model = LearnedScorer(feature_mask=feature_mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rows = []
    best_val_acc, best_state = -1.0, None

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_d, optimizer)
        val_loss, val_acc = run_epoch(model, val_d)
        rows.append(
            {
                "variant": name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            print(f"[{name}] epoch {epoch:2d}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    model.load_state_dict(best_state)
    test_loss, test_acc, test_preds = run_epoch(model, test_d, return_preds=True)
    rows.append(
        {
            "variant": name,
            "epoch": "test",
            "train_loss": np.nan,
            "train_acc": np.nan,
            "val_loss": test_loss,
            "val_acc": test_acc,
        }
    )
    print(f"[{name}] TEST accuracy (best-val checkpoint) = {test_acc:.3f}")

    w = model.net[0].weight.detach().numpy()
    col_norms = np.linalg.norm(w, axis=0)

    for fname, norm in zip(FEATURE_NAMES, col_norms):
        print(f"    input weight norm [{fname}] = {norm:.3f}")

    print()

    return rows, test_acc, test_preds


def mcnemar(correct_a: torch.Tensor, correct_b: torch.Tensor) -> tuple[int, int, float]:
    from scipy.stats import chi2

    b01 = int(((correct_a) & (~correct_b)).sum())
    b10 = int(((~correct_a) & (correct_b)).sum())

    if b01 + b10 == 0:
        return b01, b10, 1.0

    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p = 1 - chi2.cdf(stat, df=1)

    return b01, b10, p


def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(pvals)
    m = len(pvals)
    reject = [False] * m

    for rank, idx in enumerate(order):
        if pvals[idx] > alpha / (m - rank):
            break
        reject[idx] = True

    return reject


def geometric_baseline_on_test(test_d: dict) -> dict[str, float]:
    feats = test_d["features"].numpy()
    actions = test_d["actions"].numpy()
    targets = test_d["targets"].numpy()
    n = feats.shape[0]
    r01 = ((feats[:, :, 3] + 1) / 2).round().astype(np.int64)
    correct = dict.fromkeys(METHODS, 0)
    col_of = {"point": 0, "concat": 1, "chemotaxis": 2}

    for i in range(n):
        for method, col in col_of.items():
            sims = feats[i, :, col]
            order = np.argsort(-sims)[: C.TOPK]
            weights = sims[order] * (2 * r01[i, order] - 1)
            scores = np.zeros(C.N_ACTIONS)
            np.add.at(scores, actions[i, order], weights)
            pred = (
                int(scores.argmax())
                if scores.max() > 0
                else int(np.bincount(actions[i], minlength=C.N_ACTIONS).argmax())
            )
            correct[method] += int(pred == targets[i])

        order = np.arange(N_MEM)[-C.TOPK :]
        weights = feats[i, order, 3]
        scores = np.zeros(C.N_ACTIONS)
        np.add.at(scores, actions[i, order], weights)
        pred = (
            int(scores.argmax()) if scores.max() > 0 else int(np.bincount(actions[i], minlength=C.N_ACTIONS).argmax())
        )
        correct["recency"] += int(pred == targets[i])

    return {m: c / n for m, c in correct.items()}


def main():
    print("building datasets (train/val/test, disjoint seeds) ...")
    train_d = build_dataset(SEEDS_TRAIN)
    val_d = build_dataset(SEEDS_VAL)
    test_d = build_dataset(SEEDS_TEST)
    print(f"train={len(train_d['targets'])}  val={len(val_d['targets'])}  test={len(test_d['targets'])} episodes\n")

    variants = {
        "full": (True, True, True, True, True),
        "no_chemotaxis": (True, True, False, True, True),
        "chemotaxis_only": (False, False, True, True, True),
    }

    all_rows = []
    test_accs = {}
    test_preds = {}

    for name, mask in variants.items():
        rows, test_acc, preds = train_variant(name, mask, train_d, val_d, test_d)
        all_rows.extend(rows)
        test_accs[name] = test_acc
        test_preds[name] = preds

    targets = test_d["targets"]
    correct = {name: (p == targets) for name, p in test_preds.items()}
    pairs = [
        ("full", "no_chemotaxis"),
        ("full", "chemotaxis_only"),
        ("no_chemotaxis", "chemotaxis_only"),
    ]
    results = [mcnemar(correct[a], correct[b]) for a, b in pairs]
    sig = holm_bonferroni([p for _, _, p in results])

    print("Paired McNemar tests on the same 14000 test episodes (Holm-Bonferroni, family of 3):")

    for (a, b), (b01, b10, p), reject in zip(pairs, results, sig):
        print(f"    {a} vs {b}: {a}-only-right={b01}  {b}-only-right={b10}  p={p:.2e}  significant={reject}")
        all_rows.append(
            {
                "variant": f"mcnemar_{a}_vs_{b}",
                "epoch": "test",
                "train_loss": np.nan,
                "train_acc": b01,
                "val_loss": b10,
                "val_acc": p,
                "holm_significant": reject,
            }
        )

    print()

    geo = geometric_baseline_on_test(test_d)
    print("Geometric (hand-coded) baselines on the SAME test episodes:")

    for m, a in geo.items():
        print(f"    {m:12s} {a:.3f}")
        all_rows.append(
            {
                "variant": f"geometric_{m}",
                "epoch": "test",
                "train_loss": np.nan,
                "train_acc": np.nan,
                "val_loss": np.nan,
                "val_acc": a,
            }
        )

    print("\nLearned variants, test accuracy:")

    for name, acc in test_accs.items():
        print(f"    {name:16s} {acc:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "learned_scorer.csv", index=False)
    print(f"\nwritten to {RESULTS / 'learned_scorer.csv'}")


if __name__ == "__main__":
    main()
