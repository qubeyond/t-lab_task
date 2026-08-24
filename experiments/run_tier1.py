"""Tier 1: полный факторный эксперимент с kNN-vote агентом, results/tables/*.csv."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

import experiments.config as C
from agents.knn_agent import KNNVoteAgent
from env.trajectory import TrajectoryConfig, generate_trajectory
from memory.embeddings import encode_texts
from memory.methods import METHODS, predict_action

RESULTS = Path(__file__).resolve().parents[1] / C.RESULTS_DIR / "tables"
RESULTS.mkdir(parents=True, exist_ok=True)


def eval_checkpoints(T: int, burn_in: int, n_eval: int) -> np.ndarray:
    end = min(burn_in + n_eval, T)
    return np.arange(burn_in, end)


def global_majority(actions: np.ndarray, n_actions: int) -> int:
    return int(np.bincount(actions, minlength=n_actions).argmax())


def run_main_matrix() -> pd.DataFrame:
    rows = []
    agents = {m: KNNVoteAgent(m, k=C.K_DEFAULT, topk=C.TOPK, n_actions=C.N_ACTIONS) for m in METHODS}
    combos = [(mode, p, seed) for mode in C.MODES for p in C.P_GRID_MAIN for seed in C.SEEDS_MAIN]

    for mode, p, seed in tqdm(combos, desc="main_matrix"):
        cfg = TrajectoryConfig(
            mode=mode,
            p=p,
            seed=seed,
            n_steps=C.N_STEPS_TIER1,
            eps_explore=C.EPS_EXPLORE,
        )
        traj = generate_trajectory(cfg)
        E = encode_texts(traj.texts, model_name=C.ENCODER_MODEL)
        gdef = global_majority(traj.a_behavior[: C.BURN_IN], C.N_ACTIONS)
        checkpoints = eval_checkpoints(len(traj.texts), C.BURN_IN, C.N_EVAL_TIER1)
        n_mem_list = C.N_MEM_GRID if p == C.BUDGET_P else (C.N_MEM_MAIN,)

        for N_mem in n_mem_list:
            correct = dict.fromkeys(METHODS, 0)

            for t in checkpoints:
                mem_idxs = np.arange(t - N_mem, t)

                for method in METHODS:
                    pred = agents[method].act(
                        t,
                        E,
                        traj.a_behavior,
                        traj.correct,
                        mem_idxs,
                        global_default=gdef,
                    )
                    correct[method] += int(pred == traj.a_star[t])

            n_eval = len(checkpoints)

            for method in METHODS:
                rows.append(
                    {
                        "mode": mode,
                        "p": p,
                        "seed": seed,
                        "N_mem": N_mem,
                        "method": method,
                        "n_eval": n_eval,
                        "n_correct": correct[method],
                        "accuracy": correct[method] / n_eval,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "main_matrix.csv", index=False)

    return df


def run_overlap_metric() -> pd.DataFrame:
    rows = []
    combos = [(mode, p, seed) for mode in C.MODES for p in C.P_GRID_MAIN for seed in C.SEEDS_MAIN[:5]]

    for mode, p, seed in tqdm(combos, desc="overlap"):
        cfg = TrajectoryConfig(
            mode=mode,
            p=p,
            seed=seed,
            n_steps=C.N_STEPS_TIER1,
            eps_explore=C.EPS_EXPLORE,
        )
        traj = generate_trajectory(cfg)
        E = encode_texts(traj.texts, model_name=C.ENCODER_MODEL)
        checkpoints = eval_checkpoints(len(traj.texts), C.BURN_IN, C.N_EVAL_TIER1)
        overlaps = []

        for t in checkpoints:
            mem_idxs = np.arange(t - C.N_MEM_MAIN, t)
            _, chem_set = predict_action(
                "chemotaxis",
                t,
                E,
                traj.a_behavior,
                traj.correct,
                mem_idxs,
                k=C.K_DEFAULT,
                topk=C.TOPK,
                n_actions=C.N_ACTIONS,
                return_retrieved=True,
            )
            _, concat_set = predict_action(
                "concat",
                t,
                E,
                traj.a_behavior,
                traj.correct,
                mem_idxs,
                k=C.K_DEFAULT,
                topk=C.TOPK,
                n_actions=C.N_ACTIONS,
                return_retrieved=True,
            )
            overlaps.append(len(set(chem_set.tolist()) & set(concat_set.tolist())) / C.TOPK)

        rows.append(
            {
                "mode": mode,
                "p": p,
                "seed": seed,
                "N_mem": C.N_MEM_MAIN,
                "mean_overlap": float(np.mean(overlaps)),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "overlap.csv", index=False)

    return df


def run_kp_ablation() -> pd.DataFrame:
    rows = []
    combos = [(mode, p, seed) for mode in C.MODES for p in C.P_GRID_ABLATION for seed in C.SEEDS_ABLATION]

    for mode, p, seed in tqdm(combos, desc="kp_ablation"):
        cfg = TrajectoryConfig(
            mode=mode,
            p=p,
            seed=seed,
            n_steps=C.N_STEPS_ABLATION,
            eps_explore=C.EPS_EXPLORE,
        )
        traj = generate_trajectory(cfg)
        E = encode_texts(traj.texts, model_name=C.ENCODER_MODEL)
        gdef = global_majority(traj.a_behavior[: C.BURN_IN], C.N_ACTIONS)
        checkpoints = eval_checkpoints(len(traj.texts), C.BURN_IN, C.N_EVAL_ABLATION)

        for k in C.K_GRID:
            correct = 0

            for t in checkpoints:
                mem_idxs = np.arange(t - C.N_MEM_ABLATION, t)
                pred = predict_action(
                    "chemotaxis",
                    t,
                    E,
                    traj.a_behavior,
                    traj.correct,
                    mem_idxs,
                    k=k,
                    topk=C.TOPK,
                    n_actions=C.N_ACTIONS,
                    global_default=gdef,
                )
                correct += int(pred == traj.a_star[t])

            n_eval = len(checkpoints)
            rows.append(
                {
                    "mode": mode,
                    "p": p,
                    "seed": seed,
                    "k": k,
                    "n_eval": n_eval,
                    "n_correct": correct,
                    "accuracy": correct / n_eval,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "kp_ablation.csv", index=False)

    return df


def run_sanity_checks() -> pd.DataFrame:
    rows = []
    combos = [(p, seed) for p in C.SANITY_P_VALUES for seed in C.SANITY_SEEDS]

    for p, seed in tqdm(combos, desc="sanity_checks"):
        cfg = TrajectoryConfig(
            mode=C.SANITY_MODE,
            p=p,
            seed=seed,
            n_steps=C.N_STEPS_TIER1,
            eps_explore=C.EPS_EXPLORE,
        )
        traj = generate_trajectory(cfg)
        E = encode_texts(traj.texts, model_name=C.ENCODER_MODEL)
        gdef = global_majority(traj.a_behavior[: C.BURN_IN], C.N_ACTIONS)
        checkpoints = eval_checkpoints(len(traj.texts), C.BURN_IN, C.N_EVAL_TIER1)
        rng = np.random.default_rng(1000 + seed)

        shuffled_correct = traj.correct.copy()
        rng.shuffle(shuffled_correct)

        all_idx = np.arange(len(traj.texts))
        shuffled_anchor = all_idx.copy()
        rng.shuffle(shuffled_anchor)

        for control in ("none", "shuffled_r", "shuffled_d"):
            correct_counts = dict.fromkeys(METHODS, 0)

            for t in checkpoints:
                mem_idxs = np.arange(t - C.N_MEM_MAIN, t)

                for method in METHODS:
                    if control == "shuffled_r":
                        pred = predict_action(
                            method,
                            t,
                            E,
                            traj.a_behavior,
                            shuffled_correct,
                            mem_idxs,
                            k=C.K_DEFAULT,
                            topk=C.TOPK,
                            n_actions=C.N_ACTIONS,
                            global_default=gdef,
                        )
                    elif control == "shuffled_d" and method in ("chemotaxis", "concat"):
                        pred = _predict_with_shuffled_anchor(
                            method,
                            t,
                            E,
                            traj.a_behavior,
                            traj.correct,
                            mem_idxs,
                            shuffled_anchor,
                            topk=C.TOPK,
                            n_actions=C.N_ACTIONS,
                            global_default=gdef,
                        )
                    else:
                        pred = predict_action(
                            method,
                            t,
                            E,
                            traj.a_behavior,
                            traj.correct,
                            mem_idxs,
                            k=C.K_DEFAULT,
                            topk=C.TOPK,
                            n_actions=C.N_ACTIONS,
                            global_default=gdef,
                        )

                    correct_counts[method] += int(pred == traj.a_star[t])

            n_eval = len(checkpoints)

            for method in METHODS:
                rows.append(
                    {
                        "control": control,
                        "mode": C.SANITY_MODE,
                        "p": p,
                        "seed": seed,
                        "method": method,
                        "n_eval": n_eval,
                        "n_correct": correct_counts[method],
                        "accuracy": correct_counts[method] / n_eval,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "sanity_checks.csv", index=False)

    return df


def _predict_with_shuffled_anchor(
    method,
    t_eval,
    E,
    actions,
    correct01,
    mem_idxs,
    anchor_of,
    topk,
    n_actions,
    global_default,
):
    q_anchor = anchor_of[t_eval]

    if method == "concat":
        q = np.concatenate([E[t_eval], E[q_anchor]])
        C_ = np.concatenate([E[mem_idxs], E[anchor_of[mem_idxs]]], axis=1)
    else:
        q = E[t_eval] - E[q_anchor]
        C_ = E[mem_idxs] - E[anchor_of[mem_idxs]]

    qn = q / (np.linalg.norm(q) + 1e-8)
    Cn = C_ / (np.linalg.norm(C_, axis=1, keepdims=True) + 1e-8)
    sims_all = Cn @ qn
    order = np.argsort(-sims_all)[:topk]
    chosen = mem_idxs[order]
    sims = sims_all[order]
    weights = sims * (2 * correct01[chosen] - 1)
    action_scores = np.zeros(n_actions)

    for j, w in zip(chosen, weights):
        action_scores[actions[j]] += w

    if action_scores.max() <= 0:
        counts = np.bincount(actions[mem_idxs], minlength=n_actions)
        return int(np.argmax(counts)) if mem_idxs.size else global_default

    return int(np.argmax(action_scores))


if __name__ == "__main__":
    t0 = time.time()

    print("== main factorial matrix ==")
    run_main_matrix()

    print("== overlap metric ==")
    run_overlap_metric()

    print("== k x p ablation ==")
    run_kp_ablation()

    print("== sanity checks ==")
    run_sanity_checks()

    print(f"done in {time.time() - t0:.1f}s")
