"""Tier 2: срез Tier-1 с LLM-агентом, results/tables/tier2_matrix.csv, tier2_sanity_shuffled_d.csv."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

import experiments.config as C
from agents.llm_agent import LLMRouterAgent, get_llm
from env.rules import TOPICS
from env.trajectory import TrajectoryConfig, generate_trajectory
from memory.embeddings import encode_texts
from memory.methods import METHODS

RESULTS = Path(__file__).resolve().parents[1] / C.RESULTS_DIR / "tables"
RESULTS.mkdir(parents=True, exist_ok=True)

P_GRID_TIER2 = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS_TIER2 = (0, 1, 2)
N_MEM_TIER2 = C.N_MEM_MAIN
N_EVAL_TIER2 = 12
N_STEPS_TIER2 = C.BURN_IN + N_EVAL_TIER2


def eval_checkpoints(T: int, burn_in: int, n_eval: int) -> np.ndarray:
    end = min(burn_in + n_eval, T)
    return np.arange(burn_in, end)


def global_majority(actions, n_actions):
    return int(np.bincount(actions, minlength=n_actions).argmax())


def run_tier2_matrix():
    get_llm(C.LLM_MODEL, device="cpu")
    rows = []
    combos = [(mode, p, seed) for mode in C.MODES for p in P_GRID_TIER2 for seed in SEEDS_TIER2]

    for mode, p, seed in tqdm(combos, desc="tier2_matrix"):
        cfg = TrajectoryConfig(mode=mode, p=p, seed=seed, n_steps=N_STEPS_TIER2, eps_explore=C.EPS_EXPLORE)
        traj = generate_trajectory(cfg)
        E = encode_texts(traj.texts, model_name=C.ENCODER_MODEL)
        gdef = global_majority(traj.a_behavior[: C.BURN_IN], C.N_ACTIONS)
        checkpoints = eval_checkpoints(len(traj.texts), C.BURN_IN, N_EVAL_TIER2)

        agents = {
            m: LLMRouterAgent(m, list(TOPICS), k=C.K_DEFAULT, topk=C.TOPK, model_name=C.LLM_MODEL) for m in METHODS
        }
        correct = dict.fromkeys(METHODS, 0)

        for t in checkpoints:
            mem_idxs = np.arange(t - N_MEM_TIER2, t)

            for method in METHODS:
                pred = agents[method].act(t, traj.texts, E, traj.a_behavior, traj.correct, mem_idxs, gdef)
                correct[method] += int(pred == traj.a_star[t])

        n_eval = len(checkpoints)

        for method in METHODS:
            rows.append(
                {
                    "mode": mode,
                    "p": p,
                    "seed": seed,
                    "N_mem": N_MEM_TIER2,
                    "method": method,
                    "n_eval": n_eval,
                    "n_correct": correct[method],
                    "accuracy": correct[method] / n_eval,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "tier2_matrix.csv", index=False)

    return df


def run_tier2_sanity_shuffled_d():
    get_llm(C.LLM_MODEL, device="cpu")
    rows = []
    p_vals = (0.6,)
    seeds = (0, 1)

    for p in p_vals:
        for seed in seeds:
            cfg = TrajectoryConfig(mode=2, p=p, seed=seed, n_steps=N_STEPS_TIER2, eps_explore=C.EPS_EXPLORE)
            traj = generate_trajectory(cfg)
            E = encode_texts(traj.texts, model_name=C.ENCODER_MODEL)
            gdef = global_majority(traj.a_behavior[: C.BURN_IN], C.N_ACTIONS)
            checkpoints = eval_checkpoints(len(traj.texts), C.BURN_IN, N_EVAL_TIER2)
            rng = np.random.default_rng(2000 + seed)
            all_idx = np.arange(len(traj.texts))
            shuffled_anchor = all_idx.copy()
            rng.shuffle(shuffled_anchor)

            for control in ("none", "shuffled_d"):
                agent = LLMRouterAgent(
                    "chemotaxis",
                    list(TOPICS),
                    k=C.K_DEFAULT,
                    topk=C.TOPK,
                    model_name=C.LLM_MODEL,
                )
                correct = 0

                for t in checkpoints:
                    mem_idxs = np.arange(t - N_MEM_TIER2, t)

                    if control == "shuffled_d":
                        pred = _act_with_shuffled_anchor(
                            agent,
                            t,
                            traj.texts,
                            E,
                            traj.a_behavior,
                            traj.correct,
                            mem_idxs,
                            shuffled_anchor,
                        )
                    else:
                        pred = agent.act(
                            t,
                            traj.texts,
                            E,
                            traj.a_behavior,
                            traj.correct,
                            mem_idxs,
                            gdef,
                        )

                    correct += int(pred == traj.a_star[t])

                n_eval = len(checkpoints)
                rows.append(
                    {
                        "control": control,
                        "mode": 2,
                        "p": p,
                        "seed": seed,
                        "method": "chemotaxis",
                        "n_eval": n_eval,
                        "n_correct": correct,
                        "accuracy": correct / n_eval,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "tier2_sanity_shuffled_d.csv", index=False)

    return df


def _act_with_shuffled_anchor(agent, t_eval, texts, E, actions, correct01, mem_idxs, anchor_of):
    q_anchor = anchor_of[t_eval]
    d_now = E[t_eval] - E[q_anchor]
    d_mem = E[mem_idxs] - E[anchor_of[mem_idxs]]
    qn = d_now / (np.linalg.norm(d_now) + 1e-8)
    Cn = d_mem / (np.linalg.norm(d_mem, axis=1, keepdims=True) + 1e-8)
    sims = Cn @ qn
    order = np.argsort(-np.abs(sims))[: agent.topk]
    retrieved_idx = mem_idxs[order]
    retrieved = [(texts[j], agent.action_names[actions[j]], int(correct01[j])) for j in retrieved_idx]

    from agents.llm_agent import build_prompt, score_labels

    prompt = build_prompt(texts[t_eval], retrieved)
    scores = score_labels(prompt, agent.action_names, agent.model, agent.tok, agent.device)

    return int(np.argmax(scores))


if __name__ == "__main__":
    t0 = time.time()

    print("== Tier 2 LLM matrix ==")
    run_tier2_matrix()

    print("== Tier 2 LLM control 2 spot check ==")
    run_tier2_sanity_shuffled_d()

    print(f"done in {time.time() - t0:.1f}s")
