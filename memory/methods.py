"""Геометрия retrieval (rank_candidates) и голосующий слой поверх неё (predict_action)."""

from __future__ import annotations

import numpy as np

METHODS = ("recency", "point", "concat", "chemotaxis")


def _cos(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q) + 1e-8)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    return Mn @ qn


def _vector_space(method: str, t_eval: int, E: np.ndarray, idxs: np.ndarray, k: int):
    if method == "point":
        return E[t_eval], E[idxs]
    elif method == "concat":
        return np.concatenate([E[t_eval], E[t_eval - k]]), np.concatenate([E[idxs], E[idxs - k]], axis=1)
    elif method == "chemotaxis":
        return E[t_eval] - E[t_eval - k], E[idxs] - E[idxs - k]

    raise ValueError(method)


def rank_candidates(
    method: str,
    t_eval: int,
    E: np.ndarray,
    mem_idxs: np.ndarray,
    k: int = 1,
    topk: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    if method not in METHODS:
        raise ValueError(method)

    if method == "recency":
        chosen = mem_idxs[-topk:]
        return chosen, np.ones(len(chosen))

    q, C = _vector_space(method, t_eval, E, mem_idxs, k)
    sims_all = _cos(q, C)
    order = np.argsort(-sims_all)[:topk]

    return mem_idxs[order], sims_all[order]


def _majority(actions: np.ndarray, n_actions: int, default: int) -> int:
    if len(actions) == 0:
        return default

    counts = np.bincount(actions, minlength=n_actions)
    return int(np.argmax(counts))


def predict_action(
    method: str,
    t_eval: int,
    E: np.ndarray,
    actions: np.ndarray,
    correct01: np.ndarray,
    mem_idxs: np.ndarray,
    k: int = 1,
    topk: int = 5,
    n_actions: int = 6,
    global_default: int = 0,
    return_retrieved: bool = False,
):
    if method not in METHODS:
        raise ValueError(method)

    chosen, sims = rank_candidates(method, t_eval, E, mem_idxs, k=k, topk=topk)
    weights = sims * (2 * correct01[chosen] - 1)
    action_scores = np.zeros(n_actions)

    for j, w in zip(chosen, weights):
        action_scores[actions[j]] += w

    if action_scores.max() <= 0:
        pred = _majority(actions[mem_idxs], n_actions, global_default)
    else:
        pred = int(np.argmax(action_scores))

    if return_retrieved:
        return pred, chosen

    return pred
