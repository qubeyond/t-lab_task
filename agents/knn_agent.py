"""Tier-1 агент: обёртка над memory.methods.predict_action."""

from __future__ import annotations

import numpy as np

from memory.methods import predict_action


class KNNVoteAgent:
    def __init__(self, method: str, k: int = 1, topk: int = 5, n_actions: int = 6):
        self.method = method
        self.k = k
        self.topk = topk
        self.n_actions = n_actions

    def act(
        self,
        t_eval: int,
        E: np.ndarray,
        actions: np.ndarray,
        correct01: np.ndarray,
        mem_idxs: np.ndarray,
        global_default: int = 0,
        return_retrieved: bool = False,
    ):
        return predict_action(
            self.method,
            t_eval,
            E,
            actions,
            correct01,
            mem_idxs,
            k=self.k,
            topk=self.topk,
            n_actions=self.n_actions,
            global_default=global_default,
            return_retrieved=return_retrieved,
        )
