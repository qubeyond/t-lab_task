"""Обучаемая альтернатива знаковому голосованию из predict_action, feature_mask для абляций."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from memory.methods import _cos, _vector_space

FEATURE_NAMES = ("point_sim", "concat_sim", "chemotaxis_sim", "r_sign", "recency")


def extract_features(
    t_eval: int,
    E: np.ndarray,
    mem_idxs: np.ndarray,
    k: int,
    correct01: np.ndarray,
) -> np.ndarray:
    q_pt, C_pt = _vector_space("point", t_eval, E, mem_idxs, k)
    q_cc, C_cc = _vector_space("concat", t_eval, E, mem_idxs, k)
    q_ch, C_ch = _vector_space("chemotaxis", t_eval, E, mem_idxs, k)

    point_sim = _cos(q_pt, C_pt)
    concat_sim = _cos(q_cc, C_cc)
    chemotaxis_sim = _cos(q_ch, C_ch)
    r_sign = 2 * correct01[mem_idxs].astype(np.float32) - 1
    span = max(t_eval - mem_idxs.min(), 1)
    recency = (mem_idxs - mem_idxs.min()).astype(np.float32) / span

    return np.stack([point_sim, concat_sim, chemotaxis_sim, r_sign, recency], axis=1).astype(np.float32)


class LearnedScorer(nn.Module):
    def __init__(
        self,
        n_features: int = len(FEATURE_NAMES),
        hidden: int = 16,
        feature_mask: tuple[bool, ...] | None = None,
    ):
        super().__init__()

        mask = torch.ones(n_features) if feature_mask is None else torch.tensor(feature_mask, dtype=torch.float32)
        self.register_buffer("feature_mask", mask)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features * self.feature_mask).squeeze(-1)

    def action_logits(self, features: torch.Tensor, actions: torch.Tensor, n_actions: int) -> torch.Tensor:
        scores = self.forward(features)
        logits = torch.zeros(features.shape[0], n_actions, dtype=scores.dtype, device=scores.device)
        logits.scatter_add_(1, actions, scores)
        return logits
