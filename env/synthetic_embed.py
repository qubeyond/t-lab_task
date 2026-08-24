"""Управляемый embedding для mode 3: E = identity[topic] + alpha*topic*ordinal_step + style + шум."""

from __future__ import annotations

import numpy as np

from .rules import N_STYLES, N_TOPICS


def synthetic_embed(
    topics: np.ndarray,
    styles: np.ndarray,
    alpha: float = 1.0,
    dim: int = 64,
    noise: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    identity = rng.normal(size=(N_TOPICS, dim))
    identity /= np.linalg.norm(identity, axis=1, keepdims=True)
    ordinal_step = rng.normal(size=dim)
    ordinal_step /= np.linalg.norm(ordinal_step)
    style_vecs = rng.normal(size=(N_STYLES, dim))
    style_vecs /= np.linalg.norm(style_vecs, axis=1, keepdims=True)

    topic_component = identity[topics] + alpha * topics[:, None] * ordinal_step[None, :]
    E = topic_component + style_vecs[styles] + rng.normal(scale=noise, size=(len(topics), dim))
    return E.astype(np.float32)
