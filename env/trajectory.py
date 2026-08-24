"""Генератор траекторий: поток шагов (ситуация, действие, correctness) для заданных (mode, p)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rules import (
    N_STYLES,
    N_TOPICS,
    STYLES,
    TOPICS,
    build_action_table,
    render_situation,
)


@dataclass
class TrajectoryConfig:
    mode: int
    p: float
    seed: int
    n_steps: int
    eps_explore: float = 0.2
    k_env: int = 1


@dataclass
class Trajectory:
    config: TrajectoryConfig
    texts: list = field(default_factory=list)
    topics: np.ndarray = None
    styles: np.ndarray = None
    is_entry: np.ndarray = None
    a_star: np.ndarray = None
    a_behavior: np.ndarray = None
    correct: np.ndarray = None
    action_table: np.ndarray = None


def generate_trajectory(cfg: TrajectoryConfig) -> Trajectory:
    rng = np.random.default_rng(cfg.seed)
    F = build_action_table(cfg.mode, rng)

    T = cfg.n_steps
    texts = []
    topics = np.zeros(T, dtype=int)
    styles = np.zeros(T, dtype=int)
    is_entry = np.zeros(T, dtype=bool)
    a_star = np.zeros(T, dtype=int)

    cur_topic = int(rng.integers(0, N_TOPICS))
    cur_style = int(rng.integers(0, N_STYLES))
    prev_topic = cur_topic

    for t in range(T):
        if t == 0:
            jump = False
        else:
            jump = rng.random() < cfg.p

        if jump:
            choices = [i for i in range(N_TOPICS) if i != cur_topic]
            new_topic = int(rng.choice(choices))
            prev_topic = cur_topic
            cur_topic = new_topic
            cur_style = styles[t - 1]
            is_entry[t] = True
        else:
            if t > 0:
                cur_style = int(rng.integers(0, N_STYLES))
            is_entry[t] = False

        topics[t] = cur_topic
        styles[t] = cur_style
        source_topic = prev_topic if is_entry[t] else cur_topic
        a_star[t] = F[source_topic, cur_topic]
        texts.append(render_situation(TOPICS[cur_topic], STYLES[cur_style], rng))

    a_behavior = np.zeros(T, dtype=int)

    for t in range(T):
        if rng.random() < cfg.eps_explore:
            a_behavior[t] = int(rng.integers(0, N_TOPICS))
        else:
            a_behavior[t] = topics[t]

    correct = (a_behavior == a_star).astype(int)

    return Trajectory(
        config=cfg,
        texts=texts,
        topics=topics,
        styles=styles,
        is_entry=is_entry,
        a_star=a_star,
        a_behavior=a_behavior,
        correct=correct,
        action_table=F,
    )
