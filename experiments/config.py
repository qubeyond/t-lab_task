"""Фиксированные параметры экспериментов."""

from __future__ import annotations

N_ACTIONS = 6
TOPK = 5
K_DEFAULT = 1
EPS_EXPLORE = 0.2

MODES = (1, 2)

P_GRID_MAIN = (0.0, 0.15, 0.3, 0.5, 0.65, 0.8, 0.9, 1.0)
N_MEM_MAIN = 40
SEEDS_MAIN = tuple(range(10))

N_MEM_GRID = (10, 20, 40, 80)
BUDGET_P = 0.5

K_GRID = (1, 2, 4, 8)

BURN_IN = max(N_MEM_GRID) + max(K_GRID)
N_EVAL_TIER1 = 200
N_STEPS_TIER1 = BURN_IN + N_EVAL_TIER1

P_GRID_ABLATION = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS_ABLATION = tuple(range(8))
N_MEM_ABLATION = 40
N_EVAL_ABLATION = 150
N_STEPS_ABLATION = BURN_IN + N_EVAL_ABLATION

SANITY_MODE = 2
SANITY_P_VALUES = (0.3, 0.6, 0.9)
SANITY_N_MEM = 40
SANITY_SEEDS = SEEDS_MAIN

ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

RESULTS_DIR = "results"
