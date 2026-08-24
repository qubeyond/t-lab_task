"""Валидация на данных LoCoMo, results/tables/locomo_recall.csv."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from memory.embeddings import encode_texts
from memory.methods import METHODS, rank_candidates
from realdata.locomo_loader import load_locomo

DATA_PATH = Path(__file__).resolve().parents[1] / "realdata" / "data" / "locomo10.json"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "tables"
RESULTS.mkdir(parents=True, exist_ok=True)

TOPK = 5
N_MEM_GRID = (20, 50, 100, None)
K_GRID = (1, 3, 6)
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def score(retrieved_idx: np.ndarray, evidence_idx: list[int]) -> tuple[float, int]:
    hit_set = set(retrieved_idx.tolist()) & set(evidence_idx)
    recall = len(hit_set) / len(evidence_idx)
    hit = int(len(hit_set) > 0)
    return recall, hit


def main():
    conversations = load_locomo(DATA_PATH)
    rows = []

    for conv in tqdm(conversations, desc="conversations"):
        turn_emb = encode_texts(conv.texts, model_name=ENCODER_MODEL)
        T = len(conv.texts)
        question_texts = [qa["question"] for qa in conv.qas]
        question_emb = encode_texts(question_texts, model_name=ENCODER_MODEL)

        for qi, qa in enumerate(conv.qas):
            E = np.vstack([turn_emb, question_emb[qi][None, :]])
            t_eval = T

            for k in K_GRID:
                lo_bound = k

                for N_mem in N_MEM_GRID:
                    start = lo_bound if N_mem is None else max(lo_bound, T - N_mem)

                    if start >= T:
                        continue

                    mem_idxs = np.arange(start, T)

                    for method in METHODS:
                        chosen, _ = rank_candidates(method, t_eval, E, mem_idxs, k=k, topk=TOPK)
                        recall, hit = score(chosen, qa["evidence_idx"])
                        rows.append(
                            {
                                "sample_id": conv.id,
                                "category": qa["category"],
                                "method": method,
                                "N_mem": ("all" if N_mem is None else N_mem),
                                "k": k,
                                "n_evidence": len(qa["evidence_idx"]),
                                "recall": recall,
                                "hit": hit,
                            }
                        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "locomo_recall.csv", index=False)
    print(df.groupby(["N_mem", "k", "method"])[["recall", "hit"]].mean().round(3))
    print(f"\n{len(df)} rows written to {RESULTS / 'locomo_recall.csv'}")


if __name__ == "__main__":
    main()
