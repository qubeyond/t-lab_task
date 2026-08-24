"""Валидация на данных LongMemEval-S, results/tables/longmemeval_recall.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from memory.embeddings import encode_texts
from memory.methods import METHODS, rank_candidates
from realdata.longmemeval_loader import load_longmemeval

DATA_PATH = Path(__file__).resolve().parents[1] / "realdata" / "data" / "longmemeval_s.json"
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


def main(device: str | None, batch_size: int):
    dialogues = load_longmemeval(DATA_PATH)

    all_texts: list[str] = []
    turn_span: list[tuple[int, int]] = []

    for dlg in dialogues:
        start = len(all_texts)
        all_texts.extend(dlg.texts)
        turn_span.append((start, len(all_texts)))

    question_start = len(all_texts)
    all_texts.extend(dlg.qas[0]["question"] for dlg in dialogues)

    print(f"embedding {len(all_texts)} texts ({len(dialogues)} dialogues) on device={device or 'auto'} ...")
    E_all = encode_texts(all_texts, model_name=ENCODER_MODEL, batch_size=batch_size, device=device)

    rows = []

    for i, dlg in enumerate(tqdm(dialogues, desc="questions")):
        t0, t1 = turn_span[i]
        turn_emb = E_all[t0:t1]
        question_emb = E_all[question_start + i]
        T = len(dlg.texts)
        qa = dlg.qas[0]

        E = np.vstack([turn_emb, question_emb[None, :]])
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
                            "id": dlg.id,
                            "question_type": qa["question_type"],
                            "method": method,
                            "N_mem": ("all" if N_mem is None else N_mem),
                            "k": k,
                            "n_evidence": len(qa["evidence_idx"]),
                            "recall": recall,
                            "hit": hit,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "longmemeval_recall.csv", index=False)
    print(df.groupby(["N_mem", "k", "method"])[["recall", "hit"]].mean().round(3))
    print(f"\n{len(df)} rows written to {RESULTS / 'longmemeval_recall.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None, help='"cuda", "cpu", or omit for auto-detect')
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    main(device=args.device, batch_size=args.batch_size)
