"""Адаптер для LongMemEval-S: один RealDialogue на вопрос, свой haystack сессий."""

from __future__ import annotations

import json
from pathlib import Path

from realdata.schema import RealDialogue


def load_longmemeval(path: str | Path) -> list[RealDialogue]:
    raw = json.loads(Path(path).read_text())
    out = []

    for item in raw:
        qid = item["question_id"]

        if qid.endswith("_abs"):
            continue

        texts: list[str] = []
        evidence_idx: list[int] = []

        for session in item["haystack_sessions"]:
            for turn in session:
                if turn.get("has_answer"):
                    evidence_idx.append(len(texts))
                texts.append(f"{turn['role']}: {turn['content']}")

        if not evidence_idx:
            continue

        qas = [
            {
                "question": item["question"],
                "evidence_idx": sorted(set(evidence_idx)),
                "question_type": item.get("question_type"),
            }
        ]
        out.append(RealDialogue(id=qid, texts=texts, qas=qas))

    return out
