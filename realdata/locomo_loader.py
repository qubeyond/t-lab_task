"""Адаптер для LoCoMo: session-турны -> RealDialogue.texts, вопрос -> виртуальный шаг с evidence."""

from __future__ import annotations

import json
from pathlib import Path

from realdata.schema import RealDialogue


def _session_keys_in_order(conversation: dict) -> list[str]:
    keys = [k for k in conversation if k.startswith("session_") and not k.endswith("date_time")]
    return sorted(keys, key=lambda s: int(s.split("_")[1]))


def load_locomo(path: str | Path) -> list[RealDialogue]:
    raw = json.loads(Path(path).read_text())
    out = []

    for item in raw:
        conv = item["conversation"]
        texts: list[str] = []
        dia_id_to_idx: dict[str, int] = {}

        for skey in _session_keys_in_order(conv):
            for turn in conv[skey]:
                dia_id_to_idx[turn["dia_id"]] = len(texts)
                texts.append(f"{turn['speaker']}: {turn['text']}")

        qas = []

        for qa in item.get("qa", []):
            ev_idx = [dia_id_to_idx[e] for e in qa.get("evidence", []) if e in dia_id_to_idx]

            if not ev_idx:
                continue

            qas.append(
                {
                    "question": qa["question"],
                    "evidence_idx": sorted(set(ev_idx)),
                    "category": qa.get("category"),
                }
            )

        out.append(RealDialogue(id=item["sample_id"], texts=texts, qas=qas))

    return out
