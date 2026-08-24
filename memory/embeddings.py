"""Обёртка над encoder из sentence-transformers с кэшем на весь процесс."""

from __future__ import annotations

import numpy as np

_MODEL = None
_MODEL_NAME = None
_DEVICE = None
_CACHE: dict[str, np.ndarray] = {}


def get_encoder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
):
    global _MODEL, _MODEL_NAME, _DEVICE

    if _MODEL is None or _MODEL_NAME != model_name or _DEVICE != device:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(model_name, device=device)
        _MODEL_NAME = model_name
        _DEVICE = device
        _CACHE.clear()

    return _MODEL


def encode_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    model = get_encoder(model_name, device=device)
    missing = [t for t in texts if t not in _CACHE]

    if missing:
        embs = model.encode(
            missing,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        for t, e in zip(missing, embs):
            _CACHE[t] = e

    return np.stack([_CACHE[t] for t in texts]).astype(np.float32)
