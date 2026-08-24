"""Проверка среды на MiniLM-энкодере: point-similarity, камуфляж стиля, разделимость направлений."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from env.rules import N_STYLES, STYLES, TOPICS, render_situation
from memory.embeddings import encode_texts

rng = np.random.default_rng(0)


def sample_pairs(n_per_topic=8):
    texts, topic_of, style_of = [], [], []

    for ti, topic in enumerate(TOPICS):
        for _ in range(n_per_topic):
            si = int(rng.integers(0, N_STYLES))
            texts.append(render_situation(topic, STYLES[si], rng))
            topic_of.append(ti)
            style_of.append(si)

    return texts, np.array(topic_of), np.array(style_of)


def check_point_similarity_structure():
    texts, topic_of, _style_of = sample_pairs()
    E = encode_texts(texts)
    S = E @ E.T
    same_topic = []
    diff_topic = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (same_topic if topic_of[i] == topic_of[j] else diff_topic).append(S[i, j])

    same_topic, diff_topic = np.array(same_topic), np.array(diff_topic)

    print(
        f"[point similarity] same-topic mean cos={same_topic.mean():.3f} (n={len(same_topic)}), "
        f"diff-topic mean cos={diff_topic.mean():.3f} (n={len(diff_topic)})"
    )

    assert same_topic.mean() > diff_topic.mean() + 0.1, (
        "same-topic sentences should embed noticeably closer than different-topic ones"
    )

    print("point-similarity-structure OK\n")


def check_style_camouflage_signal():
    texts, topic_of, style_of = sample_pairs(n_per_topic=10)
    E = encode_texts(texts)
    S = E @ E.T
    same_style_diff_topic = []
    diff_style_diff_topic = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if topic_of[i] == topic_of[j]:
                continue

            if style_of[i] == style_of[j]:
                same_style_diff_topic.append(S[i, j])
            else:
                diff_style_diff_topic.append(S[i, j])

    a, b = np.array(same_style_diff_topic), np.array(diff_style_diff_topic)

    print(
        f"[style camouflage] diff-topic, SAME style mean cos={a.mean():.3f} (n={len(a)}), "
        f"diff-topic, DIFF style mean cos={b.mean():.3f} (n={len(b)})"
    )

    assert a.mean() > b.mean(), (
        "sharing a style wrapper should measurably raise embedding similarity even across topics "
        "-- otherwise the 'treacherous jump' construction has no real bite on point-cosine"
    )

    print("style-camouflage-signal OK\n")


def check_direction_separability():
    pairs = [(0, 1), (0, 2), (3, 4), (2, 5)]
    reps = 6
    dirs = {}

    for src, tgt in pairs:
        vecs = []

        for _ in range(reps):
            s_src = render_situation(TOPICS[src], STYLES[int(rng.integers(0, N_STYLES))], rng)
            s_tgt = render_situation(TOPICS[tgt], STYLES[int(rng.integers(0, N_STYLES))], rng)
            e_src, e_tgt = encode_texts([s_src, s_tgt])
            vecs.append(e_tgt - e_src)

        dirs[(src, tgt)] = np.stack(vecs)

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8))

    same_pair_sims, diff_pair_sims = [], []
    keys = list(dirs.keys())

    for ki, key in enumerate(keys):
        V = dirs[key]

        for i in range(reps):
            for j in range(i + 1, reps):
                same_pair_sims.append(cos(V[i], V[j]))

        for kj in range(ki + 1, len(keys)):
            V2 = dirs[keys[kj]]

            for i in range(reps):
                for j in range(reps):
                    diff_pair_sims.append(cos(V[i], V2[j]))

    same_pair_sims, diff_pair_sims = np.array(same_pair_sims), np.array(diff_pair_sims)

    print(
        f"[direction separability] same (source,target)-pair mean cos={same_pair_sims.mean():.3f}, "
        f"different-pair mean cos={diff_pair_sims.mean():.3f}"
    )

    assert same_pair_sims.mean() > diff_pair_sims.mean() + 0.1, (
        "direction vectors from the same (source,target) topic transition should be more mutually "
        "similar than direction vectors from unrelated transitions -- this is the geometric premise "
        "the chemotaxis mechanism relies on"
    )

    print("direction-separability OK\n")


if __name__ == "__main__":
    check_point_similarity_structure()
    check_style_camouflage_signal()
    check_direction_separability()

    print("ALL EMBEDDING SANITY CHECKS PASSED")
