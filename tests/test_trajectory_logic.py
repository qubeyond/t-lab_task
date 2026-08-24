"""Проверки чистой логики генератора среды, без ML-зависимостей."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from env.rules import N_TOPICS, build_action_table
from env.trajectory import TrajectoryConfig, generate_trajectory


def check_basic_shapes():
    cfg = TrajectoryConfig(mode=1, p=0.3, seed=0, n_steps=120)
    traj = generate_trajectory(cfg)

    assert len(traj.texts) == 120
    assert traj.topics.shape == (120,)
    assert traj.a_star.shape == (120,)
    assert set(np.unique(traj.topics)).issubset(set(range(N_TOPICS)))

    print("shapes OK")


def check_mode1_is_target_only():
    cfg = TrajectoryConfig(mode=1, p=0.5, seed=1, n_steps=500)
    traj = generate_trajectory(cfg)

    assert np.all(traj.a_star == traj.topics), "mode 1: a* must equal current topic always"

    print("mode 1 a*==topic OK")


def check_action_table_diagonal_is_identity():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        F = build_action_table(2, rng)
        diag = np.diag(F)

        assert np.array_equal(diag, np.arange(N_TOPICS)), f"seed={seed}: F diagonal {diag} != identity"

        off_diag_mask = ~np.eye(N_TOPICS, dtype=bool)
        col_index = np.tile(np.arange(N_TOPICS), (N_TOPICS, 1))

        assert not np.any(F[off_diag_mask] == col_index[off_diag_mask]), (
            "an off-diagonal F[i,j] equals j -- target-only guess could accidentally succeed on an entry"
        )

    print("action-table-diagonal-is-identity OK (20 seeds)")


def check_mode2_entry_vs_continuation():
    cfg = TrajectoryConfig(mode=2, p=0.5, seed=2, n_steps=2000, eps_explore=0.2)
    traj = generate_trajectory(cfg)
    F = traj.action_table

    cont = ~traj.is_entry
    diag_expected = F[traj.topics[cont], traj.topics[cont]]

    assert np.all(traj.a_star[cont] == diag_expected)
    assert np.all(traj.a_star[cont] == traj.topics[cont]), (
        "continuation steps must be trivially target-only-correct in mode 2, same as mode 1"
    )

    entry = traj.is_entry
    frac_naive_wrong_on_entry = np.mean(traj.a_star[entry] != traj.topics[entry])

    print(
        f"mode2: entries={entry.sum()}, frac where naive target-only guess is wrong = {frac_naive_wrong_on_entry:.3f}"
    )

    assert frac_naive_wrong_on_entry == 1.0, "off-diagonal F must never equal the target-only guess"

    eps, n_actions = cfg.eps_explore, F.shape[0]
    p_correct_cont = traj.correct[cont].mean()
    p_correct_entry = traj.correct[entry].mean()

    print(
        f"P(behavior correct | continuation)={p_correct_cont:.3f} (theory {(1 - eps) + eps / n_actions:.3f}), "
        f"P(behavior correct | entry)={p_correct_entry:.3f} (theory {eps / n_actions:.3f})"
    )

    assert abs(p_correct_cont - ((1 - eps) + eps / n_actions)) < 0.03
    assert abs(p_correct_entry - eps / n_actions) < 0.03

    print("mode 2 entry/continuation OK")


def check_entry_rate_scales_with_p():
    rates = {}

    for p in (0.1, 0.5, 0.9):
        cfg = TrajectoryConfig(mode=2, p=p, seed=3, n_steps=3000)
        traj = generate_trajectory(cfg)
        rates[p] = traj.is_entry[1:].mean()

    print("entry rate by p:", rates)

    assert rates[0.1] < rates[0.5] < rates[0.9]

    print("entry-rate-scales-with-p OK")


def check_permutation_f_table():
    rng = np.random.default_rng(42)
    F = build_action_table(2, rng)

    for j in range(N_TOPICS):
        assert sorted(F[:, j].tolist()) == list(range(N_TOPICS)), "each column must be a permutation"

    print("F-table permutation-per-column OK")


def check_reproducibility():
    cfg = TrajectoryConfig(mode=2, p=0.4, seed=7, n_steps=50)
    t1 = generate_trajectory(cfg)
    t2 = generate_trajectory(cfg)

    assert t1.texts == t2.texts
    assert np.array_equal(t1.a_star, t2.a_star)

    print("reproducibility (same seed -> identical trajectory) OK")


def check_camouflage_reuses_style_on_jump():
    cfg = TrajectoryConfig(mode=2, p=0.6, seed=5, n_steps=2000)
    traj = generate_trajectory(cfg)

    entry = traj.is_entry.copy()
    entry[0] = False
    same_style_entry = (traj.styles[1:][entry[1:]] == traj.styles[:-1][entry[1:]]).mean()

    cont = ~traj.is_entry
    cont[0] = False
    same_style_cont = (traj.styles[1:][cont[1:]] == traj.styles[:-1][cont[1:]]).mean()

    from env.rules import N_STYLES

    print(
        f"camouflage: P(style[t]==style[t-1] | entry)={same_style_entry:.2f}, "
        f"P(same | continuation)={same_style_cont:.2f} (chance level = 1/{N_STYLES} = {1 / N_STYLES:.2f})"
    )

    assert same_style_entry > 0.99, "entry steps must deterministically reuse previous style"
    assert same_style_cont < 0.4, "continuation steps should draw style ~freshly (near chance repeat rate)"

    print("camouflage OK")


if __name__ == "__main__":
    check_basic_shapes()
    check_mode1_is_target_only()
    check_action_table_diagonal_is_identity()
    check_mode2_entry_vs_continuation()
    check_entry_rate_scales_with_p()
    check_permutation_f_table()
    check_reproducibility()
    check_camouflage_reuses_style_on_jump()

    print("\nALL PURE-LOGIC SANITY CHECKS PASSED")
