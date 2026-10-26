"""Regression tests for the participant-level permutation test and related interval methods."""
import numpy as np
import pandas as pd
import pytest

from src.audit.stats_audit import dz_ci_noncentral
from src.models import permutation as perm


def _data(n_part=6, signal=0.0, seed=0, n_win=8):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_part):
        off = rng.normal(0, 2, 2)
        for leg in (0, 1):
            for w in range(n_win):
                rows.append((f"S{p}", leg, off[0] + rng.normal() + signal * leg, off[1] + rng.normal()))
    d = pd.DataFrame(rows, columns=["g", "y", "f1", "f2"])
    return d[["f1", "f2"]].to_numpy(), d["y"].to_numpy(), d["g"].to_numpy()


def test_exact_patterns_are_complete_and_include_identity():
    pat, exact = perm.swap_patterns(5, None, None)
    assert exact and pat.shape == (32, 5) and len({tuple(r) for r in pat}) == 32 and (~pat).all(axis=1).any()
    with pytest.raises(ValueError, match="exceed exact_max"):
        perm.swap_patterns(20, None, None)


def test_monte_carlo_patterns_are_seeded():
    a, ea = perm.swap_patterns(12, 50, np.random.default_rng(3))
    b, _ = perm.swap_patterns(12, 50, np.random.default_rng(3))
    c, _ = perm.swap_patterns(12, 50, np.random.default_rng(4))
    assert not ea and (a == b).all() and not (a == c).all() and a.shape == (50, 12)
    with pytest.raises(ValueError, match="Generator"):
        perm.swap_patterns(12, 50, None)


def test_swaps_act_on_whole_participants_and_keep_structure():
    X, y, g = _data()
    uniq, gi = np.unique(g, return_inverse=True)
    for swap in perm.swap_patterns(len(uniq), None, None)[0][[0, 5, 17, 63]]:
        yp = perm.apply_swaps(y, gi, swap)
        for k, name in enumerate(uniq):
            m = g == name
            assert (yp[m] == (1 - y[m])).all() if swap[k] else (yp[m] == y[m]).all()  # block swap only
            assert yp[m].sum() == y[m].sum()                                          # class balance of each participant kept


def test_pvalue_conventions():
    null = np.array([0.4, 0.5, 0.6, 0.7])
    assert perm.permutation_pvalue(0.6, null, exact=True) == pytest.approx(2 / 4)          # b / B (identity is part of null)
    assert perm.permutation_pvalue(0.6, null, exact=False) == pytest.approx(3 / 5)         # (b + 1) / (B + 1)
    assert perm.permutation_pvalue(0.6 + 1e-14, null, exact=True) == pytest.approx(2 / 4)  # float ties counted as ties
    assert perm.permutation_pvalue(0.99, null, exact=False) == pytest.approx(1 / 5)        # never zero
    assert perm.permutation_pvalue(0.1, null, exact=True) == 1.0


def test_exact_null_contains_observed_and_is_deterministic():
    X, y, g = _data(5, 1.5)
    mk = lambda: perm.default_model(0)  # noqa: E731
    obs = perm.logo_participant_accuracies(X, y, g, mk)[0].mean()
    pat, exact = perm.swap_patterns(5, None, None)
    n1 = perm.permutation_null(X, y, g, mk, pat, n_jobs=1)
    n2 = perm.permutation_null(X, y, g, mk, pat, n_jobs=1)
    assert (n1 == n2).all() and len(n1) == 32 and obs in n1  # identity pattern reproduces the observed statistic
    assert n1[0] == pytest.approx(obs)


def test_null_is_calibrated_without_signal_and_detects_signal():
    mk = lambda: perm.default_model(0)  # noqa: E731
    pat, _ = perm.swap_patterns(6, None, None)
    ps = []
    for s in range(24):
        X, y, g = _data(6, 0.0, seed=100 + s)
        obs = perm.logo_participant_accuracies(X, y, g, mk)[0].mean()
        ps.append(perm.permutation_pvalue(obs, perm.permutation_null(X, y, g, mk, pat, n_jobs=1), True))
    ps = np.array(ps)
    assert (ps < 0.05).mean() <= 0.25 and ps.min() >= 1 / 64          # no more false positives than expected (loose bound)
    X, y, g = _data(6, 6.0, seed=1)
    obs = perm.logo_participant_accuracies(X, y, g, mk)[0].mean()
    assert obs > 0.9
    assert perm.permutation_pvalue(obs, perm.permutation_null(X, y, g, mk, pat, n_jobs=1), True) <= 2 / 64


def test_no_heldout_information_reaches_the_model():
    """The fold's test rows must never be seen by fit(); a participant must never appear on both sides."""
    X, y, g = _data(5, 0.0, seed=2)
    X = X + np.arange(len(X))[:, None] * 1e-6  # make every row unique

    class Guard:
        train = set()

        def fit(self, A, b):
            Guard.train = {tuple(r.round(9)) for r in A}
            return self

        def predict(self, A):
            assert not any(tuple(r.round(9)) in Guard.train for r in A), "test row seen during fit"
            return np.zeros(len(A), dtype=int)

    accs, order = perm.logo_participant_accuracies(X, y, g, Guard)
    assert len(order) == 5 and accs.shape == (5,)


def test_random_guess_distribution_centered_and_narrow():
    _, _, g = _data(6, 0.0, seed=3)
    rg = perm.random_guess_distribution(g, 2000, np.random.default_rng(0))
    assert rg.mean() == pytest.approx(0.5, abs=0.01) and 0.03 < rg.std() < 0.08  # binomial spread for 6 x 16 windows


def test_noncentral_t_ci_coverage_and_extremes():
    rng = np.random.default_rng(0)
    n, true = 12, 0.5
    cover = 0
    sims = 600
    for _ in range(sims):
        d = rng.normal(true, 1.0, n)
        lo, hi = dz_ci_noncentral(d.mean() / d.std(ddof=1), n)
        cover += lo <= true <= hi
    assert 0.92 <= cover / sims <= 0.98
    lo, hi = dz_ci_noncentral(4.0, 12)           # very large effect
    assert np.isfinite([lo, hi]).all() and lo < 4.0 < hi
    assert np.isnan(dz_ci_noncentral(60.0, 12)).all()  # root outside the safeguard bracket -> NaN, never a wrong number
