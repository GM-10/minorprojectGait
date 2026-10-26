import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.models.experiments import lr_classification, select_model_features
from src.statistics.exploratory import pca_analysis, usable_features
from src.statistics.paired import (analyse_differences, bh_fdr, hedges_j, holm, paired_table, rank_biserial_paired,
                                   run_paired_analysis)


def test_bh_and_holm_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert bh_fdr(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert np.isnan(bh_fdr(np.array([np.nan, 0.5]))[0]) and bh_fdr(np.array([np.nan, 0.5]))[1] == 0.5


def test_analyse_differences_matches_scipy():
    rng = np.random.default_rng(0)
    d = rng.normal(1.0, 1.0, 12)
    r = analyse_differences(d, 0.05, 0.05, rng.integers(0, 12, (500, 12)))
    assert r["p_t"] == pytest.approx(stats.ttest_1samp(d, 0).pvalue)
    assert r["p_wilcoxon"] == pytest.approx(stats.wilcoxon(d).pvalue)
    assert r["dz"] == pytest.approx(d.mean() / d.std(ddof=1))
    assert r["hedges_gz"] == pytest.approx(r["dz"] * hedges_j(12)) and r["hedges_gz"] < r["dz"]
    lo, hi = stats.t.interval(0.95, 11, loc=d.mean(), scale=stats.sem(d))
    assert (r["ci_low"], r["ci_high"]) == pytest.approx((lo, hi))
    assert r["primary_test"] in ("paired_t", "wilcoxon") and r["n_pairs"] == 12


def test_analyse_differences_degenerate_cases():
    assert analyse_differences(np.array([1.0]), .05, .05, None)["primary_test"] == "none"
    z = analyse_differences(np.zeros(8), .05, .05, None)  # identical legs
    assert np.isnan(z["dz"]) and np.isnan(z["p_primary"]) and z["n_nonzero_diff"] == 0
    assert rank_biserial_paired(np.array([1.0, 2.0, 3.0])) == 1.0
    assert rank_biserial_paired(np.array([-1.0, -2.0])) == -1.0
    assert np.isnan(rank_biserial_paired(np.zeros(3)))


def _rec_table(n_part=8, seed=0, shift=0.0, unpaired=None):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_part):
        base = rng.normal(10, 3)
        for leg in ("left", "right"):
            if unpaired and i == unpaired and leg == "right":
                continue
            rows.append({"participant_id": f"Sub{i+1:02d}", "leg": leg,
                         "f1": base + (shift if leg == "left" else 0) + rng.normal(0, .5), "f2": rng.normal()})
    return pd.DataFrame(rows)


def test_paired_unit_is_participant_and_sign_convention(cfg):
    df = _rec_table(shift=5.0, unpaired=3)
    left, right, pairing = paired_table(df, ["f1", "f2"])
    assert len(left) == 7 and "Sub04" not in left.index and not pairing.set_index("participant_id").loc["Sub04", "paired"]
    out = run_paired_analysis(df, ["f1", "f2"], cfg)
    r = out["results"].set_index("feature")
    assert r.loc["f1", "n_pairs"] == 7 and r.loc["f1", "mean_diff"] > 3  # left - right, positive shift
    assert out["differences"]["participant_id"].nunique() == 7


def test_paired_not_inflated_by_windows(cfg):
    """Replicating each recording into many 'windows' must not change n_pairs: analysis uses one row per leg."""
    df = _rec_table()
    out = run_paired_analysis(pd.concat([df] * 20, ignore_index=True).drop_duplicates(), ["f1"], cfg)
    assert out["results"]["n_pairs"].iloc[0] == 8


def test_run_paired_exclusion_and_min_pairs(cfg):
    df = _rec_table(n_part=6)
    out = run_paired_analysis(df, ["f1"], cfg, exclude=["Sub01", "Sub02"])
    assert out["results"]["primary_test"].iloc[0] == "insufficient_pairs"  # 4 pairs < min_pairs=5


def test_pca_and_usable_features():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((30, 5)), columns=list("abcde"))
    df["const"], df["bad"] = 1.0, np.r_[np.nan, np.ones(29)]
    df["participant_id"] = ["P1", "P2"] * 15
    assert usable_features(df, list("abcde") + ["const", "bad"]) == list("abcde")
    res = pca_analysis(df, list("abcde"), n_components=3)
    assert res["scores"].shape[0] == 30 and "participant_id" in res["scores"]
    assert res["explained_variance_ratio"].sum() <= 1.0
    with pytest.raises(ValueError, match="PCA needs"):
        pca_analysis(df.head(2), list("abcde"))


def test_select_model_features():
    df = pd.DataFrame({"a__mean": [1, 2.0], "a__var": [1, 3.0], "corr__a__b": [0.1, 0.2], "a__q05": [1, 2.0]})
    assert select_model_features(df, list(df.columns), ["mean"]) == ["a__mean", "corr__a__b"]


def test_lr_classification_is_participant_grouped(cfg):
    rng = np.random.default_rng(0)
    rows = []
    for p in range(6):
        for leg in ("left", "right"):
            for w in range(6):
                rows.append({"participant_id": f"S{p}", "leg": leg, "recording_id": f"S{p}_{leg}", "window_index": w,
                             "f1": rng.normal(p), "f2": rng.normal()})  # participant identity, no leg signal
    df = pd.DataFrame(rows)
    c = {**cfg, "models": {**cfg["models"], "permutations": 5}}
    res = lr_classification(df, ["f1", "f2"], c, seed=0)
    pa = res["participant_accuracy"]
    assert len(pa) == 6 and pa["participant_id"].is_unique  # one fold per held-out participant
    m = res["metrics"]
    assert m["n_participants"] == 6 and m["n_recordings"] == 12 and m["n_windows"] == 72
    assert 0.2 <= m["mean_participant_window_accuracy"] <= 0.8  # no leg signal: near chance, not inflated


def test_paired_table_refuses_window_level_input(cfg):
    """Guard against pseudoreplication: a table with several rows per participant/leg (e.g. windows) forms no pairs."""
    rng = np.random.default_rng(0)
    rows = [{"participant_id": f"Sub{i:02d}", "leg": leg, "f1": rng.normal()} for i in range(8) for leg in ("left", "right")
            for _ in range(9)]
    df = pd.DataFrame(rows)
    left, right, pairing = paired_table(df, ["f1"])
    assert len(left) == 0 and not pairing["paired"].any()
    out = run_paired_analysis(df, ["f1"], cfg)
    assert out["results"]["n_pairs"].iloc[0] == 0 and out["results"]["primary_test"].iloc[0] == "insufficient_pairs"
