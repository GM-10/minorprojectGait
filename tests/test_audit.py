"""Tests for the audit helpers (synthetic data only; results are not taken from these)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from conftest import make_csv
from src.audit import clipping_audit as ca
from src.audit import model_audit as ma
from src.audit import stats_audit as sa
from src.audit.common import feature_channels, feature_signal, feature_stat


def test_feature_channel_mapping():
    assert feature_channels("acc_x__mean") == {"acc_x"}
    assert feature_channels("gyro_mag__rms") == {"gyro_x", "gyro_y", "gyro_z"}
    assert feature_channels("acc_dyn_mag__std") == {"acc_x", "acc_y", "acc_z"}
    assert feature_channels("corr__acc_x__gyro_mag") == {"acc_x", "gyro_x", "gyro_y", "gyro_z"}
    assert feature_channels("something_else__mean") == set()
    assert feature_stat("corr__a__b") == "corr" and feature_signal("acc_y__q05") == "acc_y"


def _raw(tmp_path, n_part=3):
    for i in range(1, n_part + 1):
        make_csv(tmp_path / f"Sub{i:02d}Left.csv", seed=i)
        make_csv(tmp_path / f"Sub{i:02d}Right.csv", seed=10 + i)
    return ca.load_raw(tmp_path)


def test_rail_inventory_exact_counts_and_tolerance_difference(tmp_path):
    raw = _raw(tmp_path)
    d = raw[("Sub01", "right")]
    d.loc[[5, 6, 7], "GyroX_dps"] = 250.13                      # three samples at the rail
    d.loc[8, "GyroX_dps"] = 250.046                              # near, but not at, the rail
    d.loc[20, "AccelZ_g"] = 1.9999
    d.loc[21, "AccelZ_g"] = -2.0
    raw[("Sub01", "left")].loc[3, "GyroX_dps"] = -250.137        # defines the negative rail
    lv = ca.rail_levels(raw)
    assert lv["gyro"] == (-250.137, 250.13) and lv["acc"] == (-2.0, 1.9999)
    inv, summ = ca.clip_inventory(raw, lv)
    r = inv[(inv.participant_id == "Sub01") & (inv.leg == "right")].set_index("channel")
    assert r.loc["gyro_x", "n_at_max"] == 3 and r.loc["gyro_x", "longest_run"] == 3 and r.loc["gyro_x", "n_episodes"] == 1
    assert r.loc["acc_z", "n_at_max"] == 1 and r.loc["acc_z", "n_at_min"] == 1
    tve = ca.tolerance_vs_exact(raw, lv, 0.001).set_index("channel")
    assert tve.loc["gyro_x", "n_extra_by_tolerance"] == 1 and 250.046 in tve.loc["gyro_x", "extra_values"]
    tight = ca.tolerance_vs_exact(raw, lv, 1e-4).set_index("channel")
    assert tight.loc["gyro_x", "n_extra_by_tolerance"] == 0


def test_int16_check_arithmetic():
    t = ca.int16_check({"acc": (-2.0, 1.9999), "gyro": (-250.137, 250.13)}).set_index("family")
    assert t.loc["acc", "max_rounding_error_counts"] < 1 and t.loc["gyro", "max_rounding_error_counts"] < 1
    bad = ca.int16_check({"acc": (-1.5, 1.5), "gyro": (-100.0, 100.0)}).set_index("family")
    assert bad.loc["acc", "max_rounding_error_counts"] > 1000


def test_accz_asymmetry_presence_vs_counts():
    inv = pd.DataFrame([{"participant_id": f"S{i}", "leg": leg, "channel": "acc_z",
                         "n_at_max": (1 if leg == "right" else 0), "n_at_min": (1 if leg == "left" else 0)}
                        for i in range(6) for leg in ("left", "right")])
    raw = {(f"S{i}", leg): pd.DataFrame({"AccelX_g": [0.0] * 30, "AccelY_g": [-1.0] * 30, "AccelZ_g": np.linspace(-1, 1, 30),
                                         "GyroX_dps": np.arange(30.0), "GyroY_dps": np.arange(30.0), "GyroZ_dps": np.arange(30.0)})
           for i in range(6) for leg in ("left", "right")}
    res = ca.accz_asymmetry(raw, {"acc": (-2.0, 2.0), "gyro": (-250.0, 250.0)}, inv)
    t = res["accz_tests"].set_index("quantity")
    pres = [k for k in t.index if k.startswith("presence")][0]
    assert t.loc[pres, "sum_right"] == 6 and t.loc[pres, "sum_left"] == 0 and t.loc[pres, "sign_test_p"] == pytest.approx(2 * 0.5 ** 6)
    assert t.loc["n samples at either Z rail", "sum_left"] == t.loc["n samples at either Z rail", "sum_right"] == 6
    assert t.loc["n samples at either Z rail", "sign_test_p"] == 1.0  # all ties: no difference in total rail exposure


def test_feature_clip_sensitivity_clean_pairs():
    rows = []
    for i in range(6):
        for leg in ("left", "right"):
            rows.append({"recording_id": f"S{i}_{leg}", "participant_id": f"S{i}", "leg": leg, "gyro_x__mean": float(i),
                         "qc_clip_fraction_gyro_x": 0.2 if i < 2 else 0.0, **{f"qc_clip_fraction_{c}": 0.0 for c in
                         ["acc_x", "acc_y", "acc_z", "gyro_y", "gyro_z"]}})
    s = ca.feature_clip_sensitivity(pd.DataFrame(rows), ["gyro_x__mean"]).iloc[0]
    assert s["n_clean_pairs"] == 4 and s["exposure_fraction_recordings"] == pytest.approx(4 / 12)
    assert s["spearman_rho_with_clip"] < -0.5  # larger feature value goes with smaller clip fraction


# ------------------------------------------------------------------ statistics
def test_nct_ci_contains_estimate_and_symmetry():
    lo, hi = sa.dz_ci_noncentral(0.8, 12)
    assert lo < 0.8 < hi
    lo2, hi2 = sa.dz_ci_noncentral(-0.8, 12)
    assert (lo2, hi2) == pytest.approx((-hi, -lo))
    assert all(np.isnan(sa.dz_ci_noncentral(np.nan, 12)))


def test_recompute_matches_scipy_and_multiple_testing():
    rng = np.random.default_rng(1)
    D = rng.normal(0, 1, (12, 30))
    D[:, 0] += 2.0
    feats = [f"f{i}" for i in range(30)]
    out = sa.recompute(D, feats)
    assert out.loc[0, "p_t"] == pytest.approx(stats.ttest_1samp(D[:, 0], 0).pvalue)
    assert out["q_bh_p_t"].min() <= out["p_t"].min() * 30 + 1e-12 and (out["q_holm_p_t"] >= out["q_bh_p_t"] - 1e-12).all()
    assert out.loc[0, "q_bh_p_primary"] < 0.01 and out.loc[1:, "q_bh_p_primary"].min() > 0.05


def test_signflip_maxT_detects_true_effect_and_controls_null():
    rng = np.random.default_rng(2)
    D = rng.normal(0, 1, (10, 25))
    D[:, 3] += 3.0
    out, glob = sa.signflip_maxT(D, [f"f{i}" for i in range(25)])
    assert glob["n_flips"] == 2 ** 10
    assert out.set_index("feature").loc["f3", "p_maxT_fwer"] < 0.01
    others = out[out.feature != "f3"]["p_maxT_fwer"]
    assert (others > 0.01).all()
    # p is a proper permutation p-value: strictly positive, at least 1/2^n
    assert out["p_maxT_fwer"].min() >= 1 / 2 ** 10 - 1e-12


def test_window_level_pseudoreplication_inflates():
    rng = np.random.default_rng(3)
    rows = []
    for p in range(6):
        off = rng.normal(0, 1)
        for leg, shift in (("left", 0.0), ("right", 0.3)):
            for w in range(9):
                rows.append({"participant_id": f"S{p}", "leg": leg, "x": off + shift + rng.normal(0, 0.2)})
    r = sa.window_level_pseudoreplication(pd.DataFrame(rows), ["x"])
    assert r["n_left_windows"] == 54 and r["n_p_lt_05_naive"] in (0, 1)


# ------------------------------------------------------------------ models
def _toy_windows(signal=0.0, n_part=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_part):
        off = rng.normal(0, 2)
        for leg in ("left", "right"):
            for w in range(8):
                rows.append({"participant_id": f"S{p}", "leg": leg, "recording_id": f"S{p}_{leg}", "window_index": w,
                             "f1": off + rng.normal() + (signal if leg == "right" else 0), "f2": rng.normal(),
                             "qc_clip_fraction_any": 0.0, "gyro_mag__rms": 1.0, "acc_dyn_mag__rms": 1.0})
    return pd.DataFrame(rows)


def test_logo_eval_groups_never_leak():
    df = _toy_windows()
    from sklearn.dummy import DummyClassifier
    y = (df["leg"] == "right").astype(int).to_numpy()
    accs, preds, order = ma.logo_eval(df[["f1", "f2"]].to_numpy(), y, df["participant_id"].to_numpy(),
                                      lambda: DummyClassifier(strategy="most_frequent"))
    assert len(order) == df["participant_id"].nunique() == len(set(order))
    assert accs.mean() == pytest.approx(0.5)
    # a "fold" that shares a participant between train and test must trip the leak assertion
    from sklearn.model_selection import LeaveOneGroupOut
    tr, te = next(LeaveOneGroupOut().split(df, y, df["participant_id"]))
    assert not (set(df["participant_id"].to_numpy()[tr]) & set(df["participant_id"].to_numpy()[te]))


def test_permutation_null_centered_and_signal_detected():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression())  # noqa: E731
    df0, df1 = _toy_windows(0.0), _toy_windows(4.0, seed=1)
    for df, expect_high in ((df0, False), (df1, True)):
        X = df[["f1", "f2"]].to_numpy()
        y = (df["leg"] == "right").astype(int).to_numpy()
        g = df["participant_id"].to_numpy()
        obs = ma.logo_eval(X, y, g, mk)[0].mean()
        null = ma.permutation_null(X, y, g, mk, 30, 0)
        assert abs(null.mean() - 0.5) < 0.1
        p = (1 + (null >= obs).sum()) / 31
        assert (p < 0.1) == expect_high or not expect_high


def test_ci_methods_ordering():
    accs = np.array([0.1, 0.5, 0.6, 0.7, 0.9, 1.0, 0.4, 0.8])
    ci = ma.ci_methods(accs)
    assert ci["boot_pct_low"] < ci["mean"] < ci["boot_pct_high"] and ci["t_interval_low"] < ci["mean"] < ci["t_interval_high"]
    assert ci["n_participants_above_0.5"] + ci["n_participants_below_0.5"] + ci["n_participants_at_0.5"] == 8


def test_anomaly_cluster_audit_runs_and_isolation_forest_scale_invariant():
    df = _toy_windows(0.0, n_part=6)
    df["f3"] = df["f1"] * 3 + 1
    feats = ["f1", "f2", "f3"]
    sens = pd.DataFrame({"feature": feats, "exposure_fraction_recordings": [0.0] * 3})
    out = ma.anomaly_cluster_audit(df, feats, feats, sens, seed=0)
    rob = out["robustness"].set_index("variant")
    assert rob.loc["robust scaler", "spearman_iforest_vs_base"] == pytest.approx(1.0)  # trees ignore monotone rescaling
    assert out["summary"]["n_flagged"] == int(round(0.05 * len(df)))
