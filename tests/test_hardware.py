"""Tests for the hardware-informed audit helpers and metadata handling (synthetic inputs)."""
import numpy as np
import pandas as pd
import pytest

from src.audit import hardware_audit as ha
from src.audit import stats_audit as sa
from src.data_validation.discovery import Recording
from src.data_validation.metadata import documented_vs_observed, participant_metadata


def _recording(pid, leg, note=""):
    return Recording(path="x", filename=f"{pid}{leg}.csv", participant_id=pid, participant_num=int(pid[3:]), leg=leg, note=note,
                     recording_id=f"{pid}_{leg}", sha256="0")


def _raw_from_counts(counts_g, counts_a, pid="Sub01", leg="left"):
    t = np.arange(1, len(counts_g) + 1) * 20
    d = pd.DataFrame({"Time_ms": t, "AccelX_g": np.round(counts_a / 16384, 4), "AccelY_g": np.round(counts_a / 16384, 4),
                      "AccelZ_g": np.round(counts_a / 16384, 4), "GyroX_dps": np.round(counts_g / 131, 3),
                      "GyroY_dps": np.round(counts_g / 131, 3), "GyroZ_dps": np.round(counts_g / 131, 3)})
    return {(pid, leg): d}


def test_quantisation_gyro_informative_accel_not(cfg):
    rng = np.random.default_rng(0)
    raw = _raw_from_counts(rng.integers(-32768, 32768, 2000), rng.integers(-32768, 32768, 2000))
    q = ha.quantisation_check(raw, cfg).set_index("channel")
    assert q.loc["gyro_x", "fraction_on_attainable_grid"] == 1.0 and q.loc["gyro_x", "informative"]
    assert q.loc["acc_x", "null_fraction_random_decimals"] > 0.95 and not q.loc["acc_x", "informative"]
    # values that are NOT integer counts / 131 (e.g. after float filtering) fall off the grid
    raw[("Sub01", "left")]["GyroX_dps"] = np.round(raw[("Sub01", "left")]["GyroX_dps"] * 0.9731 + 0.0004, 3)
    q2 = ha.quantisation_check(raw, cfg).set_index("channel")
    assert q2.loc["gyro_x", "fraction_on_attainable_grid"] < 0.5


def test_rail_expectation_matches_int16_limits(cfg):
    counts = np.array([-32768, 0, 32767, 32767, 100])
    raw = _raw_from_counts(counts, counts)
    r = ha.rail_expectation(raw, cfg)
    assert r["match"].all() and set(r["expected_csv_value"]) == {1.9999, -2.0, 250.13, -250.137}
    assert (r[(r.family == "gyro") & (r.end == "max")]["n_at_expected"] == 6).all()  # two samples x three axes


def test_plateau_table_and_wraparound(cfg):
    g = np.zeros(40, dtype=int)
    g[10:14] = 32767                      # a 4-sample plateau with a ramp in and out
    g[8], g[9], g[14] = 20000, 28000, 30000
    g[30], g[31] = 32767, -32768          # an adjacent jump rail -> opposite rail (overflow-like)
    raw = _raw_from_counts(g, np.zeros(40, dtype=int))
    pl = ha.plateau_table(raw, cfg)
    gx = pl[(pl.channel == "gyro_x") & (pl.sign == "+")].sort_values("start_idx")
    assert gx["length"].tolist() == [4, 1] and gx.iloc[0]["start_idx"] == 10
    assert gx.iloc[0]["rise_into_plateau"] > 0 and gx.iloc[0]["rise_prev"] > 0 and gx.iloc[0]["fall_out_of_plateau"] > 0
    w = ha.wraparound_check(raw, cfg).set_index("channel")
    assert w.loc["gyro_x", "n_rail_to_opposite_rail"] == 1 and w.loc["acc_x", "n_rail_to_opposite_rail"] == 0


def test_static_window_picks_quiet_segment_and_gravity_norm():
    n = 200
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"Time_ms": np.arange(1, n + 1) * 20, "AccelX_g": 0.0, "AccelY_g": -1.0, "AccelZ_g": 0.0,
                      "GyroX_dps": rng.normal(0, 50, n), "GyroY_dps": rng.normal(0, 50, n), "GyroZ_dps": rng.normal(0, 50, n)})
    d.loc[100:130, ["GyroX_dps", "GyroY_dps", "GyroZ_dps"]] = 0.5
    sl = ha.static_window(d, 25)
    assert 100 <= sl.start <= 106
    raw = {("Sub01", "left"): d, ("Sub01", "right"): d}
    st = ha.static_scale_offset(raw)
    assert st.loc[0, "acc_norm_g"] == pytest.approx(1.0)
    og = ha.orientation_diagnostics(raw, st)
    assert og["gravity"]["dominant_gravity_axis"].tolist() == ["y", "y"] and og["gravity"]["unit_y"].iloc[0] == pytest.approx(-1.0)
    assert og["angle_left_right"].iloc[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_movement_onset_detects_start():
    n = 200
    g = np.zeros(n)
    g[60:] = 100.0
    d = pd.DataFrame({"Time_ms": np.arange(1, n + 1) * 20, "AccelX_g": 0, "AccelY_g": -1, "AccelZ_g": 0,
                      "GyroX_dps": g, "GyroY_dps": 0.0, "GyroZ_dps": 0.0})
    on = ha.movement_onset({("Sub01", "left"): d}, 50.0)
    # rolling 25-sample mean exceeds 50 once 13 high samples are inside the window; the reported time is the window start
    assert 60 * 20 - 600 <= on.loc[0, "onset_ms"] <= 60 * 20 + 600


def test_sign_invariant_feature_selection():
    feats = ["acc_x__mean", "acc_x__std", "acc_x__q05", "gyro_mag__mean", "gyro_mag__q05", "corr__acc_x__acc_y",
             "gyro_z__dominant_freq_hz", "acc_y__max", "acc_z__ptp", "acc_dyn_mag__rms"]
    keep = ha.sign_invariant_features(feats)
    assert keep == ["acc_x__std", "gyro_mag__mean", "gyro_mag__q05", "gyro_z__dominant_freq_hz", "acc_z__ptp", "acc_dyn_mag__rms"]


def test_sign_invariant_features_really_are_invariant(cfg):
    """Negating an axis must leave the selected single-axis features unchanged and change the excluded ones."""
    from src.features.extract import window_features
    rng = np.random.default_rng(3)
    base = {c: rng.standard_normal(100) + (1 if c == "acc_x" else 0) for c in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]}
    flipped = dict(base)
    flipped["acc_x"] = -base["acc_x"]
    f0, f1 = window_features(base, 50.0, cfg), window_features(flipped, 50.0, cfg)
    kept = ha.sign_invariant_features([k for k in f0 if k.startswith("acc_x__")])
    assert len(kept) > 5
    for f in kept:
        assert f1[f] == pytest.approx(f0[f]), f
    assert f1["acc_x__mean"] == pytest.approx(-f0["acc_x__mean"]) and f1["corr__acc_x__acc_y"] == pytest.approx(-f0["corr__acc_x__acc_y"])


def test_trimmed_features_use_fewer_samples(cfg):
    from src.signal_processing.processing import ProcessedRecording
    n = 500
    rng = np.random.default_rng(1)
    cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    raw = pd.DataFrame(rng.standard_normal((n, 6)), columns=cols)
    proc = raw.copy()
    proc["acc_mag"] = np.sqrt((raw[cols[:3]] ** 2).sum(axis=1))
    proc["gyro_mag"] = np.sqrt((raw[cols[3:]] ** 2).sum(axis=1))
    pr = ProcessedRecording("Sub99_left", "Sub99", "left", "f.csv", 50.0, 20.0 + np.arange(n) * 20.0, raw, proc, np.zeros(n, bool),
                            pd.DataFrame(False, index=range(n), columns=cols))
    out = ha.trimmed_recording_features([pr], cfg, 2.0)
    assert out.loc[0, "n_samples"] == 400 and out.loc[0, "start_ms"] == pr.t_ms[100]


def test_paired_subset_matches_full_recompute():
    rng = np.random.default_rng(0)
    rows = [{"participant_id": f"Sub{i:02d}", "leg": leg, "a__std": rng.normal(), "b__std": rng.normal()}
            for i in range(8) for leg in ("left", "right")]
    rec = pd.DataFrame(rows)
    sub = ha.paired_subset(rec, ["a__std"], sa)
    full = ha.paired_subset(rec, ["a__std", "b__std"], sa)
    assert sub.loc[0, "p_t"] == pytest.approx(full.loc[0, "p_t"]) and ha.summarise_paired(full)["n_tested"] == 2


# ------------------------------------------------------------------ metadata
def test_participant_metadata_from_config(cfg):
    recs = [_recording("Sub04", "left", "recovering acl"), _recording("Sub04", "right", "recovering acl"),
            _recording("Sub06", "left"), _recording("Sub05", "left")]
    pm = participant_metadata(recs, cfg).set_index("participant_id")
    assert pm.loc["Sub04", "collaborator_reported_condition"] == "ACL recovery history" and pm.loc["Sub04", "has_reported_history"]
    assert pm.loc["Sub06", "has_reported_history"] and pm.loc["Sub06", "filename_note"] == ""
    assert not pm.loc["Sub05", "has_reported_history"] and pm.loc["Sub05", "collaborator_reported_condition"].startswith("described as healthy")
    assert not pm["usable_as_outcome_label"].any()
    assert (pm["source"] == "hardware_collaborator").all() and pm["verification_status"].str.startswith("not verified").all()


def test_documented_vs_observed_flags(cfg):
    rep = pd.DataFrame([{"recording_id": "a", "n_rows": 500, "duration_s": 9.98, "estimated_fs_hz": 50.0, "first_timestamp_ms": 20.0,
                         "last_timestamp_ms": 10000.0, "n_gaps": 0, "n_missing_samples_est": 0},
                        {"recording_id": "b", "n_rows": 420, "duration_s": 8.0, "estimated_fs_hz": 41.0, "first_timestamp_ms": 20.0,
                         "last_timestamp_ms": 8000.0, "n_gaps": 3, "n_missing_samples_est": 80}])
    d = documented_vs_observed(rep, cfg).set_index("recording_id")
    assert d.loc["a", "rows_within_tolerance"] and d.loc["a", "fs_matches"]
    assert not d.loc["b", "rows_within_tolerance"] and not d.loc["b", "fs_matches"]
