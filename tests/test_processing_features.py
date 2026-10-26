import numpy as np
import pandas as pd
import pytest

from src.features.extract import (extract_recording_features, feature_columns, time_domain, spectral,
                                  window_features, zero_crossing_rate, correlation)
from src.signal_processing.processing import (ProcessedRecording, lowpass, magnitude, make_windows, regularize)

FS = 50.0


# ------------------------------------------------------------------ processing
def test_regularize_fills_gap_and_flags():
    t = np.array([20, 40, 80, 100, 120.0])  # 60 ms missing
    df = pd.DataFrame({"a": [0.0, 1.0, 3.0, 4.0, 5.0]})
    grid, out, interp, info = regularize(t, df)
    assert list(grid) == [20, 40, 60, 80, 100, 120]
    assert interp.tolist() == [False, False, True, False, False, False]
    assert out["a"].tolist() == [0, 1, 2, 3, 4, 5]
    assert info["n_interpolated"] == 1


def test_regularize_rejects_nonmonotonic():
    with pytest.raises(ValueError, match="monotonic"):
        regularize(np.array([20, 40, 30, 60.0]), pd.DataFrame({"a": np.arange(4.0)}))


def test_regularize_fills_nan_and_flags():
    t = np.arange(1, 11) * 20.0
    v = np.arange(10.0)
    v[4] = np.nan
    _, out, interp, info = regularize(t, pd.DataFrame({"a": v}))
    assert out["a"][4] == 4.0 and interp[4] and info["n_nonfinite_filled"] == 1


def test_windows_basic_and_policies():
    w, info = make_windows(500, FS, 2.0, 0.5)
    assert info["window_samples"] == 100 and info["step_samples"] == 50 and len(w) == 9 and w[-1] == (400, 500)
    w2, i2 = make_windows(499, FS, 2.0, 0.5, "drop")
    assert len(w2) == 8 and i2["uncovered_tail_samples"] == 49
    w3, _ = make_windows(499, FS, 2.0, 0.5, "align_end")
    assert len(w3) == 9 and w3[-1] == (399, 499) and all(b - a == 100 for a, b in w3)


def test_windows_invalid_arguments():
    with pytest.raises(ValueError, match="exceeds recording length"):
        make_windows(50, FS, 2.0, 0.5)
    with pytest.raises(ValueError, match="overlap"):
        make_windows(500, FS, 2.0, 1.0)
    with pytest.raises(ValueError, match="below the minimum"):
        make_windows(500, FS, 0.05, 0.5)
    with pytest.raises(ValueError, match="incomplete"):
        make_windows(500, FS, 2.0, 0.5, "pad")


def test_windows_without_fs_use_sample_fallback():
    w, info = make_windows(300, None, 2.0, 0.0, fallback_window_samples=100)
    assert len(w) == 3 and "no verified sampling frequency" in info["note"]


def test_lowpass_validation_and_effect():
    t = np.arange(500) / FS
    x = np.sin(2 * np.pi * 1 * t) + np.sin(2 * np.pi * 20 * t)
    y = lowpass(x, FS, 5.0)
    assert np.std(y - np.sin(2 * np.pi * t))[()] < 0.1
    with pytest.raises(ValueError, match="Nyquist"):
        lowpass(x, FS, 30.0)
    with pytest.raises(ValueError, match="too short"):
        lowpass(x[:10], FS, 5.0)
    with pytest.raises(ValueError, match="sampling frequency"):
        lowpass(x, None, 5.0)


def test_magnitude():
    assert np.allclose(magnitude(np.array([3.0]), np.array([4.0]), np.array([0.0])), [5.0])
    with pytest.raises(ValueError):
        magnitude(np.zeros(3), np.zeros(4))


# ------------------------------------------------------------------ features
def test_time_domain_known_values(cfg):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    f = time_domain(x, 0.02, cfg["features"]["quantiles"])
    assert f["mean"] == 2.5 and f["min"] == 1 and f["max"] == 4 and f["ptp"] == 3
    assert f["var"] == pytest.approx(np.var(x, ddof=1)) and f["std"] == pytest.approx(np.std(x, ddof=1))
    assert f["rms"] == pytest.approx(np.sqrt(7.5))
    assert f["energy"] == pytest.approx(30 * 0.02)
    assert f["q50"] == 2.5


def test_time_domain_rejects_bad_input():
    with pytest.raises(ValueError, match=">= 2"):
        time_domain(np.array([1.0]), 0.02, [0.5])
    with pytest.raises(ValueError, match="Non-finite"):
        time_domain(np.array([1.0, np.nan, 2.0]), 0.02, [0.5])


def test_spectral_dominant_frequency():
    t = np.arange(500) / FS
    x = 3 + np.sin(2 * np.pi * 2.0 * t)
    s = spectral(x, FS, 64, 0.3, 3.0)
    assert s["dominant_freq_hz"] == pytest.approx(2.0, abs=0.1)
    assert s["low_band_fraction"] > 0.9
    assert s["spectral_energy"] == pytest.approx(0.5, rel=0.15)  # variance of unit sine, Hann-windowed estimate


def test_spectral_nan_when_unsupported():
    x = np.random.default_rng(0).standard_normal(500)
    assert all(np.isnan(v) for v in spectral(x, None, 64, 0.3, 3.0).values())  # unknown fs
    assert all(np.isnan(v) for v in spectral(x[:30], FS, 64, 0.3, 3.0).values())  # too short
    assert all(np.isnan(v) for v in spectral(np.ones(200), FS, 64, 0.3, 3.0).values())  # zero power


def test_zero_crossing_rate_and_unknown_fs():
    t = np.arange(500) / FS
    assert zero_crossing_rate(np.sin(2 * np.pi * 2 * t + 0.1), FS) == pytest.approx(4.0, abs=0.3)
    assert np.isnan(zero_crossing_rate(np.sin(t), None))
    assert zero_crossing_rate(np.ones(10), FS) == 0.0


def test_correlation_constant_is_nan():
    assert np.isnan(correlation(np.ones(10), np.arange(10.0)))
    assert correlation(np.arange(10.0), 2 * np.arange(10.0)) == pytest.approx(1.0)


def test_window_features_missing_channels_no_magnitudes(cfg):
    rng = np.random.default_rng(0)
    sig = {"acc_x": rng.standard_normal(100), "acc_y": rng.standard_normal(100)}
    f = window_features(sig, FS, cfg)
    assert "acc_x__mean" in f and "corr__acc_x__acc_y" in f
    assert not any(k.startswith(("gyro", "acc_mag", "acc_z")) for k in f)


def test_window_features_constant_signal_and_invalid(cfg):
    f = window_features({"acc_x": np.ones(100), "acc_y": np.arange(100.0)}, FS, cfg)
    assert f["acc_x__std"] == 0 and np.isnan(f["corr__acc_x__acc_y"]) and np.isnan(f["acc_x__dominant_freq_hz"])
    with pytest.raises(ValueError, match="equal length"):
        window_features({"acc_x": np.ones(10), "acc_y": np.ones(11)}, FS, cfg)
    with pytest.raises(ValueError, match="No signals"):
        window_features({}, FS, cfg)


def _fake_pr(n=500, fs=FS):
    rng = np.random.default_rng(1)
    cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    raw = pd.DataFrame(rng.standard_normal((n, 6)), columns=cols)
    proc = raw.copy()
    proc["acc_mag"] = np.sqrt((raw[cols[:3]] ** 2).sum(axis=1))
    proc["gyro_mag"] = np.sqrt((raw[cols[3:]] ** 2).sum(axis=1))
    return ProcessedRecording("Sub99_left", "Sub99", "left", "Sub99Left(x).csv", fs, 20.0 + np.arange(n) * 20.0, raw, proc,
                              np.zeros(n, bool), pd.DataFrame(False, index=range(n), columns=cols))


def test_extract_keeps_identifiers(cfg):
    pr = _fake_pr()
    wins, info = make_windows(500, FS, 2.0, 0.5)
    w, r = extract_recording_features(pr, wins, cfg)
    assert len(w) == 9 and len(r) == 1
    assert (w["participant_id"] == "Sub99").all() and (w["filename"] == "Sub99Left(x).csv").all()
    assert w["window_index"].tolist() == list(range(9))
    assert w.loc[1, "start_ms"] == pr.t_ms[50] and w.loc[1, "end_ms"] == pr.t_ms[149]
    assert r.loc[0, "scope"] == "recording" and r.loc[0, "n_samples"] == 500
    assert w["feature_config_id"].nunique() == 1
    feats = feature_columns(w)
    assert "gyro_mag__rms" in feats and not any(c.startswith("qc_") for c in feats) and "leg" not in feats
    assert np.isfinite(w[feats].to_numpy(dtype=float)).all()
