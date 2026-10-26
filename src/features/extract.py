"""Feature extraction for windows and whole recordings.

Conventions (also written to the report)
----------------------------------------
* ``std`` / ``var`` use ddof=1 (sample statistics within the window).
* ``energy`` = sum(x^2) * dt  (units^2 * s; for a fixed window length it is proportional to rms^2).
* ``rms`` = sqrt(mean(x^2)).  ``ptp`` = max - min.
* ``zcr_hz``: sign changes of the mean-removed signal per second (computed for axes only).
* Spectral features use a Hann-windowed one-sided PSD of the mean-removed signal; they are NaN unless a sampling
  frequency was estimated from timestamps AND the window has >= ``min_samples_for_spectrum`` samples.
* Magnitude features exist only if all three axes of that sensor are present.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.signal import get_window

from src.signal_processing.processing import ProcessedRecording

ID_COLS = ["participant_id", "leg", "filename", "recording_id", "scope", "window_index", "start_idx", "end_idx",
           "start_ms", "end_ms", "n_samples", "fs_hz", "feature_config_id"]
AXIS_SIGNALS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
MAG_SIGNALS = ["acc_mag", "acc_dyn_mag", "gyro_mag"]
STAT_NAMES = ["mean", "std", "var", "min", "max", "ptp", "rms", "energy"]


def feature_config_id(cfg: dict) -> str:
    """Short hash of the settings that influence the feature values."""
    blob = json.dumps({"processing": cfg["processing"], "features": cfg["features"]}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def time_domain(x: np.ndarray, dt_s: float, quantiles: list[float]) -> dict[str, float]:
    """Time-domain descriptors of a 1-D signal (see module docstring for conventions)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"Need a 1-D signal with >= 2 samples, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite values in signal; interpolate or drop before feature extraction")
    out = {
        "mean": x.mean(), "std": x.std(ddof=1), "var": x.var(ddof=1), "min": x.min(), "max": x.max(),
        "ptp": np.ptp(x), "rms": float(np.sqrt(np.mean(x ** 2))), "energy": float(np.sum(x ** 2) * dt_s),
    }
    for q, v in zip(quantiles, np.quantile(x, quantiles)):
        out[f"q{int(round(q * 100)):02d}"] = v
    return {k: float(v) for k, v in out.items()}


def zero_crossing_rate(x: np.ndarray, fs_hz: float | None) -> float:
    """Sign changes of the mean-removed signal per second (NaN when fs unknown)."""
    if fs_hz is None:
        return float("nan")
    s = np.sign(np.asarray(x, dtype=float) - np.mean(x))
    s = s[s != 0]
    if s.size < 2:
        return 0.0
    return float((np.diff(s) != 0).sum() / (len(x) / fs_hz))


def spectral(x: np.ndarray, fs_hz: float | None, min_samples: int, min_freq_hz: float,
             low_band_upper_hz: float) -> dict[str, float]:
    """Frequency-domain descriptors; all NaN if the sampling frequency is unknown or the window too short."""
    names = ["dominant_freq_hz", "spectral_energy", "spectral_centroid_hz", "spectral_entropy", "low_band_fraction"]
    nan = {k: float("nan") for k in names}
    x = np.asarray(x, dtype=float)
    if fs_hz is None or x.size < min_samples:
        return nan
    n = x.size
    w = get_window("hann", n)
    X = np.fft.rfft((x - x.mean()) * w)
    psd = (np.abs(X) ** 2) / (fs_hz * np.sum(w ** 2))
    psd[1:-1 if n % 2 == 0 else None] *= 2.0  # one-sided
    f = np.fft.rfftfreq(n, 1.0 / fs_hz)
    band = f >= min_freq_hz
    p = psd[band]
    total = p.sum()
    if not band.any() or total <= 0 or not np.isfinite(total):
        return nan
    fb = f[band]
    df = f[1] - f[0]
    pn = p / total
    ent = -np.sum(pn[pn > 0] * np.log(pn[pn > 0])) / np.log(len(pn)) if len(pn) > 1 else float("nan")
    return {
        "dominant_freq_hz": float(fb[np.argmax(p)]),
        "spectral_energy": float(total * df),
        "spectral_centroid_hz": float(np.sum(fb * pn)),
        "spectral_entropy": float(ent),
        "low_band_fraction": float(pn[fb < low_band_upper_hz].sum()),
    }


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; NaN if either signal is constant."""
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def window_features(signals: dict[str, np.ndarray], fs_hz: float | None, cfg: dict) -> dict[str, float]:
    """All features for one window given a dict of equally long 1-D signals.

    Missing channels simply produce no features (no imputation).
    """
    fc = cfg["features"]
    if not signals:
        raise ValueError("No signals supplied")
    if len({len(v) for v in signals.values()}) != 1:
        raise ValueError("All signals in a window must have equal length")
    dt_s = 1.0 / fs_hz if fs_hz else 1.0
    feats: dict[str, float] = {}
    for name, x in signals.items():
        for k, v in time_domain(x, dt_s, fc["quantiles"]).items():
            feats[f"{name}__{k}"] = v
        if name in AXIS_SIGNALS:
            feats[f"{name}__zcr_hz"] = zero_crossing_rate(x, fs_hz)
        for k, v in spectral(x, fs_hz, fc["min_samples_for_spectrum"], fc["spectrum_min_freq_hz"],
                             fc["low_band_upper_hz"]).items():
            feats[f"{name}__{k}"] = v
    axes = [a for a in AXIS_SIGNALS if a in signals]
    for a, b in combinations(axes, 2):
        feats[f"corr__{a}__{b}"] = correlation(signals[a], signals[b])
    if "acc_mag" in signals and "gyro_mag" in signals:
        feats["corr__acc_mag__gyro_mag"] = correlation(signals["acc_mag"], signals["gyro_mag"])
    return feats


def extract_recording_features(pr: ProcessedRecording, windows: list[tuple[int, int]], cfg: dict
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (window-level table, whole-recording table) with identifiers on every row."""
    cid = feature_config_id(cfg)
    sig_names = [c for c in AXIS_SIGNALS + MAG_SIGNALS if c in pr.proc.columns]
    arrays = {c: pr.proc[c].to_numpy() for c in sig_names}
    maxfrac = cfg["processing"]["max_interpolated_fraction_per_window"]

    def one(start: int, end: int, scope: str, widx) -> dict:
        sl = slice(start, end)
        feats = window_features({k: v[sl] for k, v in arrays.items()}, pr.fs_hz, cfg)
        frac_interp = float(pr.interpolated[sl].mean())
        row = {
            "participant_id": pr.participant_id, "leg": pr.leg, "filename": pr.filename,
            "recording_id": pr.recording_id, "scope": scope, "window_index": widx, "start_idx": start,
            "end_idx": end, "start_ms": float(pr.t_ms[start]), "end_ms": float(pr.t_ms[end - 1]),
            "n_samples": end - start, "fs_hz": pr.fs_hz, "feature_config_id": cid,
            "qc_interp_fraction": frac_interp, "qc_interp_exceeds_limit": frac_interp > maxfrac,
            "qc_clip_fraction_any": float(pr.clipped.iloc[sl].any(axis=1).mean()),
        }
        for c in pr.clipped.columns:
            row[f"qc_clip_fraction_{c}"] = float(pr.clipped[c].iloc[sl].mean())
        row.update(feats)
        return row

    win_rows = [one(s, e, "window", i) for i, (s, e) in enumerate(windows)]
    rec_row = [one(0, len(pr.t_ms), "recording", pd.NA)]
    return pd.DataFrame(win_rows), pd.DataFrame(rec_row)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns (excludes identifiers and ``qc_*`` columns)."""
    return [c for c in df.columns if c not in ID_COLS and not c.startswith("qc_")]
