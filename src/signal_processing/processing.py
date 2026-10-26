"""Documented signal-processing steps: regularization, gravity estimation, filtering, magnitudes, windowing.

Design rules
------------
* Raw values are never modified in place; every step returns new arrays.
* Nothing is filtered unless the config says so. Every decision is appended to ``ProcessedRecording.log``.
* The sampling frequency is only ever *estimated from timestamps*; if no usable timestamps exist it is ``None``
  and frequency-domain processing is refused.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from src.data_validation.channels import CHANNEL_ORDER, family
from src.data_validation.validate import clipped_mask


@dataclass
class ProcessedRecording:
    recording_id: str
    participant_id: str
    leg: str
    filename: str
    fs_hz: float | None
    t_ms: np.ndarray                 # uniform time axis (or sample index if fs_hz is None)
    raw: pd.DataFrame                # regularized raw channels, original units/labels
    proc: pd.DataFrame               # signals used for features (raw channels [+ optional filter], dyn accel, magnitudes)
    interpolated: np.ndarray         # bool, sample was filled by interpolation
    clipped: pd.DataFrame            # bool per channel: sample at an observed rail value (extreme value; saturation unconfirmed)
    log: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- filters
def lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass. Raises on invalid cutoff or too-short input."""
    if fs is None or fs <= 0:
        raise ValueError("A valid sampling frequency is required for filtering")
    nyq = fs / 2.0
    if not 0 < cutoff_hz < nyq:
        raise ValueError(f"Low-pass cutoff {cutoff_hz} Hz must be in (0, Nyquist={nyq} Hz)")
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        raise ValueError(f"Signal length {len(x)} too short for filtfilt (needs > {padlen})")
    return filtfilt(b, a, x)


def estimate_gravity(acc: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Slowly varying (posture/gravity) component of one accelerometer axis via low-pass."""
    return lowpass(acc, fs, cutoff_hz, order)


def magnitude(*axes: np.ndarray) -> np.ndarray:
    """Euclidean vector magnitude of equally long axis arrays."""
    if len({len(a) for a in axes}) != 1:
        raise ValueError("Axes must have equal length")
    return np.sqrt(np.sum(np.square(np.vstack(axes)), axis=0))


# --------------------------------------------------------------------------- regularization
def regularize(t_ms: np.ndarray, values: pd.DataFrame, gap_factor: float = 1.5
               ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, dict]:
    """Place samples on a uniform grid with the median period, linearly interpolating missing samples.

    Returns (grid_t_ms, values_on_grid, interpolated_mask, info).
    Grid points that have an original sample within 25 % of a period keep that sample's value
    and are not flagged. Non-finite values are also treated as missing and interpolated (and flagged).
    """
    t = np.asarray(t_ms, dtype=float)
    ok = np.isfinite(t)
    t, values = t[ok], values.loc[ok].reset_index(drop=True)
    if len(t) < 2:
        raise ValueError("Need at least two timestamps to regularize")
    if (np.diff(t) < 0).any():
        raise ValueError("Timestamps are not monotonic; refusing to regularize")
    keep = np.concatenate([[True], np.diff(t) > 0])
    n_dup = int((~keep).sum())
    t, values = t[keep], values.loc[keep].reset_index(drop=True)
    dt = float(np.median(np.diff(t)))
    n = int(round((t[-1] - t[0]) / dt)) + 1
    grid = t[0] + np.arange(n) * dt
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    nearest = np.minimum(np.abs(t[idx] - grid), np.abs(t[idx - 1] - grid))
    has_orig = nearest <= 0.25 * dt
    out = {}
    nan_filled = np.zeros(n, dtype=bool)
    for col in values.columns:
        v = values[col].to_numpy(dtype=float)
        fin = np.isfinite(v)
        if fin.sum() < 2:
            raise ValueError(f"Channel {col} has fewer than 2 finite samples")
        out[col] = np.interp(grid, t[fin], v[fin])
        if (~fin).any():
            bad_t = t[~fin]
            j = np.clip(np.searchsorted(grid, bad_t), 0, n - 1)
            nan_filled[j] = True
    interp = ~has_orig | nan_filled
    info = {"dt_ms": dt, "n_original": int(len(t)), "n_grid": n, "n_interpolated": int(interp.sum()),
            "n_duplicate_timestamps_dropped": n_dup, "n_nonfinite_filled": int(nan_filled.sum())}
    return grid, pd.DataFrame(out), interp, info


# --------------------------------------------------------------------------- windowing
def make_windows(n_samples: int, fs_hz: float | None, window_s: float, overlap: float,
                 incomplete: str = "drop", fallback_window_samples: int = 100,
                 min_window_samples: int = 8) -> tuple[list[tuple[int, int]], dict]:
    """Fixed-length windows as (start, end_exclusive) sample indices.

    ``incomplete``: ``drop`` ignores a tail shorter than a full window; ``align_end`` adds one extra
    full-length window ending at the last sample (so it overlaps more with its neighbour).
    A window is never shorter than the requested length.
    """
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if incomplete not in ("drop", "align_end"):
        raise ValueError(f"incomplete must be 'drop' or 'align_end', got {incomplete!r}")
    info: dict = {"window_s": window_s, "overlap": overlap, "incomplete_policy": incomplete}
    if fs_hz is None:
        win = int(fallback_window_samples)
        info["note"] = "no verified sampling frequency: window length given in samples, not seconds"
    else:
        win = int(round(window_s * fs_hz))
    if win < min_window_samples:
        raise ValueError(f"Window of {win} samples is below the minimum {min_window_samples}")
    if win > n_samples:
        raise ValueError(f"Window of {win} samples exceeds recording length {n_samples}")
    step = max(1, int(round(win * (1 - overlap))))
    starts = list(range(0, n_samples - win + 1, step))
    wins = [(s, s + win) for s in starts]
    tail = n_samples - wins[-1][1]
    if tail > 0 and incomplete == "align_end":
        wins.append((n_samples - win, n_samples))
    info.update(window_samples=win, step_samples=step, n_windows=len(wins), uncovered_tail_samples=tail
                if incomplete == "drop" else 0)
    return wins, info


# --------------------------------------------------------------------------- orchestration
def prepare_recording(rec, df: pd.DataFrame, chmap: dict, time_col: str | None, levels: dict, cfg: dict
                      ) -> ProcessedRecording:
    """Build the processed representation of one recording and log every decision."""
    pc, ing = cfg["processing"], cfg["ingestion"]
    log: dict = {"recording_id": rec.recording_id, "filename": rec.filename, "steps": []}
    chans = [c for c in CHANNEL_ORDER if c in chmap]
    if not chans:
        raise ValueError(f"{rec.filename}: no recognizable accelerometer/gyroscope channels")
    vals = pd.DataFrame({c: pd.to_numeric(df[chmap[c]["column"]], errors="coerce") for c in chans})

    fs: float | None = None
    if time_col is not None and pc["regularize_to_uniform_grid"]:
        t_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        t, raw, interp, info = regularize(t_raw, vals, ing["gap_factor"])
        fs = 1000.0 / info["dt_ms"]
        log["steps"].append({"step": "regularize_to_uniform_grid", "method": "linear interpolation of missing "
                             "samples; original samples kept", **info, "fs_hz_from_timestamps": fs})
    else:
        t = np.arange(len(vals), dtype=float)
        raw, interp = vals.copy(), np.zeros(len(vals), dtype=bool)
        log["steps"].append({"step": "no_regularization", "reason": "no timestamp column or disabled in config"})
        if time_col is not None:
            fs = 1000.0 / float(np.median(np.diff(pd.to_numeric(df[time_col], errors="coerce"))))
    log["fs_hz"] = fs

    clip = pd.DataFrame({c: clipped_mask(raw[c].to_numpy(), family(c), levels, ing["saturation"]["rel_tol"])
                         for c in chans})
    log["steps"].append({"step": "clipping_flag", "note": "samples at an observed rail value (repeated identical extreme; hardware saturation unconfirmed) are flagged, "
                         "not corrected; features computed from them are biased", "n_clipped_by_channel":
                         {c: int(clip[c].sum()) for c in chans}})

    proc = raw.copy()
    if pc["lowpass_hz"]:
        if fs is None:
            raise ValueError("lowpass_hz is set but the sampling frequency is unknown")
        for c in chans:
            proc[c] = lowpass(raw[c].to_numpy(), fs, pc["lowpass_hz"], pc["filter_order"])
        log["steps"].append({"step": "lowpass", "type": "zero-phase Butterworth", "cutoff_hz": pc["lowpass_hz"],
                             "order": pc["filter_order"]})
    else:
        log["steps"].append({"step": "lowpass", "applied": False, "reason": "lowpass_hz is null in config"})

    if all(c in chans for c in ("acc_x", "acc_y", "acc_z")):
        if fs is None:
            log["steps"].append({"step": "gravity_removal", "applied": False, "reason": "unknown fs"})
        else:
            for ax in "xyz":
                g = estimate_gravity(proc[f"acc_{ax}"].to_numpy(), fs, pc["gravity_cutoff_hz"], pc["filter_order"])
                proc[f"acc_dyn_{ax}"] = proc[f"acc_{ax}"].to_numpy() - g
            proc["acc_dyn_mag"] = magnitude(*(proc[f"acc_dyn_{a}"].to_numpy() for a in "xyz"))
            log["steps"].append({"step": "gravity_removal", "method": "subtract zero-phase low-pass estimate",
                                 "cutoff_hz": pc["gravity_cutoff_hz"], "order": pc["filter_order"],
                                 "caveat": "low-pass of a 10 s clip has edge effects; sensor orientation relative "
                                 "to gravity is NOT assumed, the estimate is data-driven"})
        proc["acc_mag"] = magnitude(*(proc[f"acc_{a}"].to_numpy() for a in "xyz"))
    else:
        log["steps"].append({"step": "acc_magnitude", "applied": False, "reason": "not all accelerometer axes present"})
    if all(c in chans for c in ("gyro_x", "gyro_y", "gyro_z")):
        proc["gyro_mag"] = magnitude(*(proc[f"gyro_{a}"].to_numpy() for a in "xyz"))
    else:
        log["steps"].append({"step": "gyro_magnitude", "applied": False, "reason": "not all gyroscope axes present"})

    return ProcessedRecording(rec.recording_id, rec.participant_id, rec.leg, rec.filename, fs, t, raw, proc,
                              interp, clip, log)
