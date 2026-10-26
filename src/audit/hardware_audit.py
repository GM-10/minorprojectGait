"""Re-audit against the collaborator-reported hardware configuration.

The collaborator reported: MPU6050 (GY-521), +-2 g and +-250 deg/s, conversion raw/16384 (g) and raw/131 (deg/s),
50 Hz, module mounted inverted on the shin, the same sensor used for both legs. Raw int16 register values, firmware
and board photographs are NOT available. Everything below therefore tests *consistency* of the CSV values with those
statements; none of it can confirm firmware behaviour, sensor initialisation or the anatomical axis mapping.
"""
from __future__ import annotations

import dataclasses
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from src.audit.clipping_audit import COLS
from src.audit.common import feature_signal, feature_stat

ACC = ["acc_x", "acc_y", "acc_z"]
GYR = ["gyro_x", "gyro_y", "gyro_z"]


def _spec(cfg: dict, fam: str) -> tuple[float, int]:
    hw = cfg["hardware"]
    if fam == "acc":
        return float(hw["conversion"]["accel_lsb_per_g"]), int(hw["csv_decimals"]["accel"])
    return float(hw["conversion"]["gyro_lsb_per_dps"]), int(hw["csv_decimals"]["gyro"])


# ----------------------------------------------------------------------------- 1. quantisation and rails
def quantisation_check(raw: dict, cfg: dict) -> pd.DataFrame:
    """Is every CSV value reproducible as round(k / divisor, decimals) for an integer k in the int16 range?

    Strong for the gyroscope (3 decimals vs a 1/131 = 0.0076 step). Uninformative for the accelerometer: its step
    (1/16384 = 0.000061) is finer than the 4-decimal CSV rounding, so almost any 4-decimal value is attainable.
    ``null_fraction`` is the share of ALL values on the CSV decimal grid (within the observed range) that are attainable.
    """
    hw = cfg["hardware"]
    rows = []
    ks = np.arange(hw["raw_int16_min"], hw["raw_int16_max"] + 1)
    for ch, col in COLS.items():
        fam = ch.split("_")[0]
        div, dec = _spec(cfg, fam)
        v = np.concatenate([d[col].to_numpy(float) for d in raw.values()])
        attain = np.unique(np.round(ks / div, dec))
        k = np.rint(v * div)
        on_grid = np.isin(np.round(v, dec), attain) & (np.abs(np.round(k / div, dec) - v) < 10 ** -(dec + 3))
        grid = np.round(np.arange(v.min(), v.max() + 10 ** -dec, 10 ** -dec), dec)
        null = float(np.isin(grid, attain).mean())
        resid = np.abs(v * div - k)
        rows.append({"channel": ch, "divisor": div, "csv_decimals": dec, "n_values": len(v),
                     "fraction_on_attainable_grid": float(on_grid.mean()), "null_fraction_random_decimals": null,
                     "informative": bool(null < 0.5), "k_min": int(k.min()), "k_max": int(k.max()),
                     "k_outside_int16": int(((k < hw["raw_int16_min"]) | (k > hw["raw_int16_max"])).sum()),
                     "max_residual_counts": float(resid.max())})
    return pd.DataFrame(rows)


def rail_expectation(raw: dict, cfg: dict) -> pd.DataFrame:
    """Expected CSV values at int16 limits under the documented formulas vs observed extremes, and counts just below."""
    hw = cfg["hardware"]
    rows = []
    for fam, chs in (("acc", ACC), ("gyro", GYR)):
        div, dec = _spec(cfg, fam)
        v = np.concatenate([d[COLS[c]].to_numpy(float) for d in raw.values() for c in chs])
        for lbl, kk, obs in (("max", hw["raw_int16_max"], v.max()), ("min", hw["raw_int16_min"], v.min())):
            exp = round(kk / div, dec)
            rows.append({"family": fam, "end": lbl, "raw_count": kk, "expected_csv_value": exp, "observed_extreme": float(obs),
                         "match": bool(np.isclose(exp, obs, atol=10 ** -dec / 2 + 1e-12)),
                         "n_at_expected": int((np.abs(v - exp) < 10 ** -dec / 2).sum()),
                         "n_in_next_5_counts": int(sum((np.abs(v - round((kk - j * np.sign(kk if kk else 1)) / div, dec)) < 10 ** -dec / 2).sum()
                                                        for j in range(1, 6)))})
    return pd.DataFrame(rows)


def _plateaus(v: np.ndarray, level: float, tol: float) -> list[tuple[int, int]]:
    m = np.abs(v - level) <= tol
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def plateau_table(raw: dict, cfg: dict) -> pd.DataFrame:
    """Every run of consecutive samples exactly at an expected rail value, with the samples around it."""
    hw = cfg["hardware"]
    rows = []
    for (pid, leg), d in raw.items():
        for ch, col in COLS.items():
            fam = ch.split("_")[0]
            div, dec = _spec(cfg, fam)
            v = d[col].to_numpy(float)
            for sign, kk in (("+", hw["raw_int16_max"]), ("-", hw["raw_int16_min"])):
                lvl = round(kk / div, dec)
                for a, b in _plateaus(v, lvl, 10 ** -dec / 2 + 1e-12):
                    before = v[a - 1] if a > 0 else np.nan
                    after = v[b + 1] if b + 1 < len(v) else np.nan
                    before2 = v[a - 2] if a > 1 else np.nan
                    rows.append({"participant_id": pid, "leg": leg, "channel": ch, "sign": sign, "start_idx": a,
                                 "length": b - a + 1, "start_ms": float(d["Time_ms"].iloc[a]), "value_before": before,
                                 "value_after": after, "rise_into_plateau": (lvl - before) * (1 if sign == "+" else -1) if np.isfinite(before) else np.nan,
                                 "rise_prev": (before - before2) * (1 if sign == "+" else -1) if np.isfinite(before2) else np.nan,
                                 "fall_out_of_plateau": (lvl - after) * (1 if sign == "+" else -1) if np.isfinite(after) else np.nan,
                                 "touches_recording_end": b + 1 == len(v), "touches_recording_start": a == 0})
    return pd.DataFrame(rows)


def plateau_summary(pl: pd.DataFrame) -> pd.DataFrame:
    g = pl.groupby(["channel", "leg"])
    out = g.agg(n_plateaus=("length", "size"), n_samples=("length", "sum"), mean_length=("length", "mean"),
                max_length=("length", "max"), share_len_ge2=("length", lambda s: (s >= 2).mean()),
                share_len_ge4=("length", lambda s: (s >= 4).mean()),
                n_participants=("participant_id", "nunique"),
                share_positive=("sign", lambda s: (s == "+").mean()),
                median_rise_into=("rise_into_plateau", "median"), median_fall_out=("fall_out_of_plateau", "median")).reset_index()
    return out


def wraparound_check(raw: dict, cfg: dict) -> pd.DataFrame:
    """int16 overflow would show as a jump from one rail to the other between adjacent samples, not as a plateau."""
    rows = []
    for ch, col in COLS.items():
        fam = ch.split("_")[0]
        span = 2 * (2 if fam == "acc" else 250)
        n_big = n_rail_to_rail = n_pairs = 0
        for d in raw.values():
            v = d[col].to_numpy(float)
            dv = np.abs(np.diff(v))
            n_pairs += len(dv)
            n_big += int((dv > 0.9 * span).sum())
            lo, hi = (-2.0, 2.0) if fam == "acc" else (-250.137, 250.13)
            n_rail_to_rail += int(((np.abs(v[:-1] - hi) < 0.02 * span) & (np.abs(v[1:] - lo) < 0.02 * span)
                                   | (np.abs(v[:-1] - lo) < 0.02 * span) & (np.abs(v[1:] - hi) < 0.02 * span)).sum())
        rows.append({"channel": ch, "n_adjacent_pairs": n_pairs, "n_jump_gt_90pct_of_full_span": n_big,
                     "n_rail_to_opposite_rail": n_rail_to_rail})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2. scale / offset / orientation from the data
def static_window(d: pd.DataFrame, n: int = 25) -> slice:
    """The n-sample window with the smallest gyroscope variance (the quietest part of the recording)."""
    g = d[[COLS[c] for c in GYR]].to_numpy(float)
    best, arg = np.inf, 0
    for i in range(0, len(g) - n + 1):
        s = g[i:i + n].std(axis=0).sum()
        if s < best:
            best, arg = s, i
    return slice(arg, arg + n)


def static_scale_offset(raw: dict) -> pd.DataFrame:
    """In the quietest 0.5 s of each recording: |accelerometer| (expect about 1 g if scale is right) and gyro offset."""
    rows = []
    for (pid, leg), d in raw.items():
        sl = static_window(d)
        a = d[[COLS[c] for c in ACC]].to_numpy(float)[sl]
        g = d[[COLS[c] for c in GYR]].to_numpy(float)[sl]
        am = a.mean(axis=0)
        rows.append({"participant_id": pid, "leg": leg, "window_start_ms": float(d["Time_ms"].iloc[sl.start]),
                     "acc_norm_g": float(np.linalg.norm(am)), "acc_x": am[0], "acc_y": am[1], "acc_z": am[2],
                     "acc_window_sd": float(a.std(axis=0).mean()), "gyro_x_mean": g[:, 0].mean(), "gyro_y_mean": g[:, 1].mean(),
                     "gyro_z_mean": g[:, 2].mean(), "gyro_window_sd": float(g.std(axis=0).mean())})
    return pd.DataFrame(rows)


def orientation_diagnostics(raw: dict, static: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Data-derived orientation facts (no anatomical mapping assumed).

    * which sensor axis is most aligned with gravity and with which sign,
    * principal axis of rotation per recording (PCA of the 3-axis gyroscope signal),
    * sign of gyro peaks per channel and leg,
    * angle between the static gravity direction of the left and right recordings of the same participant
      (how reproducible the mounting orientation is between legs; includes posture differences).
    """
    grav, pca_rows, ang = [], [], []
    for (pid, leg), d in raw.items():
        a = d[[COLS[c] for c in ACC]].to_numpy(float)
        m = a.mean(axis=0)
        u = m / np.linalg.norm(m)
        grav.append({"participant_id": pid, "leg": leg, "mean_x": m[0], "mean_y": m[1], "mean_z": m[2], "unit_x": u[0], "unit_y": u[1],
                     "unit_z": u[2], "angle_to_minus_y_deg": float(np.degrees(np.arccos(np.clip(-u[1], -1, 1)))),
                     "dominant_gravity_axis": ["x", "y", "z"][int(np.argmax(np.abs(u)))]})
        g = d[[COLS[c] for c in GYR]].to_numpy(float)
        w, V = np.linalg.eigh(np.cov(g.T))
        v1 = V[:, -1]
        v1 = v1 if v1[np.argmax(np.abs(v1))] > 0 else -v1
        pca_rows.append({"participant_id": pid, "leg": leg, "pc1_share": float(w[-1] / w.sum()), "pc1_x": v1[0], "pc1_y": v1[1], "pc1_z": v1[2],
                         "dominant_axis": ["x", "y", "z"][int(np.argmax(np.abs(v1)))],
                         "share_var_x": float(g[:, 0].var() / g.var(axis=0).sum()), "share_var_y": float(g[:, 1].var() / g.var(axis=0).sum()),
                         "share_var_z": float(g[:, 2].var() / g.var(axis=0).sum())})
    G, P = pd.DataFrame(grav), pd.DataFrame(pca_rows)
    st = static.set_index(["participant_id", "leg"])
    for pid in sorted({k[0] for k in raw}):
        vl, vr = st.loc[(pid, "left"), ["acc_x", "acc_y", "acc_z"]].to_numpy(float), st.loc[(pid, "right"), ["acc_x", "acc_y", "acc_z"]].to_numpy(float)
        c = np.dot(vl, vr) / (np.linalg.norm(vl) * np.linalg.norm(vr))
        ang.append({"participant_id": pid, "angle_between_static_gravity_left_right_deg": float(np.degrees(np.arccos(np.clip(c, -1, 1))))})
    A = pd.DataFrame(ang)
    signs = []
    for ch, col in COLS.items():
        for leg in ("left", "right"):
            v = np.concatenate([d[col].to_numpy(float) for (p, l), d in raw.items() if l == leg])
            signs.append({"channel": ch, "leg": leg, "n_gt_10pct_fullscale_pos": int((v > .1 * (2 if ch.startswith("acc") else 250)).sum()),
                          "n_lt_neg_10pct_fullscale": int((v < -.1 * (2 if ch.startswith("acc") else 250)).sum()), "mean": float(v.mean())})
    return {"gravity": G, "gyro_pca": P, "angle_left_right": A, "peak_signs": pd.DataFrame(signs)}


# ----------------------------------------------------------------------------- 3. movement onset (protocol: 3 s warning)
def movement_onset(raw: dict, threshold_dps: float, win: int = 25) -> pd.DataFrame:
    """First time at which the rolling 0.5 s mean gyroscope magnitude exceeds the threshold (heuristic, not a gait event)."""
    rows = []
    for (pid, leg), d in raw.items():
        g = np.sqrt((d[[COLS[c] for c in GYR]].to_numpy(float) ** 2).sum(axis=1))
        roll = pd.Series(g).rolling(win, min_periods=win).mean().to_numpy()
        idx = np.where(roll > threshold_dps)[0]
        rows.append({"participant_id": pid, "leg": leg, "onset_ms": float(d["Time_ms"].iloc[idx[0] - win + 1]) if len(idx) else np.nan,
                     "median_gyro_mag_first_1s": float(np.median(g[:50])), "median_gyro_mag_last_1s": float(np.median(g[-50:]))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 4. sensitivity analyses
AXIS_SIGN_INVARIANT = {"std", "var", "ptp", "rms", "energy", "zcr_hz", "dominant_freq_hz", "spectral_energy",
                       "spectral_centroid_hz", "spectral_entropy", "low_band_fraction"}
MAG_SIGNALS = {"acc_mag", "acc_dyn_mag", "gyro_mag"}


def sign_invariant_features(features: list[str]) -> list[str]:
    """Features unchanged if an axis is negated (mirror/inversion): spread, power and spectral shape of single axes and
    all statistics of vector magnitudes. Excludes axis means, minima, maxima, quantiles and any cross-correlation."""
    keep = []
    for f in features:
        sig, stat = feature_signal(f), feature_stat(f)
        if sig in MAG_SIGNALS:
            keep.append(f)
        elif sig != "corr" and stat in AXIS_SIGN_INVARIANT:
            keep.append(f)
    return keep


def trimmed_recording_features(prs: list, cfg: dict, trim_s: float) -> pd.DataFrame:
    """Whole-recording features after discarding the first ``trim_s`` seconds (filters were applied on the full signal)."""
    from src.features.extract import extract_recording_features
    rows = []
    for pr in prs:
        n0 = int(round(trim_s * pr.fs_hz))
        sl = slice(n0, None)
        t = dataclasses.replace(pr, t_ms=pr.t_ms[sl], raw=pr.raw.iloc[sl].reset_index(drop=True), proc=pr.proc.iloc[sl].reset_index(drop=True),
                                interpolated=pr.interpolated[sl], clipped=pr.clipped.iloc[sl].reset_index(drop=True))
        _, rec = extract_recording_features(t, [], cfg)
        rows.append(rec)
    return pd.concat(rows, ignore_index=True)


def paired_subset(rec: pd.DataFrame, feats: list[str], sa, alpha: float = 0.05) -> pd.DataFrame:
    """Independent paired recomputation restricted to ``feats`` (BH/Holm within the subset)."""
    pids, L, R = sa.paired_matrices(rec, feats)
    return sa.recompute(L - R, feats, alpha)


def summarise_paired(res: pd.DataFrame) -> dict:
    ok = res["p_primary"].notna()
    top = res.loc[ok].sort_values("p_primary").head(3)
    return {"n_tested": int(ok.sum()), "n_p_lt_05": int((res["p_primary"] < .05).sum()), "n_bh_lt_05": int((res["q_bh_p_primary"] < .05).sum()),
            "n_holm_lt_05": int((res["q_holm_p_primary"] < .05).sum()), "min_q_bh": float(res["q_bh_p_primary"].min()),
            "top3": [(r.feature, round(float(r.p_primary), 4)) for r in top.itertuples()]}


def paired_by_leg_order_free_summary(onset: pd.DataFrame) -> dict:
    w = onset.pivot(index="participant_id", columns="leg", values="onset_ms")
    d = (w["left"] - w["right"]).dropna()
    return {"n_pairs": int(len(d)), "mean_onset_left_ms": float(w["left"].mean()), "mean_onset_right_ms": float(w["right"].mean()),
            "median_diff_L_minus_R_ms": float(d.median()), "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if (d != 0).any() else np.nan,
            "n_missing_onset": int(onset["onset_ms"].isna().sum())}
