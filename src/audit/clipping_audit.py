"""Audit of clipping detection and of the left/right AccelZ asymmetry.

Independent of src.data_validation: reads the raw CSVs itself and uses EXACT equality with the observed
extreme values (no tolerance), then compares against the pipeline's tolerance-based flags.

Vocabulary: "observed rail value" = a value that repeats exactly at the extreme of the data. Whether this is
hardware saturation cannot be confirmed (sensor configuration undocumented); the int16 arithmetic below is
supporting evidence only.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.audit.common import AXES, feature_channels

COLS = {"acc_x": "AccelX_g", "acc_y": "AccelY_g", "acc_z": "AccelZ_g", "gyro_x": "GyroX_dps", "gyro_y": "GyroY_dps",
        "gyro_z": "GyroZ_dps"}


def load_raw(data_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    out = {}
    for f in sorted(Path(data_dir).glob("*.csv")):
        m = re.match(r"Sub(\d+)(Left|Right)", f.name)
        out[(f"Sub{int(m[1]):02d}", m[2].lower())] = pd.read_csv(f)
    return out


def _runs(mask: np.ndarray) -> list[int]:
    r, n = [], 0
    for x in mask:
        if x:
            n += 1
        elif n:
            r.append(n)
            n = 0
    if n:
        r.append(n)
    return r


def rail_levels(raw: dict) -> dict[str, tuple[float, float]]:
    """Observed extreme values per sensor family (pooled over axes, participants, legs)."""
    lv = {}
    for fam, cols in (("acc", [c for k, c in COLS.items() if k.startswith("acc")]),
                      ("gyro", [c for k, c in COLS.items() if k.startswith("gyro")])):
        v = np.concatenate([d[c].to_numpy() for d in raw.values() for c in cols])
        lv[fam] = (float(v.min()), float(v.max()))
    return lv


def int16_check(levels: dict) -> pd.DataFrame:
    """Do the extremes equal int16 register limits divided by a plausible scale factor? (supporting evidence only)"""
    rows = []
    for fam, scale, label in (("acc", 16384.0, "16384 LSB/g (+-2 g range)"), ("gyro", 131.0, "131 LSB/(deg/s) (+-250 deg/s range)")):
        lo, hi = levels[fam]
        rows.append({"family": fam, "observed_min": lo, "observed_max": hi, "assumed_scale": label,
                     "min_times_scale": lo * scale, "max_times_scale": hi * scale,
                     "int16_min": -32768, "int16_max": 32767,
                     "max_rounding_error_counts": max(abs(lo * scale + 32768), abs(hi * scale - 32767)),
                     "note": "arithmetic coincidence check only; sensor configuration is NOT documented"})
    return pd.DataFrame(rows)


def clip_inventory(raw: dict, levels: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per participant/leg/channel counts of samples at the exact observed extremes, plus run lengths."""
    rows = []
    for (pid, leg), d in raw.items():
        for ch, col in COLS.items():
            lo, hi = levels[ch.split("_")[0]]
            v = d[col].to_numpy()
            at_hi, at_lo = v == hi, v == lo
            any_ = at_hi | at_lo
            rl = _runs(any_)
            rows.append({"participant_id": pid, "leg": leg, "channel": ch, "n_samples": len(v),
                         "n_at_max": int(at_hi.sum()), "n_at_min": int(at_lo.sum()), "n_at_rail": int(any_.sum()),
                         "pct_at_rail": 100 * any_.mean(), "n_episodes": len(rl),
                         "longest_run": max(rl) if rl else 0, "recording_has_rail": bool(any_.any())})
    inv = pd.DataFrame(rows)
    summ = (inv.groupby(["channel", "leg"]).agg(n_recordings=("participant_id", "size"),
                                                 n_recordings_with_rail=("recording_has_rail", "sum"),
                                                 n_at_max=("n_at_max", "sum"), n_at_min=("n_at_min", "sum"),
                                                 n_at_rail=("n_at_rail", "sum"), n_samples=("n_samples", "sum"),
                                                 mean_run=("longest_run", "mean"), max_run=("longest_run", "max"))
            .reset_index())
    summ["pct_at_rail"] = 100 * summ["n_at_rail"] / summ["n_samples"]
    return inv, summ


def tolerance_vs_exact(raw: dict, levels: dict, rel_tol: float) -> pd.DataFrame:
    """How many samples does the pipeline's tolerance rule flag beyond exact equality with the extremes?"""
    rows = []
    for ch, col in COLS.items():
        lo, hi = levels[ch.split("_")[0]]
        v = np.concatenate([d[col].to_numpy() for d in raw.values()])
        exact = (v == lo) | (v == hi)
        tol = (np.abs(v - lo) <= rel_tol * abs(lo)) | (np.abs(v - hi) <= rel_tol * abs(hi))
        rows.append({"channel": ch, "rel_tol": rel_tol, "n_exact": int(exact.sum()), "n_tolerance": int(tol.sum()),
                     "n_extra_by_tolerance": int((tol & ~exact).sum()),
                     "extra_values": sorted(set(np.round(v[tol & ~exact], 4).tolist()))})
    return pd.DataFrame(rows)


def accz_asymmetry(raw: dict, levels: dict, inv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Why does AccelZ reach its positive extreme in all right recordings but few left ones?

    Looks at counts (not only presence), the negative rail, the tails, the gravity vector at rest and
    orientation proxies. Returns tables; interpretation is left to the report.
    """
    z = inv[inv["channel"] == "acc_z"].pivot(index="participant_id", columns="leg", values="n_at_max")
    zn = inv[inv["channel"] == "acc_z"].pivot(index="participant_id", columns="leg", values="n_at_min")
    pair = pd.DataFrame({"z_max_left": z["left"], "z_max_right": z["right"], "z_min_left": zn["left"],
                         "z_min_right": zn["right"]})
    pair["z_rail_total_left"] = pair["z_max_left"] + pair["z_min_left"]
    pair["z_rail_total_right"] = pair["z_max_right"] + pair["z_min_right"]
    res = []

    def test(name, a, b):
        d = (b - a).to_numpy(dtype=float)  # right - left
        nz = d[d != 0]
        p_w = stats.wilcoxon(d).pvalue if len(nz) else np.nan
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        p_sign = stats.binomtest(pos, pos + neg, .5).pvalue if pos + neg else 1.0  # all ties: no evidence of a difference
        res.append({"quantity": name, "sum_left": a.sum(), "sum_right": b.sum(), "n_right_gt_left": pos,
                    "n_left_gt_right": neg, "n_ties": int((d == 0).sum()), "sign_test_p": p_sign, "wilcoxon_p": p_w})

    test("n samples at Z max (counts)", pair["z_max_left"], pair["z_max_right"])
    test("n samples at Z min (counts)", pair["z_min_left"], pair["z_min_right"])
    test("n samples at either Z rail", pair["z_rail_total_left"], pair["z_rail_total_right"])
    # presence indicator (what the original report used) - exact McNemar
    l, r = pair["z_max_left"] > 0, pair["z_max_right"] > 0
    b, c = int((r & ~l).sum()), int((l & ~r).sum())
    mc = stats.binomtest(b, b + c, .5).pvalue if b + c else np.nan
    res.append({"quantity": "presence of Z max in recording (McNemar exact; indicator chosen post hoc)",
                "sum_left": int(l.sum()), "sum_right": int(r.sum()), "n_right_gt_left": b, "n_left_gt_right": c,
                "n_ties": int(12 - b - c), "sign_test_p": mc, "wilcoxon_p": np.nan})
    tests = pd.DataFrame(res)

    # tails / distribution of Z per leg (pooled), and per-participant paired quantiles
    qrows = []
    for leg in ("left", "right"):
        v = np.concatenate([d["AccelZ_g"].to_numpy() for (p, l_), d in raw.items() if l_ == leg])
        qrows.append({"leg": leg, **{f"q{q}": np.quantile(v, q / 100) for q in (1, 5, 25, 50, 75, 95, 99)},
                      "mean": v.mean(), "std": v.std(ddof=1), "n_gt_1.5g": int((v > 1.5).sum()),
                      "n_lt_-1.5g": int((v < -1.5).sum())})
    dist = pd.DataFrame(qrows)

    # orientation proxies per recording: median gravity vector over the first 25 samples, whole-record medians
    orows = []
    for (pid, leg), d in raw.items():
        a = d[[COLS["acc_x"], COLS["acc_y"], COLS["acc_z"]]].to_numpy()
        s = np.median(a[:25], axis=0)
        m = np.median(a, axis=0)
        orows.append({"participant_id": pid, "leg": leg, "start_x": s[0], "start_y": s[1], "start_z": s[2],
                      "start_norm_g": np.linalg.norm(s),
                      "start_tilt_deg_from_minus_y": np.degrees(np.arccos(np.clip(-s[1] / np.linalg.norm(s), -1, 1))),
                      "median_x": m[0], "median_y": m[1], "median_z": m[2],
                      "gyro_start_std_mean": d[[COLS[c] for c in AXES[3:]]].iloc[:25].std().mean()})
    orient = pd.DataFrame(orows)
    ow = orient.pivot(index="participant_id", columns="leg")
    orows2 = []
    for col in ("start_x", "start_z", "start_tilt_deg_from_minus_y", "median_x", "median_z", "gyro_start_std_mean"):
        d = (ow[(col, "left")] - ow[(col, "right")]).to_numpy(dtype=float)
        orows2.append({"quantity": col, "mean_left": ow[(col, "left")].mean(), "mean_right": ow[(col, "right")].mean(),
                       "mean_diff_L_minus_R": d.mean(), "t_p": stats.ttest_1samp(d, 0).pvalue,
                       "wilcoxon_p": stats.wilcoxon(d).pvalue})
    orient_tests = pd.DataFrame(orows2)

    # mounting-symmetry check: does the sign of mean mediolateral-like channels differ by leg?
    srows = []
    for ch in ("acc_x", "acc_z", "gyro_x", "gyro_y", "gyro_z"):
        per = {leg: [d[COLS[ch]].mean() for (p, l_), d in sorted(raw.items()) if l_ == leg] for leg in ("left", "right")}
        dd = np.array(per["left"]) - np.array(per["right"])
        srows.append({"channel": ch, "mean_of_means_left": np.mean(per["left"]), "mean_of_means_right": np.mean(per["right"]),
                      "n_participants_left_mean_gt_right": int((dd > 0).sum()), "paired_t_p": stats.ttest_1samp(dd, 0).pvalue})
    sign = pd.DataFrame(srows)
    return {"accz_pair_counts": pair.reset_index(), "accz_tests": tests, "accz_distribution": dist,
            "orientation_per_recording": orient, "orientation_tests": orient_tests, "channel_mean_by_leg": sign}


def feature_clip_sensitivity(rec: pd.DataFrame, features: list[str], threshold: float = 0.01) -> pd.DataFrame:
    """Per feature: exposure to rail samples and rank-correlation with the clipped fraction of its channels.

    * exposure = fraction of the 24 recordings whose involved channels have clip fraction > 0
    * spearman_rho = rank correlation, across recordings, of feature and max clip fraction of involved channels
    * n_clean_pairs = participants whose BOTH legs have clip fraction <= ``threshold`` on the involved channels
    """
    rec = rec.set_index("recording_id")
    rows = []
    for f in features:
        chs = sorted(feature_channels(f))
        cf = rec[[f"qc_clip_fraction_{c}" for c in chs]].max(axis=1) if chs else pd.Series(0.0, index=rec.index)
        ok = rec[f].notna()
        rho = stats.spearmanr(rec.loc[ok, f], cf[ok]).statistic if cf[ok].nunique() > 1 else np.nan
        clean = cf <= threshold
        pids = rec["participant_id"]
        clean_pairs = [p for p in pids.unique() if clean[pids == p].all()]
        rows.append({"feature": f, "channels": "|".join(chs), "exposure_fraction_recordings": float((cf > 0).mean()),
                     "mean_clip_fraction": float(cf.mean()), "spearman_rho_with_clip": rho,
                     "n_clean_pairs": len(clean_pairs), "clean_pair_ids": "|".join(sorted(clean_pairs))})
    return pd.DataFrame(rows)
