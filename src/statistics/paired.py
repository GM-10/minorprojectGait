"""Paired left-vs-right statistics at the participant level.

Unit of observation: one participant (n pairs). Each participant contributes ONE left value and ONE right value
per feature (whole-recording features by default), so windows are never treated as independent observations.

Sign convention: difference = LEFT - RIGHT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs are ignored and returned as NaN)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    adj = pv[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaNs ignored)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    adj = np.maximum.accumulate(pv[order] * (m - np.arange(m)))
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def hedges_j(n: int) -> float:
    """Small-sample correction factor for a paired standardized mean difference (df = n - 1)."""
    df = n - 1
    return 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else float("nan")


def rank_biserial_paired(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-), zero differences excluded."""
    d = d[d != 0]
    if d.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(d))
    wp, wm = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def analyse_differences(d: np.ndarray, alpha: float, normality_alpha: float, boot_idx: np.ndarray | None
                        ) -> dict:
    """Descriptive and inferential statistics for a vector of paired differences (left - right)."""
    d = np.asarray(d, dtype=float)
    n = d.size
    res = {"n_pairs": n, "mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "median_diff": np.nan,
           "median_ci_low": np.nan, "median_ci_high": np.nan, "sd_diff": np.nan, "dz": np.nan, "hedges_gz": np.nan,
           "dz_ci_low": np.nan, "dz_ci_high": np.nan, "rank_biserial": np.nan, "shapiro_p": np.nan,
           "normality_ok": False, "t_stat": np.nan, "p_t": np.nan, "w_stat": np.nan, "p_wilcoxon": np.nan,
           "n_nonzero_diff": int((d != 0).sum()), "primary_test": "none", "p_primary": np.nan}
    if n < 2:
        return res
    mean, sd = d.mean(), d.std(ddof=1)
    res.update(mean_diff=mean, sd_diff=sd, median_diff=float(np.median(d)))
    if sd > 0:
        se = sd / np.sqrt(n)
        tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
        res.update(ci_low=mean - tcrit * se, ci_high=mean + tcrit * se, dz=mean / sd)
        res["hedges_gz"] = res["dz"] * hedges_j(n)
        t = stats.ttest_rel(d, np.zeros(n))
        res.update(t_stat=float(t.statistic), p_t=float(t.pvalue))
        if n >= 3:
            res["shapiro_p"] = float(stats.shapiro(d).pvalue)
            res["normality_ok"] = bool(res["shapiro_p"] >= normality_alpha)
    res["rank_biserial"] = rank_biserial_paired(d)
    if res["n_nonzero_diff"] >= 1 and sd > 0:
        try:
            w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            res.update(w_stat=float(w.statistic), p_wilcoxon=float(w.pvalue))
        except ValueError:
            pass
    if boot_idx is not None and n >= 2:
        bd = d[boot_idx[:, :n] % n] if boot_idx.shape[1] != n else d[boot_idx]
        med = np.median(bd, axis=1)
        res["median_ci_low"], res["median_ci_high"] = np.quantile(med, [alpha / 2, 1 - alpha / 2])
        bsd = bd.std(axis=1, ddof=1)
        ok = bsd > 0
        if ok.any():
            dz_b = bd.mean(axis=1)[ok] / bsd[ok]
            res["dz_ci_low"], res["dz_ci_high"] = np.quantile(dz_b, [alpha / 2, 1 - alpha / 2])
    if np.isfinite(res["shapiro_p"]):
        if res["normality_ok"]:
            res["primary_test"], res["p_primary"] = "paired_t", res["p_t"]
        else:
            res["primary_test"], res["p_primary"] = "wilcoxon", res["p_wilcoxon"]
    return res


def paired_table(rec_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wide left/right tables per participant. Returns (left, right, pairing_report).

    Only participants having exactly one left and one right recording row are paired.
    """
    counts = rec_df.groupby(["participant_id", "leg"]).size().unstack(fill_value=0)
    for leg in ("left", "right"):
        if leg not in counts:
            counts[leg] = 0
    paired = counts.index[(counts["left"] == 1) & (counts["right"] == 1)]
    report = counts.reset_index()
    report["paired"] = report["participant_id"].isin(paired)
    left = rec_df[rec_df["leg"] == "left"].set_index("participant_id").loc[paired, features]
    right = rec_df[rec_df["leg"] == "right"].set_index("participant_id").loc[paired, features]
    return left, right, report


def run_paired_analysis(rec_df: pd.DataFrame, features: list[str], cfg: dict, exclude: list[str] | None = None,
                        seed: int = 42) -> dict[str, pd.DataFrame]:
    """Paired analysis over all features. ``exclude`` lists participant IDs dropped (sensitivity analysis)."""
    sc = cfg["statistics"]
    if exclude:
        rec_df = rec_df[~rec_df["participant_id"].isin(exclude)]
    left, right, pairing = paired_table(rec_df, features)
    rng = np.random.default_rng(seed)
    rows, diffs = [], {}
    for f in features:
        l, r = left[f], right[f]
        ok = l.notna() & r.notna() & np.isfinite(l.astype(float)) & np.isfinite(r.astype(float))
        d = (l[ok] - r[ok]).to_numpy(dtype=float)
        diffs[f] = (l - r).where(ok)
        n = d.size
        boot = rng.integers(0, n, size=(sc["n_bootstrap"], n)) if n >= sc["min_pairs"] else None
        if n >= sc["min_pairs"]:
            res = analyse_differences(d, sc["alpha"], sc["normality_alpha"], boot)
        else:  # keep the full column set so downstream code never sees missing columns
            res = {**analyse_differences(np.array([]), sc["alpha"], sc["normality_alpha"], None),
                   "n_pairs": n, "primary_test": "insufficient_pairs"}
        pos = (l[ok] > 0).all() and (r[ok] > 0).all()
        si = (100 * (l[ok] - r[ok]) / (0.5 * (l[ok] + r[ok]))).median() if pos and n else np.nan
        rows.append({"feature": f, "mean_left": l[ok].mean(), "mean_right": r[ok].mean(),
                     "median_symmetry_index_pct": si, **res})
    out = pd.DataFrame(rows)
    for col, name in (("p_primary", "primary"), ("p_t", "t"), ("p_wilcoxon", "wilcoxon")):
        if col in out:
            out[f"q_bh_{name}"] = bh_fdr(out[col].to_numpy(dtype=float))
    out["q_holm_primary"] = holm(out["p_primary"].to_numpy(dtype=float)) if "p_primary" in out else np.nan
    out["significant_bh_primary"] = out["q_bh_primary"] < sc["alpha"]
    out["n_features_tested"] = int(out["p_primary"].notna().sum())
    diff_df = pd.DataFrame(diffs)
    diff_df.index.name = "participant_id"
    return {"results": out, "differences": diff_df.reset_index(), "pairing": pairing,
            "excluded": pd.DataFrame({"excluded_participant": exclude or []})}
