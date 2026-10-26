"""Independent re-computation and robustness checks of the paired left/right analysis.

Nothing here imports src.statistics.paired: every statistic is recomputed from the recording-level feature table with
scipy / statsmodels so that discrepancies with the pipeline's tables reveal implementation errors.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests

ID_COLS = ["participant_id", "leg", "filename", "recording_id", "scope", "window_index", "start_idx", "end_idx",
           "start_ms", "end_ms", "n_samples", "fs_hz", "feature_config_id"]


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS and not c.startswith("qc_")]


def paired_matrices(rec: pd.DataFrame, feats: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Left and right matrices (participants x features), rows aligned by participant ID."""
    L = rec[rec["leg"] == "left"].set_index("participant_id").sort_index()
    R = rec[rec["leg"] == "right"].set_index("participant_id").sort_index()
    assert list(L.index) == list(R.index), "left/right participants do not align"
    assert L.index.is_unique and R.index.is_unique
    return list(L.index), L[feats].to_numpy(float), R[feats].to_numpy(float)


def dz_ci_noncentral(dz: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact CI for the paired standardized mean difference via the non-central t distribution.

    Computed for |dz| (the non-central t cdf is better behaved for positive t) and mirrored for negative dz.
    """
    if not np.isfinite(dz) or n < 3:
        return np.nan, np.nan
    t, df = abs(dz) * np.sqrt(n), n - 1

    def solve(prob: float) -> float:
        f = lambda lam: stats.nct.cdf(t, df, lam) - prob  # noqa: E731
        return optimize.brentq(f, t - 12, t + 12)

    try:
        lo, hi = solve(1 - alpha / 2) / np.sqrt(n), solve(alpha / 2) / np.sqrt(n)
    except ValueError:  # root outside bracket
        return np.nan, np.nan
    return (lo, hi) if dz >= 0 else (-hi, -lo)


def recompute(D: np.ndarray, feats: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """Independent per-feature statistics for difference matrix D (participants x features)."""
    rows = []
    for j, f in enumerate(feats):
        d = D[:, j]
        d = d[np.isfinite(d)]
        n = len(d)
        r = {"feature": f, "n_pairs": n}
        sd = d.std(ddof=1) if n > 1 else np.nan
        if n >= 3 and sd > 0:
            r["mean_diff"] = d.mean()
            r["dz"] = d.mean() / sd
            r["ci_low"], r["ci_high"] = stats.t.interval(1 - alpha, n - 1, loc=d.mean(), scale=stats.sem(d))
            r["p_t"] = stats.ttest_1samp(d, 0).pvalue
            r["shapiro_p"] = stats.shapiro(d).pvalue
            r["dz_nct_low"], r["dz_nct_high"] = dz_ci_noncentral(r["dz"], n, alpha)
            try:
                r["p_wilcoxon"] = stats.wilcoxon(d).pvalue
            except ValueError:
                r["p_wilcoxon"] = np.nan
            r["p_primary"] = r["p_t"] if r["shapiro_p"] >= alpha else r["p_wilcoxon"]
            r["primary_test"] = "paired_t" if r["shapiro_p"] >= alpha else "wilcoxon"
            r["has_ties_or_zeros"] = bool(len(np.unique(np.abs(d))) < n or (d == 0).any())
        rows.append(r)
    out = pd.DataFrame(rows)
    for col in ("p_primary", "p_t", "p_wilcoxon"):
        ok = out[col].notna()
        q_bh, q_holm = np.full(len(out), np.nan), np.full(len(out), np.nan)
        if ok.any():
            q_bh[ok] = multipletests(out.loc[ok, col], method="fdr_bh")[1]
            q_holm[ok] = multipletests(out.loc[ok, col], method="holm")[1]
        out[f"q_bh_{col}"], out[f"q_holm_{col}"] = q_bh, q_holm
    return out


def compare_to_pipeline(mine: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
    """Max absolute discrepancy per column between the independent and stored results."""
    m = mine.merge(stored, on="feature", suffixes=("_audit", "_pipe"))
    rows = []
    for a, b in [("mean_diff", "mean_diff"), ("dz", "dz"), ("ci_low", "ci_low"), ("ci_high", "ci_high"),
                 ("p_t", "p_t"), ("p_wilcoxon", "p_wilcoxon"), ("shapiro_p", "shapiro_p"),
                 ("p_primary", "p_primary"), ("q_bh_p_primary", "q_bh_primary"), ("q_holm_p_primary", "q_holm_primary")]:
        x = m[f"{a}_audit"] if f"{a}_audit" in m else m[a]
        y = m[f"{b}_pipe"] if f"{b}_pipe" in m else m[b]
        both = x.notna() & y.notna()
        rows.append({"column": a, "n_compared": int(both.sum()), "max_abs_diff": float((x[both] - y[both]).abs().max()),
                     "nan_mismatch": int((x.isna() != y.isna()).sum())})
    ptest = (m["primary_test_audit"] == m["primary_test_pipe"]) if "primary_test_audit" in m else pd.Series(dtype=bool)
    rows.append({"column": "primary_test (label)", "n_compared": int(len(ptest)),
                 "max_abs_diff": float((~ptest).sum()), "nan_mismatch": 0})
    return pd.DataFrame(rows)


def signflip_maxT(D: np.ndarray, feats: list[str], alpha: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Exact sign-flip permutation test across all features (2^n flips; n=12 -> 4096).

    Under H0 (no systematic left-right difference) each participant's difference is symmetric about 0, so flipping
    the sign of a participant's whole feature vector is exchangeable. Gives
    * a Westfall-Young max-|t| FWER-adjusted p-value per feature (respects feature correlation),
    * a global test using the number of features with nominal |t| > critical value.
    """
    ok = np.isfinite(D).all(axis=0) & (D.std(axis=0, ddof=1) > 0)
    X = D[:, ok]
    n = X.shape[0]
    flips = np.array(list(product([1, -1], repeat=n)), dtype=float)  # (2^n, n)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)

    def tstat(Y):  # Y: (m, n, k)
        return Y.mean(axis=1) / (Y.std(axis=1, ddof=1) / np.sqrt(n))

    obs_t = X.mean(0) / (X.std(0, ddof=1) / np.sqrt(n))
    maxT = np.empty(len(flips))
    counts = np.empty(len(flips))
    for i in range(0, len(flips), 512):
        T = np.abs(tstat(flips[i:i + 512, :, None] * X[None, :, :]))
        maxT[i:i + 512] = T.max(axis=1)
        counts[i:i + 512] = (T > tcrit).sum(axis=1)
    p_adj = np.array([(maxT >= abs(t) - 1e-12).mean() for t in obs_t])
    # per-feature exact single-feature sign-flip p (for reference)
    out = pd.DataFrame({"feature": np.array(feats)[ok], "t_obs": obs_t, "p_maxT_fwer": p_adj})
    obs_count = int((np.abs(obs_t) > tcrit).sum())
    glob = {"n_features": int(ok.sum()), "n_flips": len(flips), "observed_n_nominal_sig": obs_count,
            "null_mean_n_nominal_sig": float(counts.mean()), "null_q95_n_nominal_sig": float(np.quantile(counts, .95)),
            "null_max_n_nominal_sig": int(counts.max()),
            "global_p_count": float((counts >= obs_count).mean()),
            "min_p_maxT": float(p_adj.min())}
    return out, glob


def window_level_pseudoreplication(win: pd.DataFrame, feats: list[str]) -> dict:
    """WRONG-UNIT illustration: treat all windows as independent left vs right samples (Welch t-test).

    Shown only to quantify how much inference inflates when overlapping windows are mistaken for participants.
    """
    L, R = win[win["leg"] == "left"], win[win["leg"] == "right"]
    p = np.array([stats.ttest_ind(L[f], R[f], equal_var=False).pvalue for f in feats])
    ok = np.isfinite(p)
    q = multipletests(p[ok], method="fdr_bh")[1]
    return {"n_left_windows": len(L), "n_right_windows": len(R), "n_features": int(ok.sum()),
            "n_p_lt_05_naive": int((p[ok] < .05).sum()), "n_bh_sig_naive": int((q < .05).sum())}


def clean_pair_reanalysis(rec: pd.DataFrame, sens: pd.DataFrame, min_pairs: int = 5) -> pd.DataFrame:
    """Re-run the paired tests on participants whose both legs have <=1% clipped samples in the feature's channels."""
    L = rec[rec["leg"] == "left"].set_index("participant_id").sort_index()
    R = rec[rec["leg"] == "right"].set_index("participant_id").sort_index()
    rows = []
    for _, s in sens.iterrows():
        ids = [p for p in str(s["clean_pair_ids"]).split("|") if p]
        f = s["feature"]
        r = {"feature": f, "n_clean_pairs": len(ids)}
        if len(ids) >= min_pairs:
            d = (L.loc[ids, f] - R.loc[ids, f]).to_numpy(float)
            if d.std(ddof=1) > 0:
                r.update(mean_diff=d.mean(), dz=d.mean() / d.std(ddof=1), p_t=stats.ttest_1samp(d, 0).pvalue)
                try:
                    r["p_wilcoxon"] = stats.wilcoxon(d).pvalue
                except ValueError:
                    r["p_wilcoxon"] = np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    ok = out["p_t"].notna() if "p_t" in out else pd.Series(False, index=out.index)
    out["q_bh_p_t"] = np.nan
    if ok.any():
        out.loc[ok, "q_bh_p_t"] = multipletests(out.loc[ok, "p_t"], method="fdr_bh")[1]
    return out
