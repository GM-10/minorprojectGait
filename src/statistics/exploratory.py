"""Exploratory statistics: correlation, PCA, participant-level variability.

These functions are DESCRIPTIVE. Scalers/PCA are fitted on all supplied rows, which is acceptable for
description only; predictive code (src/models) fits any learned transform inside training folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def usable_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Drop features with any NaN/inf or zero variance."""
    out = []
    for f in features:
        v = df[f].to_numpy(dtype=float)
        if np.isfinite(v).all() and np.nanstd(v) > 0:
            out.append(f)
    return out


def correlation_matrix(df: pd.DataFrame, features: list[str], method: str = "spearman") -> pd.DataFrame:
    return df[features].corr(method=method)


def pca_analysis(df: pd.DataFrame, features: list[str], n_components: int = 10, seed: int = 42) -> dict:
    """Standardize and run PCA. Returns scores (with all non-feature columns kept), variance and loadings."""
    feats = usable_features(df, features)
    if len(feats) < 2 or len(df) < 3:
        raise ValueError(f"PCA needs >=2 usable features and >=3 rows; got {len(feats)} features, {len(df)} rows")
    X = StandardScaler().fit_transform(df[feats].to_numpy(dtype=float))
    k = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(X)
    scores = pca.transform(X)
    meta = df.drop(columns=features, errors="ignore").reset_index(drop=True)
    sc = pd.concat([meta, pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(k)])], axis=1)
    load = pd.DataFrame(pca.components_.T, index=feats, columns=[f"PC{i+1}" for i in range(k)])
    return {"scores": sc, "explained_variance_ratio": pca.explained_variance_ratio_, "loadings": load,
            "n_features_used": len(feats), "n_features_dropped": len(features) - len(feats),
            "n_rows": len(df), "n_participants": int(df["participant_id"].nunique())}


def participant_variability(win_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Between-participant vs within-recording spread for every feature (window-level input).

    * ``between_sd``: SD across participants of their mean-over-legs window-mean.
    * ``within_sd``: average (over recordings) of the SD of that feature across windows.
    * ``ratio`` = between_sd / within_sd. Windows within a recording are autocorrelated (overlap), so this is
      descriptive only and not an ICC with a valid confidence interval.
    """
    rec_mean = win_df.groupby(["participant_id", "leg"])[features].mean()
    part_mean = rec_mean.groupby("participant_id").mean()
    between = part_mean.std(ddof=1)
    within = win_df.groupby(["participant_id", "leg"])[features].std(ddof=1).mean()
    out = pd.DataFrame({"between_sd": between, "within_sd": within})
    out["ratio"] = out["between_sd"] / out["within_sd"].replace(0, np.nan)
    out["cv_between_pct"] = 100 * between / part_mean.mean().abs().replace(0, np.nan)
    return out.reset_index(names="feature")


def participant_summary(rec_df: pd.DataFrame, win_df: pd.DataFrame, headline: list[str]) -> pd.DataFrame:
    """Per participant/leg: whole-recording headline features and window-level median [IQR]."""
    base = rec_df[["participant_id", "leg", "filename"] + [h for h in headline if h in rec_df]].copy()
    for h in headline:
        if h in win_df:
            g = win_df.groupby(["participant_id", "leg"])[h]
            s = pd.DataFrame({f"{h}__win_median": g.median(), f"{h}__win_iqr": g.quantile(.75) - g.quantile(.25)})
            base = base.merge(s.reset_index(), on=["participant_id", "leg"], how="left")
    n = win_df.groupby(["participant_id", "leg"]).size().rename("n_windows").reset_index()
    return base.merge(n, on=["participant_id", "leg"], how="left")
