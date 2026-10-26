"""Optional, clearly exploratory ML experiments.

No ground-truth gait labels exist in this dataset. Consequently:
* anomaly scores are statistical outlier scores only (not clinical abnormality),
* clusters are not gait categories,
* the left/right classifier is a leakage-aware demonstration evaluated with leave-one-participant-out CV.
All learned preprocessing (scaling) is fitted inside training data only for predictive experiments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.statistics.exploratory import usable_features


def select_model_features(win_df: pd.DataFrame, features: list[str], whitelist: list[str]) -> list[str]:
    """Curated, less redundant feature subset (whitelisted statistics + all cross-channel correlations)."""
    keep = [f for f in features if f.startswith("corr__") or f.split("__")[-1] in whitelist]
    return usable_features(win_df, keep)


def anomaly_detection(win_df: pd.DataFrame, feats: list[str], cfg: dict, seed: int, noted: set[str]) -> pd.DataFrame:
    """Isolation Forest + LOF scores per window (higher = more unusual relative to THIS dataset).

    Scaling is fitted on all windows because the method is unsupervised and descriptive (no held-out evaluation).
    """
    mc = cfg["models"]
    X = StandardScaler().fit_transform(win_df[feats].to_numpy(dtype=float))
    iso = IsolationForest(n_estimators=300, contamination=mc["anomaly_contamination"], random_state=seed).fit(X)
    lof = LocalOutlierFactor(n_neighbors=min(mc["lof_neighbors"], len(X) - 1),
                             contamination=mc["anomaly_contamination"])
    lof_flag = lof.fit_predict(X)
    out = win_df[["participant_id", "leg", "filename", "recording_id", "window_index", "start_ms", "end_ms"]].copy()
    out["iforest_score"] = -iso.score_samples(X)
    out["iforest_flag"] = iso.predict(X) == -1
    out["lof_score"] = -lof.negative_outlier_factor_
    out["lof_flag"] = lof_flag == -1
    out["flagged_by_both"] = out["iforest_flag"] & out["lof_flag"]
    out["iforest_rank"] = out["iforest_score"].rank(ascending=False, method="first").astype(int)
    out["participant_has_filename_note"] = out["participant_id"].isin(noted)
    out["qc_clip_fraction_any"] = win_df["qc_clip_fraction_any"].to_numpy()
    out.attrs["spearman_if_vs_lof"] = float(spearmanr(out["iforest_score"], out["lof_score"]).statistic)
    return out.sort_values("iforest_rank").reset_index(drop=True)


def _fit_kmeans(X: np.ndarray, k: int, seed: int) -> tuple[KMeans, StandardScaler]:
    sc = StandardScaler().fit(X)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(sc.transform(X)), sc


def clustering_experiment(win_df: pd.DataFrame, feats: list[str], cfg: dict, seed: int) -> dict:
    """K-means over a range of k with silhouette, participant-subsample stability, and identity/leg overlap."""
    mc = cfg["models"]
    X = win_df[feats].to_numpy(dtype=float)
    parts = win_df["participant_id"].to_numpy()
    uniq = np.unique(parts)
    rng = np.random.default_rng(seed)
    rows, labels_by_k = [], {}
    for k in mc["kmeans_k_range"]:
        km, sc = _fit_kmeans(X, k, seed)
        lab = km.labels_
        labels_by_k[k] = lab
        aris = []
        for _ in range(mc["stability_repeats"]):
            n_sub = max(2, int(round(mc["stability_subsample"] * len(uniq))))
            a, b = (rng.choice(uniq, n_sub, replace=False) for _ in range(2))
            shared = np.isin(parts, np.intersect1d(a, b))
            if shared.sum() < 2 * k:
                continue
            la = _fit_kmeans(X[np.isin(parts, a)], k, int(rng.integers(1e6)))
            lb = _fit_kmeans(X[np.isin(parts, b)], k, int(rng.integers(1e6)))
            pa = la[0].predict(la[1].transform(X[shared]))
            pb = lb[0].predict(lb[1].transform(X[shared]))
            aris.append(adjusted_rand_score(pa, pb))
        rows.append({"k": k, "silhouette": float(silhouette_score(sc.transform(X), lab)),
                     "stability_ari_mean": float(np.mean(aris)) if aris else np.nan,
                     "stability_ari_sd": float(np.std(aris)) if aris else np.nan, "n_stability_pairs": len(aris),
                     "ari_vs_participant_id": float(adjusted_rand_score(parts, lab)),
                     "ari_vs_leg": float(adjusted_rand_score(win_df["leg"], lab))})
    summ = pd.DataFrame(rows)
    best_k = int(summ.sort_values("silhouette", ascending=False).iloc[0]["k"])
    assign = win_df[["participant_id", "leg", "recording_id", "window_index"]].copy()
    assign["cluster"] = labels_by_k[best_k]
    ct = pd.crosstab(assign["participant_id"], assign["cluster"])
    return {"summary": summ, "best_k_by_silhouette": best_k, "assignments": assign, "participant_by_cluster": ct}


def lr_classification(win_df: pd.DataFrame, feats: list[str], cfg: dict, seed: int) -> dict:
    """Leave-one-participant-out left/right classification with in-fold scaling and a participant-level label-swap test.

    Participant-level accuracy = mean window accuracy over the held-out participant's left+right windows. The
    statistic is the mean over the participants. See ``src.models.permutation`` for exactly what is permuted and the
    p-value convention (exact enumeration of all 2**n participant swaps when feasible).
    """
    from src.models import permutation as perm
    mc = cfg["models"]
    X = win_df[feats].to_numpy(dtype=float)
    y = (win_df["leg"] == "right").astype(int).to_numpy()
    groups = win_df["participant_id"].to_numpy()
    rec = win_df["recording_id"].to_numpy()
    mk = lambda: perm.default_model(seed)  # noqa: E731
    accs, order = perm.logo_participant_accuracies(X, y, groups, mk)
    preds = np.zeros(len(y), dtype=int)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        preds[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    per_part = pd.DataFrame({"participant_id": order, "window_accuracy": accs,
                             "n_windows": [int((groups == g).sum()) for g in order]})
    rec_df = pd.DataFrame({"recording_id": rec, "true": y, "pred": preds}).groupby("recording_id").agg(
        true=("true", "first"), frac_pred_right=("pred", "mean"))
    rec_df["correct"] = (rec_df["frac_pred_right"] > 0.5) == (rec_df["true"] == 1)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(accs, len(accs)).mean() for _ in range(5000)])
    obs = float(accs.mean())

    uniq = np.unique(groups)
    mode = mc.get("permutation_mode", "monte_carlo")
    patterns, exact = perm.swap_patterns(len(uniq), None if mode == "exact" else mc["permutations"],
                                         np.random.default_rng(seed + 1), mc.get("permutation_exact_max", 5000))
    null = perm.permutation_null(X, y, groups, mk, patterns)
    return {
        "participant_accuracy": per_part, "recording_decisions": rec_df.reset_index(), "permutation_null": null,
        "metrics": {
            "n_participants": int(len(uniq)), "n_recordings": int(len(np.unique(rec))), "n_windows": int(len(y)),
            "mean_participant_window_accuracy": obs,
            "bootstrap_ci95_over_participants": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "bootstrap_resamples": 5000, "bootstrap_seed": int(seed),
            "recording_level_accuracy": float(rec_df["correct"].mean()),
            "permutation_null_mean": float(null.mean()), "permutation_null_sd": float(null.std(ddof=1)),
            "permutation_null_q95": float(np.quantile(null, 0.95)),
            "permutation_p": perm.permutation_pvalue(obs, null, exact), "n_permutations": int(len(null)),
            "permutation_exact_enumeration": bool(exact),
            "permutation_convention": ("exact: p = #{null >= obs} / 2**n (identity included)" if exact else
                                       "Monte Carlo: p = (b + 1) / (B + 1)"),
            "chance_level": 0.5,
            "validation": "leave-one-participant-out; scaler fitted on training participants only; no feature selection",
        },
    }
