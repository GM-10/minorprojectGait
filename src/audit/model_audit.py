"""Audit of the left/right classifier, anomaly detection and clustering.

Re-implements the evaluation loops independently of src.models.experiments so that the reported numbers can be
cross-checked, and adds baselines and robustness variants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.models import permutation as perm


# ----------------------------------------------------------------------------- classifier
def logo_eval(X: np.ndarray, y: np.ndarray, groups: np.ndarray, make_model, check: bool = True
              ) -> tuple[np.ndarray, np.ndarray, list]:
    """Leave-one-participant-out. Returns (per-participant window accuracy, predictions, group order).

    With ``check`` asserts that no held-out participant appears in the training indices.
    """
    accs, preds, order = [], np.full(len(y), -1), []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if check:
            assert not (set(groups[tr]) & set(groups[te])), "participant leaked across train/test"
            assert len(set(groups[te])) == 1
        m = make_model().fit(X[tr], y[tr])
        p = m.predict(X[te])
        preds[te] = p
        accs.append(float((p == y[te]).mean()))
        order.append(groups[te][0])
    return np.array(accs), preds, order


def permutation_null(X, y, groups, make_model, n_perm, seed: int) -> np.ndarray:
    """Null distribution of the mean participant accuracy under participant-level left/right label swaps.

    ``n_perm`` = "exact" enumerates all 2**n swaps; an int draws that many random swaps (seeded). See
    ``src.models.permutation`` for the design, assumptions and p-value convention.
    """
    patterns, _ = perm.swap_patterns(len(np.unique(groups)), None if n_perm == "exact" else int(n_perm),
                                     np.random.default_rng(seed + 1))
    return perm.permutation_null(X, y, groups, make_model, patterns)


def ci_methods(accs: np.ndarray, seed: int = 0) -> dict:
    """Several 95% intervals for the mean of participant-level accuracies (n=12 clusters)."""
    n = len(accs)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(accs, n).mean() for _ in range(20000)])
    se = accs.std(ddof=1) / np.sqrt(n)
    tlo, thi = stats.t.interval(0.95, n - 1, loc=accs.mean(), scale=se)
    w = stats.wilcoxon(accs - 0.5, alternative="greater") if (accs != 0.5).any() else None
    return {"mean": accs.mean(), "sd_between_participants": accs.std(ddof=1), "min": accs.min(), "max": accs.max(),
            "boot_pct_low": np.quantile(boots, .025), "boot_pct_high": np.quantile(boots, .975),
            "t_interval_low": tlo, "t_interval_high": thi, "one_sample_t_vs_0.5_p": stats.ttest_1samp(accs, .5).pvalue,
            "wilcoxon_gt_0.5_p": w.pvalue if w is not None else np.nan,
            "n_participants_above_0.5": int((accs > .5).sum()), "n_participants_below_0.5": int((accs < .5).sum()),
            "n_participants_at_0.5": int((accs == .5).sum())}


def classifier_audit(win: pd.DataFrame, model_feats: list[str], sens: pd.DataFrame, seed: int, n_perm_main="exact",
                     n_perm_other="exact") -> dict:
    """Full audit of the left/right classification experiment."""
    y = (win["leg"] == "right").astype(int).to_numpy()
    groups = win["participant_id"].to_numpy()
    X = win[model_feats].to_numpy(float)
    balance = (win.groupby(["participant_id", "leg"]).size().unstack().assign(total=lambda d: d.sum(axis=1)))
    lr = lambda C=1.0: (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C, random_state=seed)))  # noqa
    exposure = sens.set_index("feature")["exposure_fraction_recordings"]
    low_exp = [f for f in model_feats if exposure.get(f, 1) < 0.5]
    no_accz = [f for f in model_feats if "acc_z" not in f]
    mag_only = [f for f in model_feats if f.split("__")[0] in ("acc_mag", "acc_dyn_mag", "gyro_mag") or f == "corr__acc_mag__gyro_mag"]
    variants = {
        "full_model (as reported)": (model_feats, lr()),
        "baseline: majority class": (model_feats, lambda: DummyClassifier(strategy="most_frequent")),
        "single feature: qc_clip_fraction_acc_z": (["qc_clip_fraction_acc_z"], lr()),
        "single feature: gyro_z__mean": (["gyro_z__mean"], lr()),
        "single feature: acc_z__max": (["acc_z__max"], lr()),
        f"low-clip-exposure features only ({len(low_exp)})": (low_exp, lr()),
        f"without any acc_z feature ({len(no_accz)})": (no_accz, lr()),
        f"magnitude-only features, rotation-invariant ({len(mag_only)})": (mag_only, lr()),
        "full_model, C=0.01": (model_feats, lr(0.01)), "full_model, C=0.1": (model_feats, lr(0.1)),
        "full_model, C=10": (model_feats, lr(10.0)),
    }
    rows, per_part, nulls = [], {}, {}
    for name, (fs, mk) in variants.items():
        Xv = win[fs].to_numpy(float)
        accs, preds, order = logo_eval(Xv, y, groups, mk)
        per_part[name] = pd.Series(accs, index=order)
        n_perm = n_perm_main if name.startswith("full_model (as") else (n_perm_other if len(fs) > 0 and "baseline" not in name else 0)
        pval, n_used = np.nan, 0
        if n_perm:
            nulls[name] = permutation_null(Xv, y, groups, mk, n_perm, seed)
            n_used = len(nulls[name])
            pval = perm.permutation_pvalue(accs.mean(), nulls[name], exact=(n_perm == "exact"))
        rows.append({"model": name, "n_features": len(fs), "mean_participant_accuracy": accs.mean(),
                     "sd_participant_accuracy": accs.std(ddof=1), "n_perm": n_used, "exact_enumeration": n_perm == "exact" and bool(n_used),
                     "permutation_p": pval, "null_sd": nulls[name].std(ddof=1) if name in nulls else np.nan,
                     "null_mean": nulls[name].mean() if name in nulls else np.nan,
                     "null_q95": np.quantile(nulls[name], .95) if name in nulls else np.nan})
    res = pd.DataFrame(rows)
    full = per_part["full_model (as reported)"]
    # full model fitted on everything (descriptive only!) to see what drives it
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)).fit(X, y)
    coef = pd.Series(m[-1].coef_[0], index=model_feats).sort_values(key=np.abs, ascending=False).head(12)
    # pair-level view: how many of the 2 recordings per participant are right
    _, preds, _ = logo_eval(X, y, groups, lr())
    d = pd.DataFrame({"rec": win["recording_id"], "p": win["participant_id"], "y": y, "pred": preds})
    rec = d.groupby(["p", "rec"]).agg(y=("y", "first"), frac_right=("pred", "mean")).reset_index()
    rec["correct"] = (rec["frac_right"] > .5) == (rec["y"] == 1)
    rec_tie = int((rec["frac_right"] == .5).sum())
    pair_correct = rec.groupby("p")["correct"].sum()
    return {"balance": balance.reset_index(), "variants": res, "per_participant": pd.DataFrame(per_part),
            "ci": pd.DataFrame([ci_methods(full.to_numpy())]), "coef": coef.rename("std_coef").reset_index(),
            "recording_level": {"accuracy": float(rec["correct"].mean()), "n": len(rec), "n_exact_ties": rec_tie,
                                "wilson95": [float(x) for x in stats.binomtest(int(rec["correct"].sum()), len(rec)).proportion_ci(.95, method="wilson")],
                                "pair_correct_counts": pair_correct.value_counts().sort_index().to_dict()},
            "null_main": nulls.get("full_model (as reported)"), "nulls": nulls,
            "random_guess": perm.random_guess_distribution(groups, 5000, np.random.default_rng(seed + 2))}


# ----------------------------------------------------------------------------- anomaly / clustering robustness
def _prep(df: pd.DataFrame, feats: list[str], scaler: str, center_participant: bool) -> np.ndarray:
    X = df[feats].copy()
    if center_participant:
        X = X - X.groupby(df["participant_id"]).transform("mean")
    Xv = X.to_numpy(float)
    sc = RobustScaler() if scaler == "robust" else StandardScaler()
    out = sc.fit_transform(Xv)
    return np.nan_to_num(out)  # features with zero spread after centring


def _if_score(X, seed, contamination=0.05):
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(X)
    return -iso.score_samples(X)


def _eta2(v: np.ndarray, g: np.ndarray) -> float:
    tot = ((v - v.mean()) ** 2).sum()
    return float(sum((g == k).sum() * (v[g == k].mean() - v.mean()) ** 2 for k in np.unique(g)) / tot)


def anomaly_cluster_audit(win: pd.DataFrame, model_feats: list[str], all_feats: list[str], sens: pd.DataFrame,
                          seed: int, contamination: float = 0.05, clip_thresh: float = 0.05) -> dict:
    """Robustness of Isolation-Forest scores and k-means (k=2) to preprocessing choices."""
    exposure = sens.set_index("feature")["exposure_fraction_recordings"]
    low_exp = [f for f in all_feats if exposure.get(f, 1) < 0.5]
    keep = (win["qc_clip_fraction_any"] <= clip_thresh).to_numpy()
    n_flag = int(round(contamination * len(win)))
    base_X = _prep(win, model_feats, "standard", False)
    base = _if_score(base_X, seed, contamination)
    base_lab = KMeans(2, n_init=10, random_state=seed).fit_predict(base_X)
    variants = {
        "robust scaler": dict(feats=model_feats, scaler="robust", center=False, mask=None),
        "participant-centred features": dict(feats=model_feats, scaler="standard", center=True, mask=None),
        f"windows with clip fraction <= {clip_thresh} only": dict(feats=model_feats, scaler="standard", center=False, mask=keep),
        f"low-clip-exposure features only ({len(low_exp)})": dict(feats=low_exp, scaler="standard", center=False, mask=None),
        "all usable features (not whitelisted)": dict(feats=all_feats, scaler="standard", center=False, mask=None),
    }
    rows = []

    def compare(name, X, mask):
        idx = np.arange(len(win)) if mask is None else np.where(mask)[0]
        s = _if_score(X, seed, contamination)
        lab = KMeans(2, n_init=10, random_state=seed).fit_predict(X)
        top_b = set(idx[np.argsort(-base[idx])[:n_flag]])
        top_v = set(idx[np.argsort(-s)[:n_flag]])
        return {"variant": name, "n_windows": len(idx), "spearman_iforest_vs_base": stats.spearmanr(base[idx], s).statistic,
                "top_k_overlap_jaccard": len(top_b & top_v) / len(top_b | top_v), "k": n_flag,
                "kmeans2_ari_vs_base": adjusted_rand_score(base_lab[idx], lab),
                "iforest_eta2_participant": _eta2(s, win["participant_id"].to_numpy()[idx])}

    for name, v in variants.items():
        Xv = _prep(win if v["mask"] is None else win[v["mask"]].reset_index(drop=True), v["feats"], v["scaler"], v["center"])
        rows.append(compare(name, Xv, v["mask"]))
    for s2 in (1, 2, 3):
        sc = _if_score(base_X, s2, contamination)
        rows.append({"variant": f"isolation-forest seed {s2}", "n_windows": len(win),
                     "spearman_iforest_vs_base": stats.spearmanr(base, sc).statistic,
                     "top_k_overlap_jaccard": len(set(np.argsort(-base)[:n_flag]) & set(np.argsort(-sc)[:n_flag])) /
                     len(set(np.argsort(-base)[:n_flag]) | set(np.argsort(-sc)[:n_flag])), "k": n_flag,
                     "kmeans2_ari_vs_base": np.nan, "iforest_eta2_participant": _eta2(sc, win["participant_id"].to_numpy())})
    rob = pd.DataFrame(rows)
    # what do windows flagged by the base model look like?
    flagged = np.zeros(len(win), bool)
    flagged[np.argsort(-base)[:n_flag]] = True
    fl = win.loc[flagged, ["recording_id", "window_index"]].sort_values(["recording_id", "window_index"])
    episodes = 0
    for _, g in fl.groupby("recording_id"):
        w = g["window_index"].to_numpy()
        episodes += 1 + int((np.diff(w) > 2).sum())  # windows overlap by 50%: neighbours up to 2 apart share samples
    shared = {"n_flagged": int(flagged.sum()), "n_episodes_if_overlapping_windows_merged": episodes,
              "base_iforest_eta2_participant": _eta2(base, win["participant_id"].to_numpy()),
              "base_iforest_eta2_leg": _eta2(base, (win["leg"] == "right").to_numpy().astype(int)),
              "spearman_score_vs_clip_fraction": float(stats.spearmanr(base, win["qc_clip_fraction_any"]).statistic),
              "spearman_score_vs_gyro_mag_rms": float(stats.spearmanr(base, win["gyro_mag__rms"]).statistic),
              "spearman_score_vs_acc_dyn_mag_rms": float(stats.spearmanr(base, win["acc_dyn_mag__rms"]).statistic)}
    cl = pd.DataFrame({"cluster": base_lab, "clip": win["qc_clip_fraction_any"], "gyro_mag_rms": win["gyro_mag__rms"],
                       "acc_dyn_mag_rms": win["acc_dyn_mag__rms"], "participant": win["participant_id"], "leg": win["leg"]})
    csum = cl.groupby("cluster").agg(n=("clip", "size"), mean_clip=("clip", "mean"), mean_gyro_mag_rms=("gyro_mag_rms", "mean"),
                                     mean_acc_dyn_mag_rms=("acc_dyn_mag_rms", "mean"),
                                     n_participants=("participant", "nunique"), frac_right=("leg", lambda s: (s == "right").mean()))
    ct = pd.crosstab(cl["participant"], cl["cluster"])
    shared["kmeans2_ari_vs_participant"] = float(adjusted_rand_score(win["participant_id"], base_lab))
    shared["kmeans2_ari_vs_leg"] = float(adjusted_rand_score(win["leg"], base_lab))
    shared["kmeans2_spearman_cluster_vs_gyro_mag_rms"] = float(stats.spearmanr(base_lab, win["gyro_mag__rms"]).statistic)
    return {"robustness": rob, "summary": shared, "cluster_profile": csum.reset_index(), "participant_by_cluster": ct.reset_index()}
