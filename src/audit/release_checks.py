"""Independent release checks: re-derive the key numbers with code that does not reuse the pipeline's implementations.

    python -m src.audit.release_checks

Writes ``reports/audit/release_checks.json`` and ``release_consistency_checks.csv``. Reads the pipeline outputs
(processed_data/, reports/) and the raw CSVs (read-only). Exit status is non-zero if any hard check fails.
"""
from __future__ import annotations

import json
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from src.config import load_config

ID_COLS = ["participant_id", "leg", "filename", "recording_id", "scope", "window_index", "start_idx", "end_idx",
           "start_ms", "end_ms", "n_samples", "fs_hz", "feature_config_id"]


def _features(df):
    return [c for c in df.columns if c not in ID_COLS and not c.startswith("qc_")]


def _model_features(win, whitelist):
    """Independent re-statement of the model-feature rule: whitelisted statistics + all cross-correlations, finite and non-constant."""
    keep = [f for f in _features(win) if f.startswith("corr__") or f.split("__")[-1] in whitelist]
    return [f for f in keep if np.isfinite(win[f].to_numpy(float)).all() and win[f].std() > 0]


def _logo(X, y, groups, seed, C=1.0):
    """Plain-loop leave-one-participant-out (no helper reuse): scaler and model are fitted on the training participants only."""
    accs = {}
    for g in sorted(set(groups)):
        te = groups == g
        tr = ~te
        assert g not in set(groups[tr]) and set(groups[te]) == {g}
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=3000, C=C, random_state=seed).fit(sc.transform(X[tr]), y[tr])
        accs[g] = float((m.predict(sc.transform(X[te])) == y[te]).mean())
    return accs


def classifier_checks(cfg, win, rep, mf, seed, n_check=120) -> dict:
    out: dict = {}
    y = (win["leg"] == "right").astype(int).to_numpy()
    groups = win["participant_id"].to_numpy()
    X = win[mf].to_numpy(float)
    accs = _logo(X, y, groups, seed)
    saved = pd.read_csv(rep / "models" / "lr_classification_participant_accuracy.csv").set_index("participant_id")["window_accuracy"]
    out["participant_accuracy_max_abs_diff_vs_saved"] = float(max(abs(accs[k] - saved[k]) for k in accs))
    obs = float(np.mean(list(accs.values())))
    out["observed_mean_participant_accuracy"] = obs
    out["class_balance"] = {"left": int((y == 0).sum()), "right": int((y == 1).sum())}
    out["n_folds"] = len(accs)
    # permutation null: re-derive a random subset of the 4096 swap patterns with independent bit decoding
    null = pd.read_csv(rep / "models" / "lr_permutation_null.csv")["mean_participant_accuracy"].to_numpy()
    parts = sorted(set(groups))
    n = len(parts)
    out["n_permutations_saved"] = int(len(null))
    out["n_patterns_expected"] = 2 ** n
    rng = np.random.default_rng(12345)
    idx = rng.choice(len(null), size=min(n_check, len(null)), replace=False)
    diffs = []
    for i in idx:
        swap = {parts[j]: bool((int(i) >> (n - 1 - j)) & 1) for j in range(n)}
        yp = np.array([1 - v if swap[g] else v for v, g in zip(y, groups)])
        diffs.append(abs(np.mean(list(_logo(X, yp, groups, seed).values())) - null[int(i)]))
    out["null_values_rechecked"] = int(len(idx))
    out["null_max_abs_diff_vs_independent_recompute"] = float(max(diffs))
    out["identity_pattern_equals_observed"] = bool(abs(null[0] - obs) < 1e-12)
    out["null_mean"], out["null_sd"], out["null_q95"] = float(null.mean()), float(null.std(ddof=1)), float(np.quantile(null, .95))
    b = int((null >= obs - 1e-12).sum())
    out["p_exact_recomputed"] = b / len(null)
    out["p_exact_b"] = b
    out["p_two_sided_symmetric_note"] = float(((null >= obs - 1e-12) | (null <= (2 * 0.5 - obs) + 1e-12)).mean())
    # confidence intervals from the saved per-participant accuracies
    a = saved.to_numpy()
    r1 = np.random.default_rng(seed)
    boots = np.array([r1.choice(a, len(a)).mean() for _ in range(5000)])
    out["bootstrap_pipeline_reproduced"] = [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]
    r2 = np.random.default_rng(0)
    b2 = np.array([r2.choice(a, len(a)).mean() for _ in range(20000)])
    out["bootstrap_audit_reproduced"] = [float(np.quantile(b2, .025)), float(np.quantile(b2, .975))]
    out["t_interval"] = [float(x) for x in stats.t.interval(0.95, len(a) - 1, loc=a.mean(), scale=stats.sem(a))]
    # jackknife-over-participants SE (a third, assumption-light view of between-participant variability)
    jk = np.array([np.delete(a, i).mean() for i in range(len(a))])
    out["jackknife_se"] = float(np.sqrt((len(a) - 1) / len(a) * ((jk - jk.mean()) ** 2).sum()))
    out["sd_between_participants"] = float(a.std(ddof=1))
    # variant files
    var = pd.read_csv(rep / "audit" / "classifier_hardware_variants.csv")
    nulls = pd.read_csv(rep / "audit" / "classifier_permutation_nulls_hardware_variants.csv")
    rows = []
    for _, r in var.iterrows():
        nv = nulls[r["variant"]].to_numpy()
        rows.append({"variant": r["variant"], "reported_p": float(r["permutation_p"]),
                     "recomputed_p": float((nv >= r["mean_participant_accuracy"] - 1e-12).mean()), "n_null": int(len(nv))})
    out["variant_p_reproduction"] = rows
    out["variant_p_max_abs_diff"] = float(max(abs(r["reported_p"] - r["recomputed_p"]) for r in rows))
    return out


def sign_and_rotation_invariance(cfg, raw_dir: Path) -> dict:
    """Numerically verify the 'sign-independent' and 'magnitude-only' feature sets on real data."""
    from src.audit.hardware_audit import sign_invariant_features
    from src.features.extract import window_features
    f = sorted(raw_dir.glob("Sub05Left*.csv"))[0]
    d = pd.read_csv(f).iloc[100:200]
    cols = {"acc_x": "AccelX_g", "acc_y": "AccelY_g", "acc_z": "AccelZ_g", "gyro_x": "GyroX_dps", "gyro_y": "GyroY_dps", "gyro_z": "GyroZ_dps"}
    base = {k: d[c].to_numpy(float) for k, c in cols.items()}

    def feats(sig):
        s = dict(sig)
        s["acc_mag"] = np.sqrt(sum(s[a] ** 2 for a in ("acc_x", "acc_y", "acc_z")))
        s["gyro_mag"] = np.sqrt(sum(s[a] ** 2 for a in ("gyro_x", "gyro_y", "gyro_z")))
        return window_features(s, 50.0, cfg)

    f0 = feats(base)
    inv = sign_invariant_features(list(f0))
    worst, n_changed_excluded = 0.0, 0
    for ax in cols:
        flipped = dict(base)
        flipped[ax] = -base[ax]
        f1 = feats(flipped)
        for k in inv:
            if np.isfinite(f0[k]):
                worst = max(worst, abs(f1[k] - f0[k]) / (abs(f0[k]) + 1e-9))
        n_changed_excluded += sum(1 for k in f0 if k not in inv and np.isfinite(f0[k]) and abs(f1[k] - f0[k]) > 1e-9)
    # random rotation of each triple: magnitude features must not change, per-axis sign-invariant features may
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    A = Q @ np.vstack([base[a] for a in ("acc_x", "acc_y", "acc_z")])
    G = Q @ np.vstack([base[a] for a in ("gyro_x", "gyro_y", "gyro_z")])
    rot = {"acc_x": A[0], "acc_y": A[1], "acc_z": A[2], "gyro_x": G[0], "gyro_y": G[1], "gyro_z": G[2]}
    f2 = feats(rot)
    mag = [k for k in f0 if k.split("__")[0] in ("acc_mag", "gyro_mag") or k == "corr__acc_mag__gyro_mag"]
    mag_worst = max(abs(f2[k] - f0[k]) / (abs(f0[k]) + 1e-9) for k in mag if np.isfinite(f0[k]))
    axis_inv = [k for k in inv if k.split("__")[0] in cols]
    axis_changed_by_rotation = int(sum(abs(f2[k] - f0[k]) > 1e-6 * (abs(f0[k]) + 1e-9) for k in axis_inv if np.isfinite(f0[k])))
    return {"n_features_checked": len(f0), "n_sign_invariant_in_set": len(inv), "max_rel_change_of_set_when_any_axis_negated": float(worst),
            "n_excluded_features_that_do_change": int(n_changed_excluded), "n_magnitude_features": len(mag),
            "max_rel_change_of_magnitude_features_under_rotation": float(mag_worst),
            "n_per_axis_sign_invariant_features_changed_by_rotation": axis_changed_by_rotation, "n_per_axis_sign_invariant": len(axis_inv)}


def paired_checks(cfg, rec, win, rep) -> dict:
    """Independent paired analysis straight from the recording-level table with scipy only."""
    out: dict = {}
    feats = _features(rec)
    L = rec[rec.leg == "left"].set_index("participant_id").sort_index()
    R = rec[rec.leg == "right"].set_index("participant_id").sort_index()
    assert list(L.index) == list(R.index) and len(L) == 12 and rec.groupby(["participant_id", "leg"]).size().max() == 1
    D = (L[feats] - R[feats]).to_numpy(float)
    out["n_pairs"], out["n_feature_columns"] = len(L), len(feats)
    sd = D.std(axis=0, ddof=1)
    ok = sd > 0
    out["n_testable"] = int(ok.sum())
    Dt = D[:, ok]
    names = np.array(feats)[ok]
    t = stats.ttest_1samp(Dt, 0, axis=0)
    pw = np.array([stats.wilcoxon(Dt[:, j]).pvalue if (Dt[:, j] != 0).any() else np.nan for j in range(Dt.shape[1])])
    sh = np.array([stats.shapiro(Dt[:, j]).pvalue for j in range(Dt.shape[1])])
    prim = np.where(sh >= 0.05, t.pvalue, pw)
    out["n_paired_t"], out["n_wilcoxon"] = int((sh >= .05).sum()), int((sh < .05).sum())
    out["n_p_lt_05"] = int((prim < .05).sum())
    out["bh_min_q"] = float(multipletests(prim, method="fdr_bh")[1].min())
    out["holm_min_q"] = float(multipletests(prim, method="holm")[1].min())
    out["n_bh_sig"] = int((multipletests(prim, method="fdr_bh")[1] < .05).sum())
    # exact sign-flip with scipy t-statistics (independent of the closed-form version used before)
    tcrit = stats.t.ppf(.975, 11)
    obs_t = t.statistic
    counts, maxT = [], []
    for flips in product([1, -1], repeat=12):
        tt = stats.ttest_1samp(Dt * np.array(flips)[:, None], 0, axis=0).statistic
        counts.append(int((np.abs(tt) > tcrit).sum()))
        maxT.append(np.abs(tt).max())
    counts, maxT = np.array(counts), np.array(maxT)
    obs_n = int((np.abs(obs_t) > tcrit).sum())
    out["signflip"] = {"observed_n": obs_n, "null_mean": float(counts.mean()), "null_q95": float(np.quantile(counts, .95)),
                       "global_p": float((counts >= obs_n).mean()), "min_fwer_p": float(min((maxT >= abs(x) - 1e-12).mean() for x in obs_t))}
    # sign-invariant subset
    from src.audit.hardware_audit import sign_invariant_features
    si = [f for f in sign_invariant_features(feats) if f in set(names)]
    idx = [list(names).index(f) for f in si]
    ps = prim[idx]
    out["n_sign_invariant"] = len(sign_invariant_features(feats))
    out["n_sign_invariant_testable"] = len(si)
    out["sign_invariant"] = {"n_p_lt_05": int((ps < .05).sum()), "n_bh_sig": int((multipletests(ps, method="fdr_bh")[1] < .05).sum()),
                             "min_q": float(multipletests(ps, method="fdr_bh")[1].min())}
    # exclude Sub04, Sub06
    keep = ~L.index.isin(["Sub04", "Sub06"])
    Dx = Dt[keep]
    px = np.where(np.array([stats.shapiro(Dx[:, j]).pvalue for j in range(Dx.shape[1])]) >= .05,
                  stats.ttest_1samp(Dx, 0, axis=0).pvalue, np.array([stats.wilcoxon(Dx[:, j]).pvalue if (Dx[:, j] != 0).any() else np.nan for j in range(Dx.shape[1])]))
    okx = np.isfinite(px)
    out["exclude_sub04_sub06"] = {"n_pairs": int(keep.sum()), "n_p_lt_05": int((px[okx] < .05).sum()),
                                  "n_bh_sig": int((multipletests(px[okx], method="fdr_bh")[1] < .05).sum())}
    saved_x = pd.read_csv(rep / "tables" / "paired_stats_excluding_noted_participants.csv")
    out["exclude_matches_pipeline_n_p_lt_05"] = int((saved_x["p_primary"] < .05).sum())
    # window-level data are never used as the paired unit
    out["window_rows_per_participant_leg"] = int(win.groupby(["participant_id", "leg"]).size().min())
    out["recording_rows_per_participant_leg"] = int(rec.groupby(["participant_id", "leg"]).size().max())
    return out


def trimmed_check(rep: Path, raw_dir: Path) -> dict:
    """Recompute gyro_mag__rms over t >= 2.02 s directly from raw CSVs (recordings without gaps) and compare to the saved table."""
    tab = pd.read_csv(rep / "audit" / "trimmed_first_2s_recording_features.csv")
    saved = tab.set_index("recording_id")["gyro_mag__rms"]
    diffs = {}
    for f in sorted(raw_dir.glob("*.csv")):
        d = pd.read_csv(f)
        if len(d) != 500 or d["Time_ms"].diff().dropna().ne(20).any() or d["Time_ms"].iloc[0] != 20:
            continue
        g = np.sqrt((d[["GyroX_dps", "GyroY_dps", "GyroZ_dps"]].to_numpy(float) ** 2).sum(axis=1))[100:]
        m = re.match(r"Sub(\d+)(Left|Right)", f.name)
        rid = f"Sub{int(m[1]):02d}_{m[2].lower()}"
        diffs[rid] = abs(float(np.sqrt(np.mean(g ** 2))) - float(saved[rid]))
    return {"n_recordings_checked": len(diffs), "max_abs_diff_vs_saved": float(max(diffs.values())),
            "n_samples_in_saved_trimmed_table": sorted(tab["n_samples"].unique().tolist())}


def hardware_text_checks(root: Path) -> list[dict]:
    """Text-level safeguards: statements that must not appear unqualified in the reports."""
    docs = {p.name: p.read_text(encoding="utf-8") for p in [root / "reports" / "final_report.md", root / "reports" / "scientific_audit.md",
                                                             root / "docs" / "data_collection.md"] if p.exists()}
    forbidden = [r"confirmed (hardware )?saturation(?! is not| not)", r"proves? that", r"eliminat\w+ (the )?mounting", r"validated injury",
                 r"pathological gait", r"is equivalent", r"demonstrates? (a )?left"]
    rows = []
    for name, txt in docs.items():
        for pat in forbidden:
            hits = [m.group(0) for m in re.finditer(pat, txt, flags=re.I)]
            rows.append({"file": name, "pattern": pat, "n_hits": len(hits), "examples": "; ".join(hits[:3])})
    return rows


def consistency_checks(cfg, root: Path) -> pd.DataFrame:
    """Every number in the checklist is derived from the saved tables and searched for in the report text."""
    rep = Path(cfg["paths"]["reports_dir"])
    fr = (rep / "final_report.md").read_text(encoding="utf-8")
    sa_txt = (rep / "scientific_audit.md").read_text(encoding="utf-8")
    dq = json.loads((rep / "data_quality" / "data_quality_summary.json").read_text(encoding="utf-8"))
    rq = pd.read_csv(rep / "data_quality" / "data_quality_recordings.csv")
    rs = json.loads((rep / "run_summary.json").read_text(encoding="utf-8"))
    pr = pd.read_csv(rep / "tables" / "paired_stats_all_participants.csv")
    au = json.loads((rep / "audit" / "audit_summary.json").read_text(encoding="utf-8"))
    lr = rs["models"]["lr_classification"]
    an = rs["models"]["anomaly"]
    cs = pd.DataFrame(rs["models"]["clustering"]["summary"])
    fv = pd.read_csv(rep / "audit" / "classifier_hardware_variants.csv").set_index("variant")
    clip = rq["fraction_samples_with_any_clipped_channel"]
    hw = au["hardware"]
    ci = lr["bootstrap_ci95_over_participants"]
    si, mo = "sign-invariant model features only", "magnitude-only model features (rotation-invariant)"
    nt = int(pr["p_primary"].notna().sum())
    checks = [
        ("final_report", "files", f"**{dq['n_files']} CSV files**"), ("final_report", "participants", f"**{dq['n_participants']} participants**"),
        ("final_report", "pairs", f"**{dq['n_valid_pairs']} candidate-valid left/right pairs**"),
        ("final_report", "rows range", f"Rows per file: {dq['rows_range'][0]}-{dq['rows_range'][1]}"),
        ("final_report", "duration range", f"duration {rq['duration_s'].min():.2f}-{rq['duration_s'].max():.2f} s"),
        ("final_report", "fs", f"{dq['estimated_fs_hz_range'][0]:.1f}-{dq['estimated_fs_hz_range'][1]:.1f} Hz"),
        ("final_report", "clipped mean/max", f"{100*clip.mean():.1f}% of samples on average (max {100*clip.max():.1f}% in {rq.loc[clip.idxmax(), 'recording_id']})"),
        ("final_report", "windows/recordings", f"{rs['n_windows']} windows from {rs['n_recordings']} recordings"),
        ("final_report", "features per row", f"({rs['n_features']} per row)"),
        ("final_report", "features tested", f"Features tested: {nt}"),
        ("final_report", "raw p<.05", f"**{int((pr['p_primary'] < .05).sum())}** of {nt}"),
        ("final_report", "BH", f"After BH correction: **{int((pr['q_bh_primary'] < .05).sum())}**"),
        ("final_report", "min q", f"Smallest BH q = {pr['q_bh_primary'].min():.2f}"),
        ("final_report", "signflip global p", f"global p = {au['stats']['signflip']['global_p_count']:.2f}"),
        ("final_report", "pca", f"PC1-PC2 explain {100*rs['pca']['evr'][0]:.1f}% and {100*rs['pca']['evr'][1]:.1f}%"),
        ("final_report", "anomaly flagged", f"{an['n_flagged_iforest']} windows flagged"), ("final_report", "anomaly both", f"{an['n_flagged_both']} by both methods"),
        ("final_report", "kmeans silhouette", f"max {cs['silhouette'].max():.2f}"),
        ("final_report", "clf accuracy", f"**{lr['mean_participant_window_accuracy']:.2f}**"), ("final_report", "clf interval", f"{ci[0]:.2f}-{ci[1]:.2f}"),
        ("final_report", "clf null q95", f"95th pct {lr['permutation_null_q95']:.2f}"), ("final_report", "clf p", f"permutation p = {lr['permutation_p']:.3f}"),
        ("final_report", "clf n perm", f"{lr['n_permutations']} participant-level label swaps"),
        ("final_report", "sign-invariant clf", f"{fv.loc[si, 'mean_participant_accuracy']:.2f} (exact permutation p = {fv.loc[si, 'permutation_p']:.2f})"),
        ("final_report", "magnitude-only clf", f"{fv.loc[mo, 'mean_participant_accuracy']:.2f} (p = {fv.loc[mo, 'permutation_p']:.2f})"),
        ("final_report", "sign-invariant paired", f"gives {hw['paired_sign_invariant']['n_p_lt_05']} raw p < 0.05 and {hw['paired_sign_invariant']['n_bh_lt_05']} BH-significant"),
        ("scientific_audit", "exact p", f"{lr['permutation_p']:.3f}"),
        ("scientific_audit", "b/B", f"{int(round(lr['permutation_p'] * lr['n_permutations']))}/{lr['n_permutations']}"),
        ("scientific_audit", "null q95", f"{lr['permutation_null_q95']:.3f}"), ("scientific_audit", "null sd", f"SD {lr['permutation_null_sd']:.3f}"),
        ("scientific_audit", "sign-inv acc", f"{fv.loc[si, 'mean_participant_accuracy']:.3f}"),
        ("scientific_audit", "mag-only acc", f"{fv.loc[mo, 'mean_participant_accuracy']:.3f}"),
        ("scientific_audit", "min BH q", f"{pr['q_bh_primary'].min():.2f}"), ("scientific_audit", "13 of 183", f"{int((pr['p_primary'] < .05).sum())} of {nt}"),
        ("scientific_audit", "plateaus", f"{hw['n_plateaus']} runs"), ("scientific_audit", "sign-inv n", f"{hw['n_sign_invariant_features']} of 183"),
        ("scientific_audit", "gyro grid null", f"{100 * hw['gyro_null_fraction']:.1f} %"),
        ("scientific_audit", "static |a|", f"{hw['static_acc_norm_mean']:.3f} g"),
        ("scientific_audit", "L/R gravity angle", f"{hw['angle_left_right_static_gravity_deg']['median']:.1f}"),
    ]
    rows = []
    for doc, name, exp in checks:
        txt = fr if doc == "final_report" else sa_txt
        rows.append({"document": doc, "claim": name, "expected_string_from_tables": exp, "found": exp in txt})
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    root = Path(cfg["paths"]["reports_dir"]).parent
    rep = Path(cfg["paths"]["reports_dir"])
    proc = Path(cfg["paths"]["processed_dir"])
    win = pd.read_csv(proc / "features_windows.csv")
    rec = pd.read_csv(proc / "features_recordings.csv")
    seed = cfg["random_seed"]
    mf = _model_features(win, cfg["models"]["feature_stat_whitelist"])
    res = {"n_model_features": len(mf), "seed": seed}
    res["classifier"] = classifier_checks(cfg, win, rep, mf, seed)
    res["invariance"] = sign_and_rotation_invariance(cfg, Path(cfg["paths"]["raw_data_dir"]))
    res["paired"] = paired_checks(cfg, rec, win, rep)
    res["trimmed_gyro_mag_rms_from_raw"] = trimmed_check(rep, Path(cfg["paths"]["raw_data_dir"]))
    res["hardware_arithmetic"] = {"accel_full_scale_g": 32768 / cfg["hardware"]["conversion"]["accel_lsb_per_g"],
                                  "gyro_full_scale_dps": 32768 / cfg["hardware"]["conversion"]["gyro_lsb_per_dps"],
                                  "matches_stated_ranges": bool(abs(32768 / cfg["hardware"]["conversion"]["accel_lsb_per_g"] - cfg["hardware"]["accel_range_g"]) < 1e-9
                                                                and abs(32768 / cfg["hardware"]["conversion"]["gyro_lsb_per_dps"] - cfg["hardware"]["gyro_range_dps"]) < 0.2)}
    res["text_checks"] = hardware_text_checks(root)
    cc = consistency_checks(cfg, root)
    cc.to_csv(rep / "audit" / "release_consistency_checks.csv", index=False)
    res["consistency"] = {"n_checks": int(len(cc)), "n_found": int(cc["found"].sum()), "missing": cc.loc[~cc["found"], "claim"].tolist()}
    with open(rep / "audit" / "release_checks.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: v for k, v in res.items() if k not in ("text_checks",)}, indent=1, default=str)[:6000])
    c = res["classifier"]
    hard = [c["participant_accuracy_max_abs_diff_vs_saved"] < 1e-9, c["null_max_abs_diff_vs_independent_recompute"] < 1e-9,
            c["identity_pattern_equals_observed"], c["n_permutations_saved"] == c["n_patterns_expected"], c["variant_p_max_abs_diff"] < 1e-12,
            res["invariance"]["max_rel_change_of_set_when_any_axis_negated"] < 1e-9, res["invariance"]["max_rel_change_of_magnitude_features_under_rotation"] < 1e-9,
            res["trimmed_gyro_mag_rms_from_raw"]["max_abs_diff_vs_saved"] < 1e-9, res["consistency"]["n_found"] == res["consistency"]["n_checks"]]
    return 0 if all(hard) else 1


if __name__ == "__main__":
    sys.exit(main())
