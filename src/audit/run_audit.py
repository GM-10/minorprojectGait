"""Run the scientific audit and write machine-readable results to reports/audit/.

    python -m src.audit.run_audit

Reads the pipeline's outputs (processed_data/, reports/tables/) and the raw CSVs (read-only); never modifies them.
The narrative is in reports/scientific_audit.md.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit import clipping_audit as ca
from src.audit import hardware_audit as ha
from src.audit import model_audit as ma
from src.audit import stats_audit as sa
from src.config import load_config
from src.models import experiments as ml
from src.statistics import exploratory as expl


def main() -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    seed = cfg["random_seed"]
    out = Path(cfg["paths"]["reports_dir"]) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    proc = Path(cfg["paths"]["processed_dir"])
    rec = pd.read_csv(proc / "features_recordings.csv")
    win = pd.read_csv(proc / "features_windows.csv")
    stored = pd.read_csv(Path(cfg["paths"]["reports_dir"]) / "tables" / "paired_stats_all_participants.csv")
    summary: dict = {}

    # ---- 1 + 2: clipping and AccelZ
    raw = ca.load_raw(cfg["paths"]["raw_data_dir"])
    levels = ca.rail_levels(raw)
    inv, by_ch_leg = ca.clip_inventory(raw, levels)
    inv.to_csv(out / "clip_inventory_per_recording_channel.csv", index=False)
    by_ch_leg.to_csv(out / "clip_summary_by_channel_leg.csv", index=False)
    ca.int16_check(levels).to_csv(out / "rail_int16_arithmetic.csv", index=False)
    tve = pd.concat([ca.tolerance_vs_exact(raw, levels, tol) for tol in (0.001, cfg["ingestion"]["saturation"]["rel_tol"])])
    tve.to_csv(out / "tolerance_vs_exact_rails.csv", index=False)  # 0.001 = original pipeline setting
    for k, v in ca.accz_asymmetry(raw, levels, inv).items():
        v.to_csv(out / f"{k}.csv", index=False)
    summary["rail_levels"] = levels
    summary["pct_samples_at_rail_by_channel"] = (inv.groupby("channel")["n_at_rail"].sum() /
                                                  inv.groupby("channel")["n_samples"].sum() * 100).round(3).to_dict()
    summary["extra_samples_flagged_by_tolerance_rule"] = {str(t): int(g["n_extra_by_tolerance"].sum()) for t, g in tve.groupby("rel_tol")}

    feats = sa.feature_cols(rec)
    sens = ca.feature_clip_sensitivity(rec, feats)
    sens.to_csv(out / "feature_clip_sensitivity.csv", index=False)

    # ---- 3: statistics
    pids, L, R = sa.paired_matrices(rec, feats)
    mine = sa.recompute(L - R, feats)
    mine.to_csv(out / "paired_recomputed_independent.csv", index=False)
    cmp = sa.compare_to_pipeline(mine, stored)
    cmp.to_csv(out / "paired_recompute_vs_pipeline.csv", index=False)
    flip, glob = sa.signflip_maxT(L - R, feats)
    flip.to_csv(out / "paired_signflip_maxT.csv", index=False)
    clean = sa.clean_pair_reanalysis(rec, sens)
    clean.to_csv(out / "paired_clean_pairs_only.csv", index=False)
    pseudo = sa.window_level_pseudoreplication(win, sa.feature_cols(win))
    top = mine.sort_values("p_primary").head(15)[["feature", "p_t", "p_wilcoxon", "primary_test", "p_primary",
                                                  "q_bh_p_primary", "q_holm_p_primary", "dz", "dz_nct_low", "dz_nct_high"]]
    top = top.merge(sens[["feature", "channels", "exposure_fraction_recordings", "n_clean_pairs"]], on="feature") \
            .merge(flip[["feature", "p_maxT_fwer"]], on="feature", how="left") \
            .merge(clean[["feature", "p_t", "dz"]].rename(columns={"p_t": "p_t_clean_pairs", "dz": "dz_clean_pairs"}),
                   on="feature", how="left")
    top.to_csv(out / "paired_top15_with_robustness.csv", index=False)
    summary["stats"] = {"n_feature_columns": len(feats), "n_tested": int(mine["p_primary"].notna().sum()),
                        "untested": [f for f, ok in zip(feats, mine["p_primary"].notna()) if not ok],
                        "n_pairs": len(pids), "participants_unique": len(set(pids)) == len(pids),
                        "primary_test_counts": mine["primary_test"].value_counts().to_dict(),
                        "n_p_lt_05_primary": int((mine["p_primary"] < .05).sum()),
                        "n_p_t_lt_05": int((mine["p_t"] < .05).sum()), "n_p_w_lt_05": int((mine["p_wilcoxon"] < .05).sum()),
                        "min_q_bh_primary": float(mine["q_bh_p_primary"].min()),
                        "min_q_holm_primary": float(mine["q_holm_p_primary"].min()),
                        "max_abs_discrepancy_vs_pipeline": float(cmp.loc[cmp["column"] != "primary_test (label)", "max_abs_diff"].max()),
                        "signflip": glob, "window_level_wrong_unit": pseudo,
                        "n_features_exposure_gt_half": int((sens["exposure_fraction_recordings"] > .5).sum()),
                        "n_features_clean_reanalysed": int(clean["p_t"].notna().sum()),
                        "clean_p_lt_05": int((clean["p_t"] < .05).sum()), "clean_bh_sig": int((clean["q_bh_p_t"] < .05).sum()),
                        "dz_ci_width_nct_mean": float((mine["dz_nct_high"] - mine["dz_nct_low"]).mean()),
                        "dz_ci_width_boot_mean": float((stored["dz_ci_high"] - stored["dz_ci_low"]).mean())}

    # ---- 6: hardware-informed re-audit (collaborator-reported configuration; raw counts/firmware unavailable)
    quant = ha.quantisation_check(raw, cfg)
    quant.to_csv(out / "hardware_quantisation_check.csv", index=False)
    rail_exp = ha.rail_expectation(raw, cfg)
    rail_exp.to_csv(out / "hardware_rail_expectation.csv", index=False)
    pl = ha.plateau_table(raw, cfg)
    pl.to_csv(out / "hardware_plateaus.csv", index=False)
    ha.plateau_summary(pl).to_csv(out / "hardware_plateau_summary.csv", index=False)
    pl.groupby(["participant_id", "leg", "channel"]).agg(n_plateaus=("length", "size"), n_samples=("length", "sum"),
                                                          max_length=("length", "max")).reset_index().to_csv(
        out / "hardware_plateaus_by_recording_channel.csv", index=False)
    wrap = ha.wraparound_check(raw, cfg)
    wrap.to_csv(out / "hardware_wraparound_check.csv", index=False)
    static = ha.static_scale_offset(raw)
    static.to_csv(out / "hardware_static_window_scale_offset.csv", index=False)
    od = ha.orientation_diagnostics(raw, static)
    for k, v in od.items():
        v.to_csv(out / f"hardware_orientation_{k}.csv", index=False)
    onset = ha.movement_onset(raw, cfg["protocol"]["movement_onset_rms_threshold_dps"])
    onset.to_csv(out / "hardware_movement_onset.csv", index=False)
    feats_si = ha.sign_invariant_features(feats)
    res_si = ha.paired_subset(rec, feats_si, sa)
    res_si.to_csv(out / "paired_sign_invariant_features.csv", index=False)
    from src.data_validation.validate import run_validation
    from src.pipeline import build_features
    prs, _, _, _ = build_features(cfg, run_validation(cfg))
    trim_s = 2.0
    rec_trim = ha.trimmed_recording_features(prs, cfg, trim_s)
    rec_trim.to_csv(out / "trimmed_first_2s_recording_features.csv", index=False)
    res_trim = ha.paired_subset(rec_trim, feats, sa)
    res_trim.to_csv(out / "paired_trimmed_first_2s.csv", index=False)
    res_trim_si = ha.paired_subset(rec_trim, feats_si, sa)
    res_trim_si.to_csv(out / "paired_trimmed_sign_invariant.csv", index=False)
    feats_mag = [f for f in feats if f.split("__")[0] in ("acc_mag", "acc_dyn_mag", "gyro_mag") or f == "corr__acc_mag__gyro_mag"]
    res_mag = ha.paired_subset(rec, feats_mag, sa)
    res_mag.to_csv(out / "paired_magnitude_only_features.csv", index=False)
    full_res = sa.recompute(L - R, feats)
    summary["hardware"] = {
        "gyro_on_grid_fraction": quant.loc[quant["channel"].str.startswith("gyro"), "fraction_on_attainable_grid"].tolist(),
        "gyro_null_fraction": float(quant.loc[quant["channel"] == "gyro_x", "null_fraction_random_decimals"].iloc[0]),
        "accel_null_fraction": float(quant.loc[quant["channel"] == "acc_x", "null_fraction_random_decimals"].iloc[0]),
        "k_outside_int16": int(quant["k_outside_int16"].sum()),
        "rail_expectation_all_match": bool(rail_exp["match"].all()),
        "n_plateaus": int(len(pl)),
        "static_acc_norm_mean": float(static["acc_norm_g"].mean()), "static_acc_norm_sd": float(static["acc_norm_g"].std()),
        "static_acc_norm_range": [float(static["acc_norm_g"].min()), float(static["acc_norm_g"].max())],
        "gravity_dominant_axis_counts": od["gravity"]["dominant_gravity_axis"].value_counts().to_dict(),
        "gravity_unit_y_mean": float(od["gravity"]["unit_y"].mean()),
        "angle_left_right_static_gravity_deg": {"median": float(od["angle_left_right"].iloc[:, 1].median()),
                                                "max": float(od["angle_left_right"].iloc[:, 1].max())},
        "onset": ha.paired_by_leg_order_free_summary(onset),
        "n_recordings_onset_le_100ms": int((onset["onset_ms"] <= 100).sum()),
        "paired_full": ha.summarise_paired(full_res),
        "paired_sign_invariant": ha.summarise_paired(res_si),
        "n_sign_invariant_features": len(feats_si),
        "paired_trimmed_2s": ha.summarise_paired(res_trim),
        "n_magnitude_only_features": len(feats_mag), "paired_magnitude_only": ha.summarise_paired(res_mag),
        "paired_trimmed_sign_invariant": ha.summarise_paired(res_trim_si),
    }

    # ---- 4 + 5: models
    uf = expl.usable_features(win, sa.feature_cols(win))
    mf = ml.select_model_features(win, uf, cfg["models"]["feature_stat_whitelist"])
    clf = ma.classifier_audit(win, mf, sens, seed)
    clf["balance"].to_csv(out / "classifier_class_balance.csv", index=False)
    clf["variants"].to_csv(out / "classifier_variants_and_baselines.csv", index=False)
    clf["per_participant"].to_csv(out / "classifier_per_participant_accuracy.csv")
    clf["ci"].to_csv(out / "classifier_ci_methods.csv", index=False)
    clf["coef"].to_csv(out / "classifier_top_coefficients_descriptive.csv", index=False)
    summary["classifier"] = {"recording_level": clf["recording_level"], "n_model_features": len(mf)}
    anc = ma.anomaly_cluster_audit(win, mf, uf, sens, seed, cfg["models"]["anomaly_contamination"])
    anc["robustness"].to_csv(out / "anomaly_cluster_robustness.csv", index=False)
    anc["cluster_profile"].to_csv(out / "kmeans2_cluster_profile.csv", index=False)
    anc["participant_by_cluster"].to_csv(out / "kmeans2_participant_by_cluster.csv", index=False)
    summary["anomaly_cluster"] = anc["summary"]
    # classifier restricted to features that do not depend on axis sign, and without the first 2 s of each recording
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.models import permutation as perm
    mk = lambda: perm.default_model(seed)  # noqa: E731
    si_set = set(feats_si)
    mag_set = [f for f in mf if f.split("__")[0] in ("acc_mag", "acc_dyn_mag", "gyro_mag") or f == "corr__acc_mag__gyro_mag"]
    variants = {"all model features (reported)": (mf, win), "sign-invariant model features only": ([f for f in mf if f in si_set], win),
                "magnitude-only model features (rotation-invariant)": (mag_set, win),
                "model features, windows starting >= 2 s": (mf, win[win["start_ms"] >= 2000].reset_index(drop=True))}
    crow, hw_nulls = [], {}
    for name, (fs, w) in variants.items():
        y = (w["leg"] == "right").astype(int).to_numpy()
        g = w["participant_id"].to_numpy()
        X = w[fs].to_numpy(float)
        accs = ma.logo_eval(X, y, g, mk)[0]
        null = clf["null_main"] if name.startswith("all model") else ma.permutation_null(X, y, g, mk, "exact", seed)
        hw_nulls[name] = null
        crow.append({"variant": name, "n_features": len(fs), "n_windows": len(w), "mean_participant_accuracy": accs.mean(),
                     "sd": accs.std(ddof=1), "n_permutations": len(null), "exact_enumeration": True, "null_mean": float(null.mean()),
                     "null_sd": float(null.std(ddof=1)), "null_q95": float(np.quantile(null, .95)),
                     "permutation_p": perm.permutation_pvalue(accs.mean(), null, exact=True)})
    pd.DataFrame(hw_nulls).to_csv(out / "classifier_permutation_nulls_hardware_variants.csv", index=False)
    pd.DataFrame(clf["nulls"]).to_csv(out / "classifier_permutation_nulls_variants.csv", index=False)
    pd.DataFrame({"random_guess_mean_participant_accuracy": clf["random_guess"]}).to_csv(out / "classifier_random_guess_distribution.csv", index=False)
    summary["classifier"]["random_guess"] = {"mean": float(clf["random_guess"].mean()), "sd": float(clf["random_guess"].std(ddof=1)),
                                            "q95": float(np.quantile(clf["random_guess"], .95)), "n_sim": int(len(clf["random_guess"]))}
    pd.DataFrame(crow).to_csv(out / "classifier_hardware_variants.csv", index=False)
    summary["hardware"]["classifier_variants"] = crow
    with open(out / "audit_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"audit tables written to {out}")


if __name__ == "__main__":
    main()
