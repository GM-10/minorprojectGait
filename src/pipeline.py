"""End-to-end pipeline. Run from the project root:

    python -m src.pipeline                  # all stages
    python -m src.pipeline --stages validate eda features stats explore models report
    python -m src.pipeline --config configs/default.yaml

Source CSVs are only read. All outputs go to data/processed and reports/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data_validation.validate import run_validation
from src.eda.eda import run_eda
from src.features.extract import extract_recording_features, feature_columns, feature_config_id
from src.models import experiments as ml
from src.signal_processing.processing import make_windows, prepare_recording
from src.statistics import exploratory as expl
from src.statistics.paired import run_paired_analysis
from src.visualization import plots

log = logging.getLogger("gait")
STAGES = ["validate", "eda", "features", "stats", "explore", "models", "audit", "report"]


def snapshot_hashes(data_dir: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(data_dir).rglob("*.csv"))}


def build_features(cfg: dict, val: dict) -> tuple[list, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Process every recording, window it and extract features. Returns (processed, windows, recordings, logs)."""
    prs, wins, recs, logs = [], [], [], []
    pc = cfg["processing"]
    for r in val["recs"]:
        rid = r.recording_id
        pr = prepare_recording(r, val["frames"][rid], val["chmaps"][rid], val["time_cols"][rid], val["levels"], cfg)
        windows, winfo = make_windows(len(pr.t_ms), pr.fs_hz, pc["window_s"], pc["overlap"],
                                      pc["incomplete_final_window"], pc["fallback_window_samples"])
        pr.log["windowing"] = winfo
        w, rr = extract_recording_features(pr, windows, cfg)
        prs.append(pr)
        wins.append(w)
        recs.append(rr)
        logs.append(pr.log)
    return prs, pd.concat(wins, ignore_index=True), pd.concat(recs, ignore_index=True), logs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    seed = cfg["random_seed"]
    np.random.seed(seed)
    rep, proc_dir = Path(cfg["paths"]["reports_dir"]), Path(cfg["paths"]["processed_dir"])
    tab = rep / "tables"
    for d in (rep, proc_dir, tab):
        d.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(cfg["paths"]["raw_data_dir"]).resolve()
    for key in ("processed_dir", "reports_dir"):
        out_p = Path(cfg["paths"][key]).resolve()
        if out_p == raw_dir or raw_dir in out_p.parents:
            raise ValueError(f"paths.{key} ({out_p}) must not be inside the raw data directory ({raw_dir})")
    before = snapshot_hashes(cfg["paths"]["raw_data_dir"])
    log.info("validate: %d source CSVs hashed", len(before))
    val = run_validation(cfg)
    s = val["summary"]
    log.info("validated %d files / %d participants / %d candidate pairs", s["n_files"], s["n_participants"],
             s["n_valid_pairs"])

    prs, win_df, rec_df, plogs = build_features(cfg, val)
    with open(rep / "processing_log.json", "w", encoding="utf-8") as fh:
        json.dump(plogs, fh, indent=2, default=str)
    win_df.to_csv(proc_dir / "features_windows.csv", index=False)
    rec_df.to_csv(proc_dir / "features_recordings.csv", index=False)
    with open(proc_dir / "feature_config.json", "w", encoding="utf-8") as fh:
        json.dump({"feature_config_id": feature_config_id(cfg), "processing": cfg["processing"],
                   "features": cfg["features"], "random_seed": seed}, fh, indent=2)
    feats = feature_columns(rec_df)
    log.info("features: %d windows, %d recordings, %d feature columns", len(win_df), len(rec_df), len(feats))
    fig_dir = rep / "figures"
    dpi = cfg["eda"]["dpi"]
    figs: dict[str, list[str]] = {}
    out: dict = {"n_windows": len(win_df), "n_recordings": len(rec_df), "n_features": len(feats)}
    noted = {r.participant_id for r in val["recs"] if r.note}

    if "eda" in args.stages:
        figs.update(run_eda(cfg, val, prs, rec_df))
        log.info("eda figures written")

    if "stats" in args.stages:
        hl = [f for f in cfg["statistics"]["headline_features"] if f in feats]
        main_res = run_paired_analysis(rec_df, feats, cfg, seed=seed)
        sens_res = run_paired_analysis(rec_df, feats, cfg, exclude=sorted(noted), seed=seed)
        main_res["results"].to_csv(tab / "paired_stats_all_participants.csv", index=False)
        main_res["differences"].to_csv(tab / "paired_differences_per_participant.csv", index=False)
        main_res["pairing"].to_csv(tab / "paired_pairing_report.csv", index=False)
        sens_res["results"].to_csv(tab / "paired_stats_excluding_noted_participants.csv", index=False)
        # sensitivity: window-median per recording instead of whole-recording features
        wmed = win_df.groupby(["participant_id", "leg"])[feats].median().reset_index()
        wmed_res = run_paired_analysis(wmed, feats, cfg, seed=seed)
        wmed_res["results"].to_csv(tab / "paired_stats_window_median_sensitivity.csv", index=False)
        left = rec_df[rec_df.leg == "left"].set_index("participant_id")[feats]
        right = rec_df[rec_df.leg == "right"].set_index("participant_id")[feats]
        pids = main_res["differences"]["participant_id"]
        figs["paired"] = [
            plots.paired_slopes(left.loc[pids], right.loc[pids], main_res["results"], hl,
                                fig_dir / "paired" / "paired_slopes_headline.png", dpi),
            plots.difference_plot(main_res["differences"].set_index("participant_id"), main_res["results"], hl,
                                  fig_dir / "paired" / "paired_differences_headline.png", dpi),
            plots.effect_size_bars(main_res["results"], fig_dir / "paired" / "effect_sizes_top.png", dpi)]
        r = main_res["results"]
        out["paired"] = {
            "n_pairs_max": int(r["n_pairs"].max()), "n_features_tested": int(r["p_primary"].notna().sum()),
            "n_sig_bh": int(r["significant_bh_primary"].sum()), "n_sig_holm": int((r["q_holm_primary"] < .05).sum()),
            "n_raw_p_lt_05": int((r["p_primary"] < .05).sum()),
            "n_normal_diffs": int(r["normality_ok"].sum()),
            "sens_excl_noted": {"excluded": sorted(noted), "n_pairs": int(sens_res["results"]["n_pairs"].max()),
                                "n_sig_bh": int(sens_res["results"]["significant_bh_primary"].sum()),
                                "n_raw_p_lt_05": int((sens_res["results"]["p_primary"] < .05).sum())},
            "sens_window_median": {"n_sig_bh": int(wmed_res["results"]["significant_bh_primary"].sum()),
                                   "n_raw_p_lt_05": int((wmed_res["results"]["p_primary"] < .05).sum())}}
        psum = expl.participant_summary(rec_df, win_df, hl)
        psum.to_csv(tab / "participant_level_summary.csv", index=False)
        log.info("paired stats written (n_pairs=%d, features=%d, BH-significant=%d)", out["paired"]["n_pairs_max"],
                 out["paired"]["n_features_tested"], out["paired"]["n_sig_bh"])

    if "explore" in args.stages:
        uf = expl.usable_features(win_df, feats)
        hm_feats = [f for f in uf if f.split("__")[-1] in ("mean", "std", "rms", "ptp", "dominant_freq_hz")
                    and f.split("__")[0] in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "acc_mag",
                                             "acc_dyn_mag", "gyro_mag")]
        corr = expl.correlation_matrix(win_df, hm_feats)
        corr.to_csv(tab / "feature_correlation_spearman.csv")
        pca = expl.pca_analysis(win_df[["participant_id", "leg", "recording_id", "window_index"] + uf], uf,
                                seed=seed)
        pca["scores"].to_csv(tab / "pca_scores_windows.csv", index=False)
        pca["loadings"].to_csv(tab / "pca_loadings.csv")
        var = expl.participant_variability(win_df, uf)
        var.to_csv(tab / "participant_variability.csv", index=False)
        fe = fig_dir / "exploration"
        figs["explore"] = [
            plots.feature_boxplots(win_df, [f for f in cfg["statistics"]["headline_features"] if f in feats],
                                   fe / "feature_boxplots_headline.png", dpi),
            plots.corr_heatmap(corr, fe / "feature_correlation_heatmap.png", dpi,
                               "Spearman correlation of window-level features (windows are autocorrelated)"),
            plots.pca_plots(pca, fe / "pca_scatter.png", dpi)]
        out["pca"] = {"evr": [float(x) for x in pca["explained_variance_ratio"][:5]],
                      "n_features": pca["n_features_used"], "n_dropped": pca["n_features_dropped"]}
        log.info("exploration written")

    if "models" in args.stages:
        mc = cfg["models"]
        n_part = int(win_df["participant_id"].nunique())
        uf = expl.usable_features(win_df, feats)
        mf = ml.select_model_features(win_df, uf, mc["feature_stat_whitelist"])
        out["models"] = {"n_participants": n_part, "n_windows": len(win_df), "n_model_features": len(mf)}
        mdir = rep / "models"
        mdir.mkdir(exist_ok=True)
        if n_part < mc["min_participants_for_models"]:
            out["models"]["skipped"] = (f"only {n_part} participants (< {mc['min_participants_for_models']}); "
                                        "descriptive statistics only")
            log.warning(out["models"]["skipped"])
        else:
            if mc["run_anomaly"]:
                an = ml.anomaly_detection(win_df, mf, cfg, seed, noted)
                an.to_csv(mdir / "anomaly_scores_windows.csv", index=False)
                an.head(mc["n_top_anomalies"]).to_csv(mdir / "anomaly_top_windows_for_inspection.csv", index=False)
                figs["anomaly"] = [plots.anomaly_plot(an, fig_dir / "models" / "anomaly_scores.png", dpi)]
                out["models"]["anomaly"] = {
                    "spearman_iforest_vs_lof": an.attrs["spearman_if_vs_lof"],
                    "n_flagged_iforest": int(an["iforest_flag"].sum()), "n_flagged_both": int(an["flagged_by_both"].sum()),
                    "flagged_by_participant": an[an["iforest_flag"]].groupby("participant_id").size().to_dict()}
            if mc["run_clustering"]:
                cl = ml.clustering_experiment(win_df, mf, cfg, seed)
                cl["summary"].to_csv(mdir / "clustering_summary.csv", index=False)
                cl["participant_by_cluster"].to_csv(mdir / "clustering_participant_by_cluster.csv")
                cl["assignments"].to_csv(mdir / "clustering_assignments.csv", index=False)
                out["models"]["clustering"] = {"summary": cl["summary"].to_dict("records"),
                                               "best_k_by_silhouette": cl["best_k_by_silhouette"]}
            if mc["run_lr_classification"]:
                lr = ml.lr_classification(win_df, mf, cfg, seed)
                lr["participant_accuracy"].to_csv(mdir / "lr_classification_participant_accuracy.csv", index=False)
                lr["recording_decisions"].to_csv(mdir / "lr_classification_recording_decisions.csv", index=False)
                pd.DataFrame({"mean_participant_accuracy": lr["permutation_null"]}).to_csv(mdir / "lr_permutation_null.csv", index=False)
                out["models"]["lr_classification"] = lr["metrics"]
            out["models"]["injury_classifier"] = ("not attempted: collaborator-reported histories exist for only 2 of 12 participants "
                                                  "(Sub04 ACL recovery, Sub06 ankle twist; unverified, no severity/side/timing/diagnosis); no outcome label")
            out["models"]["tinyml"] = "not attempted (no validated prediction target)"
        log.info("models: %s", json.dumps(out["models"], default=str)[:300])

    if "audit" in args.stages:
        from src.audit.run_audit import main as run_audit
        run_audit()
        log.info("audit tables written to %s", rep / "audit")

    after = snapshot_hashes(cfg["paths"]["raw_data_dir"])
    out["source_csvs_unchanged"] = before == after
    if not out["source_csvs_unchanged"]:
        raise RuntimeError("Source CSVs changed during the run - this must never happen")
    out["figures"] = figs
    with open(rep / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    if "report" in args.stages:
        from src.report import write_report
        write_report(cfg, val, win_df, rec_df, out)
        log.info("final report written to %s", rep / "final_report.md")
    log.info("done; source CSVs unchanged = %s", out["source_csvs_unchanged"])


if __name__ == "__main__":
    main()
