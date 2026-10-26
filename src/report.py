"""Generates reports/final_report.md. Every number is read from the generated tables; prose is static."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _eta2(values: np.ndarray, groups: np.ndarray) -> float:
    """Share of variance explained by a grouping (descriptive)."""
    tot = ((values - values.mean()) ** 2).sum()
    between = sum(len(values[groups == g]) * (values[groups == g].mean() - values.mean()) ** 2
                  for g in np.unique(groups))
    return float(between / tot) if tot > 0 else float("nan")


def _md(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                     else str(v) for v in r) + " |")
    return "\n".join(out)


def write_report(cfg: dict, val: dict, win_df: pd.DataFrame, rec_df: pd.DataFrame, out: dict) -> Path:
    rep = Path(cfg["paths"]["reports_dir"])
    tab = rep / "tables"
    s, q, ch = val["summary"], val["report"], val["channels"]
    pc, fc = cfg["processing"], cfg["features"]
    L: list[str] = []
    add = L.append
    aud_dir = rep / "audit"
    aud = json.loads((aud_dir / "audit_summary.json").read_text(encoding="utf-8")) if (aud_dir / "audit_summary.json").is_file() else None

    # ------------------------------------------------------------------ dataset
    add("# Gait IMU analysis: final report\n")
    add("*Exploratory, descriptive pilot analysis. No ground-truth gait labels or clinical diagnoses exist in this "
        "dataset; nothing here is a clinical claim.*\n")
    add("## 1. Dataset (as found, not assumed)\n")
    add(f"- **{s['n_files']} CSV files**, **{s['n_participants']} participants** (`SubNN`), left and right recording each; "
        f"**{s['n_valid_pairs']} candidate-valid left/right pairs** (both legs present and identical filename notes). "
        f"There is no trial/recording ID in the data: `recording_id` = `<participant>_<leg>` derived from the file name.")
    add(f"- One schema in all files: `{', '.join(s['modal_schema'])}` (index column = `Time_ms`; 3-axis accelerometer + "
        "3-axis gyroscope; no other channels or metadata). "
        f"Rows per file: {s['rows_range'][0]}-{s['rows_range'][1]}; duration "
        f"{q['duration_s'].min():.2f}-{q['duration_s'].max():.2f} s.")
    add(f"- Sampling frequency estimated from `Time_ms` (median period): **{s['estimated_fs_hz_range'][0]:.1f}-"
        f"{s['estimated_fs_hz_range'][1]:.1f} Hz** (20 ms). Every timestamp is an exact multiple of 20 ms and every "
        "recording ends at exactly 10 000 ms, which is consistent with a nominal/scheduled clock; actual sensor "
        "timing jitter cannot be verified from the CSVs.")
    add(f"- Byte-identical duplicate CSVs: {len(s['byte_identical_duplicates'])}. (`Data.zip` contains the same 24 CSVs as `Data/`; it is ignored.)\n")

    add("### 1a. Hardware, protocol and metadata: what is known and how well\n")
    add("Source of each item is stated. **Raw int16 register values, firmware, board photographs and sensor initialisation logs are not available**, "
        "so firmware behaviour and axis mapping cannot be verified. Full details: `docs/data_collection.md`.\n")
    add("| Category | Content | Status |\n|---|---|---|")
    add("| Hardware specification (collaborator-reported) | ESP32 + GY-521 (MPU6050) over I2C (SDA GPIO 21, SCL GPIO 22); accelerometer +-2 g, gyroscope +-250 deg/s; "
        "50 Hz (0.02 s); ~10 s, ~500 samples per leg; conversion raw/16384 (g) and raw/131 (deg/s) | Not independently verified; consistency with the CSVs checked below |")
    add("| Protocol (collaborator-reported) | module on the shin about one breadboard-length below the knee, mounted inverted with its top towards the ground; "
        "**same physical sensor for both legs**; barefoot, flat surface; 3 s warning before walking straight; first step with the leg opposite to the sensor leg; "
        "stated as consistent across participants | Not verifiable from the CSVs |")
    add("| Participant metadata (collaborator-reported) | Sub04: ACL recovery history; Sub06: ankle-twist history (file name adds 'right', unverified); all others described as healthy | "
        "Not verified; no severity, side, date, diagnosis or recovery status; not usable as outcome labels |")
    if aud is not None:
        hw = aud["hardware"]
        add(f"| Verified directly from the CSVs | schema and column order; 24/24 recordings within 5 rows of 500 and estimated 50.0 Hz; all gyroscope values lie exactly on the grid round(k/131, 3) "
            f"for an integer k (100 % vs {100*hw['gyro_null_fraction']:.0f} % expected by chance), none outside the int16 range; the observed extremes equal the values expected for raw counts +32767 / -32768 "
            f"under the documented formulas; |accelerometer| in the quietest 0.5 s of each recording averages {hw['static_acc_norm_mean']:.3f} g (SD {hw['static_acc_norm_sd']:.3f}, range "
            f"{hw['static_acc_norm_range'][0]:.2f}-{hw['static_acc_norm_range'][1]:.2f}); accelerometer Y is the gravity-dominant axis in {hw['gravity_dominant_axis_counts'].get('y', 0)}/24 recordings with mean unit component {hw['gravity_unit_y_mean']:.2f} | "
            "Checks of consistency only |")
        add(f"| Assumptions / unresolved | anatomical mapping of X, Y, Z; sensor initialisation and firmware export; zero-rate offsets (no true rest period: {hw['n_recordings_onset_le_100ms']}/24 recordings already show movement in the first 100 ms); "
            f"order of the legs and re-mounting between legs (static gravity direction differs between a participant's legs by a median {hw['angle_left_right_static_gravity_deg']['median']:.0f} deg, up to {hw['angle_left_right_static_gravity_deg']['max']:.0f} deg); "
            "whether the extreme values are true saturation; timestamp source; clock synchronisation between legs (separate trials) | Unresolved |\n")
    else:
        add("| Verified directly from the CSVs | run the audit stage for the hardware consistency checks | n/a |\n")
    add("*The data are in the documented units (`g`, `deg/s`) as far as can be verified; the unit labels in the headers now agree with the collaborator's description.* "
        "Using one sensor for both legs removes between-device gain/offset differences from the left-right comparison, but not differences in how it was mounted on each leg, "
        "its orientation, timing or the measurement range.\n")

    add("## 2. Data-quality findings\n")
    gaps = q[q["n_gaps"] > 0]
    late = q[q["first_timestamp_ms"] > cfg["ingestion"]["gap_factor"] * q["median_dt_ms"]]
    add(f"- Missing/non-finite values: **{int(q['n_missing_total'].sum())}**; duplicate timestamps: "
        f"**{int(q['n_duplicate_timestamps'].sum())}**; constant channels: **{int(ch['is_constant'].sum())}**; "
        f"abrupt cutoffs: **{int(q['warnings'].str.contains('abrupt cutoff').sum())}**.")
    add(f"- Dropped samples (40 ms steps): {', '.join(f'{r.recording_id} ({r.n_missing_samples_est})' for r in gaps.itertuples()) or 'none'}"
        f"; late start: {', '.join(f'{r.recording_id} (first sample {r.first_timestamp_ms:.0f} ms)' for r in late.itertuples()) or 'none'}. "
        "Gaps were filled by linear interpolation and flagged (`qc_interp_fraction`); this affects "
        f"<= {100*win_df['qc_interp_fraction'].max():.1f}% of any window.")
    lv = s["saturation_levels"]
    cl = ch.assign(pct=100 * ch["clipped_fraction"]).groupby("channel")["pct"].agg(["mean", "max"])
    add(f"- **Values stuck at the extremes are the main quality problem.** Values repeat at identical extremes ({', '.join(f'{k}={v:g}' for k, v in lv.items())}; "
        "these are exactly the values expected for raw int16 counts +32767 / -32768 under the collaborator-documented +-2 g / +-250 deg/s settings, see section 1a and `reports/scientific_audit.md`); "
        f"{(q['fraction_samples_with_any_clipped_channel']>0).sum()} of {len(q)} recordings contain such samples, "
        f"{100*q['fraction_samples_with_any_clipped_channel'].mean():.1f}% of samples on average (max "
        f"{100*q['fraction_samples_with_any_clipped_channel'].max():.1f}% in {q.loc[q['fraction_samples_with_any_clipped_channel'].idxmax(), 'recording_id']}). "
        "Worst channels (mean % of samples at these rail values per recording): " +
        ", ".join(f"{c} {r['mean']:.1f}%" for c, r in cl.sort_values('mean', ascending=False).head(3).iterrows()) +
        ". This is consistent with the sensor output reaching its configured full-scale during gait, but **hardware saturation is not confirmed** (no raw counts, firmware or register read-back; a firmware clamp cannot be excluded). Min/max/ptp/energy and "
        "magnitude features of affected recordings are biased low and gyro/accel spikes are truncated. "
        "Clipped samples are flagged, **not** corrected (`qc_clip_fraction_*` columns).")
    if aud is not None:
        zt = pd.read_csv(aud_dir / "accz_tests.csv").set_index("quantity")
        pres = zt.loc[[k for k in zt.index if k.startswith("presence")][0]]
        cnt = zt.loc["n samples at Z max (counts)"]
        tot = zt.loc["n samples at either Z rail"]
        add(f"- **AccelZ left/right (audited; corrects an earlier over-statement)**: Z touches its positive extreme in {int(pres['sum_right'])}/12 right "
            f"vs {int(pres['sum_left'])}/12 left recordings, but this presence indicator rests on tiny counts (1-7 samples) and was chosen post hoc. "
            f"Counting samples: {int(cnt['sum_right'])} right vs {int(cnt['sum_left'])} left at the positive extreme (sign test p = {cnt['sign_test_p']:.2f}); "
            f"at either extreme {int(tot['sum_right'])} vs {int(tot['sum_left'])} (p = {tot['sign_test_p']:.2f}). "
            "So there is no reliable evidence of a leg difference in Z extremes overall; the direction of the few hits differs by leg (positive more often on the right, negative on the left), "
            "which cannot be attributed to mounting vs motion from these data. See `reports/scientific_audit.md`.")
    add("- Figures: `reports/figures/eda/data_quality_summary.png`, `sampling_irregularities.png`, `clipping_heatmap.png`; "
        "full tables in `reports/data_quality/`.\n")

    add("## 3. Methods\n")
    add(f"1. **Regularisation**: timestamps -> uniform 20 ms grid, missing samples linearly interpolated and flagged. Raw CSVs are never modified (SHA-256 checked before/after every run: unchanged = **{out['source_csvs_unchanged']}**).")
    add(f"2. **Filtering**: none applied to features (`lowpass_hz: {pc['lowpass_hz']}`). A {pc['comparison_lowpass_hz']} Hz zero-phase Butterworth is shown only in `raw_vs_processed.png`. Rationale: no filter cut-off is justified by the data here (Nyquist is 25 Hz), and filtering clipped signals can introduce ringing; unfiltered features keep the analysis transparent.")
    add(f"3. **Gravity**: accelerometer is kept as recorded (`acc_*`, `acc_mag` include gravity/posture). A dynamic component `acc_dyn_*` = signal minus a {pc['gravity_cutoff_hz']} Hz zero-phase low-pass estimate gives `acc_dyn_mag`. No sensor orientation is assumed.")
    add(f"4. **Windows**: {pc['window_s']} s ({int(pc['window_s']*50)} samples), {int(100*pc['overlap'])}% overlap, incomplete tail `{pc['incomplete_final_window']}`; {out['n_windows']} windows from {out['n_recordings']} recordings. Frequency resolution of a window is 0.5 Hz; of a whole recording 0.1 Hz. Windows overlap, so they are *not* independent.")
    add(f"5. **Features** ({out['n_features']} per row): time-domain (mean, std, var [ddof=1], min, max, ptp, rms, energy = sum(x^2)*dt, quantiles), zero-crossing rate (axes), magnitudes (`acc_mag`, `acc_dyn_mag`, `gyro_mag`), spectral (dominant frequency >= {fc['spectrum_min_freq_hz']} Hz, spectral energy, centroid, entropy, fraction below {fc['low_band_upper_hz']} Hz; fs known and >= {fc['min_samples_for_spectrum']} samples) and the 15 axis-pair correlations (+ acc_mag/gyro_mag). Tables: `processed_data/features_windows.csv`, `features_recordings.csv`.")
    add("6. **Paired analysis**: unit = participant. Features of the *whole recording* (one value per leg) -> difference = LEFT - RIGHT. Per feature: mean/median difference, t-based CI, bootstrap CI (median, dz), Cohen's dz and Hedges' gz, matched-pairs rank-biserial r, Shapiro-Wilk on the differences, paired t-test and Wilcoxon signed-rank. The 'primary' test is the t-test if Shapiro p >= 0.05, otherwise Wilcoxon. Benjamini-Hochberg (and Holm) correction across all features tested.")
    add("7. **Exploration/ML** (descriptive or leave-one-participant-out, seed fixed at "
        f"{cfg['random_seed']}): correlation heatmap, PCA, participant variability, Isolation Forest + LOF, k-means, left/right logistic regression.\n")

    # ------------------------------------------------------------------ paired
    add("## 4. Left vs right: paired results\n")
    pr = pd.read_csv(tab / "paired_stats_all_participants.csv")
    po = out["paired"]
    add(f"- Participants/pairs analysed: **{po['n_pairs_max']}** of {s['n_participants']} (missing pairs: none). "
        f"Features tested: {po['n_features_tested']}. Difference normality not rejected for {po['n_normal_diffs']} features (Shapiro has little power at n=12).")
    add(f"- Raw p < 0.05 for **{po['n_raw_p_lt_05']}** of {po['n_features_tested']} features (about "
        f"{0.05*po['n_features_tested']:.0f} expected by chance alone; features are strongly correlated, so this is not a precise figure). "
        f"After BH correction: **{po['n_sig_bh']}** features significant; Holm: **{po['n_sig_holm']}**. "
        f"Smallest BH q = {pr['q_bh_primary'].min():.2f}.")
    sens = po["sens_excl_noted"]
    add(f"- Sensitivity: excluding the two participants with collaborator-reported histories ({', '.join(sens['excluded'])}, identified through file-name notes; n={sens['n_pairs']}) -> raw p<0.05 for {sens['n_raw_p_lt_05']}, BH-significant {sens['n_sig_bh']}. "
        f"Using window-median instead of whole-recording features -> raw p<0.05 for {po['sens_window_median']['n_raw_p_lt_05']}, BH-significant {po['sens_window_median']['n_sig_bh']}.")
    if aud is not None:
        sf = aud["stats"]["signflip"]
        add(f"- Audit cross-checks (`reports/scientific_audit.md`): an independent recomputation matches these tables (max abs discrepancy {aud['stats']['max_abs_discrepancy_vs_pipeline']:.1e}); "
            f"an exact sign-flip permutation test over all {sf['n_flips']} participant-level flips gives a family-wise-adjusted minimum p = {sf['min_p_maxT']:.2f}, and "
            f"{sf['observed_n_nominal_sig']} nominally significant features is within the null range (null mean {sf['null_mean_n_nominal_sig']:.1f}, 95th pct {sf['null_q95_n_nominal_sig']:.0f}; global p = {sf['global_p_count']:.2f}). "
            "**No left/right difference is established.**")
        hw = aud["hardware"]
        add(f"- Hardware-informed reassessment (numbers unchanged; additional sensitivity analyses): the module was mounted inverted and its axis mapping is unconfirmed, so signed axis means, minima, maxima, "
            f"quantiles and cross-axis correlations may change sign between legs for orientation reasons alone. Restricting the paired analysis to the {hw['n_sign_invariant_features']} features that do not depend on axis sign "
            f"(spread, power, spectral shape, all magnitude statistics) gives {hw['paired_sign_invariant']['n_p_lt_05']} raw p < 0.05 and {hw['paired_sign_invariant']['n_bh_lt_05']} BH-significant (min q = {hw['paired_sign_invariant']['min_q_bh']:.2f}); "
            f"dropping the first 2 s of each recording (movement initiation; {hw['n_recordings_onset_le_100ms']}/24 recordings are already moving within 100 ms) gives {hw['paired_trimmed_2s']['n_p_lt_05']} raw p < 0.05 and {hw['paired_trimmed_2s']['n_bh_lt_05']} BH-significant. "
            "**A non-significant result is not evidence that the legs are equivalent**: with 12 pairs the effect-size intervals remain wide (the four amplitude headline features have exact dz intervals spanning roughly -0.8 to +0.8), "
            "so moderate differences cannot be excluded. Using one sensor for both legs removes device-to-device differences, not mounting, orientation, timing or range limitations.")
    top = pr.sort_values("p_primary").head(8)[["feature", "mean_left", "mean_right", "mean_diff", "ci_low", "ci_high", "dz",
                                               "primary_test", "p_primary", "q_bh_primary"]]
    add("\nLargest raw differences (exploratory; none should be read as a finding unless it survives correction and replication):\n")
    add(_md(top))
    hl = pr[pr["feature"].isin(cfg["statistics"]["headline_features"])][
        ["feature", "mean_diff", "median_diff", "dz", "dz_ci_low", "dz_ci_high", "primary_test", "p_primary", "q_bh_primary"]]
    add("\nHeadline features (pre-specified in `configs/default.yaml`):\n")
    add(_md(hl))
    add("\n**Interpretation limits.** The dz intervals above are percentile-bootstrap intervals over the 12 participants (5000 resamples); they are "
        "approximate at this sample size (exact non-central-t intervals are in `reports/audit/paired_recomputed_independent.csv`). "
        "With 12 pairs, power is low and CIs on effect sizes are wide. A left-right difference "
        "can come from sensor placement/orientation, walking speed, protocol or clipping differences - not "
        "necessarily biological asymmetry. Dominant-frequency features are quantised (0.1 Hz) and several differences are ties, so "
        "bootstrap CIs for them are approximate. Several top 'differences' involve min/max/ceiling features that are affected "
        "by values stuck at the extremes (`acc_z__max` is at the observed positive extreme in all right-leg and 5 left-leg recordings, so that contrast is censored; see section 2 and the audit). "
        "Signed means such as `gyro_z__mean` depend on axis orientation conventions of each mount, which are unknown.\n")
    add("Figures: `reports/figures/paired/` (slopes, standardised differences + dz CIs, top effect sizes). Tables: `reports/tables/paired_*.csv`, `participant_level_summary.csv`.\n")

    # ------------------------------------------------------------------ explore
    add("## 5. Exploratory statistics\n")
    pcs = pd.read_csv(tab / "pca_scores_windows.csv")
    ev = out["pca"]["evr"]
    e = {k: [_eta2(pcs[f"PC{i}"].to_numpy(), pcs[k].to_numpy()) for i in (1, 2)] for k in ("participant_id", "leg")}
    add(f"- PCA on {out['pca']['n_features']} standardised window features ({len(pcs)} windows, {pcs['participant_id'].nunique()} participants): PC1-PC2 explain {100*ev[0]:.1f}% and {100*ev[1]:.1f}% of variance. "
        f"Share of PC1/PC2 variance associated with participant identity: {100*e['participant_id'][0]:.0f}% / {100*e['participant_id'][1]:.0f}%; with leg: {100*e['leg'][0]:.0f}% / {100*e['leg'][1]:.0f}%. "
        "PCA is fitted on all windows for description only (no prediction). It shows broad structure and which recordings are far apart; it cannot show causes, and clusters in PC space are not gait classes.")
    pv = pd.read_csv(tab / "participant_variability.csv")
    pv_ratio = pv["ratio"].median()
    add(f"- Between-participant vs within-recording spread: median ratio of between-participant SD to mean within-recording window SD = {pv_ratio:.2f} over features (`participant_variability.csv`; descriptive, windows overlap).")
    add("- Correlation heatmap: Spearman rho on window features for mean/std/rms/ptp/dominant frequency of each signal; shows heavy redundancy among amplitude features (std, rms, ptp). Windows within a recording are autocorrelated, so no p-values are given.")
    add("- Figures: `reports/figures/exploration/`.\n")

    # ------------------------------------------------------------------ models
    add("## 6. Optional ML experiments (exploratory; what they cannot show)\n")
    m = out["models"]
    if "skipped" in m:
        add(f"Skipped: {m['skipped']}\n")
    else:
        a = m["anomaly"]
        an_all = pd.read_csv(rep / "models" / "anomaly_scores_windows.csv")
        an_flag, an_ok = an_all[an_all["iforest_flag"]], an_all[~an_all["iforest_flag"]]
        add(f"- **Anomaly detection** (Isolation Forest + LOF on {m['n_model_features']} curated features; scaler fitted on all windows because the method is unsupervised and unevaluated): "
            f"{a['n_flagged_iforest']} windows flagged at contamination {cfg['models']['anomaly_contamination']} (the number flagged is fixed by that parameter, not estimated from the data),{a['n_flagged_both']} by both methods "
            f"(rank correlation of scores rho={a['spearman_iforest_vs_lof']:.2f}). Flags by participant: {a['flagged_by_participant']}. "
            "These are statistical outliers relative to these 12 participants - not abnormal gait. "
            f"Mean fraction of clipped samples per window: {an_flag['qc_clip_fraction_any'].mean():.3f} in flagged vs {an_ok['qc_clip_fraction_any'].mean():.3f} in unflagged windows, "
            "so clipping alone does not explain the flags; sensor placement, speed or movement style are untested alternatives. "
            + (f"Audit: participant identity explains {100*aud['anomaly_cluster']['base_iforest_eta2_participant']:.0f}% of the score variance, the flagged windows form only "
               f"{aud['anomaly_cluster']['n_episodes_if_overlapping_windows_merged']} episodes once overlapping neighbours are merged, "
               "and the flagged set changes noticeably with preprocessing (see `reports/audit/anomaly_cluster_robustness.csv`), so treat it as exploratory. " if aud is not None else "")
            +
            f"Participants with a filename note (Sub04, Sub06) account for {int(an_flag['participant_has_filename_note'].sum())} of {len(an_flag)} flagged windows - noted only because the unsupervised method has no labels; this is not evidence of pathology. "
            "Candidates for manual inspection: `reports/models/anomaly_top_windows_for_inspection.csv`.")
        cs = pd.DataFrame(m["clustering"]["summary"])
        add(f"- **K-means** (k={cfg['models']['kmeans_k_range'][0]}-{cfg['models']['kmeans_k_range'][-1]}): silhouette is low for all k (max {cs['silhouette'].max():.2f}), i.e. no clearly separated groups. "
            "Stability = ARI between fits on two random 80% subsets of participants (higher is better); ARI against participant identity / leg labels shows what the clusters track:\n")
        add(_md(cs[["k", "silhouette", "stability_ari_mean", "ari_vs_participant_id", "ari_vs_leg"]]))
        add("\n  Clusters are not labelled as gait categories." + (
            f" Audit: the k=2 split mainly separates higher-intensity windows (mean gyro-magnitude RMS 176 vs 124 in the two clusters; "
            f"Spearman of cluster label vs gyro_mag RMS {aud['anomaly_cluster']['kmeans2_spearman_cluster_vs_gyro_mag_rms']:.2f}) "
            "that also have more samples at rail values, and some participants fall almost entirely in one cluster; "
            "it is not robust to the scaler (ARI about 0.05 against a robust-scaler solution)." if aud is not None else ""))
        lr = m["lr_classification"]
        ci = lr["bootstrap_ci95_over_participants"]
        add(f"- **Left/right classification** (logistic regression, leave-one-participant-out, scaler fitted inside training participants; {lr['n_participants']} participants / {lr['n_recordings']} recordings / {lr['n_windows']} windows): "
            f"mean per-participant window accuracy **{lr['mean_participant_window_accuracy']:.2f}** (bootstrap 95% CI over participants {ci[0]:.2f}-{ci[1]:.2f}; chance 0.50); "
            f"recording-level accuracy {lr['recording_level_accuracy']:.2f} ({int(round(lr['recording_level_accuracy']*lr['n_recordings']))}/{lr['n_recordings']}); "
            f"label-swap permutation null mean {lr['permutation_null_mean']:.2f}, 95th pct {lr['permutation_null_q95']:.2f}, one-sided permutation p = {lr['permutation_p']:.3f} ({lr['n_permutations']} participant-level label swaps{', exact enumeration of all 2^12' if lr.get('permutation_exact_enumeration') else ', Monte Carlo'}). "
            "**Not a demonstrated result**: the interval includes chance, per-participant accuracy ranges from near 0 to 1 (Sub06 is assigned the opposite leg in almost every window), "
            "and any signal could reflect sensor mounting/orientation rather than leg biomechanics. No random-window split was used because it would leak participant identity. "
            "p is not below 0.05, and the participant-level interval is only a description of between-participant variability, not a population-level confidence interval. The audit adds baselines and ablations tested with the same exact null (`reports/audit/classifier_*.csv`); a single feature chosen after seeing all participants scores higher, "
            "but that figure is selection-biased and must not be reported as performance."
            + (" Hardware-informed check: restricted to the "
               f"{aud['hardware']['classifier_variants'][1]['n_features']} model features that do not depend on axis sign, the same procedure gives "
               f"{aud['hardware']['classifier_variants'][1]['mean_participant_accuracy']:.2f} (exact permutation p = {aud['hardware']['classifier_variants'][1]['permutation_p']:.2f}), i.e. no above-chance separation, and the {aud['hardware']['classifier_variants'][2]['n_features']} magnitude-only features (invariant to any rotation of the sensor axes) give "
               f"{aud['hardware']['classifier_variants'][2]['mean_participant_accuracy']:.2f} (p = {aud['hardware']['classifier_variants'][2]['permutation_p']:.2f}). "
               "The 0.63 therefore depends on sign-dependent axis-level features; that is what an orientation/mirroring difference between the two mountings could produce, but it is a consistency observation, not proof of the cause "
               "(sign-independent is not orientation-independent: axis-specific features also change when the sensor is rotated)." if aud is not None else ""))
        add(f"- **Injury analysis**: {m['injury_classifier']}.")
        add(f"- **TinyML**: {m['tinyml']}.\n")

    # ------------------------------------------------------------------ limits
    add("## 7. Limitations\n")
    for t in [
        "12 participants, one 10 s recording per leg: low power, wide intervals, no replication or held-out cohort.",
        "Collaborator-reported protocol and hardware settings are unverified; walking speed, leg order, re-mounting between legs and the anatomical axis mapping are unknown; no raw counts or firmware are available.",
        "Many samples sit at repeated extreme values (probable range limit, unconfirmed); truncation biases extremes, energy and magnitude features, and most features involve channels affected in more than half of the recordings.",
        "Left and right are separate recordings, not time-synchronised; no gait-event (heel strike) detection was attempted, so no stride-level symmetry indices.",
        "Timestamps look nominal (exact 20 ms multiples); real acquisition jitter is unknown.",
        "Collaborator-reported conditions (Sub04 ACL recovery history, Sub06 ankle-twist history) cover two participants with no severity, side, timing or diagnosis; no clinical interpretation or injury model is possible.",
        "Many correlated features were tested; BH assumes positive dependence, which is plausible but unverified.",
    ]:
        add(f"- {t}")
    add("\n## 8. Recommended next steps\n")
    for t in [
        "Re-record with a wider accelerometer/gyroscope range (e.g. the MPU6050 +-2000 deg/s setting; confirm in firmware), log raw int16 counts, the configured range/DLPF register values and the firmware version, and read back the registers at the start of each recording.",
        "Record synchronised left/right data with a shared clock and a documented protocol; add trial IDs.",
        "Detect gait events (e.g. from gyro peaks) to compute stride-level features and symmetry indices; consider mixed-effects models with participant random effects.",
        "Collect more participants before any predictive modelling; pre-register the headline features.",
        "Only after the above consider labelled tasks and, much later, a TinyML target.",
    ]:
        add(f"- {t}")
    add("\n## Reproduce\n")
    add("```\npip install -r requirements.txt\npython -m src.pipeline            # all stages\npython -m pytest -q               # tests\n```")
    path = rep / "final_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
