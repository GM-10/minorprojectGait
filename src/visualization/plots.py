"""Plotting utilities. Every function saves a PNG and closes the figure.

Units shown on axes are the labels declared in the CSV column names (e.g. ``g``, ``deg/s``); they are NOT
verified against sensor metadata, which is stated in figure footers where relevant.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

LEG_COLORS = {"left": "#0072B2", "right": "#D55E00"}
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
UNIT_NOTE = "units = column-name labels (unverified)"
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def _label(ch: str, units: dict[str, str]) -> str:
    kind, ax = ch.split("_")[0], ch.split("_")[-1].upper()
    return f"{'Accel' if kind == 'acc' else 'Gyro'} {ax} [{units.get(ch, '?')}]"


def _save(fig, path: Path, dpi: int, footer: str | None = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(fig, "_suptitle", None) is not None:
        fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    if footer:
        fig.text(0.99, 0.003, footer, ha="right", va="bottom", fontsize=7, color="gray")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def six_channel(prs: list, units: dict, path: Path, dpi: int, title: str) -> str:
    """Available channels vs time for one participant's recordings (legs overlaid)."""
    chans = [c for c in CHANNELS if all(c in p.raw.columns for p in prs)]
    fig, axes = plt.subplots(len(chans), 1, figsize=(10, 1.7 * len(chans)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, chans):
        for p in prs:
            ax.plot(p.t_ms / 1000, p.raw[c], lw=0.9, color=LEG_COLORS[p.leg], label=p.leg)
        ax.set_ylabel(_label(c, units), fontsize=8)
    axes[0].legend(ncol=2, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time [s] (from Time_ms)")
    fig.suptitle(title)
    return _save(fig, path, dpi, UNIT_NOTE)


def across_participants(prs: list, leg: str, units: dict, path: Path, dpi: int) -> str:
    """Same channel across all participants (one leg), one panel per channel."""
    sel = [p for p in prs if p.leg == leg]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    cmap = plt.get_cmap("tab20")
    for ax, c in zip(axes.T.ravel(), CHANNELS):
        for i, p in enumerate(sel):
            if c in p.raw.columns:
                ax.plot(p.t_ms / 1000, p.raw[c], lw=0.6, alpha=0.8, color=cmap(i % 20), label=p.participant_id)
        ax.set_ylabel(_label(c, units), fontsize=8)
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    axes[0, 1].legend(ncol=3, fontsize=6, loc="upper right")
    fig.suptitle(f"{leg.capitalize()}-leg recordings: same channel across participants")
    return _save(fig, path, dpi, UNIT_NOTE)


def distributions(raw_long: pd.DataFrame, units: dict, path_hist: Path, path_box: Path, dpi: int) -> tuple[str, str]:
    """Histograms (by leg) and per-participant box plots (by leg) for each channel."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), CHANNELS):
        sub = raw_long[raw_long["channel"] == c]
        if sub.empty:
            ax.axis("off")
            continue
        sns.histplot(data=sub, x="value", hue="leg", palette=LEG_COLORS, bins=60, stat="density",
                     common_norm=False, element="step", ax=ax)
        ax.set_xlabel(_label(c, units))
    fig.suptitle("Sample-value distributions by leg (pooled over participants; samples are autocorrelated)")
    p1 = _save(fig, path_hist, dpi, UNIT_NOTE)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for ax, c in zip(axes.T.ravel(), CHANNELS):
        sub = raw_long[raw_long["channel"] == c]
        if sub.empty:
            ax.axis("off")
            continue
        sns.boxplot(data=sub, x="participant_id", y="value", hue="leg", palette=LEG_COLORS, fliersize=1.5,
                    linewidth=0.7, ax=ax)
        ax.set_ylabel(_label(c, units), fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        if ax.get_legend():
            ax.get_legend().remove()
    h = [plt.Rectangle((0, 0), 1, 1, color=LEG_COLORS[k]) for k in ("left", "right")]
    fig.legend(h, ["left", "right"], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Per-recording value distributions (box = quartiles, whiskers 1.5 IQR; values at +-2 g / +-250 deg/s are observed rail values)")
    return p1, _save(fig, path_box, dpi, UNIT_NOTE)


def sampling_overview(report: pd.DataFrame, prs: list, path: Path, dpi: int) -> str:
    """Rows/gaps per recording and where interpolated samples fall in time."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.6]})
    r = report.sort_values(["participant_id", "leg"]).reset_index(drop=True)
    ypos = np.arange(len(r))
    axes[0].barh(ypos, r["n_rows"], color=[LEG_COLORS[l] for l in r["leg"]])
    axes[0].set_yticks(ypos, r["recording_id"], fontsize=7)
    axes[0].set_xlim(r["n_rows"].min() - 5, r["n_rows"].max() + 2)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Rows in CSV (axis truncated)")
    axes[0].set_title("Row counts (500 = no dropped sample)")
    order = {p.recording_id: p for p in prs}
    t_max = max(p.t_ms.max() for p in prs) / 1000
    for i, rid in enumerate(r["recording_id"]):
        p = order[rid]
        axes[1].hlines(i, p.t_ms.min() / 1000, p.t_ms.max() / 1000, color="lightgray", lw=4)
        ti = p.t_ms[p.interpolated] / 1000
        axes[1].plot(ti, np.full_like(ti, i), "|", color="crimson", ms=10, mew=2)
    axes[1].set_yticks(ypos, r["recording_id"], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, t_max + 0.2)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Recording span; red ticks = samples missing in the CSV and interpolated")
    return _save(fig, path, dpi)


def clipping_heatmap(ch_df: pd.DataFrame, path: Path, dpi: int) -> str:
    """Percentage of samples at an observed rail value (repeated identical extreme), per recording x channel."""
    m = ch_df.pivot_table(index="recording_id", columns="channel", values="clipped_fraction") * 100
    m = m[[c for c in CHANNELS if c in m.columns]]
    fig, ax = plt.subplots(figsize=(7, 9))
    sns.heatmap(m, annot=True, fmt=".1f", cmap="Reds", cbar_kws={"label": "% samples at observed rail value"}, ax=ax)
    ax.set_title("Observed rail values (repeated identical extremes; hardware saturation not confirmed)")
    ax.set_ylabel("")
    return _save(fig, path, dpi)


def magnitude_variability(feat_rec: pd.DataFrame, path: Path, dpi: int) -> str:
    """Mean and variability (std) of the vector magnitudes per recording."""
    cols = [c for c in ("gyro_mag", "acc_dyn_mag", "acc_mag") if f"{c}__mean" in feat_rec]
    fig, axes = plt.subplots(2, len(cols), figsize=(4.2 * len(cols), 7), squeeze=False)
    for j, c in enumerate(cols):
        for i, stat in enumerate(("mean", "std")):
            ax = axes[i, j]
            sns.barplot(data=feat_rec, x="participant_id", y=f"{c}__{stat}", hue="leg", palette=LEG_COLORS, ax=ax)
            unit = "g" if c.startswith("acc") else "deg/s"
            ax.set_title(f"{c} {stat} [{unit}]", fontsize=9)
            ax.tick_params(axis="x", rotation=90, labelsize=7)
            ax.set_xlabel("")
            if ax.get_legend():
                ax.get_legend().remove()
    h = [plt.Rectangle((0, 0), 1, 1, color=LEG_COLORS[k]) for k in ("left", "right")]
    fig.legend(h, ["left", "right"], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Magnitude level and variability per recording (whole-recording features)")
    return _save(fig, path, dpi, UNIT_NOTE)


def lr_overlay_grid(prs: list, signal: str, path: Path, dpi: int, ylabel: str) -> str:
    """Per-participant overlay of left vs right for one derived signal (small multiples)."""
    pids = sorted({p.participant_id for p in prs})
    ncol = 4
    nrow = int(np.ceil(len(pids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.4 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, pid in zip(axes.ravel(), pids):
        for p in prs:
            if p.participant_id == pid and signal in p.proc.columns:
                ax.plot(p.t_ms / 1000, p.proc[signal], lw=0.8, color=LEG_COLORS[p.leg], label=p.leg, alpha=.9)
        ax.set_title(pid, fontsize=9)
    for ax in axes.ravel()[len(pids):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle(f"Left vs right, {signal} (time axes are not synchronised between legs: separate recordings)")
    return _save(fig, path, dpi, UNIT_NOTE)


def raw_vs_processed(pr, cutoff_hz: float, cfg: dict, units: dict, path: Path, dpi: int) -> str:
    """Raw vs (a) low-pass (demo only) and (b) gravity-removed accelerometer for one recording."""
    from src.signal_processing.processing import estimate_gravity, lowpass
    fs, t = pr.fs_hz, pr.t_ms / 1000
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    x = pr.raw["acc_y"].to_numpy()
    axes[0].plot(t, x, lw=0.8, color="gray", label="raw acc_y")
    axes[0].plot(t, estimate_gravity(x, fs, cfg["processing"]["gravity_cutoff_hz"]), color="k", lw=1.5,
                 label=f"low-pass {cfg['processing']['gravity_cutoff_hz']} Hz (gravity/posture estimate)")
    axes[0].set_ylabel(_label("acc_y", units))
    axes[0].legend(fontsize=8)
    axes[1].plot(t, pr.proc["acc_dyn_y"], lw=0.8, color="#009E73", label="acc_y minus gravity estimate")
    axes[1].set_ylabel(_label("acc_y", units))
    axes[1].legend(fontsize=8)
    g = pr.raw["gyro_x"].to_numpy()
    axes[2].plot(t, g, lw=0.8, color="gray", label="raw gyro_x")
    axes[2].plot(t, lowpass(g, fs, cutoff_hz, cfg["processing"]["filter_order"]), color="#CC79A7", lw=1.3,
                 label=f"{cutoff_hz} Hz zero-phase Butterworth (comparison only; NOT applied to features)")
    axes[2].set_ylabel(_label("gyro_x", units))
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("Time [s]")
    fig.suptitle(f"Raw vs processed signals: {pr.recording_id}")
    return _save(fig, path, dpi, UNIT_NOTE)


# ---------------------------------------------------------------- paired statistics
def paired_slopes(left: pd.DataFrame, right: pd.DataFrame, res: pd.DataFrame, features: list[str], path: Path,
                  dpi: int) -> str:
    """Left->right line per participant for each headline feature."""
    features = [f for f in features if f in left]
    ncol = 3
    nrow = int(np.ceil(len(features) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.6 * nrow), squeeze=False)
    for ax, f in zip(axes.ravel(), features):
        for pid in left.index:
            ax.plot([0, 1], [left.loc[pid, f], right.loc[pid, f]], "-o", color="gray", lw=0.8, ms=3, alpha=0.7)
        ax.plot([0, 1], [left[f].mean(), right[f].mean()], "-s", color="k", lw=2, ms=6, label="mean")
        ax.set_xticks([0, 1], ["left", "right"])
        r = res[res["feature"] == f].iloc[0]
        ax.set_title(f"{f}\nn={int(r['n_pairs'])}, dz={r['dz']:.2f}, p({r['primary_test']})={r['p_primary']:.3f}, "
                     f"q_BH={r['q_bh_primary']:.2f}", fontsize=8)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    fig.suptitle("Paired left-right values per participant (each line = one participant; whole-recording features in the "
                 "feature's own units: g or deg/s)")
    return _save(fig, path, dpi)


def difference_plot(diffs: pd.DataFrame, res: pd.DataFrame, features: list[str], path: Path, dpi: int) -> str:
    """Distribution of standardized paired differences (left-right)/SD for headline features, per participant."""
    features = [f for f in features if f in diffs]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.9 * len(features) + 2), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, f in enumerate(features):
        d = diffs[f].dropna()
        z = d / d.std(ddof=1) if d.std(ddof=1) > 0 else d * np.nan
        axes[0].scatter(z, np.full(len(z), i), color="gray", s=18, alpha=.8)
        axes[0].plot(z.mean(), i, "ks", ms=7)
    axes[0].axvline(0, color="crimson", lw=1)
    axes[0].set_yticks(range(len(features)), features, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("(left - right) / SD of differences; black square = mean")
    axes[0].set_title("Per-participant differences (standardized)")
    r = res.set_index("feature").loc[features]
    y = np.arange(len(features))
    axes[1].errorbar(r["dz"], y, xerr=[r["dz"] - r["dz_ci_low"], r["dz_ci_high"] - r["dz"]], fmt="o", color="k",
                     capsize=3)
    axes[1].axvline(0, color="crimson", lw=1)
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Cohen's dz with bootstrap 95% CI")
    axes[1].set_title("Effect size (n small: wide CIs)")
    return _save(fig, path, dpi)


def effect_size_bars(res: pd.DataFrame, path: Path, dpi: int, top: int = 25) -> str:
    """Top features by |dz| coloured by BH significance."""
    r = res.dropna(subset=["dz"]).reindex(res["dz"].abs().sort_values(ascending=False).index).dropna(subset=["dz"]).head(top)
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(r) + 1.5))
    colors = np.where(r["significant_bh_primary"], "#D55E00", "#999999")
    ax.barh(range(len(r)), r["dz"], color=colors)
    ax.set_yticks(range(len(r)), r["feature"], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"Cohen's dz (left - right); orange = BH q < 0.05 (n = {int(res['significant_bh_primary'].sum())} features)")
    ax.set_title(f"Largest paired effect sizes (top {len(r)} of {int(res['dz'].notna().sum())} features with a defined dz)")
    return _save(fig, path, dpi)


# ---------------------------------------------------------------- exploratory
def feature_boxplots(win_df: pd.DataFrame, features: list[str], path: Path, dpi: int) -> str:
    """Window-level feature distributions by participant and leg for headline features."""
    features = [f for f in features if f in win_df]
    ncol = 2
    nrow = int(np.ceil(len(features) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow), squeeze=False)
    for ax, f in zip(axes.ravel(), features):
        sns.boxplot(data=win_df, x="participant_id", y=f, hue="leg", palette=LEG_COLORS, fliersize=2, linewidth=0.7,
                    ax=ax)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.set_xlabel("")
        ax.set_title(f + ("  [values on a 0.5 Hz grid]" if f.endswith("dominant_freq_hz") else ""), fontsize=9)
        if ax.get_legend():
            ax.get_legend().remove()
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    h = [plt.Rectangle((0, 0), 1, 1, color=LEG_COLORS[k]) for k in ("left", "right")]
    fig.legend(h, ["left", "right"], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Window-level feature distributions per participant and leg (9 windows per recording, 50% overlap: not independent)")
    return _save(fig, path, dpi)


def corr_heatmap(corr: pd.DataFrame, path: Path, dpi: int, title: str) -> str:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(7, 0.28 * n), max(6, 0.26 * n)))
    sns.heatmap(corr, cmap="vlag", vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"label": "Spearman rho"},
                xticklabels=True, yticklabels=True)
    ax.tick_params(labelsize=6)
    ax.set_title(title)
    return _save(fig, path, dpi)


def pca_plots(pca: dict, path: Path, dpi: int) -> str:
    """PC1-PC2 scatter by leg and by participant, plus scree."""
    sc, evr = pca["scores"], pca["explained_variance_ratio"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    sns.scatterplot(data=sc, x="PC1", y="PC2", hue="leg", palette=LEG_COLORS, s=22, ax=axes[0])
    axes[0].set_title("Windows by leg")
    sns.scatterplot(data=sc, x="PC1", y="PC2", hue="participant_id", palette="tab20", s=22, ax=axes[1], legend=True)
    axes[1].legend(fontsize=6, ncol=2, loc="best")
    axes[1].set_title("Windows by participant")
    for ax in axes[:2]:
        ax.set_xlabel(f"PC1 ({100*evr[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({100*evr[1]:.1f}% var)")
    axes[2].bar(range(1, len(evr) + 1), 100 * evr, color="#56B4E9")
    axes[2].plot(range(1, len(evr) + 1), 100 * np.cumsum(evr), "k-o", ms=3)
    axes[2].set_xlabel("Component")
    axes[2].set_ylabel("% variance (bars) / cumulative (line)")
    axes[2].set_title("Scree")
    fig.suptitle(f"PCA of standardized window features ({pca['n_features_used']} features, "
                 f"{pca['n_rows']} windows, {pca['n_participants']} participants) - descriptive only")
    return _save(fig, path, dpi)


def anomaly_plot(an: pd.DataFrame, path: Path, dpi: int) -> str:
    """Isolation-forest score of every window, by participant, plus score distribution by participant."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    a = an.sort_values(["participant_id", "leg", "window_index"])
    sns.stripplot(data=a, x="participant_id", y="iforest_score", hue="leg", palette=LEG_COLORS, dodge=True, size=3.5,
                  ax=axes[0])
    thr = a.loc[a["iforest_flag"], "iforest_score"].min()
    if np.isfinite(thr):
        axes[0].axhline(thr, color="k", ls="--", lw=0.8, label="flag threshold")
    axes[0].tick_params(axis="x", rotation=90, labelsize=8)
    axes[0].set_ylabel("Isolation-forest score (higher = more unusual)")
    axes[0].set_xlabel("")
    axes[0].set_title("Window scores by participant")
    sns.histplot(a["iforest_score"], bins=30, color="#56B4E9", ax=axes[1])
    axes[1].set_xlabel("Isolation-forest score")
    fig.suptitle(f"Unsupervised outlier scores ({int(a['iforest_flag'].sum())} of {len(a)} windows flagged; "
                 "participant identity explains much of the score): statistical outliers only, NOT clinical abnormality")
    return _save(fig, path, dpi)


def quality_summary_plot(report: pd.DataFrame, path: Path, dpi: int) -> str:
    """Compact data-quality panel: duration, sample count, missingness, clipping per recording."""
    r = report.sort_values(["participant_id", "leg"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True)
    y = np.arange(len(r))
    cols = [LEG_COLORS[l] for l in r["leg"]]
    axes[0].barh(y, r["duration_s"], color=cols)
    axes[0].set_title("Duration [s]")
    axes[1].barh(y, r["n_rows"], color=cols)
    axes[1].set_title("Rows (axis truncated)")
    axes[1].set_xlim(r["n_rows"].min() - 3, r["n_rows"].max() + 1)
    axes[2].barh(y, r["n_missing_total"] + r["n_missing_samples_est"], color=cols)
    axes[2].set_title("Missing values + dropped samples")
    axes[3].barh(y, 100 * r["fraction_samples_with_any_clipped_channel"], color=cols)
    axes[3].set_title("% samples at an observed rail value (any channel)")
    axes[0].set_yticks(y, r["recording_id"], fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle(f"Data-quality summary (est. fs: {r['estimated_fs_hz'].min():.1f}-{r['estimated_fs_hz'].max():.1f} Hz "
                 "from timestamps; blue = left, orange = right)")
    return _save(fig, path, dpi)
