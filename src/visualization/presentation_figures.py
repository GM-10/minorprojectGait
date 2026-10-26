"""Figures made for the final presentation. Run on demand (not part of the pipeline):

    python -m src.visualization.presentation_figures

Everything is read from saved outputs (``reports/``) or the raw CSVs (read-only); nothing is recomputed or invented, and no
pipeline output is modified. Figures are written to ``reports/figures/presentation/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402

NAVY, GREY, TEAL, ORANGE = "#122039", "#5C6C7E", "#14918B", "#E8832A"
LEFT, RIGHT = "#0072B2", "#D55E00"
CH = [("AccelX_g", "Accel X [g]", 2.0), ("AccelY_g", "Accel Y [g]", 2.0), ("AccelZ_g", "Accel Z [g]", 2.0),
      ("GyroX_dps", "Gyro X [deg/s]", 250.0), ("GyroY_dps", "Gyro Y [deg/s]", 250.0), ("GyroZ_dps", "Gyro Z [deg/s]", 250.0)]


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#9AA5B1", "axes.labelcolor": NAVY, "xtick.color": GREY,
                         "ytick.color": GREY, "text.color": NAVY, "axes.spines.top": False, "axes.spines.right": False})


def six_channel_two_legs(raw_dir: Path, pid: str, out: Path) -> str:
    """All six channels of one participant, left and right recording overlaid (separate, unsynchronised recordings)."""
    _style()
    d = {leg: pd.read_csv(next(Path(raw_dir).glob(f"{pid}{leg.capitalize()}*.csv"))) for leg in ("left", "right")}
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 4.45), sharex=True)
    for k, (col, label, rail) in enumerate(CH):
        ax = axes[k % 3, k // 3]
        for leg, color in (("left", LEFT), ("right", RIGHT)):
            ax.plot(d[leg]["Time_ms"] / 1000, d[leg][col], lw=0.7, color=color, label=f"{leg} leg" if k == 0 else None)
        for s in (-1, 1):
            ax.axhline(s * rail, color=GREY, lw=0.6, ls=(0, (4, 3)))
        ax.set_ylim(-rail * 1.15, rail * 1.15)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
        if k % 3 == 2:
            ax.set_xlabel("Time [s] (from Time_ms)", fontsize=8)
    axes[0, 0].legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    fig.suptitle(f"{pid}: left and right recording (separate trials); dashed lines = reported full scale (+-2 g, +-250 deg/s)",
                 fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    return str(out)


def classifier_evaluation(rep: Path, out: Path) -> str:
    """Per-participant leave-one-participant-out accuracy next to the exact permutation null."""
    _style()
    m = json.loads((rep / "run_summary.json").read_text(encoding="utf-8"))["models"]["lr_classification"]
    acc = pd.read_csv(rep / "models" / "lr_classification_participant_accuracy.csv")
    null = pd.read_csv(rep / "models" / "lr_permutation_null.csv")["mean_participant_accuracy"].to_numpy()
    obs = m["mean_participant_window_accuracy"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.8, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(acc))
    a.bar(x, acc["window_accuracy"], color=TEAL, width=0.72)
    a.axhline(0.5, color=GREY, lw=0.9, ls=(0, (4, 3)))
    a.axhline(obs, color=ORANGE, lw=1.3)
    a.set_xticks(x, [p.replace("Sub", "") for p in acc["participant_id"]], fontsize=7)
    a.set_ylim(0, 1.05)
    a.set_xlabel("held-out participant", fontsize=8)
    a.set_ylabel("window accuracy", fontsize=8)
    a.tick_params(axis="y", labelsize=7)
    a.set_title(f"Leave-one-participant-out (orange: mean {obs:.2f}; dashed: chance)", fontsize=7.5, color=NAVY)
    b.hist(null, bins=26, color="#B8C4D0", edgecolor="white")
    b.axvline(obs, color=ORANGE, lw=1.6)
    b.axvline(m["permutation_null_q95"], color=GREY, lw=1, ls=(0, (4, 3)))
    b.set_ylim(0, 560)
    b.set_xlabel("mean accuracy under label swaps", fontsize=8)
    b.set_ylabel("count of 4096 swaps", fontsize=8)
    b.tick_params(labelsize=7)
    b.set_title(f"Exact permutation test: p = {m['permutation_p']:.3f}", fontsize=8.5, color=NAVY)
    b.text(0.03, 0.95, f"orange: observed {obs:.2f}\ndashed: null 95th pct {m['permutation_null_q95']:.2f}", transform=b.transAxes,
           ha="left", va="top", fontsize=7, color=NAVY)
    fig.tight_layout(pad=0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    return str(out)


def pca_clusters(rep: Path, out: Path) -> str:
    """PC1-PC2 of the window features, coloured by the saved k-means assignment (best k by silhouette)."""
    _style()
    rs = json.loads((rep / "run_summary.json").read_text(encoding="utf-8"))
    evr = rs["pca"]["evr"]
    k = rs["models"]["clustering"]["best_k_by_silhouette"]
    sil = next(r["silhouette"] for r in rs["models"]["clustering"]["summary"] if r["k"] == k)
    sc = pd.read_csv(rep / "tables" / "pca_scores_windows.csv")
    cl = pd.read_csv(rep / "models" / "clustering_assignments.csv")
    d = sc.merge(cl[["recording_id", "window_index", "cluster"]], on=["recording_id", "window_index"], how="inner")
    assert len(d) == len(sc) == len(cl), "PCA scores and cluster assignments do not align"
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    colors = [TEAL, ORANGE, "#2E64A8", "#7A5C99"]
    for c in sorted(d["cluster"].unique()):
        s = d[d["cluster"] == c]
        ax.scatter(s["PC1"], s["PC2"], s=11, color=colors[int(c) % 4], alpha=0.85, lw=0, label=f"cluster {int(c) + 1} (n = {len(s)})")
    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% of variance)", fontsize=8)
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title(f"{len(d)} windows from {d['participant_id'].nunique()} participants; k-means k = {k}, silhouette {sil:.2f} (weak)",
                 fontsize=8, color=NAVY)
    fig.tight_layout(pad=0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    return str(out)


def main() -> None:
    cfg = load_config()
    rep = Path(cfg["paths"]["reports_dir"])
    out = rep / "figures" / "presentation"
    print(six_channel_two_legs(cfg["paths"]["raw_data_dir"], "Sub02", out / "slide07_six_channel_Sub02.png"))
    print(classifier_evaluation(rep, out / "slide10_classifier_evaluation.png"))
    print(pca_clusters(rep, out / "slide10_pca_clusters.png"))


if __name__ == "__main__":
    main()
