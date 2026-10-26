"""EDA figure generation (a small, configurable set of representative plots, not one figure per file)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization import plots


def pick_representative(feat_rec: pd.DataFrame, n: int) -> list[str]:
    """Participants spanning low / median / high gyro-magnitude RMS (data-driven, not hand-picked)."""
    col = "gyro_mag__rms"
    if col not in feat_rec:
        return sorted(feat_rec["participant_id"].unique())[:n]
    m = feat_rec.groupby("participant_id")[col].mean().sort_values()
    idx = sorted({0, len(m) // 2, len(m) - 1})[:max(n, 1)] if n >= 3 else list(range(min(n, len(m))))
    return [m.index[i] for i in idx]


def run_eda(cfg: dict, val: dict, prs: list, feat_rec: pd.DataFrame) -> dict[str, list[str]]:
    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "eda"
    dpi = cfg["eda"]["dpi"]
    units = {}
    for cm in val["chmaps"].values():
        for ch, info in cm.items():
            units.setdefault(ch, info["declared_unit"])
    made: dict[str, list[str]] = {}

    reps = pick_representative(feat_rec, cfg["eda"]["representative_n"])
    made["representative_six_channel"] = [
        plots.six_channel([p for p in prs if p.participant_id == pid], units,
                          fig_dir / f"timeseries_channels_{pid}.png", dpi,
                          f"{pid}: all sensor channels (representative; chosen by gyro-magnitude RMS rank)")
        for pid in reps]
    made["across_participants"] = [plots.across_participants(prs, leg, units, fig_dir / f"across_participants_{leg}.png",
                                                             dpi) for leg in ("left", "right")]

    rows = []
    for p in prs:
        for c in p.raw.columns:
            rows.append(pd.DataFrame({"participant_id": p.participant_id, "leg": p.leg, "channel": c,
                                      "value": p.raw[c].to_numpy()}))
    long = pd.concat(rows, ignore_index=True)
    made["distributions"] = list(plots.distributions(long, units, fig_dir / "hist_by_leg.png",
                                                     fig_dir / "box_by_participant.png", dpi))
    q = long.groupby(["participant_id", "leg", "channel"])["value"].quantile([0, .05, .25, .5, .75, .95, 1]).unstack()
    q.columns = [f"q{int(c*100):02d}" for c in q.columns]
    tab_dir = Path(cfg["paths"]["reports_dir"]) / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    q.reset_index().to_csv(tab_dir / "channel_quantiles.csv", index=False)

    made["magnitude"] = [plots.magnitude_variability(feat_rec, fig_dir / "magnitude_variability.png", dpi)]
    made["quality"] = [plots.quality_summary_plot(val["report"], fig_dir / "data_quality_summary.png", dpi),
                       plots.sampling_overview(val["report"], prs, fig_dir / "sampling_irregularities.png", dpi),
                       plots.clipping_heatmap(val["channels"], fig_dir / "clipping_heatmap.png", dpi)]
    made["left_vs_right"] = [
        plots.lr_overlay_grid(prs, "gyro_mag", fig_dir / "left_vs_right_gyro_magnitude.png", dpi,
                              f"gyro magnitude [{units.get('gyro_x', '?')}]"),
        plots.lr_overlay_grid(prs, "acc_dyn_mag", fig_dir / "left_vs_right_acc_dynamic_magnitude.png", dpi,
                              f"acc_dyn_mag [{units.get('acc_x', '?')}]")]
    ref = next((p for p in prs if p.participant_id == reps[0] and "acc_dyn_y" in p.proc.columns), None)
    if ref is not None:
        made["raw_vs_processed"] = [plots.raw_vs_processed(ref, cfg["processing"]["comparison_lowpass_hz"], cfg, units,
                                                           fig_dir / "raw_vs_processed.png", dpi)]
    return made
