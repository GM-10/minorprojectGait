"""Per-recording data-quality validation and report generation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_validation.channels import detect_channels, family, find_time_column
from src.data_validation.metadata import documented_vs_observed, participant_metadata
from src.data_validation.discovery import (Recording, build_pairs, discover_recordings,
                                           find_duplicate_files)


def load_csv(path: str) -> pd.DataFrame:
    """Read a source CSV (read-only)."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path}: file contains no data rows")
    return df


def detect_saturation_levels(frames: dict[str, pd.DataFrame], chmaps: dict[str, dict],
                             min_count: int, rel_tol: float) -> dict[tuple[str, str], float]:
    """Empirical saturation levels per sensor family and sign.

    A level is reported when the most extreme value of a family (pooled over all
    recordings and axes) is hit, within ``rel_tol``, at least ``min_count`` times.
    This is evidence of clipping in the *data*; it does not identify the configured
    sensor range.
    """
    pooled: dict[str, list[np.ndarray]] = {}
    for rid, df in frames.items():
        for ch, info in chmaps[rid].items():
            pooled.setdefault(family(ch), []).append(
                pd.to_numeric(df[info["column"]], errors="coerce").to_numpy(dtype=float))
    levels: dict[tuple[str, str], float] = {}
    for fam, arrs in pooled.items():
        v = np.concatenate(arrs)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        for sign, ext in (("pos", v.max()), ("neg", v.min())):
            if ext == 0:
                continue
            n_hit = int((np.abs(v - ext) <= rel_tol * abs(ext)).sum())
            if n_hit >= min_count:
                levels[(fam, sign)] = float(ext)
    return levels


def clipped_mask(x: np.ndarray, fam: str, levels: dict[tuple[str, str], float], rel_tol: float) -> np.ndarray:
    """Boolean mask of samples sitting at an empirical saturation level."""
    m = np.zeros(x.shape, dtype=bool)
    for sign in ("pos", "neg"):
        lvl = levels.get((fam, sign))
        if lvl is not None:
            m |= np.abs(x - lvl) <= rel_tol * abs(lvl)
    return m


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return np.zeros_like(x) if mad == 0 else 0.6745 * (x - med) / mad


def validate_recording(rec: Recording, df: pd.DataFrame, chmap: dict, time_col: str | None,
                       levels: dict, schema_modal: tuple, cohort_median_duration_ms: float | None,
                       cfg: dict) -> tuple[dict, list[dict]]:
    """Return (recording-level report row, list of channel-level rows)."""
    ing = cfg["ingestion"]
    warnings: list[str] = []
    cols = list(df.columns)
    row: dict = {**{k: v for k, v in rec.to_dict().items() if k not in ("path", "sha256")},
                 "n_rows": len(df), "columns": "|".join(cols)}

    if tuple(cols) != schema_modal:
        warnings.append(f"schema differs from the most common schema {list(schema_modal)}")
    missing_ch = [c for c in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z") if c not in chmap]
    if missing_ch:
        warnings.append(f"missing sensor channels: {missing_ch}")
    row["channels_found"] = "|".join(chmap)
    row["declared_units"] = "|".join(f"{c}={i['declared_unit']}" for c, i in chmap.items())
    row["units_status"] = "declared in column names only; NOT verified (no calibration/metadata file)"

    numeric = {}
    n_nonnumeric = 0
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        n_nonnumeric += int(s.isna().sum() - df[col].isna().sum())
        numeric[col] = s
    num = pd.DataFrame(numeric)
    row["n_missing_total"] = int(df.isna().sum().sum())
    row["missing_fraction"] = float(df.isna().sum().sum() / df.size)
    row["n_nonnumeric_values"] = n_nonnumeric
    row["n_inf_values"] = int(np.isinf(num.to_numpy(dtype=float)).sum())
    if row["n_missing_total"]:
        warnings.append(f"{row['n_missing_total']} missing values")
    if n_nonnumeric:
        warnings.append(f"{n_nonnumeric} non-numeric values")
    if row["n_inf_values"]:
        warnings.append(f"{row['n_inf_values']} infinite values")

    # ---- timing
    row.update(has_timestamps=time_col is not None, time_column=time_col, duration_s=np.nan,
               median_dt_ms=np.nan, estimated_fs_hz=np.nan, n_gaps=0, n_missing_samples_est=0,
               n_duplicate_timestamps=0, n_nonmonotonic=0, dt_cv=np.nan, timestamps_on_nominal_grid=np.nan,
               first_timestamp_ms=np.nan, last_timestamp_ms=np.nan, expected_rows_from_span=np.nan)
    if time_col is None:
        warnings.append("no timestamp column: sampling frequency cannot be verified from the CSV alone")
    else:
        t = num[time_col].to_numpy(dtype=float)
        tf = t[np.isfinite(t)]
        dt = np.diff(tf)
        if len(dt) and (dt > 0).any():
            med = float(np.median(dt[dt > 0]))
            row["median_dt_ms"] = med
            row["estimated_fs_hz"] = 1000.0 / med
            row["first_timestamp_ms"], row["last_timestamp_ms"] = float(tf[0]), float(tf[-1])
            row["duration_s"] = float((tf[-1] - tf[0]) / 1000.0)
            row["n_duplicate_timestamps"] = int((dt == 0).sum())
            row["n_nonmonotonic"] = int((dt < 0).sum())
            pos = dt[dt > 0]
            row["dt_cv"] = float(pos.std() / pos.mean())
            gaps = dt > ing["gap_factor"] * med
            row["n_gaps"] = int(gaps.sum())
            row["n_missing_samples_est"] = int(np.round(dt[gaps] / med - 1).sum())
            row["expected_rows_from_span"] = int(round((tf[-1] - tf[0]) / med)) + 1
            rem = np.mod(tf, med)
            row["timestamps_on_nominal_grid"] = bool(np.all(np.minimum(rem, med - rem) < 1e-6))
            if row["n_duplicate_timestamps"]:
                warnings.append(f"{row['n_duplicate_timestamps']} duplicated timestamps")
            if row["n_nonmonotonic"]:
                warnings.append(f"{row['n_nonmonotonic']} non-monotonic timestamps")
            if row["n_gaps"]:
                warnings.append(f"{row['n_gaps']} sampling gaps (~{row['n_missing_samples_est']} dropped samples)")
            if tf[0] > ing["gap_factor"] * med:
                warnings.append(f"first sample at {tf[0]:.0f} ms (leading samples missing or recording starts late)")
            if row["timestamps_on_nominal_grid"]:
                warnings.append("all timestamps are exact multiples of the median period: consistent with a "
                                "nominal/scheduled clock; true acquisition jitter cannot be verified")
            if cohort_median_duration_ms and tf[-1] < ing["expected_cutoff_fraction"] * cohort_median_duration_ms:
                warnings.append(f"recording ends at {tf[-1]:.0f} ms, well before the cohort median "
                                f"{cohort_median_duration_ms:.0f} ms (possible abrupt cutoff)")
        else:
            warnings.append("timestamps present but unusable (no positive increments)")

    # ---- per-channel quality
    ch_rows = []
    n_clipped_any = np.zeros(len(df), dtype=bool)
    for ch, info in chmap.items():
        x = num[info["column"]].to_numpy(dtype=float)
        fin = np.isfinite(x)
        xf = x[fin]
        clip = clipped_mask(x, family(ch), levels, ing["saturation"]["rel_tol"]) & fin
        n_clipped_any |= clip
        rz = _robust_z(xf) if xf.size else xf
        crow = {
            "recording_id": rec.recording_id, "participant_id": rec.participant_id, "leg": rec.leg,
            "channel": ch, "column": info["column"], "declared_unit": info["declared_unit"],
            "n_missing": int((~fin).sum()), "min": float(xf.min()) if xf.size else np.nan,
            "max": float(xf.max()) if xf.size else np.nan, "mean": float(xf.mean()) if xf.size else np.nan,
            "std": float(xf.std(ddof=1)) if xf.size > 1 else np.nan,
            "n_unique": int(np.unique(xf).size), "is_constant": bool(np.unique(xf).size <= 1),
            "n_clipped": int(clip.sum()), "clipped_fraction": float(clip.mean()),
            "n_extreme_robust_z": int((np.abs(rz) > ing["robust_z_extreme"]).sum()),
        }
        ch_rows.append(crow)
        if crow["is_constant"]:
            warnings.append(f"{ch} is constant")
    clip_ch = [r for r in ch_rows if r["n_clipped"] > 0]
    row["n_samples_with_any_clipped_channel"] = int(n_clipped_any.sum())
    row["fraction_samples_with_any_clipped_channel"] = float(n_clipped_any.mean())
    if clip_ch:
        worst = max(clip_ch, key=lambda r: r["clipped_fraction"])
        warnings.append(f"{int(n_clipped_any.sum())} samples ({100*n_clipped_any.mean():.1f}%) sit at an "
                        f"observed rail value (repeated identical extreme; hardware saturation NOT confirmed) in at least one channel (worst: {worst['channel']} "
                        f"{100*worst['clipped_fraction']:.1f}%)")
    row["n_warnings"] = len(warnings)
    row["warnings"] = " ; ".join(warnings)
    return row, ch_rows


def run_validation(cfg: dict) -> dict:
    """Discover, validate and write the data-quality outputs. Returns in-memory objects for later stages."""
    out_dir = Path(cfg["paths"]["reports_dir"]) / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs, unparsed = discover_recordings(cfg["paths"]["raw_data_dir"], cfg["ingestion"]["filename_regex"])
    frames = {r.recording_id: load_csv(r.path) for r in recs}
    time_cols = {rid: find_time_column(list(df.columns), cfg["ingestion"]["time_column_regex"])
                 for rid, df in frames.items()}
    chmaps = {rid: detect_channels(list(df.columns)) for rid, df in frames.items()}

    sat = cfg["ingestion"]["saturation"]
    levels = detect_saturation_levels(frames, chmaps, sat["min_count"], sat["rel_tol"])
    schema_modal = pd.Series([tuple(df.columns) for df in frames.values()]).mode().iloc[0]
    durs = [pd.to_numeric(frames[r.recording_id][time_cols[r.recording_id]], errors="coerce").max()
            for r in recs if time_cols[r.recording_id]]
    cohort_med = float(np.median(durs)) if durs else None

    rows, ch_rows = [], []
    for r in recs:
        rid = r.recording_id
        row, crs = validate_recording(r, frames[rid], chmaps[rid], time_cols[rid], levels,
                                      tuple(schema_modal), cohort_med, cfg)
        rows.append(row)
        ch_rows += crs
    report = pd.DataFrame(rows)
    dups = find_duplicate_files(recs)
    pairs = build_pairs(recs)
    report["is_byte_duplicate"] = report["filename"].isin([b for _, b in dups])
    ch_df = pd.DataFrame(ch_rows)

    report.to_csv(out_dir / "data_quality_recordings.csv", index=False)
    ch_df.to_csv(out_dir / "data_quality_channels.csv", index=False)
    pairs.to_csv(out_dir / "pairs.csv", index=False)
    pmeta = participant_metadata(recs, cfg)
    pmeta.to_csv(out_dir / "participant_metadata.csv", index=False)
    docobs = documented_vs_observed(report, cfg)
    docobs.to_csv(out_dir / "documented_vs_observed.csv", index=False)
    summary = {
        "n_files": len(recs), "n_participants": int(report["participant_id"].nunique()),
        "n_valid_pairs": int(pairs["valid_pair"].sum()), "unparsed_files": unparsed,
        "byte_identical_duplicates": dups, "modal_schema": list(schema_modal),
        "n_schemas": int(len({tuple(df.columns) for df in frames.values()})),
        "saturation_levels": {f"{k[0]}_{k[1]}": v for k, v in levels.items()},
        "estimated_fs_hz_range": [float(report["estimated_fs_hz"].min()), float(report["estimated_fs_hz"].max())],
        "rows_range": [int(report["n_rows"].min()), int(report["n_rows"].max())],
        "files_with_warnings": int((report["n_warnings"] > 0).sum()),
        "notes_in_filenames_unverified": {r.filename: r.note for r in recs if r.note},
        "collaborator_reported_conditions": {r["participant_id"]: r["collaborator_reported_condition"] for _, r in pmeta.iterrows() if r["has_reported_history"]},
        "documented_vs_observed": {"n_recordings": len(docobs), "rows_within_tolerance": int(docobs["rows_within_tolerance"].sum()), "fs_matches": int(docobs["fs_matches"].sum())},
    }
    with open(out_dir / "data_quality_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    _write_markdown(out_dir / "data_quality_report.md", report, pairs, summary)
    return {"recs": recs, "frames": frames, "chmaps": chmaps, "time_cols": time_cols, "levels": levels,
            "report": report, "channels": ch_df, "pairs": pairs, "summary": summary}


def _write_markdown(path: Path, report: pd.DataFrame, pairs: pd.DataFrame, s: dict) -> None:
    L = ["# Data-quality report", "",
         f"- Files: **{s['n_files']}**; participants: **{s['n_participants']}**; "
         f"candidate-valid left/right pairs: **{s['n_valid_pairs']}**",
         f"- Distinct column schemas: {s['n_schemas']} (`{', '.join(s['modal_schema'])}`)",
         f"- Rows per file: {s['rows_range'][0]}-{s['rows_range'][1]}; estimated sampling frequency from timestamps: "
         f"{s['estimated_fs_hz_range'][0]:.2f}-{s['estimated_fs_hz_range'][1]:.2f} Hz",
         f"- Byte-identical duplicate files: {s['byte_identical_duplicates'] or 'none'}",
         f"- Observed rail values (repeated identical extremes; hardware saturation is NOT confirmed, sensor configuration undocumented): {s['saturation_levels'] or 'none detected'}",
         "- Units are taken from column-name suffixes only (`_g`, `_dps`); no calibration or sensor-range metadata is "
         "available, so they are **unverified**.",
         "- Filename notes (unverified, not diagnoses): " + (json.dumps(s["notes_in_filenames_unverified"]) or "none"),
         "- Collaborator-reported participant conditions (provided by the hardware collaborator; not independently verified; no severity, side, timing or diagnosis): " + json.dumps(s["collaborator_reported_conditions"]),
         f"- Documented acquisition settings (collaborator): rows within tolerance of the expected count in {s['documented_vs_observed']['rows_within_tolerance']}/{s['documented_vs_observed']['n_recordings']} recordings; estimated sampling frequency matches 50 Hz in {s['documented_vs_observed']['fs_matches']}/{s['documented_vs_observed']['n_recordings']}.",
         "", "## Per-recording summary", "",
         "| file | participant | leg | rows | duration s | fs Hz | gaps | missing | clipped samples % | warnings |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in report.iterrows():
        L.append(f"| {r['filename']} | {r['participant_id']} | {r['leg']} | {r['n_rows']} | {r['duration_s']:.2f} | "
                 f"{r['estimated_fs_hz']:.1f} | {r['n_gaps']} | {r['n_missing_total']} | "
                 f"{100*r['fraction_samples_with_any_clipped_channel']:.1f} | {r['n_warnings']} |")
    L += ["", "## Warnings by recording", ""]
    for _, r in report.iterrows():
        if r["warnings"]:
            L.append(f"**{r['filename']}**")
            L += [f"- {w.strip()}" for w in r["warnings"].split(" ; ")]
            L.append("")
    L += ["## Pairing", "", "| participant | left file | right file | both legs | notes match | note | valid pair |",
          "|---|---|---|---|---|---|---|"]
    for _, p in pairs.iterrows():
        L.append(f"| {p['participant_id']} | {p['left_file']} | {p['right_file']} | {p['has_both_legs']} | "
                 f"{p['notes_match']} | {p['note']} | {p['valid_pair']} |")
    path.write_text("\n".join(L), encoding="utf-8")
