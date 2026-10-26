"""Collaborator-provided participant metadata and documented-vs-observed hardware expectations.

Nothing here is inferred from the data: the metadata come from ``configs/default.yaml`` (provenance recorded)
and are cross-referenced with the notes parsed from the file names, which are a separate, unverified source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_validation.discovery import Recording


def participant_metadata(recs: list[Recording], cfg: dict) -> pd.DataFrame:
    """One row per participant: collaborator description, file-name note, provenance and verification status."""
    pm = cfg["participant_metadata"]
    rows = []
    for pid in sorted({r.participant_id for r in recs}):
        notes = sorted({r.note for r in recs if r.participant_id == pid and r.note})
        cond = pm["conditions"].get(pid)
        rows.append({
            "participant_id": pid,
            "collaborator_reported_condition": cond if cond else pm["default_description"],
            "has_reported_history": cond is not None,
            "filename_note": "|".join(notes),
            "filename_vs_collaborator": ("both sources report a history; any further detail (e.g. a side in the file name) exists only in the unverified file name" if notes and cond else
                                         "no filename note" if not notes and not cond else "MISMATCH: check"),
            "source": pm["source"], "verification_status": pm["verification"],
            "usable_as_outcome_label": False,
            "reason": "2 of 12 participants, no severity/side/timing/diagnosis; descriptive context only",
        })
    return pd.DataFrame(rows)


def documented_vs_observed(report: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compare per-recording observations with the collaborator-documented acquisition settings."""
    hw = cfg["hardware"]
    rows = []
    for _, r in report.iterrows():
        rows.append({
            "recording_id": r["recording_id"], "n_rows": r["n_rows"],
            "expected_rows": hw["expected_samples"],
            "rows_within_tolerance": abs(r["n_rows"] - hw["expected_samples"]) <= hw["expected_samples_tolerance"],
            "duration_s": r["duration_s"], "expected_duration_s": hw["recording_duration_s"],
            "estimated_fs_hz": r["estimated_fs_hz"], "documented_fs_hz": hw["sampling_frequency_hz"],
            "fs_matches": bool(np.isclose(r["estimated_fs_hz"], hw["sampling_frequency_hz"], rtol=0.01)),
            "first_timestamp_ms": r["first_timestamp_ms"], "last_timestamp_ms": r["last_timestamp_ms"],
            "n_gaps": r["n_gaps"], "n_missing_samples_est": r["n_missing_samples_est"],
        })
    return pd.DataFrame(rows)
