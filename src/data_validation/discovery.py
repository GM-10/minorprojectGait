"""Discovery of recordings, filename parsing, duplicate detection and left/right pairing.

Source CSVs are only ever opened for reading.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Recording:
    """Metadata parsed from a file name. ``note`` is unverified free text."""
    path: str
    filename: str
    participant_id: str       # e.g. "Sub04"
    participant_num: int
    leg: str                  # "left" | "right"
    note: str                 # text in parentheses, "" if none (UNVERIFIED metadata)
    recording_id: str         # derived from filename only: no trial id exists in the data
    sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_filename(filename: str, pattern: str) -> dict | None:
    """Parse participant, leg and note from a file name. Returns None if it does not match."""
    m = re.match(pattern, filename, flags=re.IGNORECASE)
    if not m:
        return None
    pid_num = int(m.group("pid"))
    return {
        "participant_id": f"Sub{pid_num:02d}",
        "participant_num": pid_num,
        "leg": m.group("leg").lower(),
        "note": (m.group("note") or "").strip(),
    }


def discover_recordings(data_dir: Path, pattern: str) -> tuple[list[Recording], list[str]]:
    """Find every CSV under ``data_dir`` (recursively).

    Returns (recordings, unparsed_filenames). Files whose names do not match are
    reported, not silently dropped.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    recs: list[Recording] = []
    unparsed: list[str] = []
    for p in sorted(data_dir.rglob("*.csv")):
        meta = parse_filename(p.name, pattern)
        if meta is None:
            unparsed.append(str(p))
            continue
        recs.append(Recording(
            path=str(p), filename=p.name, recording_id=f"{meta['participant_id']}_{meta['leg']}",
            sha256=file_sha256(p), **meta))
    if not recs:
        raise FileNotFoundError(f"No parsable CSV recordings under {data_dir}")
    ids = [r.recording_id for r in recs]
    dup_ids = sorted({i for i in ids if ids.count(i) > 1})
    if dup_ids:
        raise ValueError(f"Multiple files map to the same participant/leg: {dup_ids}")
    return recs, unparsed


def find_duplicate_files(recs: list[Recording]) -> list[tuple[str, str]]:
    """Pairs of files with byte-identical content."""
    seen: dict[str, str] = {}
    dups = []
    for r in recs:
        if r.sha256 in seen:
            dups.append((seen[r.sha256], r.filename))
        else:
            seen[r.sha256] = r.filename
    return dups


def build_pairs(recs: list[Recording]) -> pd.DataFrame:
    """One row per participant: which legs exist, whether notes agree, whether a valid pair exists.

    A pair is *candidate-valid* when both a left and a right file exist and their
    filename notes are identical. Protocol equivalence cannot be verified from the CSVs.
    """
    rows = []
    for pid in sorted({r.participant_id for r in recs}):
        left = next((r for r in recs if r.participant_id == pid and r.leg == "left"), None)
        right = next((r for r in recs if r.participant_id == pid and r.leg == "right"), None)
        notes_match = bool(left and right and left.note == right.note)
        rows.append({
            "participant_id": pid,
            "left_file": left.filename if left else None,
            "right_file": right.filename if right else None,
            "has_both_legs": bool(left and right),
            "notes_match": notes_match,
            "note": (left.note if left else right.note if right else ""),
            "valid_pair": bool(left and right and notes_match),
        })
    return pd.DataFrame(rows)
