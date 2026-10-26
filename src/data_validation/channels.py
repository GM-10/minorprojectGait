"""Channel detection from column names. Units are *declared* labels, never verified."""
from __future__ import annotations

import re

CHANNEL_ORDER = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
_PATTERNS = {
    "acc": re.compile(r"^(accel|acc)[_ ]?([xyz])(?:[_ ]?(.*))?$", re.I),
    "gyro": re.compile(r"^(gyro|gyr)[_ ]?([xyz])(?:[_ ]?(.*))?$", re.I),
}
_UNIT_LABELS = {"g": "g", "dps": "deg/s", "deg/s": "deg/s", "rad/s": "rad/s", "rads": "rad/s",
                "ms2": "m/s^2", "m/s2": "m/s^2", "raw": "raw counts"}


def detect_channels(columns: list[str]) -> dict[str, dict]:
    """Map canonical channel names (acc_x ...) to original column + header-declared unit label."""
    found: dict[str, dict] = {}
    for col in columns:
        for kind, pat in _PATTERNS.items():
            m = pat.match(col.strip())
            if m:
                suffix = (m.group(3) or "").strip().lower()
                found[f"{kind}_{m.group(2).lower()}"] = {
                    "column": col,
                    "declared_unit": _UNIT_LABELS.get(suffix, "unknown"),
                }
    return {k: found[k] for k in CHANNEL_ORDER if k in found}


def find_time_column(columns: list[str], pattern: str) -> str | None:
    for col in columns:
        if re.match(pattern, col.strip(), flags=re.I):
            return col
    return None


def family(channel: str) -> str:
    return channel.split("_")[0]
