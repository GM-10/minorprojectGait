"""Shared helpers for the scientific audit."""
from __future__ import annotations

import re

AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
_MAG = {"acc_mag": ["acc_x", "acc_y", "acc_z"], "acc_dyn_mag": ["acc_x", "acc_y", "acc_z"],
        "gyro_mag": ["gyro_x", "gyro_y", "gyro_z"]}


def signal_channels(signal: str) -> set[str]:
    """Raw channels that contribute to a derived signal name (``acc_mag`` -> its three axes)."""
    if signal in AXES:
        return {signal}
    if signal in _MAG:
        return set(_MAG[signal])
    return set()


def feature_channels(feature: str) -> set[str]:
    """Raw channels that contribute to a feature column (``corr__acc_x__gyro_x`` -> both axes)."""
    if feature.startswith("corr__"):
        out: set[str] = set()
        for part in feature.split("__")[1:]:
            out |= signal_channels(part)
        return out
    return signal_channels(feature.split("__")[0])


def feature_stat(feature: str) -> str:
    return "corr" if feature.startswith("corr__") else feature.split("__")[-1]


def feature_signal(feature: str) -> str:
    return "corr" if feature.startswith("corr__") else feature.split("__")[0]
