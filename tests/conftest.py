import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_csv(path: Path, n: int = 500, dt: int = 20, drop: tuple[int, ...] = (), seed: int = 0, const: str | None = None,
             time: bool = True) -> Path:
    """Write a synthetic recording CSV with the real schema (tests only; never used for results)."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1) * dt
    f = 1.0
    sig = np.sin(2 * np.pi * f * t / 1000)
    df = pd.DataFrame({
        "Time_ms": t, "AccelX_g": 0.1 * sig + 0.01 * rng.standard_normal(n), "AccelY_g": -1 + 0.2 * sig,
        "AccelZ_g": 0.05 * rng.standard_normal(n), "GyroX_dps": 100 * sig + rng.standard_normal(n),
        "GyroY_dps": 50 * np.cos(2 * np.pi * f * t / 1000), "GyroZ_dps": 10 * rng.standard_normal(n)})
    if const:
        df[const] = 0.5
    df = df.drop(index=list(drop)).reset_index(drop=True)
    if not time:
        df = df.drop(columns="Time_ms")
    df.to_csv(path, index=False)
    return path
