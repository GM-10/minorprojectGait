"""Integration test on the real dataset (skipped if Data/ is absent). Source CSVs are only hashed and read."""
import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.data_validation.validate import run_validation
from src.features.extract import feature_columns
from src.pipeline import build_features, snapshot_hashes

DATA = PROJECT_ROOT / "Data"
pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="Data/ not present")


def test_real_data_end_to_end(cfg, tmp_path):
    c = {**cfg, "paths": {**cfg["paths"], "reports_dir": tmp_path / "rep", "processed_dir": tmp_path / "proc"}}
    before = snapshot_hashes(DATA)
    val = run_validation(c)
    prs, win, rec, logs = build_features(c, val)
    assert snapshot_hashes(DATA) == before  # sources untouched
    assert not any(p.is_relative_to(DATA) for p in tmp_path.rglob("*")) and not (DATA / "processed").exists()
    assert len(rec) == len(val["recs"]) == 24 and rec["participant_id"].nunique() == 12
    assert set(win["recording_id"]) == set(rec["recording_id"])
    # identifiers retained and consistent with the original file names
    assert set(win["filename"]) == {r.filename for r in val["recs"]}
    assert (win.groupby("recording_id")["leg"].nunique() == 1).all()
    assert (win["end_idx"] - win["start_idx"] == win["n_samples"]).all() and win["n_samples"].nunique() == 1
    assert (win["fs_hz"] == 50).all() and win["feature_config_id"].nunique() == 1
    feats = feature_columns(win)
    assert np.isfinite(win[feats].to_numpy(dtype=float)).all()
    assert len(logs) == 24 and all("windowing" in l for l in logs)
