import numpy as np
import pandas as pd
import pytest

from conftest import make_csv
from src.data_validation.channels import detect_channels
from src.data_validation.discovery import (build_pairs, discover_recordings, find_duplicate_files, parse_filename)
from src.data_validation.validate import clipped_mask, detect_saturation_levels, run_validation

PAT = r'^Sub(?P<pid>\d+)(?P<leg>Left|Right)(?:\((?P<note>.*)\))?\.csv$'


def test_parse_filename_with_and_without_note():
    a = parse_filename("Sub04Left(recovering acl).csv", PAT)
    assert a == {"participant_id": "Sub04", "participant_num": 4, "leg": "left", "note": "recovering acl"}
    b = parse_filename("Sub12Right.csv", PAT)
    assert b["leg"] == "right" and b["note"] == "" and b["participant_id"] == "Sub12"
    assert parse_filename("notes.csv", PAT) is None


def test_detect_channels_and_declared_units():
    ch = detect_channels(["Time_ms", "AccelX_g", "AccelY_g", "AccelZ_g", "GyroX_dps", "GyroY_dps", "GyroZ_dps"])
    assert list(ch) == ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    assert ch["acc_x"]["declared_unit"] == "g" and ch["gyro_z"]["declared_unit"] == "deg/s"
    assert "gyro_x" not in detect_channels(["AccelX_g"])
    assert detect_channels(["AccelX"])["acc_x"]["declared_unit"] == "unknown"


def test_discovery_pairs_and_duplicates(tmp_path):
    make_csv(tmp_path / "Sub01Left.csv", seed=1)
    make_csv(tmp_path / "Sub01Right.csv", seed=2)
    make_csv(tmp_path / "Sub02Left(note a).csv", seed=3)
    make_csv(tmp_path / "Sub02Right(note b).csv", seed=4)
    make_csv(tmp_path / "Sub03Left.csv", seed=5)
    (tmp_path / "Sub03Left_copy.csv").write_bytes((tmp_path / "Sub03Left.csv").read_bytes())
    make_csv(tmp_path / "other.csv")
    recs, unparsed = discover_recordings(tmp_path, PAT)
    assert len(recs) == 5 and len(unparsed) == 2  # copy file and other.csv do not match the pattern
    pairs = build_pairs(recs).set_index("participant_id")
    assert pairs.loc["Sub01", "valid_pair"]
    assert pairs.loc["Sub02", "has_both_legs"] and not pairs.loc["Sub02", "valid_pair"]  # notes differ
    assert not pairs.loc["Sub03", "has_both_legs"]
    assert find_duplicate_files(recs) == []


def test_duplicate_participant_leg_raises(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    make_csv(tmp_path / "a" / "Sub01Left.csv")
    make_csv(tmp_path / "b" / "Sub01Left.csv")
    with pytest.raises(ValueError, match="same participant/leg"):
        discover_recordings(tmp_path, PAT)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_recordings(tmp_path / "nope", PAT)


def test_saturation_detection():
    x = np.r_[np.full(20, 2.0), np.linspace(-1, 1, 100)]
    df = pd.DataFrame({"AccelX_g": x})
    ch = {"r": {"acc_x": {"column": "AccelX_g", "declared_unit": "g"}}}
    lv = detect_saturation_levels({"r": df}, ch, min_count=10, rel_tol=0.001)
    assert ("acc", "pos") in lv and ("acc", "neg") not in lv
    m = clipped_mask(x, "acc", lv, 0.001)
    assert m[:20].all() and m[20:].sum() == 0


def test_validation_flags_gaps_and_no_timestamp(tmp_path, cfg):
    make_csv(tmp_path / "Sub01Left.csv", drop=(10, 11, 200))
    make_csv(tmp_path / "Sub01Right.csv", time=False, const="GyroZ_dps")
    c = {**cfg, "paths": {**cfg["paths"], "raw_data_dir": tmp_path, "reports_dir": tmp_path / "rep"}}
    out = run_validation(c)
    rep = out["report"].set_index("recording_id")
    left, right = rep.loc["Sub01_left"], rep.loc["Sub01_right"]
    assert left["n_gaps"] == 2 and left["n_missing_samples_est"] == 3
    assert abs(left["estimated_fs_hz"] - 50) < 1e-6
    assert not right["has_timestamps"] and np.isnan(right["estimated_fs_hz"])
    assert "cannot be verified" in right["warnings"]
    ch = out["channels"].set_index(["recording_id", "channel"])
    assert ch.loc[("Sub01_right", "gyro_z"), "is_constant"]
    assert (tmp_path / "rep" / "data_quality" / "data_quality_report.md").is_file()
