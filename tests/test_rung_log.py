"""Tests for the per-rung metric log."""
from nuslam.eval import format_rungs, load_rungs, record_rung


def test_record_and_load_roundtrip(tmp_path):
    p = tmp_path / "rungs.json"
    record_rung(p, "3.1", {"psnr": 24.5, "ate_rmse": 2.2}, notes="frozen poses")
    record_rung(p, "3.2", {"psnr": 25.1, "ate_rmse": 2.6})
    data = load_rungs(p)
    assert set(data) == {"3.1", "3.2"}
    assert data["3.1"]["psnr"] == 24.5
    assert data["3.1"]["_notes"] == "frozen poses"


def test_record_replaces_same_rung(tmp_path):
    p = tmp_path / "rungs.json"
    record_rung(p, "3.1", {"psnr": 20.0})
    record_rung(p, "3.1", {"psnr": 24.0})
    assert load_rungs(p)["3.1"]["psnr"] == 24.0


def test_load_missing_is_empty(tmp_path):
    assert load_rungs(tmp_path / "nope.json") == {}


def test_format_shows_baseline_deltas(tmp_path):
    p = tmp_path / "rungs.json"
    record_rung(p, "3.1", {"psnr": 24.0, "ate_rmse": 2.0})
    record_rung(p, "3.2", {"psnr": 25.0, "ate_rmse": 2.5})
    table = format_rungs(load_rungs(p), baseline="3.1")
    assert "psnr" in table and "ate_rmse" in table
    assert "3.1" in table and "3.2" in table
    # the 3.2 column shows the signed delta from the 3.1 baseline
    assert "+1" in table or "+0.5" in table


def test_format_empty():
    assert "no rungs" in format_rungs({})
