"""Tests for the per-stage metric log."""
from nuslam.eval import format_metrics_log, load_metrics_log, record_metrics


def test_record_and_load_roundtrip(tmp_path):
    p = tmp_path / "metrics.json"
    record_metrics(p, "photometric", {"psnr": 24.5, "ate_rmse": 2.2}, notes="poses fixed")
    record_metrics(p, "free-poses", {"psnr": 25.1, "ate_rmse": 2.6})
    data = load_metrics_log(p)
    assert set(data) == {"photometric", "free-poses"}
    assert data["photometric"]["psnr"] == 24.5
    assert data["photometric"]["_notes"] == "poses fixed"


def test_record_replaces_same_stage(tmp_path):
    p = tmp_path / "metrics.json"
    record_metrics(p, "photometric", {"psnr": 20.0})
    record_metrics(p, "photometric", {"psnr": 24.0})
    assert load_metrics_log(p)["photometric"]["psnr"] == 24.0


def test_load_missing_is_empty(tmp_path):
    assert load_metrics_log(tmp_path / "nope.json") == {}


def test_format_shows_baseline_deltas(tmp_path):
    p = tmp_path / "metrics.json"
    record_metrics(p, "photometric", {"psnr": 24.0, "ate_rmse": 2.0})
    record_metrics(p, "free-poses", {"psnr": 25.0, "ate_rmse": 2.5})
    table = format_metrics_log(load_metrics_log(p), baseline="photometric")
    assert "psnr" in table and "ate_rmse" in table
    assert "photometric" in table and "free-poses" in table
    assert "+1" in table or "+0.5" in table


def test_format_empty():
    assert "no metrics" in format_metrics_log({})
