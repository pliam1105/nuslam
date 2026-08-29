"""Shared test fixtures: src on path, dataroot discovery."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_DATAROOT = REPO_ROOT / "data" / "nuscenes"


def _has_mini() -> bool:
    return (_DATAROOT / "v1.0-mini").is_dir()


@pytest.fixture(scope="session")
def dataroot() -> Path:
    if not _has_mini():
        pytest.skip("nuScenes v1.0-mini not present under data/nuscenes")
    return _DATAROOT
