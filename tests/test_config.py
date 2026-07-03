"""Tests for path/config resolution."""

import importlib

import pytest

pytestmark = pytest.mark.m0


def test_data_dir_env_override(monkeypatch, tmp_path):
    """FORAGER_DATA_DIR points DATA_DIR (and STATE_PATH) at a shared directory."""
    from src import config as cfg

    monkeypatch.setenv("FORAGER_DATA_DIR", str(tmp_path))
    try:
        importlib.reload(cfg)
        assert tmp_path == cfg.DATA_DIR
        assert tmp_path / "state.json" == cfg.STATE_PATH
    finally:
        monkeypatch.delenv("FORAGER_DATA_DIR", raising=False)
        importlib.reload(cfg)  # restore default so other tests are unaffected
