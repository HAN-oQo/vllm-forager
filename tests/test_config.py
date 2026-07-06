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


def test_validate_roles_rejects_unknown_role():
    """T3.17 review: a typo'd role must fail loud, not silently fall through every role-keyed
    lookup's default branch."""
    from src import config as cfg

    with pytest.raises(ValueError, match="invalid role"):
        cfg._validate_roles([{"slug": "o/r", "role": "Source"}])


def test_validate_roles_accepts_every_known_role():
    from src import config as cfg

    cfg._validate_roles([{"slug": f"o/{role}", "role": role} for role in cfg._VALID_ROLES])


def test_live_repos_all_have_valid_roles():
    """The real config.REPOS list itself passed `_validate_roles` at import time (this module
    would have failed to import otherwise) -- this test just documents that guarantee."""
    from src import config as cfg

    cfg._validate_roles(cfg.REPOS)  # re-run explicitly; raises on regression
