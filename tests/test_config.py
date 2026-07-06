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


def test_validate_repos_rejects_unknown_role():
    """T3.17 review: a typo'd role must fail loud, not silently fall through every role-keyed
    lookup's default branch."""
    from src import config as cfg

    with pytest.raises(ValueError, match="invalid role"):
        cfg._validate_repos([{"slug": "o/r", "role": "Source", "domain": "engine"}])


def test_validate_repos_rejects_unknown_domain():
    """T3.18 review: a typo'd/mis-cased domain (e.g. `"RL"`) must fail loud too -- otherwise it
    silently crashes `scripts/repos-md.py` (a hard `repo["domain"]` lookup) and mis-scopes
    `parity.py`/`scout.py`'s defensive `.get("domain")` reads with no error anywhere."""
    from src import config as cfg

    with pytest.raises(ValueError, match="invalid domain"):
        cfg._validate_repos([{"slug": "o/r", "role": "primary", "domain": "RL"}])


def test_validate_repos_rejects_duplicate_slug():
    from src import config as cfg

    with pytest.raises(ValueError, match="duplicate slug"):
        cfg._validate_repos(
            [
                {"slug": "o/r", "role": "primary", "domain": "engine"},
                {"slug": "o/r", "role": "source", "domain": "rl"},
            ]
        )


def test_validate_repos_accepts_every_known_role_and_domain():
    from src import config as cfg

    cfg._validate_repos(
        [
            {"slug": f"o/{role}-{domain}", "role": role, "domain": domain}
            for role in cfg._VALID_ROLES
            for domain in cfg._VALID_DOMAINS
        ]
    )


def test_live_repos_all_pass_validation():
    """The real config.REPOS list itself passed `_validate_repos` at import time (this module
    would have failed to import otherwise) -- this test just documents that guarantee."""
    from src import config as cfg

    cfg._validate_repos(cfg.REPOS)  # re-run explicitly; raises on regression
