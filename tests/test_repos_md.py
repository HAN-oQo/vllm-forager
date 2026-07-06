"""Tests for scripts/repos-md.py (T3.18) — offline & deterministic.

The script's filename has a hyphen (matches the DEVPLAN todo's own naming), so it can't be
`import`ed as a module — it's exercised as a subprocess instead, same as any other CLI tool.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from src import config

pytestmark = pytest.mark.m0

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "repos-md.py"
PLAN_MD = REPO_ROOT / "docs" / "PLAN.md"


def _run_script() -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return result.stdout


def test_repos_md_lists_every_repo_with_its_role() -> None:
    """Every `config.REPOS` slug appears, tagged with its own role — not any other repo's."""
    output = _run_script()
    for repo in config.REPOS:
        assert f"`{repo['slug']}` — {repo['role']}" in output


def test_repos_md_groups_each_repo_under_its_own_domain() -> None:
    """Each repo's slug line appears after its own domain heading, not another domain's."""
    output = _run_script()
    for repo in config.REPOS:
        domain_idx = output.index(f"**{repo['domain']}**")
        slug_idx = output.index(f"`{repo['slug']}`")
        assert domain_idx < slug_idx


def test_repos_md_total_count_matches_config_repos() -> None:
    output = _run_script()
    assert f"{len(config.REPOS)} tracked repos" in output


def test_plan_md_embedded_block_matches_the_script_exactly() -> None:
    """Regression guardrail (the todo's own "so it can't drift"): `docs/PLAN.md`'s
    `<!-- repos-md:start -->`/`<!-- repos-md:end -->` block must be byte-identical to the
    script's current output — a `config.REPOS` change that isn't followed by regenerating and
    re-pasting this block now fails the test suite instead of silently going stale."""
    plan_text = PLAN_MD.read_text()
    start = plan_text.index("<!-- repos-md:start -->") + len("<!-- repos-md:start -->")
    end = plan_text.index("<!-- repos-md:end -->")
    embedded = plan_text[start:end].strip("\n") + "\n"

    assert embedded == _run_script()
