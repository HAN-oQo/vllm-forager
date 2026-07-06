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

_START = "<!-- repos-md:start -->"
_END = "<!-- repos-md:end -->"


@pytest.fixture(scope="module")
def script_output() -> str:
    """The script's stdout, computed once and shared by every test below (each spawns a fresh
    Python interpreter; the output is identical and deterministic across calls in a run, so
    there's nothing to gain from re-spawning it once per assertion)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return result.stdout


def test_repos_md_lists_every_repo_with_its_role(script_output: str) -> None:
    """Every `config.REPOS` slug appears, tagged with its own role — not any other repo's."""
    for repo in config.REPOS:
        assert f"`{repo['slug']}` — {repo['role']}" in script_output


def test_repos_md_groups_each_repo_under_its_own_domain(script_output: str) -> None:
    """Each repo's slug line appears after its own domain heading, not another domain's."""
    for repo in config.REPOS:
        domain_idx = script_output.index(f"**{repo['domain']}**")
        slug_idx = script_output.index(f"`{repo['slug']}`")
        assert domain_idx < slug_idx


def test_repos_md_total_count_matches_config_repos(script_output: str) -> None:
    assert f"{len(config.REPOS)} tracked repos" in script_output


def test_plan_md_has_exactly_one_marker_pair() -> None:
    """Regression guard for the drift-check below: if a future PLAN.md edit mentions the
    literal marker text again (e.g. prose explaining this mechanism), `.index()`-based
    extraction would silently lock onto the wrong occurrence instead of failing loudly."""
    plan_text = PLAN_MD.read_text()
    assert plan_text.count(_START) == 1
    assert plan_text.count(_END) == 1


def test_plan_md_embedded_block_matches_the_script_exactly(script_output: str) -> None:
    """Regression guardrail (the todo's own "so it can't drift"): `docs/PLAN.md`'s
    `<!-- repos-md:start -->`/`<!-- repos-md:end -->` block must be byte-identical to the
    script's current output — a `config.REPOS` change that isn't followed by regenerating and
    re-pasting this block (or running `python scripts/repos-md.py --sync`, which the
    `repos-md-sync` pre-commit hook already does automatically) now fails the test suite
    instead of silently going stale."""
    plan_text = PLAN_MD.read_text()
    start = plan_text.index(_START) + len(_START)
    end = plan_text.index(_END)
    embedded = plan_text[start:end].strip("\n") + "\n"

    assert embedded == script_output


# --------------------------------------------------------------------- --sync (pre-commit hook)


def _run_sync(target: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--sync", str(target)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_sync_rewrites_a_stale_block_and_exits_nonzero(tmp_path: Path, script_output: str) -> None:
    """The `repos-md-sync` pre-commit hook's whole point: a stale block gets rewritten in
    place, and the process exits non-zero (black/ruff's own "auto-fixed, re-stage" convention)
    so the commit doesn't silently succeed with stale content."""
    target = tmp_path / "PLAN.md"
    target.write_text(f"before\n\n{_START}\nSTALE CONTENT\n{_END}\n\nafter\n")

    result = _run_sync(target)

    assert result.returncode == 1
    new_text = target.read_text()
    assert "STALE CONTENT" not in new_text
    assert "before\n" in new_text and "\nafter\n" in new_text  # surrounding content preserved
    assert script_output.strip("\n") in new_text


def test_sync_is_a_noop_once_in_sync(tmp_path: Path) -> None:
    """A second `--sync` run against an already-current file changes nothing and exits 0."""
    target = tmp_path / "PLAN.md"
    target.write_text(f"before\n\n{_START}\nSTALE CONTENT\n{_END}\n\nafter\n")
    _run_sync(target)  # first run: rewrites it (exit 1) -- see test above
    before = target.read_text()

    result = _run_sync(target)

    assert result.returncode == 0
    assert target.read_text() == before


def test_sync_raises_on_missing_or_duplicate_markers(tmp_path: Path) -> None:
    target = tmp_path / "PLAN.md"
    target.write_text("no markers here at all\n")

    result = _run_sync(target)

    assert result.returncode != 0
    assert "expected exactly one" in result.stderr
