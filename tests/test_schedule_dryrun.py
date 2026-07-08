"""Scheduler dry-run test (T4.2) — offline & deterministic.

Per the DEVPLAN todo: a dry-run tick runs end-to-end without error, with every real agent
CLI stubbed out — this is what `scripts/orchestrator.sh` invokes under cron/tmux on
ce-master, so `python -m src.orchestrator --once` itself (argument parsing, store
resolution, `run_tick`, the printed summary) has to work, not just `run_tick` in isolation
(already covered by `tests/test_orchestrator.py`).
"""

from __future__ import annotations

import pytest

from src import analyze as analyze_module
from src import candidates as candidates_module
from src import collector as collector_module
from src import forecast as forecast_module
from src import orchestrator, policy, taxonomy
from src import report as report_module
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m4


@pytest.fixture
def store(tmp_path) -> JsonlStore:
    """A KB with an active policy -- the minimum `run_tick` needs to pin a version at all."""
    s = JsonlStore(tmp_path)
    taxonomy.create_taxonomy(s, ["rocm-build"])
    policy.create_policy(s, scoring_weights={}, prompt_templates={}, active_taxonomy_version=1)
    return s


@pytest.fixture
def _stub_real_agents(monkeypatch: pytest.MonkeyPatch, store: JsonlStore) -> list[str]:
    """Stub every real agent CLI `main()` `_real_stages()` wires up, recording call order, and
    point `orchestrator.get_store()` at the fixture's scratch KB -- so a dry-run tick never
    touches the network, an LLM, or the ambient (real) data dir."""
    calls: list[str] = []
    monkeypatch.setattr(orchestrator, "get_store", lambda: store)
    monkeypatch.setattr(collector_module, "main", lambda argv: calls.append("collect") or None)
    monkeypatch.setattr(analyze_module, "main", lambda argv: calls.append("analyze") or 0)
    monkeypatch.setattr(forecast_module, "main", lambda argv: calls.append("forecast") or 0)
    monkeypatch.setattr(report_module, "main", lambda argv: calls.append("report") or 0)
    monkeypatch.setattr(candidates_module, "main", lambda argv: calls.append("candidates") or 0)
    return calls


def test_orchestrator_cli_dry_run_completes_without_error(
    store: JsonlStore, capsys, _stub_real_agents: list[str]
) -> None:
    """`python -m src.orchestrator --once` end-to-end: a fresh KB's first tick runs collect +
    intel + contribution (all due/triggered on a never-run store), stubbed agents included,
    and exits 0 with a summary line -- no network/LLM call anywhere in this path."""
    store.upsert_items(
        [
            {
                "repo": "o/r",
                "number": 1,
                "type": "issue",
                "state": "open",
                "title": "fix",
                "labels": ["good first issue"],
                "url": "http://x/o/r/1",
                "body": "",
            }
        ]
    )

    rc = orchestrator.main(["--once"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "policy@1" in out
    assert set(_stub_real_agents) == {"collect", "analyze", "forecast", "report", "candidates"}


def test_orchestrator_cli_dry_run_second_tick_skips_undue_cadence_stages(
    capsys, _stub_real_agents: list[str]
) -> None:
    """A second tick immediately after the first must not re-run collect/intel (not due
    yet) -- only re-runs contribution if it's still triggered."""
    orchestrator.main(["--once"])
    capsys.readouterr()  # discard the first tick's output; only the second tick is asserted on
    _stub_real_agents.clear()

    rc = orchestrator.main(["--once"])

    out = capsys.readouterr().out
    assert rc == 0
    assert _stub_real_agents == []
    # Check "collect"/"intel" specifically appear in the printed `skipped` dict, not just that
    # the word "skipped" is present -- main()'s summary line always contains "skipped: " even
    # when the dict is empty, so that substring alone wouldn't prove either stage was skipped.
    assert "'collect':" in out
    assert "'intel':" in out


def test_orchestrator_cli_without_once_prints_usage_and_exits_nonzero(capsys) -> None:
    rc = orchestrator.main([])
    assert rc == 2
    assert "only --once is supported" in capsys.readouterr().err
