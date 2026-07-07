"""Tests for the Scout CLI (T2.5) — offline & deterministic.

``main()`` on a tmp store prints the ranked candidate queue. ``--per-domain-limit`` (T3.19
support) bounds a dry run over a KB with many matching issues.
"""

import pytest

from src import candidates, llm, parity
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m2


def _item(repo: str, number: int, title: str, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": title,
        "state": "open",
        "labels": ["good first issue"],
        "url": f"http://x/{repo}/{number}",
        "body": "",
    }
    rec.update(overrides)
    return rec


@pytest.fixture(autouse=True)
def _config_repos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        parity.config, "REPOS", [{"slug": "o/r", "role": "primary", "domain": "speech"}]
    )


def _reply(risk="low", effort="low", impact="low") -> dict:
    return {"risk": risk, "effort": effort, "impact": impact}


def test_main_prints_ranked_candidates(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "fix typo")])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())

    rc = candidates.main(["--data-dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("1 candidate(s)")
    assert "good-first-issue" in out


def test_main_no_candidates_prints_zero(tmp_path, capsys) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([])
    rc = candidates.main(["--data-dir", str(tmp_path)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "0 candidate(s)"


def test_main_per_domain_limit_caps_candidates(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """T3.19: `--per-domain-limit` bounds a dry run over a KB with many matching issues."""
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", n, f"fix typo {n}") for n in range(1, 6)])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())

    rc = candidates.main(["--data-dir", str(tmp_path), "--per-domain-limit", "2"])

    assert rc == 0
    assert capsys.readouterr().out.strip().startswith("2 candidate(s)")


def test_main_per_domain_limit_rejects_non_positive(capsys) -> None:
    with pytest.raises(SystemExit):
        candidates.main(["--per-domain-limit", "0"])
    assert "must be a positive int" in capsys.readouterr().err
