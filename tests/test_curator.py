"""Tests for the Curator agent (T2.3) — offline & deterministic.

Per the DEVPLAN todo: a category with no activity for N weeks → flagged retire; a novel
cluster → proposed new category. Also covers weeks-since-active math, clustering behavior,
and per-cluster naming failure exclusion.
"""

from datetime import datetime, timezone

import pytest

from src import llm, taxonomy
from src.agents import curator
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m2

_NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)  # 2026-W23


def _item(repo: str, number: int, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": "x",
        "state": "open",
        "created_at": "2026-01-01T00:00:00Z",
        "url": f"https://github.com/{repo}/issues/{number}",
    }
    rec.update(overrides)
    return rec


# --------------------------------------------------------------------- weeks_since_active


def test_weeks_since_active_never_active_is_inf() -> None:
    assert curator.weeks_since_active("build", {}, now=_NOW) == float("inf")


def test_weeks_since_active_computes_gap() -> None:
    series = {"build": {"2026-W01": 3}}
    weeks = curator.weeks_since_active("build", series, now=_NOW)
    # 2026-W01 to 2026-W23 is 22 weeks
    assert weeks == pytest.approx(22, abs=1)


def test_weeks_since_active_recent_is_small() -> None:
    series = {"build": {"2026-W22": 1}}
    weeks = curator.weeks_since_active("build", series, now=_NOW)
    assert weeks < 2


# --------------------------------------------------------------------- propose_retirements


def test_propose_retirements_flags_inactive_and_spares_active(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    taxonomy.create_taxonomy(store, ["build", "quantization"])
    store.upsert_items(
        [
            # "build" active this week
            _item("o/r", 1, category="build", created_at="2026-05-30T00:00:00Z"),
            # "quantization" last active in W01 -- long stale by W23
            _item("o/r", 2, category="quantization", created_at="2026-01-02T00:00:00Z"),
        ]
    )

    proposals = curator.propose_retirements(store, inactive_weeks=8, now=_NOW)

    assert [p.category for p in proposals] == ["quantization"]
    assert proposals[0].weeks_inactive >= 8


def test_propose_retirements_never_active_category_flagged(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    taxonomy.create_taxonomy(store, ["build", "never-used"])
    store.upsert_items([_item("o/r", 1, category="build", created_at="2026-05-30T00:00:00Z")])

    proposals = curator.propose_retirements(store, inactive_weeks=8, now=_NOW)

    assert [p.category for p in proposals] == ["never-used"]
    assert proposals[0].weeks_inactive == float("inf")


def test_propose_retirements_none_when_all_active(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    taxonomy.create_taxonomy(store, ["build"])
    store.upsert_items([_item("o/r", 1, category="build", created_at="2026-05-30T00:00:00Z")])

    assert curator.propose_retirements(store, inactive_weeks=8, now=_NOW) == []


# --------------------------------------------------------------------- _cluster_items


def test_cluster_items_groups_similar_separates_dissimilar() -> None:
    items = [
        _item("o/r", 1, title="constrained decoding json schema grammar issue"),
        _item("o/r", 2, title="constrained decoding json schema grammar bug"),
        _item("o/r", 3, title="power throttle bios firmware voltage regression"),
        _item("o/r", 4, title="power throttle bios firmware voltage crash"),
    ]

    clusters = curator._cluster_items(items)

    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 2]


def test_cluster_items_empty_returns_empty() -> None:
    assert curator._cluster_items([]) == []


# --------------------------------------------------------------------- propose_new_categories


def _other_cluster(n: int, *, theme: str, repo: str = "o/r") -> list[dict]:
    return [_item(repo, i, title=f"{theme} issue {i}", category="Other") for i in range(1, n + 1)]


def test_propose_new_categories_names_large_cluster(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(_other_cluster(5, theme="constrained decoding json schema grammar"))
    monkeypatch.setattr(llm, "complete", lambda prompt, **kwargs: {"name": "structured-output"})

    proposals = curator.propose_new_categories(store, min_cluster_size=5)

    assert len(proposals) == 1
    assert proposals[0].name == "structured-output"
    assert proposals[0].size == 5
    assert len(proposals[0].evidence) == 5


def test_propose_new_categories_skips_cluster_below_min_size(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(_other_cluster(3, theme="constrained decoding json schema grammar"))
    monkeypatch.setattr(llm, "complete", lambda prompt, **kwargs: {"name": "structured-output"})

    assert curator.propose_new_categories(store, min_cluster_size=5) == []


def test_propose_new_categories_skips_unnamed_cluster(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(_other_cluster(5, theme="constrained decoding json schema grammar"))
    monkeypatch.setattr(llm, "complete", lambda prompt, **kwargs: {"name": ""})

    assert curator.propose_new_categories(store, min_cluster_size=5) == []


def test_propose_new_categories_ignores_non_other_items(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    items = _other_cluster(5, theme="constrained decoding json schema grammar")
    for item in items:
        item["category"] = "build"  # already classified -- not Other
    store.upsert_items(items)
    monkeypatch.setattr(llm, "complete", lambda prompt, **kwargs: {"name": "structured-output"})

    assert curator.propose_new_categories(store, min_cluster_size=5) == []


def test_propose_new_categories_no_other_items_returns_empty(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    assert curator.propose_new_categories(store) == []
