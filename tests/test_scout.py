"""Tests for the Scout agent (T2.5) — offline & deterministic.

Per the DEVPLAN todo: fixture items → every ranked candidate carries risk, effort, and impact
scores + evidence present; ranking reflects the combination of all three, not risk alone (a
low-risk/low-impact item doesn't outrank a medium-risk/high-impact one).
"""

import pytest

from src import llm, parity
from src.agents import scout
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m2


def _item(repo: str, number: int, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": "x",
        "state": "open",
        "labels": [],
        "url": f"https://github.com/{repo}/issues/{number}",
    }
    rec.update(overrides)
    return rec


def _reply(risk="low", effort="low", impact="low") -> dict:
    return {"risk": risk, "effort": effort, "impact": impact}


# --------------------------------------------------------------------- discover_candidates


def test_discover_candidates_from_parity_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    gap = parity.Gap(
        capability="fp8-kv-cache",
        target_engine="vllm-project/vllm-omni",
        source_engine="sgl-project/sglang",
        evidence="https://github.com/sgl-project/sglang/pull/1",
    )

    candidates = scout.discover_candidates([], [gap])

    assert len(candidates) == 1
    assert candidates[0].source == "parity-gap"
    assert candidates[0].evidence == "https://github.com/sgl-project/sglang/pull/1"
    assert "fp8-kv-cache" in candidates[0].title


def test_discover_candidates_skips_gap_with_no_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a Gap with evidence=None (parity.py's own documented case when no shipped
    item's URL resolves) used to produce a Candidate(evidence="") -- a fabricated-empty
    citation, not real evidence. Such a gap is now skipped instead of ranked."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    gap = parity.Gap(
        capability="fp8-kv-cache",
        target_engine="vllm-project/vllm-omni",
        source_engine="sgl-project/sglang",
        evidence=None,
    )

    assert scout.discover_candidates([], [gap]) == []


def test_discover_candidates_good_first_issue_hyphenated_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="fix typo", labels=["good-first-issue"])]

    candidates = scout.discover_candidates(items)

    assert len(candidates) == 1
    assert candidates[0].source == "good-first-issue"


def test_discover_candidates_good_first_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="fix typo", labels=["good first issue"])]

    candidates = scout.discover_candidates(items)

    assert len(candidates) == 1
    assert candidates[0].source == "good-first-issue"
    assert candidates[0].evidence == "https://github.com/o/r/issues/1"


def test_discover_candidates_rocm_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="hipBLAS build fails on gfx90a")]

    candidates = scout.discover_candidates(items)

    assert len(candidates) == 1
    assert candidates[0].source == "rocm-reproducible"


def test_discover_candidates_ignores_closed_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="hipBLAS bug on gfx90a", state="closed")]

    assert scout.discover_candidates(items) == []


def test_discover_candidates_ignores_prs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="hipBLAS fix on gfx90a", type="pr")]

    assert scout.discover_candidates(items) == []


def test_discover_candidates_ignores_unrelated_open_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="typo in docs")]

    assert scout.discover_candidates(items) == []


def test_discover_candidates_dedupes_across_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """An item matching both good-first-issue AND rocm-reproducible is only counted once."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="hipBLAS bug on gfx90a", labels=["good first issue"])]

    candidates = scout.discover_candidates(items)

    assert len(candidates) == 1
    assert candidates[0].source == "good-first-issue"  # checked first


def test_discover_candidates_ranking_reflects_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEVPLAN's named scenario: a low-risk/low-impact item doesn't outrank a
    medium-risk/high-impact one."""

    def fake_complete(prompt: str, **kwargs) -> dict:
        if "low-impact-item" in prompt:
            return _reply(risk="low", effort="low", impact="low")
        return _reply(risk="medium", effort="high", impact="high")

    monkeypatch.setattr(llm, "complete", fake_complete)
    items = [
        _item("o/r", 1, title="low-impact-item hipBLAS", labels=["good first issue"]),
        _item("o/r", 2, title="high-impact-item hipBLAS", labels=["good first issue"]),
    ]

    candidates = scout.discover_candidates(items)

    assert [c.title for c in candidates] == ["high-impact-item hipBLAS", "low-impact-item hipBLAS"]


def test_discover_candidates_skips_llm_failure_keeps_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def flaky_complete(prompt: str, **kwargs) -> dict:
        if "bad" in prompt:
            raise llm.LLMError("simulated transient failure")
        return _reply()

    monkeypatch.setattr(llm, "complete", flaky_complete)
    items = [
        _item("o/r", 1, title="bad hipBLAS issue", labels=["good first issue"]),
        _item("o/r", 2, title="good hipBLAS issue", labels=["good first issue"]),
    ]

    candidates = scout.discover_candidates(items)

    assert len(candidates) == 1
    assert candidates[0].title == "good hipBLAS issue"


def test_discover_candidates_skips_malformed_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"risk": "extreme"})
    items = [_item("o/r", 1, title="hipBLAS bug", labels=["good first issue"])]

    assert scout.discover_candidates(items) == []


def test_discover_candidates_empty_returns_empty() -> None:
    assert scout.discover_candidates([]) == []


# --------------------------------------------------------------------- T3.16 boost signals


def test_discover_candidates_rocm_speech_outranks_non_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEVPLAN's T3.16 scenario: a `[ROCm] whisper decode` vllm issue outranks a generic
    feature — same LLM scores, so only the ROCm∩speech boost can explain the order."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [
        _item(
            "vllm-project/vllm",
            1,
            title="generic feature request",
            labels=["good first issue"],
        ),
        _item(
            "vllm-project/vllm",
            2,
            title="[ROCm] whisper decode is broken on gfx90a",
            labels=["good first issue"],
        ),
    ]

    candidates = scout.discover_candidates(items)

    assert [c.title for c in candidates] == [
        "[ROCm] whisper decode is broken on gfx90a",
        "generic feature request",
    ]


def test_discover_candidates_rocm_speech_boost_needs_speech_domain_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ROCm∩speech boost only applies on a tracked `"speech"`-domain repo (vllm) -- the
    same matching text on an untracked repo gets no boost."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="[ROCm] whisper decode is broken on gfx90a")]

    candidates = scout.discover_candidates(items)

    assert len(candidates) == 1
    assert candidates[0].boost == 0


def test_discover_candidates_merge_velocity_affects_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEVPLAN's T3.16 scenario: merge-velocity affects order -- two otherwise-identical
    candidates on different repos rank by which repo merges PRs faster."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [
        _item(
            "fast/repo",
            1,
            type="pr",
            state="closed",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
        ),
        _item(
            "slow/repo",
            1,
            type="pr",
            state="closed",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-06-01T00:00:00Z",
        ),
        _item("fast/repo", 2, title="hipBLAS bug on gfx90a"),
        _item("slow/repo", 2, title="hipBLAS bug on gfx90a"),
    ]

    candidates = scout.discover_candidates(items)

    assert [(c.evidence, c.boost) for c in candidates] == [
        ("https://github.com/fast/repo/issues/2", 1),
        ("https://github.com/slow/repo/issues/2", 0),
    ]


def test_discover_candidates_merge_velocity_needs_at_least_two_repos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a single repo's mean merge time is trivially its own median, so it must not
    win the merge-velocity boost by default -- even a genuinely slow, lone repo shouldn't look
    'faster than typical' just because there's nothing else in the batch to compare it to."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [
        _item(
            "only/repo",
            1,
            type="pr",
            state="closed",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",  # a full year to merge -- unambiguously slow
        ),
        _item("only/repo", 2, title="hipBLAS bug on gfx90a"),
    ]

    candidates = scout.discover_candidates(items)

    assert len(candidates) == 1
    assert candidates[0].boost == 0


def test_merge_velocity_by_repo_skips_unparseable_timestamps() -> None:
    """Regression: a non-str `created_at`/`updated_at` (e.g. a malformed record) must be
    skipped, not raise -- this used to crash `_parse_ts` with an uncaught AttributeError."""
    items = [
        _item(
            "o/r",
            1,
            type="pr",
            state="closed",
            created_at=1700000000,
            updated_at="2024-01-01T00:00:00Z",
        ),
        _item(
            "o/r",
            2,
            type="pr",
            state="closed",
            created_at="not-a-timestamp",
            updated_at="2024-01-01T00:00:00Z",
        ),
    ]

    assert scout._merge_velocity_by_repo(items) == {}


def test_merge_velocity_by_repo_treats_naive_timestamp_as_utc() -> None:
    """Regression: a `created_at`/`updated_at` pair where one has a `Z` suffix and the other
    doesn't used to raise `TypeError` on subtraction (naive vs. aware datetimes) -- both must
    parse as UTC so the pair can still be subtracted."""
    items = [
        _item(
            "o/r",
            1,
            type="pr",
            state="closed",
            created_at="2024-01-01T00:00:00",  # no "Z"
            updated_at="2024-01-02T00:00:00Z",
        )
    ]

    assert scout._merge_velocity_by_repo(items) == {"o/r": 1.0}


# --------------------------------------------------------------------- Candidate/priority


def test_candidate_rejects_invalid_level() -> None:
    with pytest.raises(scout.ScoutError, match="risk must be one of"):
        scout.Candidate(
            title="x", source="parity-gap", risk="extreme", effort="low", impact="low", evidence=""
        )


def test_priority_low_risk_low_impact_never_outranks_medium_risk_high_impact() -> None:
    low = scout.Candidate(
        title="a", source="x", risk="low", effort="low", impact="low", evidence=""
    )
    for effort in ("low", "medium", "high"):
        medium_high = scout.Candidate(
            title="b", source="x", risk="medium", effort=effort, impact="high", evidence=""
        )
        assert medium_high.priority > low.priority


def test_priority_boost_never_lets_low_impact_outrank_medium_risk_high_impact() -> None:
    """Regression: `boost` is a pure tiebreak between identically-scored candidates, not a
    lever that can flip a genuine risk/effort/impact difference -- this held even at boost's
    current maximum (+4) on the low side against boost=0 on the medium-risk/high-impact side,
    which a naive `+ self.boost` (rather than scaling risk/effort/impact first) did not."""
    low = scout.Candidate(
        title="a", source="x", risk="low", effort="low", impact="low", evidence="", boost=4
    )
    for effort in ("low", "medium", "high"):
        medium_high = scout.Candidate(
            title="b", source="x", risk="medium", effort=effort, impact="high", evidence="", boost=0
        )
        assert medium_high.priority > low.priority


def test_priority_boost_breaks_ties_between_identical_risk_effort_impact() -> None:
    """`boost` still does its job: among two candidates with the identical risk/effort/impact
    combination, the higher-boosted one ranks first."""
    plain = scout.Candidate(
        title="a", source="x", risk="low", effort="low", impact="medium", evidence="", boost=0
    )
    boosted = scout.Candidate(
        title="b", source="x", risk="low", effort="low", impact="medium", evidence="", boost=2
    )
    assert boosted.priority > plain.priority


# --------------------------------------------------------------------- discover_from_store


def test_discover_from_store(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("vllm-project/vllm", 1, title="hipBLAS bug on gfx90a", category="build"),
            _item(
                "sgl-project/sglang",  # a real, non-"primary" tracked engine -- no ROCm fork
                2,
                title="shipped fp8 kv-cache",
                type="pr",
                state="closed",
                category="quantization > FP8",
            ),
        ]
    )
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())

    candidates = scout.discover_from_store(store)

    sources = {c.source for c in candidates}
    assert "rocm-reproducible" in sources
    assert "parity-gap" in sources


def test_discover_from_store_degrades_gracefully_on_parity_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a config.REPOS misconfiguration (no primary role) used to crash the whole
    run via an uncaught parity.ParityError, losing the good-first-issue/rocm-reproducible
    sources too, even though neither depends on parity gaps at all."""
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, title="hipBLAS bug on gfx90a")])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    monkeypatch.setattr(parity.config, "REPOS", [{"slug": "o/r", "role": "source"}])

    candidates = scout.discover_from_store(store)

    assert len(candidates) == 1
    assert candidates[0].source == "rocm-reproducible"
