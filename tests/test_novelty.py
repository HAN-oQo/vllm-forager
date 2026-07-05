"""Tests for the novelty / dedup filter (T2.7) — offline & deterministic.

Per the DEVPLAN todo: a near-duplicate candidate is rejected; a genuinely new one passes
(embedding + judge mocked).
"""

import pytest

from src import llm, novelty
from src.novelty import NoveltyError, PriorAttempt, check_novelty

pytestmark = pytest.mark.m2


def _fake_embed_texts(vectors_by_text: dict[str, list[float]]):
    def _embed(texts: list[str]) -> list[list[float]]:
        return [vectors_by_text[text] for text in texts]

    return _embed


_PRIOR = PriorAttempt(title="Port fp8 KV cache to upstream", body="", evidence="https://x/1")


def _mock_embeddings(
    monkeypatch: pytest.MonkeyPatch, *, candidate_text: str, similarity: str
) -> None:
    """`similarity` is "near" (identical vectors) or "far" (orthogonal vectors)."""
    prior_text = novelty._candidate_text(_PRIOR.title, _PRIOR.body)
    if similarity == "near":
        vectors = {candidate_text: [1.0, 0.0], prior_text: [1.0, 0.0]}
    else:
        vectors = {candidate_text: [1.0, 0.0], prior_text: [0.0, 1.0]}
    monkeypatch.setattr(novelty.embed, "embed_texts", _fake_embed_texts(vectors))


# --------------------------------------------------------------------- check_novelty


def test_rejects_near_duplicate_by_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    title, body = "Port fp8 KV cache", ""
    candidate_text = novelty._candidate_text(title, body)
    _mock_embeddings(monkeypatch, candidate_text=candidate_text, similarity="near")

    def _fail_if_called(*a: object, **k: object) -> None:
        raise AssertionError("judge should not be called when embedding already rejects")

    monkeypatch.setattr(llm, "complete", _fail_if_called)

    verdict = check_novelty(title, body, [_PRIOR])

    assert verdict.is_novel is False
    assert verdict.similar_to == _PRIOR
    assert "embedding similarity" in verdict.reason


def test_passes_genuinely_new_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    title, body = "Add speculative decoding support", ""
    candidate_text = novelty._candidate_text(title, body)
    _mock_embeddings(monkeypatch, candidate_text=candidate_text, similarity="far")
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"redundant": False})

    verdict = check_novelty(title, body, [_PRIOR])

    assert verdict.is_novel is True
    assert verdict.similar_to is None
    assert verdict.reason == "novel"


def test_rejected_by_judge_when_embedding_score_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    title, body = "Port fp8 KV cache (reworded)", ""
    candidate_text = novelty._candidate_text(title, body)
    _mock_embeddings(monkeypatch, candidate_text=candidate_text, similarity="far")
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"redundant": True})

    verdict = check_novelty(title, body, [_PRIOR])

    assert verdict.is_novel is False
    assert verdict.similar_to == _PRIOR
    assert verdict.reason == "LLM novelty judge ruled it redundant"


def test_no_prior_attempts_is_trivially_novel(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_if_called(*a: object, **k: object) -> None:
        raise AssertionError("embedding/judge should not run with no prior attempts")

    monkeypatch.setattr(novelty.embed, "embed_texts", _fail_if_called)
    monkeypatch.setattr(llm, "complete", _fail_if_called)

    verdict = check_novelty("Anything", "", [])

    assert verdict.is_novel is True
    assert verdict.similar_to is None


def test_judge_failure_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    title, body = "Port fp8 KV cache (reworded)", ""
    candidate_text = novelty._candidate_text(title, body)
    _mock_embeddings(monkeypatch, candidate_text=candidate_text, similarity="far")

    def _raise(*a: object, **k: object) -> None:
        raise llm.LLMError("provider unavailable")

    monkeypatch.setattr(llm, "complete", _raise)

    verdict = check_novelty(title, body, [_PRIOR])

    assert verdict.is_novel is True


def test_judge_malformed_reply_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    title, body = "Port fp8 KV cache (reworded)", ""
    candidate_text = novelty._candidate_text(title, body)
    _mock_embeddings(monkeypatch, candidate_text=candidate_text, similarity="far")
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"unexpected": "shape"})

    verdict = check_novelty(title, body, [_PRIOR])

    assert verdict.is_novel is True


def test_most_similar_picks_best_match_among_several(monkeypatch: pytest.MonkeyPatch) -> None:
    title, body = "Add speculative decoding support", ""
    candidate_text = novelty._candidate_text(title, body)
    near = PriorAttempt(title="Add speculative decoding", body="")
    far = PriorAttempt(title="Unrelated memory leak fix", body="")
    vectors = {
        candidate_text: [1.0, 0.0],
        novelty._candidate_text(near.title, near.body): [1.0, 0.0],
        novelty._candidate_text(far.title, far.body): [0.0, 1.0],
    }
    monkeypatch.setattr(novelty.embed, "embed_texts", _fake_embed_texts(vectors))

    verdict = check_novelty(title, body, [far, near])

    assert verdict.is_novel is False
    assert verdict.similar_to == near


def test_similarity_threshold_out_of_range_raises() -> None:
    with pytest.raises(NoveltyError, match=r"similarity_threshold must be in \[0, 1\]"):
        check_novelty("x", "", [_PRIOR], similarity_threshold=1.5)
