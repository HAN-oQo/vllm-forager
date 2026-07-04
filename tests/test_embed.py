"""Tests for pluggable embeddings + NN search (T1.1) — offline & deterministic.

The ``local`` provider is mocked at its transport boundary (``requests.post``); the ``hash``
provider needs no mocking since it's already offline. NN correctness is checked with a
deterministic fixture "model" (a hand-built embed_fn), per the DEVPLAN todo: "with a
deterministic fixture/mock model, NN of a query returns the semantically closer of two docs."
"""

import pytest
import requests

from src import embed

pytestmark = pytest.mark.m1


class _FakeResp:
    """Minimal stand-in for a requests.Response (json / raise_for_status)."""

    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self) -> dict:
        return self._payload


def test_embed_hash_deterministic_and_shaped() -> None:
    vectors = embed.embed_texts(["hipBLAS build fails on gfx90a", "unrelated cat picture"])
    assert len(vectors) == 2
    assert all(len(v) == embed.HASH_DIM for v in vectors)
    # same input -> same output, every time
    assert embed.embed_texts(["hipBLAS build fails on gfx90a"])[0] == vectors[0]


def test_embed_hash_empty_input() -> None:
    assert embed.embed_texts([]) == []


def test_embed_unknown_provider_raises() -> None:
    with pytest.raises(embed.EmbedError, match="unknown EMBED_PROVIDER"):
        embed.embed_texts(["x"], provider="not-a-real-provider")


def test_embed_local_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(embed.EmbedError, match="LLM_BASE_URL"):
        embed.embed_texts(["x"], provider="local")


def test_embed_local_calls_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    captured = {}

    def fake_post(url: str, headers: dict, json: dict, timeout: float) -> _FakeResp:
        captured["url"] = url
        captured["json"] = json
        return _FakeResp({"data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]})

    monkeypatch.setattr(requests, "post", fake_post)
    vectors = embed.embed_texts(["a", "b"], provider="local")
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert captured["url"] == "http://localhost:8000/v1/embeddings"
    assert captured["json"]["input"] == ["a", "b"]


def test_embed_local_bad_shape_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResp({"nope": True}))
    with pytest.raises(embed.EmbedError, match="unexpected shape"):
        embed.embed_texts(["x"], provider="local")


def test_cosine_similarity_identical_orthogonal_and_zero() -> None:
    assert embed.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert embed.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert embed.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_build_index_length_mismatch_raises() -> None:
    with pytest.raises(embed.EmbedError, match="length mismatch"):
        embed.build_index(["a", "b"], [[1.0, 0.0]])


def test_nearest_returns_closer_doc_first() -> None:
    # "rocm-build" doc and "unrelated" doc live in clearly separated directions; the query
    # sits close to "rocm-build" — NN must rank it first regardless of insertion order.
    index = embed.build_index(
        ids=["unrelated", "rocm-build"],
        vectors=[[0.0, 1.0], [1.0, 0.0]],
    )
    results = embed.nearest(index, query_vector=[0.9, 0.1], k=2)
    assert [id_ for id_, _ in results] == ["rocm-build", "unrelated"]
    assert results[0][1] > results[1][1]


def test_search_uses_embed_fn_and_orders_by_similarity() -> None:
    """A deterministic fixture "model": docs about the same topic as the query score higher."""

    corpus = {
        "hipblas-build-fail": "hipBLAS build fails to link on gfx90a MI250",
        "unrelated-typo": "fix a typo in the README",
    }
    ids = list(corpus)
    vectors = embed.embed_texts(list(corpus.values()), provider="hash")
    index = embed.build_index(ids, vectors)

    def fixture_model(texts: list[str]) -> list[list[float]]:
        return embed.embed_texts(texts, provider="hash")

    results = embed.search(index, "hipBLAS linker error on gfx90a", embed_fn=fixture_model, k=2)
    assert results[0][0] == "hipblas-build-fail"
    assert results[0][1] > results[1][1]
