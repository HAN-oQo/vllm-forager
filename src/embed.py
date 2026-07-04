"""Pluggable text embeddings + brute-force nearest-neighbor search (T1.1).

Every downstream RAG/classification stage needs "embed some text, find the nearest items"
without caring which embedding backend produced the vectors — mirrors the ``src/llm.py``
pattern: one contract (:func:`embed_texts`), dispatch by env var, agents never hardcode a
provider.

===============  ================================================  =========================
``EMBED_PROVIDER``  backend                                        config
===============  ================================================  =========================
``hash``          deterministic offline bag-of-words hashing        none — always works
``local``         OpenAI-compatible ``/embeddings`` (vLLM server)   ``LLM_BASE_URL``
===============  ================================================  =========================

``hash`` is the default: it needs no credentials or running server, so the pipeline always
has *some* embedding signal from the start. It captures only lexical overlap (not semantics)
— swap to ``local`` (a real embedding model served over vLLM) once one is available, without
touching any caller.

The nearest-neighbor index (:class:`EmbedIndex`) is an in-memory brute-force cosine search —
a placeholder for a real vector store (pgvector or LanceDB; see the open question in
``docs/CONTEXT.md``). Brute force is O(n) per query, which is plenty fast at M1's item volume
and keeps this module dependency-free until a real backend is chosen.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # works even if python-dotenv isn't installed
    pass

DEFAULT_PROVIDER = "hash"
HASH_DIM = 256  # fixed vector width for the "hash" provider


class EmbedError(RuntimeError):
    """Any embedding failure: unknown provider, missing config, or a transport error."""


def _env_float(name: str, default: float) -> float:
    """Read a float env var; raise :class:`EmbedError` (not a bare ValueError) if malformed."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise EmbedError(f"{name}={raw!r} is not a valid number") from exc


def _embed_hash(texts: list[str]) -> list[list[float]]:
    """Deterministic, dependency-free bag-of-words hashing embedding.

    Each whitespace token is hashed into one of ``HASH_DIM`` buckets; the resulting
    bag-of-words vector is L2-normalized so cosine similarity behaves like a (coarse)
    lexical-overlap signal. Same text always produces the same vector.
    """
    vectors = []
    for text in texts:
        vec = [0.0] * HASH_DIM
        for token in text.lower().split():
            idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % HASH_DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


def _embed_local(texts: list[str]) -> list[list[float]]:
    """Call an OpenAI-compatible ``/embeddings`` endpoint (a local vLLM embedding server)."""
    base = os.getenv("LLM_BASE_URL")
    if not base:
        raise EmbedError("local provider requires LLM_BASE_URL (OpenAI-compatible endpoint)")
    model = os.getenv("EMBED_MODEL") or "default"
    headers = {"content-type": "application/json"}
    key = os.getenv("LLM_API_KEY")  # optional — vLLM can be run with an --api-key
    if key:
        headers["Authorization"] = f"Bearer {key}"
    timeout = _env_float("EMBED_TIMEOUT", 30.0)
    try:
        resp = requests.post(
            f"{base.rstrip('/')}/embeddings",
            headers=headers,
            json={"model": model, "input": texts},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:  # covers HTTPError, Timeout, ConnectionError
        raise EmbedError(f"HTTP request to {base} failed: {exc}") from exc
    try:
        data = resp.json()
        # Sort by the OpenAI-compatible response's own `index` field rather than trusting
        # list order — a server that reorders/batches results would otherwise silently
        # misalign vectors against the requested texts (and downstream ids in build_index).
        items = sorted(data["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in items]
    except (KeyError, ValueError, TypeError) as exc:
        raise EmbedError(f"local endpoint returned an unexpected shape: {exc}") from exc
    if len(vectors) != len(texts):
        raise EmbedError(
            f"local endpoint returned {len(vectors)} embeddings for {len(texts)} inputs"
        )
    return vectors


# Provider dispatch table — the single place that maps EMBED_PROVIDER → implementation.
_PROVIDERS: dict[str, Callable[[list[str]], list[list[float]]]] = {
    "hash": _embed_hash,
    "local": _embed_local,
}


def embed_texts(texts: list[str], *, provider: str | None = None) -> list[list[float]]:
    """Embed a batch of texts into vectors.

    Args:
        texts: Texts to embed (e.g. issue/PR titles + bodies, or taxonomy labels).
        provider: Override ``EMBED_PROVIDER`` for this call (mainly for tests).

    Returns:
        One vector per input text, same order. ``[]`` for empty input.

    Raises:
        EmbedError: unknown provider, missing config (``local`` without ``LLM_BASE_URL``),
            or a transport/shape error from the backend.
    """
    resolved = provider or os.getenv("EMBED_PROVIDER") or DEFAULT_PROVIDER
    fn = _PROVIDERS.get(resolved)
    if fn is None:
        raise EmbedError(
            f"unknown EMBED_PROVIDER {resolved!r}; expected one of {sorted(_PROVIDERS)}"
        )
    if not texts:
        return []
    return fn(texts)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors; ``0.0`` if either is all-zero."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class EmbedIndex:
    """An in-memory brute-force nearest-neighbor index: parallel ids + vectors.

    Validated in ``__post_init__`` (not just the ``build_index`` factory) so there is exactly
    one way to end up with a well-formed index, however it's constructed.

    Raises:
        EmbedError: ``ids``/``vectors`` have different lengths, or the vectors don't all
            share one dimensionality.
    """

    ids: list[str]
    vectors: list[list[float]]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.vectors):
            raise EmbedError(
                f"ids ({len(self.ids)}) and vectors ({len(self.vectors)}) length mismatch"
            )
        dims = {len(vec) for vec in self.vectors}
        if len(dims) > 1:
            raise EmbedError(f"vectors have inconsistent dimensionality: {sorted(dims)}")


def build_index(ids: list[str], vectors: list[list[float]]) -> EmbedIndex:
    """Build an :class:`EmbedIndex` from parallel id/vector lists (see its own validation)."""
    return EmbedIndex(ids=list(ids), vectors=list(vectors))


def nearest(index: EmbedIndex, query_vector: list[float], k: int = 5) -> list[tuple[str, float]]:
    """Return the top-``k`` ``(id, similarity)`` pairs in ``index``, sorted by similarity desc.

    Raises:
        EmbedError: ``k`` is negative.
    """
    if k < 0:
        raise EmbedError(f"k must be >= 0, got {k}")
    scored = [
        (id_, cosine_similarity(query_vector, vec))
        for id_, vec in zip(index.ids, index.vectors, strict=True)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def search(
    index: EmbedIndex,
    query_text: str,
    *,
    embed_fn: Callable[[list[str]], list[list[float]]] = embed_texts,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Embed ``query_text`` with ``embed_fn`` and return its top-``k`` neighbors in ``index``.

    Raises:
        EmbedError: ``embed_fn`` returns something other than exactly one vector.
    """
    vectors = embed_fn([query_text])
    if len(vectors) != 1:
        raise EmbedError(f"embed_fn returned {len(vectors)} vectors for 1 query text")
    return nearest(index, vectors[0], k=k)
