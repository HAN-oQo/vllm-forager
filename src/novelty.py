"""Novelty / dedup filter before expensive MI250 evaluation (T2.7).

Borrowed from ShinkaEvolve: the single biggest sample-efficiency lever for an evolutionary/
iterative search over expensive candidates is not re-attempting one that's already been tried.
Here, "expensive" means an MI250 build + verify (M3's Engineer, T3.2/T3.3) — this module is the
gate in front of that cost, not a replacement for it. It answers one question per candidate:
*"has something like this already been tried?"* — via two independent, either-one-rejects
signals, exactly matching the DEVPLAN's own "near-duplicates ... (embedding similarity ≥
threshold) **or** an LLM-as-novelty-judge rules it redundant" framing:

- **Embedding similarity** (:mod:`src.embed`) against every :class:`PriorAttempt` — cheap,
  catches lexically near-identical candidates (the common case: the same issue/PR re-surfacing,
  a trivially-reworded title). Checked first since it's a single batched embedding call with
  no LLM cost.
- **LLM-as-novelty-judge** — checked only against the *single most similar* prior attempt (not
  every one — one call per candidate, the same per-item-cost shape T2.5's Scout already
  established), and only if the embedding score didn't already clear the threshold. Catches a
  semantically-duplicate-but-lexically-different candidate the embedding signal alone would
  miss (paraphrased title, different framing of the same underlying fix).

Like T2.6's bandit, this module is a standalone, store-agnostic algorithm tested against
synthetic history (see the DEVPLAN's own Test bullet), not yet wired to a real prior-attempt
log — M3's Engineer (T3.2/T3.3, not yet built) is what would actually persist each build
attempt as a :class:`PriorAttempt` and call :func:`check_novelty` before spending MI250 time on
the next one. Designing that persistence now, against an unbuilt consumer's unspecified shape,
would be premature (the same judgment call the DEVPLAN itself makes for T2.6 → T3.x).

A judge-call failure (``llm.LLMError`` or a malformed reply) fails **open** — the candidate is
not rejected on the judge's account, only logged — matching every sibling agent's per-item LLM
failure isolation (grader/curator/scout): a judge outage should not silently block real
candidates from ever reaching the (already-gating) human review, only lose the extra signal the
judge would have added on top of the embedding check.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from . import embed, llm

# Illustrative, tunable similarity cutoff (see DEVPLAN's own "0.95-similar" example) — not
# derived from any measured embedding distribution. With the default "hash" embed provider
# (lexical bag-of-words), near-identical text scores well above this; genuinely different
# candidates score well below it.
DEFAULT_SIMILARITY_THRESHOLD = 0.9

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"redundant": {"type": "boolean"}},
    "required": ["redundant"],
}


class NoveltyError(RuntimeError):
    """``similarity_threshold`` was outside ``[0, 1]``."""


@dataclass(frozen=True)
class PriorAttempt:
    """One earlier contribution attempt to check a new candidate against.

    `evidence` (optional) is a source link for whichever attempt this was (e.g. the PR the
    prior attempt produced) — carried through to :class:`NoveltyVerdict` so a caller can cite
    *why* a candidate was rejected, not just that it was, matching the CLAUDE.md evidence
    principle.
    """

    title: str
    body: str = ""
    evidence: str = ""


@dataclass(frozen=True)
class NoveltyVerdict:
    """The outcome of :func:`check_novelty`.

    `similar_to` is the matching :class:`PriorAttempt` when `is_novel` is `False` (via either
    signal); `None` when there was nothing to compare against or nothing matched.
    """

    is_novel: bool
    reason: str
    similar_to: PriorAttempt | None = None


def _candidate_text(title: str, body: str) -> str:
    return f"{title}\n\n{body}".strip()


def _most_similar(
    candidate_text: str, prior_attempts: list[PriorAttempt]
) -> tuple[PriorAttempt, float] | None:
    """The `PriorAttempt` whose text is most cosine-similar to `candidate_text`, and that
    score — `None` if `prior_attempts` is empty. One batched embedding call covers the
    candidate and every prior attempt together, rather than one call per comparison."""
    if not prior_attempts:
        return None
    texts = [candidate_text] + [_candidate_text(p.title, p.body) for p in prior_attempts]
    vectors = embed.embed_texts(texts)
    candidate_vector, prior_vectors = vectors[0], vectors[1:]
    scored = [
        (attempt, embed.cosine_similarity(candidate_vector, vector))
        for attempt, vector in zip(prior_attempts, prior_vectors, strict=True)
    ]
    return max(scored, key=lambda pair: pair[1])


def _judge_prompt(title: str, body: str, prior: PriorAttempt) -> str:
    return (
        "Two vLLM/ROCm open-source contribution candidates below. Has candidate A already been "
        "tried, in substance, by candidate B — same underlying fix/port, even if worded "
        "differently? Reply with `redundant` (boolean).\n\n"
        f"Candidate A (new): {title}\n\n{body[:2000]}\n\n"
        f"Candidate B (prior attempt): {prior.title}\n\n{prior.body[:2000]}"
    )


def _judge_redundant(title: str, body: str, prior: PriorAttempt) -> bool | None:
    """Whether the LLM judge rules `prior` covers `title`/`body` already — `None` if the call
    failed or the reply didn't shape into a boolean (see module docstring: fails open)."""
    try:
        reply = llm.complete(_judge_prompt(title, body, prior), json_schema=_JUDGE_SCHEMA)
    except llm.LLMError as exc:
        print(f"novelty: judge call failed for {title!r}: {exc}", file=sys.stderr)
        return None
    if not isinstance(reply, dict):
        return None
    redundant = reply.get("redundant")
    return redundant if isinstance(redundant, bool) else None


def check_novelty(
    title: str,
    body: str,
    prior_attempts: list[PriorAttempt],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> NoveltyVerdict:
    """Whether a `title`/`body` candidate is novel against `prior_attempts` — see module
    docstring for the two either-one-rejects signals.

    Raises:
        NoveltyError: `similarity_threshold` isn't in ``[0, 1]``.
    """
    if not 0.0 <= similarity_threshold <= 1.0:
        raise NoveltyError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")

    match = _most_similar(_candidate_text(title, body), prior_attempts)
    if match is None:
        return NoveltyVerdict(is_novel=True, reason="no prior attempts to compare against")

    prior, score = match
    if score >= similarity_threshold:
        return NoveltyVerdict(
            is_novel=False,
            reason=f"embedding similarity {score:.2f} >= threshold {similarity_threshold:.2f}",
            similar_to=prior,
        )

    if _judge_redundant(title, body, prior):
        return NoveltyVerdict(
            is_novel=False, reason="LLM novelty judge ruled it redundant", similar_to=prior
        )

    return NoveltyVerdict(is_novel=True, reason="novel")
