"""Novelty / dedup filter before expensive MI250 evaluation (T2.7).

Borrowed from ShinkaEvolve: the single biggest sample-efficiency lever for an evolutionary/
iterative search over expensive candidates is not re-attempting one that's already been tried.
Here, "expensive" means an MI250 build + verify (M3's Engineer, T3.2/T3.3) — this module is the
gate in front of that cost, not a replacement for it. It answers one question per candidate:
*"has something like this already been tried?"* — via two independent, either-one-rejects
signals, exactly matching the DEVPLAN's own "near-duplicates ... (embedding similarity ≥
threshold) **or** an LLM-as-novelty-judge rules it redundant" framing:

- **Embedding similarity** (:mod:`src.embed`) against every evidenced :class:`PriorAttempt` —
  cheap, catches lexically near-identical candidates (the common case: the same issue/PR
  re-surfacing, a trivially-reworded title). Checked first since it's a single batched embedding
  call with no LLM cost. Only :class:`PriorAttempt`\\ s carrying non-empty `evidence` are ever
  compared against — matching the CLAUDE.md evidence principle at the same decision point
  :func:`~src.agents.scout.discover_candidates` already applies it: a rejection a human can't
  trace to a source isn't one worth acting on. A candidate with no evidenced prior attempts to
  compare against is trivially novel.
- **LLM-as-novelty-judge** — checked only against the *single most similar* evidenced prior
  attempt (not every one — one call per candidate, the same per-item-cost shape T2.5's Scout
  already established), and only if the embedding score didn't already clear the threshold.
  Catches a semantically-duplicate-but-lexically-different candidate the embedding signal alone
  would miss (paraphrased title, different framing of the same underlying fix).

Both signals fail **open** on a backend outage — an ``embed.EmbedError`` (embedding backend
down/misconfigured) or an ``llm.LLMError``/malformed reply (judge backend down) never raises out
of :func:`check_novelty`, only logs and treats the candidate as novel for that signal — matching
every sibling agent's per-item failure isolation (grader/curator/scout): a backend outage should
not silently block real candidates from ever reaching the (already-gating) human review, only
lose that signal on top of whatever the other one still provides. :class:`NoveltyVerdict.reason`
distinguishes a judge-confirmed "novel" from "novel because the judge was unavailable" so a
caller/human can tell how much of a given verdict rests on a degraded signal.

Like T2.6's bandit, this module is a standalone, store-agnostic algorithm tested against
synthetic history (see the DEVPLAN's own Test bullet), not yet wired to a real prior-attempt
log — M3's Engineer (T3.2/T3.3, not yet built) is what would actually persist each build
attempt as a :class:`PriorAttempt` and call :func:`check_novelty` before spending MI250 time on
the next one. Designing that persistence now, against an unbuilt consumer's unspecified shape,
would be premature (the same judgment call the DEVPLAN itself makes for T2.6 → T3.x).

Known limitations, not fixed here (the same "premature to design against an unbuilt consumer"
judgment call as above):
- The judge is only ever consulted against the single most-similar-by-embedding prior attempt,
  not the top-K or the whole pool — a real duplicate that happens to score lower than an
  unrelated-but-lexically-overlapping prior attempt (plausible with the coarse "hash" bag-of-
  words provider) is never shown to the judge. Revisiting this needs a real, sizeable
  prior-attempt pool to even observe the failure mode against.
- :func:`_most_similar` re-embeds every evidenced prior attempt from scratch on every
  :func:`check_novelty` call (one batched call, but with no cross-call caching of prior-attempt
  vectors) — real cost once a real, growing prior-attempt log exists and this is called once per
  scouted candidate; no such caller exists yet to size the actual cost against.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from . import embed, llm

# Illustrative, tunable similarity cutoff (see DEVPLAN's own "0.95-similar" example) — not
# derived from any measured embedding distribution. With the default "hash" embed provider
# (lexical bag-of-words), near-identical text scores well above this; genuinely different
# candidates score well below it. A plain module constant, not a config.py setting — the same
# choice curator.py's own `_DEFAULT_SIMILARITY_THRESHOLD` already makes for its clustering cutoff.
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

    `evidence` is a source link for whichever attempt this was (e.g. the PR the prior attempt
    produced). An attempt with no `evidence` is never compared against — see module docstring —
    so it can never be why a candidate was rejected without a human being able to check the
    source (the CLAUDE.md evidence principle).
    """

    title: str
    body: str = ""
    evidence: str = ""


@dataclass(frozen=True)
class NoveltyVerdict:
    """The outcome of :func:`check_novelty`.

    `similar_to` is the matching :class:`PriorAttempt` when `is_novel` is `False` (via either
    signal); `None` when there was nothing evidenced to compare against or nothing matched.
    """

    is_novel: bool
    reason: str
    similar_to: PriorAttempt | None = None


def _candidate_text(title: str, body: str) -> str:
    """Text to embed for `title`/`body` — `body` is treated as `""` if falsy (e.g. `None`, as a
    GitHub issue/PR body routinely is), matching every sibling agent's own `item.get("body") or
    ""` normalization of the same field."""
    return f"{title}\n\n{body or ''}".strip()


def _most_similar(
    candidate_text: str, prior_attempts: list[PriorAttempt]
) -> tuple[PriorAttempt, float] | None:
    """The `PriorAttempt` whose text is most cosine-similar to `candidate_text`, and that
    score — `None` if `prior_attempts` is empty. One batched embedding call covers the
    candidate and every prior attempt together, rather than one call per comparison. Built on
    :func:`~src.embed.build_index`/:func:`~src.embed.nearest` (the same nearest-neighbor lookup
    :mod:`src.rag_eval` already uses for this exact "closest match to a query" shape), rather
    than a hand-rolled scan.

    Raises:
        embed.EmbedError: the embedding backend failed or is misconfigured (left to the caller
            to decide whether/how to fail open — see module docstring).
    """
    if not prior_attempts:
        return None
    texts = [candidate_text] + [_candidate_text(p.title, p.body) for p in prior_attempts]
    vectors = embed.embed_texts(texts)
    candidate_vector, prior_vectors = vectors[0], vectors[1:]
    index = embed.build_index(
        ids=[str(i) for i in range(len(prior_attempts))], vectors=prior_vectors
    )
    [(best_id, score)] = embed.nearest(index, candidate_vector, k=1)
    return prior_attempts[int(best_id)], score


def _judge_prompt(title: str, body: str, prior: PriorAttempt) -> str:
    return (
        "Two vLLM/ROCm open-source contribution candidates below. Has candidate A already been "
        "tried, in substance, by candidate B — same underlying fix/port, even if worded "
        "differently? Reply with `redundant` (boolean).\n\n"
        f"Candidate A (new): {title}\n\n{(body or '')[:2000]}\n\n"
        f"Candidate B (prior attempt): {prior.title}\n\n{(prior.body or '')[:2000]}"
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
    docstring for the two either-one-rejects signals, the evidence-principle filtering, and the
    embedding/judge fail-open behavior.

    Raises:
        NoveltyError: `similarity_threshold` isn't in ``[0, 1]``.
    """
    if not 0.0 <= similarity_threshold <= 1.0:
        raise NoveltyError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")

    evidenced = [p for p in prior_attempts if p.evidence]
    if not evidenced:
        return NoveltyVerdict(
            is_novel=True, reason="no evidenced prior attempts to compare against"
        )

    try:
        match = _most_similar(_candidate_text(title, body), evidenced)
    except embed.EmbedError as exc:
        print(f"novelty: embedding check failed for {title!r}: {exc}", file=sys.stderr)
        return NoveltyVerdict(is_novel=True, reason="embedding check unavailable, skipped")
    assert match is not None  # evidenced is non-empty, so _most_similar always finds one

    prior, score = match
    if score >= similarity_threshold:
        return NoveltyVerdict(
            is_novel=False,
            reason=f"embedding similarity {score:.2f} >= threshold {similarity_threshold:.2f}",
            similar_to=prior,
        )

    judge_verdict = _judge_redundant(title, body, prior)
    if judge_verdict:
        return NoveltyVerdict(
            is_novel=False, reason="LLM novelty judge ruled it redundant", similar_to=prior
        )
    if judge_verdict is None:
        return NoveltyVerdict(
            is_novel=True, reason="novel (embedding below threshold; judge unavailable)"
        )
    return NoveltyVerdict(is_novel=True, reason="novel")
