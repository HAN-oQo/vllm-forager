"""RAG evaluation guardrail (T1.8, guardrail 2): a trustworthy score for retrieval + citations.

CLAUDE.md's Guardrails section: "a hand-labeled golden set + retrieval metrics (Recall@k /
MRR / nDCG) and groundedness metrics (faithfulness / hallucination-rate), thresholded and
drift-tracked — so a report's citations are numbers you can trust, not hope." Two concerns,
kept separate:

- **Retrieval quality** (:func:`recall_at_k`, :func:`mrr`, :func:`ndcg_at_k`): pure metric
  math over a ranked id list vs. a hand-labeled relevant-id set — no LLM, no embedding
  backend, fully deterministic. :func:`evaluate_golden_set` wires these to a real
  :mod:`src.embed` index; the default ``hash`` provider (T1.1) makes even that path offline
  and deterministic — only a genuinely live semantic-embedding backend is integration-tagged.
- **Generation trustworthiness** (:func:`answer_query`, :func:`judge_faithfulness`): every
  answer either cites real retrieved evidence or explicitly says so — the hallucination guard
  in :func:`answer_query` refuses to answer (never calling the LLM at all) when nothing
  clears the relevance threshold, so "an absent-topic query returns 'no evidence'" is a
  code-level guarantee, not something hoped for from the model.

Scores are logged to the KB's generic state map as an append-only run history
(``rag_eval@1``, ``rag_eval@2``, ... + ``rag_eval_count``, mirroring :mod:`src.agents.
forecaster`'s prediction log) so drift is trackable over time (T5.7's dashboard panel). Unlike
T1.5's per-*item* prediction log, this log grows once per **evaluation run** (a scheduled,
infrequent cadence), not once per classified item — the same low-growth-rate reasoning that
makes :mod:`src.taxonomy`/:mod:`src.policy`'s state-map idiom appropriate applies here too.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from . import embed, llm
from .store.base import Store

_EmbedFn = Callable[[list[str]], list[list[float]]]

_RUN_KEY_PREFIX = "rag_eval@"
_COUNT_KEY = "rag_eval_count"
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Per DEVPLAN's own example thresholds.
RECALL_THRESHOLD = 0.8
HALLUCINATION_THRESHOLD = 0.0

# A candidate counts as "relevant enough to answer from" only above this cosine similarity —
# below it, answer_query() refuses rather than guessing. Tuned for the offline "hash" embed
# provider's typical in-topic vs. off-topic separation (see tests/test_rag_eval.py).
_RELEVANCE_THRESHOLD = 0.05

_NO_EVIDENCE_ANSWER = "No evidence found for this query in the knowledge base."

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}
_FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {"faithful": {"type": "boolean"}},
    "required": ["faithful"],
}


class RagEvalError(RuntimeError):
    """Any RAG-eval failure: a corrupt KB record, or a malformed golden-set entry."""


# --------------------------------------------------------------------- retrieval metrics


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of `relevant_ids` present in the top-`k` of `ranked_ids`.

    An empty `relevant_ids` is vacuously perfect recall (``1.0``) — there is nothing to miss.
    """
    if not relevant_ids:
        return 1.0
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant id in `ranked_ids` (``0.0`` if none appear)."""
    for i, id_ in enumerate(ranked_ids, start=1):
        if id_ in relevant_ids:
            return 1.0 / i
    return 0.0


def _dcg(relevances: list[int]) -> float:
    """Discounted cumulative gain for a rank-ordered list of binary (1/0) relevances."""
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(relevances, start=1) if rel)


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized discounted cumulative gain at `k` (binary relevance).

    ``0.0`` when `relevant_ids` is empty — nDCG is undefined with nothing to rank well.
    """
    actual = [1 if id_ in relevant_ids else 0 for id_ in ranked_ids[:k]]
    ideal = [1] * min(len(relevant_ids), k) + [0] * max(0, k - len(relevant_ids))
    idcg = _dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return _dcg(actual) / idcg


# --------------------------------------------------------------------- generation metrics


def citation_accuracy(claims: list[dict]) -> float:
    """Fraction of `claims` (each ``{"evidence": [...], ...}``) that carry ≥1 citation.

    ``1.0`` for an empty `claims` list — nothing was claimed, so nothing is uncited.
    """
    if not claims:
        return 1.0
    cited = sum(1 for claim in claims if claim.get("evidence"))
    return cited / len(claims)


def answer_query(
    index: embed.EmbedIndex,
    query: str,
    *,
    embed_fn: _EmbedFn = embed.embed_texts,
    k: int = 5,
    threshold: float = _RELEVANCE_THRESHOLD,
) -> dict:
    """Answer `query` from `index`, grounded in retrieved evidence — or refuse.

    The hallucination guard: if nothing in `index` clears `threshold`, this returns
    :data:`_NO_EVIDENCE_ANSWER` with empty evidence **without calling the LLM at all** — an
    absent-topic query can't be answered correctly by asking the model to try anyway, so the
    guarantee lives in code, not in hoping the model declines.

    Returns ``{"answer": str, "evidence": list[str]}`` (the ids retrieved and cited).

    Raises:
        llm.LLMError: the completion call failed (only reachable when evidence was found).
    """
    results = embed.search(index, query, embed_fn=embed_fn, k=k)
    relevant = [id_ for id_, score in results if score >= threshold]
    if not relevant:
        return {"answer": _NO_EVIDENCE_ANSWER, "evidence": []}

    prompt = (
        f"Answer this question using ONLY the following evidence (cite nothing else): "
        f"{query}\n\nEvidence:\n" + "\n".join(f"- {id_}" for id_ in relevant)
    )
    reply = llm.complete(prompt, json_schema=_ANSWER_SCHEMA)
    answer = reply.get("answer") if isinstance(reply, dict) else None
    return {
        "answer": answer if isinstance(answer, str) and answer else _NO_EVIDENCE_ANSWER,
        "evidence": relevant,
    }


def judge_faithfulness(answer: str, evidence: list[str]) -> bool:
    """Ask an LLM judge whether `answer` is fully supported by `evidence` (no fabrication).

    Raises:
        llm.LLMError: the completion call failed.
    """
    prompt = (
        "Is the following answer FULLY supported by the given evidence, with no fabricated "
        f"claims? Answer: {answer!r}\n\nEvidence:\n" + "\n".join(f"- {e}" for e in evidence)
    )
    reply = llm.complete(prompt, json_schema=_FAITHFULNESS_SCHEMA)
    return bool(reply.get("faithful")) if isinstance(reply, dict) else False


# --------------------------------------------------------------------- golden-set evaluation


@dataclass(frozen=True)
class RagEvalScore:
    """One evaluation run's aggregate scores, validated and thresholded at construction time."""

    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    citation_accuracy: float
    hallucination_rate: float
    created_at: str

    def __post_init__(self) -> None:
        for name in ("recall_at_k", "mrr", "ndcg_at_k", "citation_accuracy"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise RagEvalError(f"{name} must be in [0, 1], got {value}")
        if not 0.0 <= self.hallucination_rate <= 1.0:
            raise RagEvalError(
                f"hallucination_rate must be in [0, 1], got {self.hallucination_rate}"
            )

    @property
    def passed(self) -> bool:
        """Whether this run clears both guardrail thresholds (DEVPLAN's example values)."""
        return (
            self.recall_at_k >= RECALL_THRESHOLD
            and self.hallucination_rate <= HALLUCINATION_THRESHOLD
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "recall_at_k": self.recall_at_k,
                "mrr": self.mrr,
                "ndcg_at_k": self.ndcg_at_k,
                "citation_accuracy": self.citation_accuracy,
                "hallucination_rate": self.hallucination_rate,
                "created_at": self.created_at,
            }
        )

    @staticmethod
    def from_json(raw: str) -> RagEvalScore:
        try:
            data = json.loads(raw)
            return RagEvalScore(
                recall_at_k=data["recall_at_k"],
                mrr=data["mrr"],
                ndcg_at_k=data["ndcg_at_k"],
                citation_accuracy=data["citation_accuracy"],
                hallucination_rate=data["hallucination_rate"],
                created_at=data["created_at"],
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise RagEvalError(f"corrupt rag_eval record: {exc}") from exc


def evaluate_golden_set(
    index: embed.EmbedIndex,
    golden: list[dict],
    *,
    k: int = 10,
    embed_fn: _EmbedFn = embed.embed_texts,
    threshold: float = _RELEVANCE_THRESHOLD,
    now: datetime | None = None,
) -> RagEvalScore:
    """Run every golden query through retrieval + generation; return the aggregate score.

    `golden` is a list of ``{"query": str, "relevant_ids": list[str]}`` records (loaded from
    ``tests/rag_eval/golden.jsonl``). A query with an empty ``relevant_ids`` is an
    **absent-topic probe**: it's only used to measure the hallucination rate (did
    :func:`answer_query` correctly refuse?), not retrieval quality (recall/MRR/nDCG are
    undefined with nothing to retrieve).

    A per-query failure (``llm.LLMError`` from the generation step) is skipped and logged,
    not raised — one bad query shouldn't abort the whole evaluation run.

    Raises:
        RagEvalError: `golden` is empty, or a resulting score is out of range.
    """
    if not golden:
        raise RagEvalError("golden set is empty — nothing to evaluate")

    retrieval_scores: list[dict] = []
    claims: list[dict] = []
    hallucinations = 0
    absent_topic_queries = 0

    for record in golden:
        query = record["query"]
        relevant_ids = set(record.get("relevant_ids") or [])
        ranked = [id_ for id_, _score in embed.search(index, query, embed_fn=embed_fn, k=k)]

        if relevant_ids:
            retrieval_scores.append(
                {
                    "recall": recall_at_k(ranked, relevant_ids, k),
                    "mrr": mrr(ranked, relevant_ids),
                    "ndcg": ndcg_at_k(ranked, relevant_ids, k),
                }
            )
        else:
            absent_topic_queries += 1

        try:
            result = answer_query(index, query, embed_fn=embed_fn, k=k, threshold=threshold)
        except llm.LLMError as exc:
            print(f"rag_eval: skipping generation for {query!r}: {exc}", file=sys.stderr)
            continue

        if result["evidence"]:
            # A refusal ("no evidence", empty evidence) isn't a claim needing a citation — it's
            # the correct outcome for a query with nothing relevant, already scored above via
            # hallucination_rate; only count actually-answered queries toward citation_accuracy.
            claims.append({"query": query, "evidence": result["evidence"]})
        if not relevant_ids and result["evidence"]:
            hallucinations += 1  # fabricated a citation for a genuinely absent topic

    when = now or datetime.now(timezone.utc)
    n_retrieval = len(retrieval_scores) or 1
    return RagEvalScore(
        recall_at_k=sum(s["recall"] for s in retrieval_scores) / n_retrieval,
        mrr=sum(s["mrr"] for s in retrieval_scores) / n_retrieval,
        ndcg_at_k=sum(s["ndcg"] for s in retrieval_scores) / n_retrieval,
        citation_accuracy=citation_accuracy(claims),
        hallucination_rate=(hallucinations / absent_topic_queries) if absent_topic_queries else 0.0,
        created_at=when.strftime(_TS_FORMAT),
    )


# --------------------------------------------------------------------- KB run history


def _parse_count(raw: str | None) -> int:
    """Parse the run-count index; raise :class:`RagEvalError` (not ValueError)."""
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError as exc:
        raise RagEvalError(f"corrupt rag_eval_count {raw!r}") from exc


def record_score(store: Store, score: RagEvalScore) -> int:
    """Append `score` to the KB's rag-eval run history; return its 1-based sequence number."""
    next_id = _parse_count(store.get_state(_COUNT_KEY)) + 1
    store.set_state(f"{_RUN_KEY_PREFIX}{next_id}", score.to_json())
    store.set_state(_COUNT_KEY, str(next_id))
    return next_id


def list_scores(store: Store) -> list[RagEvalScore]:
    """Return every recorded evaluation run, oldest first."""
    count = _parse_count(store.get_state(_COUNT_KEY))
    scores = []
    for i in range(1, count + 1):
        stored = store.get_state(f"{_RUN_KEY_PREFIX}{i}")
        if stored is not None:
            scores.append(RagEvalScore.from_json(stored))
    return scores
