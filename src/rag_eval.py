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
  code-level guarantee, not something hoped for from the model. :func:`judge_faithfulness`
  catches the complementary failure ``citation_accuracy`` can't: an on-topic answer that cites
  a real evidence id but fabricates claims that id doesn't actually support.

Scores are logged to the KB's generic state map as an append-only run history
(``rag_eval@1``, ``rag_eval@2``, ... + ``rag_eval_count``, mirroring :mod:`src.agents.
forecaster`'s prediction log) so drift is trackable over time (T5.7's dashboard panel). Unlike
T1.5's per-*item* prediction log, this log grows once per **evaluation run** (a scheduled,
infrequent cadence), not once per classified item — the same low-growth-rate reasoning that
makes :mod:`src.taxonomy`/:mod:`src.policy`'s state-map idiom appropriate applies here too.

Known limitation (not fixed here, matching every sibling with this shape): :func:`record_score`
is an unguarded read-then-write with no locking — two concurrent evaluation runs against the
same store can silently clobber each other's history entry. See :mod:`src.taxonomy`'s module
docstring for the same caveat and why a real fix belongs at the Store layer.
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

# Per DEVPLAN's own example thresholds. Faithfulness has no DEVPLAN example value; 1.0 (zero
# tolerance for an unfaithful answer) mirrors HALLUCINATION_THRESHOLD's zero-tolerance spirit.
RECALL_THRESHOLD = 0.8
HALLUCINATION_THRESHOLD = 0.0
FAITHFULNESS_THRESHOLD = 1.0

# A candidate counts as "relevant enough to answer from" only above this cosine similarity —
# below it, answer_query() refuses rather than guessing. Tuned for the offline "hash" embed
# provider's typical in-topic vs. off-topic separation (see tests/test_rag_eval.py). A real
# semantic embedding backend's similarity distribution differs; this constant would need
# re-tuning before EMBED_PROVIDER=local is used with this guardrail.
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


def _reply_field(reply: object, key: str) -> object | None:
    """Safely pull `key` out of an ``llm.complete`` reply that's supposed to be a dict.

    ``llm.py``'s JSON mode is instruct-and-parse, not schema-validated — the model can return
    any JSON shape despite the schema request, so every field pulled from a reply must be
    checked, not trusted.
    """
    return reply.get(key) if isinstance(reply, dict) else None


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
    results: list[tuple[str, float]] | None = None,
) -> dict:
    """Answer `query` from `index`, grounded in retrieved evidence — or refuse.

    The hallucination guard: if nothing clears `threshold`, this returns
    :data:`_NO_EVIDENCE_ANSWER` with empty evidence **without calling the LLM at all** — an
    absent-topic query can't be answered correctly by asking the model to try anyway, so the
    guarantee lives in code, not in hoping the model declines.

    `results` lets a caller that already ran retrieval (e.g. :func:`evaluate_golden_set`,
    which also needs the ranked ids for recall/MRR/nDCG) pass it straight in instead of
    searching `index` a second time; omit it to have this function search on its own.

    Returns ``{"answer": str, "evidence": list[str]}`` (the ids retrieved and cited).

    Raises:
        embed.EmbedError: retrieval failed (only when `results` isn't supplied).
        llm.LLMError: the completion call failed (only reachable when evidence was found).
    """
    if results is None:
        results = embed.search(index, query, embed_fn=embed_fn, k=k)
    relevant = [id_ for id_, score in results if score >= threshold]
    if not relevant:
        return {"answer": _NO_EVIDENCE_ANSWER, "evidence": []}

    prompt = (
        f"Answer this question using ONLY the following evidence (cite nothing else): "
        f"{query}\n\nEvidence:\n" + "\n".join(f"- {id_}" for id_ in relevant)
    )
    reply = llm.complete(prompt, json_schema=_ANSWER_SCHEMA)
    answer = _reply_field(reply, "answer")
    return {
        "answer": answer if isinstance(answer, str) and answer else _NO_EVIDENCE_ANSWER,
        "evidence": relevant,
    }


def judge_faithfulness(answer: str, evidence: list[str]) -> bool:
    """Ask an LLM judge whether `answer` is fully supported by `evidence` (no fabrication).

    Only an explicit boolean ``True`` in the reply counts as faithful — a malformed or
    unexpected-typed reply (e.g. the string ``"false"``, which is truthy in Python but means
    the opposite) defaults to ``False``, never trusted as a pass.

    Raises:
        llm.LLMError: the completion call failed.
    """
    prompt = (
        "Is the following answer FULLY supported by the given evidence, with no fabricated "
        f"claims? Answer: {answer!r}\n\nEvidence:\n" + "\n".join(f"- {e}" for e in evidence)
    )
    reply = llm.complete(prompt, json_schema=_FAITHFULNESS_SCHEMA)
    return _reply_field(reply, "faithful") is True


# --------------------------------------------------------------------- golden-set evaluation


@dataclass(frozen=True)
class RagEvalScore:
    """One evaluation run's aggregate scores, validated and thresholded at construction time."""

    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    citation_accuracy: float
    hallucination_rate: float
    faithfulness: float
    created_at: str

    def __post_init__(self) -> None:
        for name, value in (
            ("recall_at_k", self.recall_at_k),
            ("mrr", self.mrr),
            ("ndcg_at_k", self.ndcg_at_k),
            ("citation_accuracy", self.citation_accuracy),
            ("hallucination_rate", self.hallucination_rate),
            ("faithfulness", self.faithfulness),
        ):
            if not 0.0 <= value <= 1.0:
                raise RagEvalError(f"{name} must be in [0, 1], got {value}")

    @property
    def passed(self) -> bool:
        """Whether this run clears every guardrail threshold."""
        return (
            self.recall_at_k >= RECALL_THRESHOLD
            and self.hallucination_rate <= HALLUCINATION_THRESHOLD
            and self.faithfulness >= FAITHFULNESS_THRESHOLD
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "recall_at_k": self.recall_at_k,
                "mrr": self.mrr,
                "ndcg_at_k": self.ndcg_at_k,
                "citation_accuracy": self.citation_accuracy,
                "hallucination_rate": self.hallucination_rate,
                "faithfulness": self.faithfulness,
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
                faithfulness=data["faithfulness"],
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
    vacuously perfect, ``1.0``, when there's nothing to retrieve — consistent with
    :func:`recall_at_k`'s/:func:`citation_accuracy`'s own convention for the same case). If
    the golden set has no real-topic queries, or no absent-topic probes, that's logged: a run
    that never exercises one half of the guardrail shouldn't silently look identical to one
    that exercised it and passed.

    A per-record failure (a malformed record, ``embed.EmbedError``, or ``llm.LLMError``) is
    skipped and logged, not raised — one bad record shouldn't abort the whole evaluation run.

    Raises:
        RagEvalError: `golden` is empty, or a resulting score is out of range.
    """
    if not golden:
        raise RagEvalError("golden set is empty — nothing to evaluate")

    recall_sum = mrr_sum = ndcg_sum = 0.0
    n_retrieval = 0
    absent_topic_queries = 0
    claims: list[dict] = []
    faithfulness_votes: list[bool] = []
    hallucinations = 0

    for record in golden:
        try:
            query = record["query"]
            relevant_ids = set(record.get("relevant_ids") or [])
            results = embed.search(index, query, embed_fn=embed_fn, k=k)
            ranked = [id_ for id_, _score in results]

            if relevant_ids:
                recall_sum += recall_at_k(ranked, relevant_ids, k)
                mrr_sum += mrr(ranked, relevant_ids)
                ndcg_sum += ndcg_at_k(ranked, relevant_ids, k)
                n_retrieval += 1
            else:
                absent_topic_queries += 1

            result = answer_query(
                index, query, embed_fn=embed_fn, k=k, threshold=threshold, results=results
            )

            if result["evidence"]:
                # A refusal ("no evidence") isn't a claim needing a citation — it's the
                # correct outcome for a query with nothing relevant, already scored via
                # hallucination_rate below; only actually-answered queries count here.
                claims.append({"query": query, "evidence": result["evidence"]})
                try:
                    faithfulness_votes.append(
                        judge_faithfulness(result["answer"], result["evidence"])
                    )
                except llm.LLMError as exc:
                    print(
                        f"rag_eval: skipping faithfulness judge for {query!r}: {exc}",
                        file=sys.stderr,
                    )
                if not relevant_ids:
                    hallucinations += 1  # fabricated a citation for a genuinely absent topic
        except (llm.LLMError, embed.EmbedError, KeyError) as exc:
            print(f"rag_eval: skipping golden record {record!r}: {exc}", file=sys.stderr)
            continue

    if n_retrieval == 0:
        print(
            "rag_eval: golden set has no real-topic queries — recall/MRR/nDCG are vacuous",
            file=sys.stderr,
        )
    if absent_topic_queries == 0:
        print(
            "rag_eval: golden set has no absent-topic probes — hallucination_rate is unmeasured",
            file=sys.stderr,
        )

    when = now or datetime.now(timezone.utc)
    return RagEvalScore(
        recall_at_k=(recall_sum / n_retrieval) if n_retrieval else 1.0,
        mrr=(mrr_sum / n_retrieval) if n_retrieval else 1.0,
        ndcg_at_k=(ndcg_sum / n_retrieval) if n_retrieval else 1.0,
        citation_accuracy=citation_accuracy(claims),
        hallucination_rate=(hallucinations / absent_topic_queries) if absent_topic_queries else 0.0,
        faithfulness=(
            (sum(faithfulness_votes) / len(faithfulness_votes)) if faithfulness_votes else 1.0
        ),
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
