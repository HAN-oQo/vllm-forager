"""Tests for the RAG evaluation guardrail (T1.8) — offline & deterministic.

Per the DEVPLAN todo: offline metric math on a fixed ranked list (known Recall@k/MRR/nDCG),
every claim carries a citation, an absent-topic query ⇒ "no evidence". Live retrieval against
a real semantic-embedding backend + a live LLM-judge call are ``@pytest.mark.integration``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import embed, llm, rag_eval
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m1

_GOLDEN_PATH = Path(__file__).parent / "rag_eval" / "golden.jsonl"
_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

# A small synthetic corpus with clearly separated vocabulary per topic, matching
# tests/rag_eval/golden.jsonl's queries/relevant_ids — the "hash" embed provider (T1.1) is
# purely lexical, so distinct topics need distinct words to be distinguishable offline.
_CORPUS_TEXT = {
    "o/r#1": "hipBLAS build fails on gfx90a MI250 with ROCm 6 toolchain link error",
    "o/r#2": "ROCm build broken after toolchain upgrade hipBLAS symbols missing",
    "o/r#3": "Add FP8 quantization support for MI300 kernels",
    "o/r#4": "INT4 quantization accuracy regression on quantized models",
    "o/r#5": "fix a typo in the README installation instructions",
}


def _load_golden() -> list[dict]:
    with _GOLDEN_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _corpus_index() -> embed.EmbedIndex:
    ids = list(_CORPUS_TEXT)
    vectors = embed.embed_texts(list(_CORPUS_TEXT.values()))
    return embed.build_index(ids, vectors)


# --------------------------------------------------------------------- retrieval metrics


def test_recall_at_k_fixed_ranked_list() -> None:
    """The DEVPLAN's named scenario: metric math on a fixed ranked list."""
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d", "z"}  # "z" isn't retrieved at all
    assert rag_eval.recall_at_k(ranked, relevant, k=5) == pytest.approx(2 / 3)
    assert rag_eval.recall_at_k(ranked, relevant, k=2) == pytest.approx(1 / 3)  # only "b" in top 2


def test_recall_at_k_empty_relevant_set_is_vacuously_perfect() -> None:
    assert rag_eval.recall_at_k(["a", "b"], set(), k=5) == 1.0


def test_mrr_fixed_ranked_list() -> None:
    assert rag_eval.mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert rag_eval.mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)
    assert rag_eval.mrr(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_at_k_fixed_ranked_list() -> None:
    # relevant items at ranks 1 and 3; ideal would have both relevant items at ranks 1 and 2.
    ranked = ["a", "b", "c"]
    relevant = {"a", "c"}
    dcg = 1 / 1 + 1 / __import__("math").log2(4)  # rank 1 and rank 3
    idcg = 1 / 1 + 1 / __import__("math").log2(3)  # ideal: ranks 1 and 2
    assert rag_eval.ndcg_at_k(ranked, relevant, k=3) == pytest.approx(dcg / idcg)


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    assert rag_eval.ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_empty_relevant_set_is_zero() -> None:
    assert rag_eval.ndcg_at_k(["a", "b"], set(), k=5) == 0.0


# --------------------------------------------------------------------- generation metrics


def test_citation_accuracy_mixed_claims() -> None:
    claims = [
        {"text": "a", "evidence": ["http://x/1"]},
        {"text": "b", "evidence": []},
        {"text": "c", "evidence": ["http://x/3"]},
    ]
    assert rag_eval.citation_accuracy(claims) == pytest.approx(2 / 3)


def test_citation_accuracy_empty_claims_is_perfect() -> None:
    assert rag_eval.citation_accuracy([]) == 1.0


def test_answer_query_absent_topic_returns_no_evidence_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DEVPLAN's named scenario: an absent-topic query ⇒ "no evidence"."""

    def boom(*a, **k):
        raise AssertionError("llm.complete should not be called for an absent-topic query")

    monkeypatch.setattr(llm, "complete", boom)
    index = _corpus_index()

    result = rag_eval.answer_query(index, "kubernetes helm chart deployment configuration")

    assert result == {"answer": rag_eval._NO_EVIDENCE_ANSWER, "evidence": []}


def test_answer_query_with_evidence_calls_llm_and_cites(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"answer": "It's a hipBLAS link error."})
    index = _corpus_index()

    result = rag_eval.answer_query(index, "hipBLAS build fails on gfx90a")

    assert result["answer"] == "It's a hipBLAS link error."
    assert result["evidence"]  # at least one citation
    assert set(result["evidence"]) <= set(_CORPUS_TEXT)


def test_judge_faithfulness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"faithful": True})
    assert rag_eval.judge_faithfulness("answer", ["evidence"]) is True

    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"faithful": False})
    assert rag_eval.judge_faithfulness("answer", ["evidence"]) is False


def test_judge_faithfulness_rejects_non_bool_truthy_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed reply (e.g. the JSON string "false") must never coerce to faithful=True."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"faithful": "false"})
    assert rag_eval.judge_faithfulness("answer", ["evidence"]) is False

    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"unexpected": "shape"})
    assert rag_eval.judge_faithfulness("answer", ["evidence"]) is False


# --------------------------------------------------------------------- golden-set evaluation


def _fake_complete_factory(answer: str, faithful: bool):
    """Build a fake `llm.complete` that answers both the answer-generation prompt and the
    faithfulness-judge prompt correctly, distinguishing them by their distinctive wording."""

    def fake(prompt: str, **kwargs) -> dict:
        if "FULLY supported" in prompt:
            return {"faithful": faithful}
        return {"answer": answer}

    return fake


def test_evaluate_golden_set_scores_retrieval_and_hallucination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", _fake_complete_factory("a grounded answer", True))
    index = _corpus_index()
    golden = _load_golden()

    score = rag_eval.evaluate_golden_set(index, golden, k=5, now=_NOW)

    # both real-topic queries should retrieve their relevant items perfectly from this
    # deliberately well-separated corpus
    assert score.recall_at_k == pytest.approx(1.0)
    assert score.mrr == pytest.approx(1.0)
    assert score.ndcg_at_k == pytest.approx(1.0)
    # the absent-topic probe found nothing relevant -> answer_query refused -> no hallucination
    assert score.hallucination_rate == 0.0
    assert score.citation_accuracy == pytest.approx(1.0)
    assert score.faithfulness == pytest.approx(1.0)
    assert score.passed is True
    assert score.created_at == "2026-01-01T00:00:00Z"


def test_evaluate_golden_set_flags_hallucination_when_guard_bypassed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If retrieval ever DID surface evidence for an absent-topic query, that's a
    hallucination — verified by lowering the threshold so the guard doesn't kick in."""
    monkeypatch.setattr(llm, "complete", _fake_complete_factory("a fabricated answer", True))
    index = _corpus_index()
    golden = _load_golden()

    score = rag_eval.evaluate_golden_set(index, golden, k=5, threshold=-1.0, now=_NOW)

    assert score.hallucination_rate == pytest.approx(1.0)
    assert score.passed is False  # hallucination_rate > HALLUCINATION_THRESHOLD


def test_evaluate_golden_set_unfaithful_answer_fails_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An answer judged unfaithful must fail `passed` even when retrieval is perfect."""
    monkeypatch.setattr(llm, "complete", _fake_complete_factory("a grounded answer", False))
    index = _corpus_index()
    golden = _load_golden()

    score = rag_eval.evaluate_golden_set(index, golden, k=5, now=_NOW)

    assert score.faithfulness == pytest.approx(0.0)
    assert score.passed is False


def test_evaluate_golden_set_skips_failing_query_and_keeps_going(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def flaky(prompt: str, **kwargs) -> dict:
        if "FP8" in prompt:
            raise llm.LLMError("simulated failure")
        if "FULLY supported" in prompt:
            return {"faithful": True}
        return {"answer": "a grounded answer"}

    monkeypatch.setattr(llm, "complete", flaky)
    index = _corpus_index()
    golden = _load_golden()

    score = rag_eval.evaluate_golden_set(index, golden, k=5, now=_NOW)

    # retrieval metrics are unaffected by a generation-side failure — the failing record's
    # retrieval score was already accumulated before its generation step raised
    assert score.recall_at_k == pytest.approx(1.0)


def test_evaluate_golden_set_skips_malformed_record_and_keeps_going(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", _fake_complete_factory("a grounded answer", True))
    index = _corpus_index()
    golden = [{"relevant_ids": ["o/r#1"]}, *_load_golden()]  # missing "query" key

    score = rag_eval.evaluate_golden_set(index, golden, k=5, now=_NOW)

    assert score.recall_at_k == pytest.approx(1.0)


def test_evaluate_golden_set_empty_raises() -> None:
    with pytest.raises(rag_eval.RagEvalError, match="golden set is empty"):
        rag_eval.evaluate_golden_set(_corpus_index(), [])


def test_evaluate_golden_set_no_absent_topic_probes_logs_and_defaults_hallucination(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(llm, "complete", _fake_complete_factory("a grounded answer", True))
    index = _corpus_index()
    golden = [rec for rec in _load_golden() if rec.get("relevant_ids")]

    score = rag_eval.evaluate_golden_set(index, golden, k=5, now=_NOW)

    assert score.hallucination_rate == 0.0
    assert "hallucination_rate is unmeasured" in capsys.readouterr().err


# --------------------------------------------------------------------- RagEvalScore + KB log


def _score(**overrides) -> rag_eval.RagEvalScore:
    base = {
        "recall_at_k": 0.9,
        "mrr": 0.8,
        "ndcg_at_k": 0.85,
        "citation_accuracy": 1.0,
        "hallucination_rate": 0.0,
        "faithfulness": 1.0,
        "created_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return rag_eval.RagEvalScore(**base)


def test_rag_eval_score_passed_thresholds() -> None:
    assert _score(recall_at_k=0.8, hallucination_rate=0.0, faithfulness=1.0).passed is True
    assert _score(recall_at_k=0.79).passed is False
    assert _score(hallucination_rate=0.01).passed is False
    assert _score(faithfulness=0.99).passed is False


@pytest.mark.parametrize(
    "bad_field",
    ["recall_at_k", "mrr", "ndcg_at_k", "citation_accuracy", "hallucination_rate", "faithfulness"],
)
def test_rag_eval_score_out_of_range_raises(bad_field: str) -> None:
    with pytest.raises(rag_eval.RagEvalError, match="must be in \\[0, 1\\]"):
        _score(**{bad_field: 1.5})


def test_rag_eval_score_json_roundtrip() -> None:
    original = _score()
    assert rag_eval.RagEvalScore.from_json(original.to_json()) == original


def test_rag_eval_score_from_json_corrupt_raises() -> None:
    with pytest.raises(rag_eval.RagEvalError, match="corrupt rag_eval record"):
        rag_eval.RagEvalScore.from_json("not json")

    with pytest.raises(rag_eval.RagEvalError, match="corrupt rag_eval record"):
        rag_eval.RagEvalScore.from_json(json.dumps({"recall_at_k": 0.9}))  # missing fields


def test_record_and_list_scores_roundtrip(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    s1 = _score(recall_at_k=0.9)
    s2 = _score(recall_at_k=0.7)
    assert rag_eval.record_score(store, s1) == 1
    assert rag_eval.record_score(store, s2) == 2
    assert rag_eval.list_scores(store) == [s1, s2]


def test_list_scores_corrupt_count_raises(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.set_state("rag_eval_count", "not-a-number")
    with pytest.raises(rag_eval.RagEvalError, match="corrupt rag_eval_count"):
        rag_eval.list_scores(store)


# --------------------------------------------------------------------- live smoke


@pytest.mark.integration
def test_golden_set_live_smoke():
    # A real end-to-end pass: retrieval (hash provider) + a live LLM answer + a live
    # LLM-judge faithfulness check. Needs local Claude Code auth; skipped by default.
    index = _corpus_index()
    golden = _load_golden()
    score = rag_eval.evaluate_golden_set(index, golden, k=5)
    assert score.recall_at_k > 0.0
    assert score.hallucination_rate == 0.0

    result = rag_eval.answer_query(index, "hipBLAS build fails on gfx90a")
    assert result["evidence"]
    faithful = rag_eval.judge_faithfulness(result["answer"], result["evidence"])
    assert isinstance(faithful, bool)
