"""Tests for the Analyst agent (T1.4, path-classifying since T1.5.2, domain-rooted since
T3.15) — offline & deterministic.

Per the DEVPLAN todo: items from rl/omni/speech repos land in the right domain subtree (an
OpenRLHF PPO issue -> ``rl > post-training > PPO``; a diffusers scheduler bug -> ``omni >
diffusion > scheduler``). Also covers: mocked ``llm.complete`` -> item gets a valid subpath
under its domain, citation preserved; an off-taxonomy answer is normalized, not rejected; an
untracked repo (no known domain) falls back to ``Other`` with no LLM call; a domain with no
registered subcategories yet falls back to just the domain, also with no LLM call;
per-item failure isolation; delta-only reclassification; and the ordering of the taxonomy
lookup vs. the pending-items check.
"""

import pytest

from src import llm, parity
from src.agents import analyst
from src.store.jsonl_store import JsonlStore
from src.taxonomy import Taxonomy, TaxonomyError, create_taxonomy

pytestmark = pytest.mark.m1

# A small, stable set of domain-tagged repos -- "o/r" is deliberately left untracked (no
# config.REPOS entry) so it exercises the "unknown domain" fallback in tests that use it.
_REPOS = [
    {"slug": "verl-project/verl", "role": "source", "domain": "rl"},
    {"slug": "huggingface/diffusers", "role": "source", "domain": "omni"},
    {"slug": "vllm-project/vllm", "role": "primary", "domain": "speech"},
]


@pytest.fixture(autouse=True)
def _config_repos(monkeypatch: pytest.MonkeyPatch) -> None:
    # analyst._domain_of_item delegates to parity._domain_of (T3.15 reuse), which reads
    # parity's own `config` reference -- patch REPOS there, not on analyst (which no longer
    # imports config directly).
    monkeypatch.setattr(parity.config, "REPOS", _REPOS)


def _item(repo: str, number: int, title: str, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": title,
        "state": "open",
        "labels": [],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "url": f"http://x/{repo}/{number}",
        "body": "",
    }
    rec.update(overrides)
    return rec


def _boom(*args, **kwargs):
    raise AssertionError("llm.complete should not be called")


# --------------------------------------------------------------------- classify_item


def test_classify_item_sets_path_and_category_and_preserves_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_complete(prompt: str, *, json_schema: dict) -> dict:
        captured["prompt"] = prompt
        captured["json_schema"] = json_schema
        return {"path": ["post-training", "PPO"]}

    monkeypatch.setattr(llm, "complete", fake_complete)
    taxonomy = Taxonomy(version=1, categories=(("rl", "post-training", "PPO"), ("rl", "eval")))
    item = _item("verl-project/verl", 1, "PPO trainer regression")

    result = analyst.classify_item(item, taxonomy)

    assert result["path"] == ["rl", "post-training", "PPO"]
    assert result["category"] == "rl > post-training > PPO"
    assert result["taxonomy_version"] == 1
    assert result["url"] == item["url"]  # evidence citation preserved
    assert result["title"] == item["title"]  # original fields untouched
    # the schema/prompt actually reach llm.complete, not just decorative constants
    assert captured["json_schema"] == analyst._PATH_SCHEMA
    assert "PPO trainer regression" in captured["prompt"]
    assert "post-training > PPO" in captured["prompt"]


def test_classify_item_matches_the_devplan_worked_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DEVPLAN's own e.g.: an OpenRLHF-shaped PPO issue -> rl > post-training > PPO; a
    diffusers scheduler bug -> omni > diffusion > scheduler."""
    monkeypatch.setattr(llm, "complete", lambda prompt, **k: {"path": ["post-training", "PPO"]})
    rl_taxonomy = Taxonomy(version=1, categories=(("rl", "post-training", "PPO"),))
    rl_result = analyst.classify_item(
        _item("verl-project/verl", 1, "PPO reward normalization bug"), rl_taxonomy
    )
    assert rl_result["path"] == ["rl", "post-training", "PPO"]

    monkeypatch.setattr(llm, "complete", lambda prompt, **k: {"path": ["diffusion", "scheduler"]})
    omni_taxonomy = Taxonomy(version=1, categories=(("omni", "diffusion", "scheduler"),))
    omni_result = analyst.classify_item(
        _item("huggingface/diffusers", 2, "flow-matching scheduler off by one"), omni_taxonomy
    )
    assert omni_result["path"] == ["omni", "diffusion", "scheduler"]

    monkeypatch.setattr(llm, "complete", lambda prompt, **k: {"path": ["rocm-build"]})
    speech_taxonomy = Taxonomy(version=1, categories=(("speech", "rocm-build"),))
    speech_result = analyst.classify_item(
        _item("vllm-project/vllm", 3, "hipBLAS build fails on gfx90a"), speech_taxonomy
    )
    assert speech_result["path"] == ["speech", "rocm-build"]


def test_classify_item_stops_at_the_first_level_that_drifts_off_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1.5.2's named scenario: an off-taxonomy answer is normalized, not rejected outright —
    the valid prefix (here, just the domain) is kept."""
    monkeypatch.setattr(
        llm, "complete", lambda *a, **k: {"path": ["post-training", "not-a-real-child"]}
    )
    taxonomy = Taxonomy(version=1, categories=(("rl", "post-training"),))

    result = analyst.classify_item(_item("verl-project/verl", 1, "x"), taxonomy)

    assert result["path"] == ["rl", "post-training"]
    assert result["category"] == "rl > post-training"


def test_classify_item_path_match_is_case_and_whitespace_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors taxonomy.add_category's own casefold+strip normalization."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": [" POST-TRAINING "]})
    taxonomy = Taxonomy(version=1, categories=(("rl", "post-training"),))

    result = analyst.classify_item(_item("verl-project/verl", 1, "x"), taxonomy)

    # the taxonomy's own canonical spelling is used, not the model's raw casing
    assert result["path"] == ["rl", "post-training"]


def test_classify_item_unknown_sublevel_falls_back_to_domain_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": ["not-a-real-category"]})
    taxonomy = Taxonomy(version=1, categories=(("rl", "post-training"),))

    result = analyst.classify_item(_item("verl-project/verl", 1, "x"), taxonomy)

    assert result["path"] == ["rl"]
    assert result["category"] == "rl"


def test_classify_item_non_dict_reply_falls_back_to_domain_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "not json")
    taxonomy = Taxonomy(version=1, categories=(("rl", "post-training"),))
    result = analyst.classify_item(_item("verl-project/verl", 1, "x"), taxonomy)
    assert result["path"] == ["rl"]


def test_classify_item_empty_path_reply_falls_back_to_domain_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": []})
    taxonomy = Taxonomy(version=1, categories=(("rl", "post-training"),))
    result = analyst.classify_item(_item("verl-project/verl", 1, "x"), taxonomy)
    assert result["path"] == ["rl"]


def test_classify_item_untracked_repo_falls_back_to_other_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repo not in config.REPOS (e.g. a stale/retired one collected before T3.14 dropped
    it) has no known domain -- Other, and no LLM call (there's nothing to classify against)."""
    monkeypatch.setattr(llm, "complete", _boom)
    taxonomy = Taxonomy(version=1, categories=(("rl", "post-training"),))

    result = analyst.classify_item(_item("o/r", 1, "x"), taxonomy)

    assert result["path"] == [analyst.OTHER]
    assert result["category"] == analyst.OTHER


def test_classify_item_domain_with_no_subcategories_yet_returns_domain_only_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A known domain with nothing registered under it yet (e.g. right after T3.15 ships,
    before the Curator proposes anything for "omni") skips the LLM call entirely -- there's
    nothing to validate an answer against."""
    monkeypatch.setattr(llm, "complete", _boom)
    taxonomy = Taxonomy(version=1, categories=(("rl", "post-training"),))  # nothing under omni

    result = analyst.classify_item(_item("huggingface/diffusers", 1, "x"), taxonomy)

    assert result["path"] == ["omni"]
    assert result["category"] == "omni"


def test_validated_subpath_non_string_element_returns_empty() -> None:
    taxonomy = Taxonomy(version=1, categories=(("rl", "post-training"),))
    assert analyst._validated_subpath(["post-training", 5], taxonomy, prefix=("rl",)) == ()


def test_prompt_truncates_long_body() -> None:
    long_body = "x" * (analyst._BODY_CHARS + 500)
    taxonomy = Taxonomy(version=1, categories=(("rl", "a"),))
    prompt = analyst._path_prompt(
        _item("verl-project/verl", 1, "t", body=long_body), taxonomy, "rl"
    )
    assert "x" * analyst._BODY_CHARS in prompt
    assert "x" * (analyst._BODY_CHARS + 1) not in prompt


def test_prompt_never_shows_the_domain_prefix_in_known_subpaths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taxonomy = Taxonomy(version=1, categories=(("rl", "post-training", "PPO"), ("omni", "x")))
    prompt = analyst._path_prompt(_item("verl-project/verl", 1, "t"), taxonomy, "rl")
    assert "post-training > PPO" in prompt
    assert "omni" not in prompt  # a different domain's subpaths must not leak in


# --------------------------------------------------------------------- analyze_store


def test_analyze_store_classifies_only_pending_items(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, [["rl", "post-training"], ["rl", "eval"]])
    store.upsert_items(
        [
            _item("verl-project/verl", 1, "PPO trainer regression"),
            _item(
                "verl-project/verl",
                2,
                "already classified",
                path=["rl", "eval"],
                category="rl > eval",
                taxonomy_version=1,
            ),
        ]
    )
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": ["post-training"]})

    classified = analyst.analyze_store(store)

    assert len(classified) == 1
    assert classified[0]["number"] == 1
    assert classified[0]["path"] == ["rl", "post-training"]
    # the store itself now reflects the new classification, and the pre-classified item
    # (#2) was left untouched by this run
    stored = {item["number"]: item for item in store.query()}
    assert stored[1]["path"] == ["rl", "post-training"]
    assert stored[2]["path"] == ["rl", "eval"]


def test_analyze_store_backfills_path_for_a_pre_t1_5_2_flat_category_item(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The delta check switched from "no category" to "no path" — an item classified by the
    pre-T1.5.2 Analyst (has `category`, no `path`) gets re-classified once to backfill it."""
    store = JsonlStore(tmp_path)
    create_taxonomy(store, [["rl", "post-training"]])
    store.upsert_items(
        [
            _item(
                "verl-project/verl",
                1,
                "legacy item",
                category="post-training",
                taxonomy_version=1,
            )
        ]
    )
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": ["post-training"]})

    classified = analyst.analyze_store(store)

    assert len(classified) == 1
    assert classified[0]["path"] == ["rl", "post-training"]


def test_analyze_store_skips_failing_item_and_persists_the_rest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: one item's LLMError used to discard the whole batch's progress."""
    store = JsonlStore(tmp_path)
    create_taxonomy(store, [["rl", "post-training"]])
    store.upsert_items(
        [
            _item("verl-project/verl", 1, "good item"),
            _item("verl-project/verl", 2, "bad item"),
        ],
    )

    def flaky_complete(prompt: str, **kwargs) -> dict:
        if "bad item" in prompt:
            raise llm.LLMError("simulated transient failure")
        return {"path": ["post-training"]}

    monkeypatch.setattr(llm, "complete", flaky_complete)

    classified = analyst.analyze_store(store)

    assert len(classified) == 1
    assert classified[0]["number"] == 1
    stored = {item["number"]: item for item in store.query()}
    assert stored[1]["path"] == ["rl", "post-training"]
    assert "path" not in stored[2]  # left pending for a future retry


def test_analyze_store_untracked_repo_marks_other_without_calling_llm(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, [["rl", "post-training"]])
    store.upsert_items([_item("o/r", 1, "x")])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(llm, "complete", _boom)
        classified = analyst.analyze_store(store)

    assert classified[0]["path"] == [analyst.OTHER]
    assert classified[0]["category"] == analyst.OTHER


def test_analyze_store_domain_with_no_subcategories_marks_domain_only_without_calling_llm(
    tmp_path,
) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, [])  # nothing registered under any domain yet
    store.upsert_items([_item("verl-project/verl", 1, "x")])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(llm, "complete", _boom)
        classified = analyst.analyze_store(store)

    assert classified[0]["path"] == ["rl"]
    assert classified[0]["category"] == "rl"


def test_analyze_store_no_pending_items_returns_empty(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, [["rl", "post-training"]])
    store.upsert_items(
        [
            _item(
                "verl-project/verl",
                1,
                "x",
                path=["rl", "post-training"],
                category="rl > post-training",
                taxonomy_version=1,
            )
        ]
    )
    assert analyst.analyze_store(store) == []


def test_analyze_store_no_pending_items_and_no_taxonomy_returns_empty(tmp_path) -> None:
    """The pending check runs before the taxonomy lookup, so an empty store never raises."""
    store = JsonlStore(tmp_path)
    assert analyst.analyze_store(store) == []


def test_analyze_store_pending_items_but_no_taxonomy_raises(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "x")])
    with pytest.raises(TaxonomyError, match="no taxonomy exists"):
        analyst.analyze_store(store)
