"""Tests for the parity matrix (T2.4/T3.14) — offline & deterministic.

Per the DEVPLAN todo: synthetic capability signals → matrix cell populated + "present in
engine A, missing in target B" gap flagged, across any tracked engines (not a hardcoded
ROCm/vllm fork).
"""

import pytest

from src import parity

pytestmark = pytest.mark.m2


def _item(repo: str, number: int, category: str, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "pr",
        "title": "x",
        "state": "closed",
        "category": category,
        "url": f"https://github.com/{repo}/pull/{number}",
    }
    rec.update(overrides)
    return rec


# --------------------------------------------------------------------- build_matrix


def test_build_matrix_present_when_shipped() -> None:
    items = [_item("engine-a", 1, "fp8-kv-cache", state="closed", type="pr")]

    cells = parity.build_matrix(items)

    assert len(cells) == 1
    assert cells[0] == parity.ParityCell(
        engine="engine-a",
        capability="fp8-kv-cache",
        present=True,
        evidence="https://github.com/engine-a/pull/1",
    )


def test_build_matrix_absent_when_only_open_pr() -> None:
    items = [_item("engine-a", 1, "fp8-kv-cache", state="open", type="pr")]

    cells = parity.build_matrix(items)

    assert cells[0].present is False
    assert cells[0].evidence is None


def test_build_matrix_absent_when_only_an_issue() -> None:
    items = [_item("engine-a", 1, "fp8-kv-cache", type="issue", state="closed")]

    cells = parity.build_matrix(items)

    assert cells[0].present is False
    assert cells[0].evidence is None


def test_build_matrix_excludes_uncategorized_items() -> None:
    items = [_item("engine-a", 1, "fp8-kv-cache")]
    del items[0]["category"]

    assert parity.build_matrix(items) == []


def test_build_matrix_excludes_non_string_category() -> None:
    items = [_item("engine-a", 1, category=["not-a-string"])]

    assert parity.build_matrix(items) == []


def test_build_matrix_excludes_items_with_no_repo() -> None:
    items = [_item("engine-a", 1, "fp8-kv-cache")]
    del items[0]["repo"]

    assert parity.build_matrix(items) == []


def test_build_matrix_one_cell_per_engine_capability_pair() -> None:
    items = [
        _item("engine-a", 1, "fp8-kv-cache"),
        _item("engine-a", 2, "fp8-kv-cache", state="open"),  # same cell, still present overall
        _item("engine-b", 3, "fp8-kv-cache", state="open"),  # different engine
    ]

    cells = {(c.engine, c.capability): c for c in parity.build_matrix(items)}

    assert cells[("engine-a", "fp8-kv-cache")].present is True
    assert cells[("engine-b", "fp8-kv-cache")].present is False


def test_build_matrix_evidence_prefers_most_recently_shipped_item() -> None:
    items = [
        _item("engine-a", 1, "fp8-kv-cache", created_at="2026-01-01T00:00:00Z"),
        _item("engine-a", 2, "fp8-kv-cache", created_at="2026-06-01T00:00:00Z"),
    ]

    cells = parity.build_matrix(items)

    assert cells[0].evidence == "https://github.com/engine-a/pull/2"


def test_build_matrix_evidence_falls_back_when_newest_has_no_resolvable_url() -> None:
    """Regression: evidence used to be taken from the first shipped item unconditionally,
    reporting no evidence at all when that one happened to be unresolvable even though an
    older shipped item in the same cell had a perfectly good URL."""
    newest_unresolvable = _item("engine-a", 1, "fp8-kv-cache", created_at="2026-06-01T00:00:00Z")
    del newest_unresolvable["url"]
    del newest_unresolvable["number"]  # repo stays -- needed for bucketing into the same cell
    older_resolvable = _item("engine-a", 2, "fp8-kv-cache", created_at="2026-01-01T00:00:00Z")

    cells = parity.build_matrix([newest_unresolvable, older_resolvable])

    assert cells[0].evidence == "https://github.com/engine-a/pull/2"


# --------------------------------------------------------------------- _default_engine


def test_default_engine_raises_clear_error_for_unknown_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parity.config, "REPOS", [{"slug": "o/r", "role": "source"}])
    with pytest.raises(parity.ParityError, match="no config.REPOS entry has role 'primary'"):
        parity.find_gaps([])


def test_engines_with_role_returns_every_matching_slug_in_file_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parity.config,
        "REPOS",
        [
            {"slug": "a", "role": "primary"},
            {"slug": "b", "role": "source"},
            {"slug": "c", "role": "primary"},
        ],
    )
    assert parity._engines_with_role("primary") == ["a", "c"]


# --------------------------------------------------------------------- find_gaps


def test_find_gaps_flags_source_present_target_missing() -> None:
    """Matches the DEVPLAN's own worked example: a capability shipped on SGLang but missing on
    vllm-omni is flagged -- no ROCm/vllm fork needed."""
    items = [
        _item("sgl-project/sglang", 1, "flash-attn-3", state="closed"),
        _item("vllm-project/vllm-omni", 2, "flash-attn-3", state="open"),
    ]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(
        cells, source_engines=["sgl-project/sglang"], target_engine="vllm-project/vllm-omni"
    )

    assert gaps == [
        parity.Gap(
            capability="flash-attn-3",
            target_engine="vllm-project/vllm-omni",
            source_engine="sgl-project/sglang",
            evidence="https://github.com/sgl-project/sglang/pull/1",
        )
    ]


def test_find_gaps_flags_when_target_has_no_cell_at_all() -> None:
    items = [_item("sgl-project/sglang", 1, "flash-attn-3", state="closed")]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(
        cells, source_engines=["sgl-project/sglang"], target_engine="vllm-project/vllm-omni"
    )

    assert len(gaps) == 1
    assert gaps[0].capability == "flash-attn-3"


def test_find_gaps_no_gap_when_both_shipped() -> None:
    items = [
        _item("sgl-project/sglang", 1, "flash-attn-3", state="closed"),
        _item("vllm-project/vllm-omni", 2, "flash-attn-3", state="closed"),
    ]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(
        cells, source_engines=["sgl-project/sglang"], target_engine="vllm-project/vllm-omni"
    )

    assert gaps == []


def test_find_gaps_no_gap_when_source_does_not_have_it() -> None:
    items = [_item("vllm-project/vllm-omni", 1, "flash-attn-3", state="closed")]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(
        cells, source_engines=["sgl-project/sglang"], target_engine="vllm-project/vllm-omni"
    )

    assert gaps == []


def test_find_gaps_handles_a_source_engine_missing_from_cells_entirely() -> None:
    """Regression: a source engine with no data yet (never classified/shipped anything) is
    handled like any other "not present" case, not an error, when multiple source engines are
    checked and only some of them have cells."""
    items = [_item("sgl-project/sglang", 1, "flash-attn-3", state="closed")]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(
        cells,
        source_engines=["ai-dynamo/dynamo", "sgl-project/sglang"],  # dynamo has no cells at all
        target_engine="vllm-project/vllm-omni",
    )

    assert [g.capability for g in gaps] == ["flash-attn-3"]
    assert gaps[0].source_engine == "sgl-project/sglang"


def test_find_gaps_any_role_can_be_a_source_not_just_a_hardcoded_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T3.14: the fork-only `role: "fork"` premise is gone -- any tracked engine, of any role,
    can supply a capability a target is missing."""
    monkeypatch.setattr(
        parity.config,
        "REPOS",
        [
            {"slug": "vllm-project/vllm-omni", "role": "primary", "domain": "omni"},
            {"slug": "huggingface/diffusers", "role": "source", "domain": "omni"},
        ],
    )
    items = [_item("huggingface/diffusers", 1, "flow-matching-scheduler", state="closed")]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(cells)

    assert [g.capability for g in gaps] == ["flow-matching-scheduler"]
    assert gaps[0].source_engine == "huggingface/diffusers"
    assert gaps[0].target_engine == "vllm-project/vllm-omni"


def test_find_gaps_default_source_engines_excludes_unrelated_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a code-review finding on this rework -- before it, there was exactly one
    fork and one upstream, so no domain could ever cross. Generalizing to "every other tracked
    engine" without a domain filter would flag an RL-only capability as a "gap" on the
    speech-focused vllm target, a semantically meaningless cross-domain port suggestion."""
    monkeypatch.setattr(
        parity.config,
        "REPOS",
        [
            {"slug": "vllm-project/vllm", "role": "primary", "domain": "speech"},
            {"slug": "vllm-project/vime", "role": "primary", "domain": "rl"},
            {"slug": "verl-project/verl", "role": "source", "domain": "rl"},
        ],
    )
    items = [_item("vllm-project/vime", 1, "ppo-trainer", state="closed")]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(cells, target_engine="vllm-project/vllm")

    assert gaps == []


def test_find_gaps_default_source_engines_includes_same_domain_ecosystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parity.config,
        "REPOS",
        [
            {"slug": "vllm-project/vime", "role": "primary", "domain": "rl"},
            {"slug": "verl-project/verl", "role": "source", "domain": "rl"},
        ],
    )
    items = [_item("verl-project/verl", 1, "ppo-trainer", state="closed")]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(cells, target_engine="vllm-project/vime")

    assert [g.capability for g in gaps] == ["ppo-trainer"]


def test_find_gaps_default_source_engines_includes_engine_domain_regardless_of_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An "engine"-domain repo (SGLang/Dynamo/llm-d) is a generic cross-domain comparison
    baseline, not tied to any one target's ecosystem -- matches this todo's own worked example
    (SGLang -> vllm-omni, "engine" domain vs. "omni" domain)."""
    monkeypatch.setattr(
        parity.config,
        "REPOS",
        [
            {"slug": "vllm-project/vime", "role": "primary", "domain": "rl"},
            {"slug": "sgl-project/sglang", "role": "parity", "domain": "engine"},
        ],
    )
    items = [_item("sgl-project/sglang", 1, "flash-attn-3", state="closed")]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(cells, target_engine="vllm-project/vime")

    assert [g.capability for g in gaps] == ["flash-attn-3"]


def test_find_gaps_for_all_targets_excludes_cross_domain_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression, end to end through find_gaps_for_all_targets: a capability vime (domain
    "rl") ships must not be flagged as missing on vllm (domain "speech") or vllm-omni (domain
    "omni")."""
    monkeypatch.setattr(
        parity.config,
        "REPOS",
        [
            {"slug": "vllm-project/vllm", "role": "primary", "domain": "speech"},
            {"slug": "vllm-project/vllm-omni", "role": "primary", "domain": "omni"},
            {"slug": "vllm-project/vime", "role": "primary", "domain": "rl"},
        ],
    )
    items = [_item("vllm-project/vime", 1, "ppo-trainer", state="closed")]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps_for_all_targets(cells)

    assert gaps == []


def test_find_gaps_uses_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """No explicit target_engine/source_engines -- target falls back to config.REPOS's first
    "primary" role; source_engines falls back to every other tracked engine."""
    monkeypatch.setattr(
        parity.config,
        "REPOS",
        [
            {"slug": "vllm-project/vllm-omni", "role": "primary", "domain": "omni"},
            {"slug": "sgl-project/sglang", "role": "parity", "domain": "engine"},
        ],
    )
    items = [
        _item("sgl-project/sglang", 1, "flash-attn-3", state="closed"),
        _item("vllm-project/vllm-omni", 2, "flash-attn-3", state="open"),
    ]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(cells)

    assert [g.capability for g in gaps] == ["flash-attn-3"]


# --------------------------------------------------------------------- find_gaps_for_all_targets


def test_find_gaps_for_all_targets_checks_every_primary_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: with multiple contribution targets (vllm, vllm-omni, vime), a gap check
    against a single default target would silently miss what the other targets are missing."""
    monkeypatch.setattr(
        parity.config,
        "REPOS",
        [
            {"slug": "vllm-project/vllm", "role": "primary", "domain": "speech"},
            {"slug": "vllm-project/vllm-omni", "role": "primary", "domain": "omni"},
            {"slug": "sgl-project/sglang", "role": "parity", "domain": "engine"},
        ],
    )
    items = [_item("sgl-project/sglang", 1, "flash-attn-3", state="closed")]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps_for_all_targets(cells)

    targets = {g.target_engine for g in gaps}
    assert targets == {"vllm-project/vllm", "vllm-project/vllm-omni"}
    assert all(g.capability == "flash-attn-3" for g in gaps)


def test_find_gaps_for_all_targets_raises_when_no_primary_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parity.config, "REPOS", [{"slug": "o/r", "role": "source"}])
    with pytest.raises(parity.ParityError, match="no config.REPOS entry has role 'primary'"):
        parity.find_gaps_for_all_targets([])
