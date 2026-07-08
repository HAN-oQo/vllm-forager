"""Scout agent (T2.5): candidate discovery + risk/effort/impact ranking.

This is M2's "*where can I contribute?*" output — it turns T2.4's parity gaps and the raw
GitHub signal into a single ranked queue a human (or M3's Engineer) can pick up from, top to
bottom. Every candidate is scored on **three independent dimensions** (risk, effort, impact —
each ``"low"``/``"medium"``/``"high"``), not one collapsed tier: a low-risk/low-impact
candidate isn't obviously worth doing before a medium-risk/high-impact one, and effort is what
tells you whether something is schedulable soon or someday.

Three candidate sources, per the DEVPLAN spec:

- **parity gap** (:func:`~src.parity.find_gaps_for_all_targets`) — a capability shipped on any
  tracked engine but missing on one of this project's ``"primary"`` contribution targets
  (vllm, vllm-omni, vime — T3.14 generalized this off the old hardcoded ROCm-fork-vs-upstream
  premise); the clearest, already-evidenced port candidate. A gap with no resolvable evidence
  (:attr:`~src.parity.Gap.evidence` is `None` — see that module's own docstring for when) is
  skipped, not surfaced with a fabricated-empty citation: the evidence principle means a
  candidate a human can't check a source for isn't a candidate worth ranking.
- **good-first-issue** — an open issue labeled ``"good first issue"`` (a real, common GitHub
  label — matched with hyphens/underscores normalized to spaces first, since repos spell this
  inconsistently: ``"good-first-issue"``, ``"Good First Issue"``, etc.) on any tracked repo.
- **ROCm-reproducible** — an open issue :func:`~src.agents.reporter.categorize` files under
  ``"ROCm / AMD"`` (the same word-boundary-safe keyword match the v0 report already uses) — a
  real bug on the hardware this project cares about, worth reproducing and fixing. Known
  inconsistency, not fixed here: :mod:`src.stats` already has its *own*, independent (and
  looser — labels-only substring, not this word-boundary title+labels+body match) notion of
  "ROCm relevant"; the two aren't reconciled, so an item can count as ROCm-relevant for one and
  not the other.

An item matching more than one source (e.g. a "good first issue" that's also ROCm-tagged) is
only scored once, under whichever source is checked first — deliberately not double-counted,
since ranking the same underlying work twice would misrepresent the queue's actual size.

**ROCm∩speech boost + per-repo signals (T3.16):** on top of the LLM's risk/effort/impact score,
every candidate gets a small deterministic ``boost`` added straight into :attr:`Candidate.priority`
— computed from facts already in the KB, not another LLM call, so it stays cheap and testable
offline:

- **ROCm∩speech** (+2): the candidate's repo is in the ``"speech"`` domain (config-derived via
  :func:`~src.parity._domain_of` — currently just ``vllm-project/vllm``, but not restricted to
  ``"primary"``-role repos, so a future ``"speech"``-domain ``"source"``/``"parity"`` entry
  boosts too) *and* its text matches **both** :data:`~src.config.ROCM_HINTS` and
  :data:`~src.config.SPEECH_HINTS` — the intersection :data:`~src.config.SPEECH_HINTS`'s own
  docstring names as vLLM's current top priority.
- **Edge-applicability** (+1): the candidate's repo domain is inference/serving-shaped work an
  MI250 (an inference-class accelerator) can plausibly run (``speech``/``omni``/``engine``) —
  not ``rl`` (post-training work, typically trained on large clusters rather than deployed at
  the edge) and not an untracked repo.
- **Merge-velocity** (+1): the candidate's repo ships (merges) PRs faster than the run's own
  median repo (:func:`_merge_velocity_by_repo` + :func:`_merge_velocity_median`, from
  `updated_at - created_at` on already-merged PRs in this same item batch — the same
  ``state == "closed"``-as-merged approximation :func:`~src.agents.reporter.is_merged` already
  documents). A repo with no merge history in this batch, or a batch with fewer than two repos
  to compare, gets no boost either way — unknown, not assumed slow (a *single* repo's velocity
  is trivially its own median regardless of how fast or slow it actually is, so it's excluded
  rather than always winning the boost).

A parity-gap candidate is boosted against its `target_engine` (where the port would land), using
its own synthesized title as the boost's text source — there's no real issue/PR body for a gap.

Risk/effort/impact are judged by one ``llm.complete`` call per candidate — like T1.4/T1.5/
T2.1/T2.3's own qualitative calls, these aren't reducible to a deterministic field check. A
candidate whose scoring call fails (``llm.LLMError`` or a malformed reply) is skipped and
logged, not allowed to abort the whole queue — the by-now-established per-item failure
isolation every sibling agent applies. Deliberately not batched the way T1.6's chunked report
claims or T2.3's single per-cluster naming call are: those group many *uniform* items behind
one call; a heterogeneous candidate list (three different sources, each needing its own
title/body context) doesn't have an obvious single-prompt shape to batch into, so this is one
call per candidate, and cost scales with candidate count, not with what's new since the last
run — real for a KB with many open ROCm-relevant/good-first issues, acceptable for this first
M2 cut.

Known limitations, not fixed here:
- Like :mod:`~src.agents.curator`'s proposals, this queue is recomputed fresh from the item
  list every call (:func:`discover_from_store` makes four passes over it: the store scan,
  :func:`~src.parity.build_matrix`, this module's own :func:`_merge_velocity_by_repo` scan
  (T3.16), and the discovery loop itself), with no persistence of what a human already picked
  up, and no dedup against a near-duplicate candidate seen before (T2.7's job).
- :func:`discover_from_store` computes parity gaps via
  :func:`~src.parity.find_gaps_for_all_targets`'s own ``config.REPOS``-derived defaults; if no
  entry has a ``"primary"`` role, :class:`~src.parity.ParityError` is caught and logged here
  rather than left to crash the whole run — unlike T2.2's policy-missing case, a parity
  misconfiguration shouldn't take down the two GitHub-signal sources that don't depend on it at
  all.
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import config, llm
from .. import parity as parity_module
from ..store.base import Store
from .reporter import _compile, _haystack, categorize, evidence_url, is_merged

_LEVELS = ("low", "medium", "high")
_LEVEL_SCORE = {level: score for score, level in enumerate(_LEVELS, start=1)}

_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "risk": {"type": "string", "enum": list(_LEVELS)},
        "effort": {"type": "string", "enum": list(_LEVELS)},
        "impact": {"type": "string", "enum": list(_LEVELS)},
    },
    "required": ["risk", "effort", "impact"],
}

# T3.16 boost signals — see module docstring's "ROCm∩speech boost + per-repo signals" section.
_ROCM_PATTERN = _compile(config.ROCM_HINTS)
_SPEECH_PATTERN = _compile(config.SPEECH_HINTS)

# Domains an MI250 (an inference-class accelerator) can plausibly help with -- inference/serving
# work, not "rl" (post-training, typically trained on large clusters, not deployed at the edge).
_EDGE_APPLICABLE_DOMAINS = frozenset({"speech", "omni", "engine"})

# Scale factor for Candidate.priority (see its own docstring): keeps `boost` a pure tiebreak
# among equal risk/effort/impact combinations, never able to overturn a genuine difference
# between them, since the smallest possible gap between two distinct risk/effort/impact scores
# (1, see _LEVEL_SCORE) times this scale (10) exceeds boost's current maximum (+4) many times
# over -- with headroom for future boost signals too, as long as their sum stays under 10.
_PRIORITY_SCALE = 10


class ScoutError(RuntimeError):
    """A candidate was constructed with a risk/effort/impact value outside `_LEVELS`."""


@dataclass(frozen=True)
class Candidate:
    """One ranked contribution candidate.

    `priority` combines all three LLM-judged dimensions plus `boost` into one sort key — impact
    weighted most heavily, risk and effort each a mild penalty, `boost` added on top (see module
    docstring's "ROCm∩speech boost + per-repo signals" section). It's a ranking aid, not a claim
    of precise value.
    """

    title: str
    source: str
    risk: str
    effort: str
    impact: str
    evidence: str
    boost: int = 0

    def __post_init__(self) -> None:
        for name, value in (("risk", self.risk), ("effort", self.effort), ("impact", self.impact)):
            if value not in _LEVEL_SCORE:
                raise ScoutError(f"{name} must be one of {_LEVELS}, got {value!r}")

    @property
    def priority(self) -> int:
        """(3×impact − risk − effort) × :data:`_PRIORITY_SCALE` + boost — weighted so impact
        dominates enough to satisfy the DEVPLAN's own acceptance criterion (a low-risk/low-impact
        candidate never outranks a medium-risk/high-impact one, for any effort value — see this
        module's own test), while still letting a cheap, low-risk medium-impact candidate outrank
        an expensive, risky high-impact one: a deliberate tiebreak, not a gap, since effort is
        what makes a candidate schedulable soon rather than someday (see module docstring).

        `boost` (T3.16) is added only *after* scaling the risk/effort/impact term by
        :data:`_PRIORITY_SCALE` — not summed in directly — so it can only ever break a tie
        between two candidates with the **identical** risk/effort/impact combination; it can
        never overturn a genuine difference between them (a plain, unscaled sum let a max-boosted
        low-risk/low-impact candidate outrank a medium-risk/high-impact one, contradicting the
        acceptance criterion above — a real regression caught in review, not a hypothetical).
        """
        base = 3 * _LEVEL_SCORE[self.impact] - _LEVEL_SCORE[self.risk] - _LEVEL_SCORE[self.effort]
        return base * _PRIORITY_SCALE + self.boost


def _score_prompt(title: str, body: str, source: str) -> str:
    return (
        "Score this vLLM/ROCm open-source contribution candidate on three independent "
        "dimensions, each exactly 'low', 'medium', or 'high':\n"
        "- risk: how likely a fix/port here introduces regressions or hits something "
        "technically tricky.\n"
        "- effort: how much implementation work this looks like.\n"
        "- impact: how much this would matter to ROCm/vLLM users if done.\n\n"
        f"Source: {source}\nTitle: {title}\n\nBody: {body[:2000]}\n\n"
        "Reply with `risk`, `effort`, `impact`."
    )


def _score(title: str, body: str, source: str) -> dict[str, str] | None:
    """`{"risk": ..., "effort": ..., "impact": ...}`, or `None` if the call failed or the
    model's reply didn't shape into three valid levels."""
    try:
        reply = llm.complete(_score_prompt(title, body, source), json_schema=_SCORE_SCHEMA)
    except llm.LLMError as exc:
        print(f"scout: skipping {source} candidate {title!r}: {exc}", file=sys.stderr)
        return None
    if not isinstance(reply, dict):
        return None
    scores: dict[str, str] = {}
    for key in ("risk", "effort", "impact"):
        value = reply.get(key)
        if value not in _LEVEL_SCORE:
            return None
        scores[key] = str(value)
    return scores


def _parse_ts(raw: object) -> datetime | None:
    """Best-effort UTC timestamp parse for :func:`_merge_velocity_by_repo` — `None` (not a
    raised error) on anything missing, non-``str``, or malformed, so one bad timestamp drops
    just that item from the average rather than aborting the whole ranking (the same per-item
    degradation this module's LLM-scoring path already applies). A parsed value with no ``Z``/
    offset is treated as UTC (like :func:`~src.agents.forecaster.parse_ts` already does for its
    own timestamps) rather than left timezone-naive — two naive/aware values would otherwise
    raise `TypeError` on subtraction in :func:`_merge_velocity_by_repo`, which is exactly the
    "abort the whole ranking" outcome this function exists to avoid.
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _rocm_speech_boost(item: dict) -> int:
    """+2 if `item`'s repo is in the ``"speech"`` domain (config-derived via
    :func:`~src.parity._domain_of` — not restricted to `"primary"`-role repos, so a future
    ``"speech"``-domain ``"source"``/``"parity"`` entry gets boosted too, not just today's one
    ``"primary"`` speech target) AND its title/labels/body match **both** `config.ROCM_HINTS`
    and `config.SPEECH_HINTS`; 0 otherwise. Computed fresh from `config.REPOS` on every call
    (via `_domain_of`), not cached at import time, so a test that monkeypatches `config.REPOS`
    (as :mod:`tests.test_scout` already does for parity errors) is honored here too."""
    if parity_module._domain_of(item.get("repo") or "") != "speech":
        return 0
    text = _haystack(item)
    if _ROCM_PATTERN.search(text) and _SPEECH_PATTERN.search(text):
        return 2
    return 0


def _edge_applicability_boost(item: dict) -> int:
    """+1 if `item`'s repo domain is in :data:`_EDGE_APPLICABLE_DOMAINS`; 0 for `"rl"` or an
    untracked repo (:func:`~src.parity._domain_of` returns `None`)."""
    domain = parity_module._domain_of(item.get("repo") or "")
    return 1 if domain in _EDGE_APPLICABLE_DOMAINS else 0


def _merge_velocity_by_repo(items: list[dict]) -> dict[str, float]:
    """Mean days-to-merge (`updated_at - created_at`, the same closed-as-merged approximation
    :func:`~src.agents.reporter.is_merged` already documents) for every repo with at least one
    shipped PR in `items` with parseable timestamps — the fewer days, the faster that repo
    accepts work. A repo with no shipped PRs, or none with parseable timestamps, has no entry.
    """
    days_by_repo: dict[str, list[float]] = defaultdict(list)
    for item in items:
        if not is_merged(item):
            continue
        repo = item.get("repo")
        created = _parse_ts(item.get("created_at"))
        updated = _parse_ts(item.get("updated_at"))
        if not repo or created is None or updated is None:
            continue
        days_by_repo[repo].append((updated - created).total_seconds() / 86400)
    return {repo: sum(days) / len(days) for repo, days in days_by_repo.items() if days}


def _merge_velocity_median(velocity_by_repo: dict[str, float]) -> float | None:
    """Median of every repo's mean merge time, or `None` with fewer than two repos to compare —
    one repo's velocity is trivially its own median regardless of how fast or slow it actually
    is, so a single-repo batch has no real "faster than typical" comparison to make. Computed
    once per :func:`discover_candidates` call, not once per candidate."""
    if len(velocity_by_repo) < 2:
        return None
    return statistics.median(velocity_by_repo.values())


def _merge_velocity_boost(
    repo: str | None, velocity_by_repo: dict[str, float], median: float | None
) -> int:
    """+1 if `repo` merges PRs at or faster than `median` (a relative, run-local "faster than
    typical" signal, not a fixed day threshold); 0 for a slower repo, an unknown repo, or when
    `median` is `None` (fewer than two repos had merge-velocity data this run)."""
    if repo is None or median is None or repo not in velocity_by_repo:
        return 0
    return 1 if velocity_by_repo[repo] <= median else 0


def _boost_for(
    item: dict, velocity_by_repo: dict[str, float], velocity_median: float | None
) -> int:
    """Sum of every T3.16 boost signal for `item` (a real issue/PR, or a synthetic
    ``{"repo": ..., "title": ..., "body": ""}`` stand-in for a parity gap)."""
    return (
        _rocm_speech_boost(item)
        + _edge_applicability_boost(item)
        + _merge_velocity_boost(item.get("repo"), velocity_by_repo, velocity_median)
    )


def _candidate_source(item: dict) -> str | None:
    """Which of T2.5's two GitHub-signal sources `item` qualifies for, or `None` for neither."""
    labels = [
        str(label).lower().replace("-", " ").replace("_", " ") for label in item.get("labels") or []
    ]
    if any("good first issue" in label for label in labels):
        return "good-first-issue"
    if categorize(item) == "ROCm / AMD":
        return "rocm-reproducible"
    return None


def _try_score_and_append(
    candidates: list[Candidate],
    *,
    title: str,
    body: str,
    source: str,
    evidence: str,
    boost: int = 0,
) -> None:
    """Score `(title, body, source)`; append a :class:`Candidate` (carrying `boost`, T3.16) to
    `candidates` if it scored (silently a no-op otherwise — :func:`_score` already logged why)."""
    scores = _score(title, body, source)
    if scores is not None:
        candidates.append(
            Candidate(title=title, source=source, evidence=evidence, boost=boost, **scores)
        )


def _sample_matched_per_domain(items: list[dict], limit: int) -> list[dict]:
    """At most `limit` open, good-first-issue/ROCm-reproducible-eligible issues **per domain**
    (:func:`~src.parity._domain_of`), preserving `items`' own relative order.

    Matching is re-derived here (open+type check, :func:`_candidate_source`) so the cap only
    counts against items that would actually reach an LLM scoring call — bounds the cost of a
    KB with hundreds of matching issues (e.g. right after a multi-domain retarget, T3.19) the
    same way :func:`~src.agents.analyst._sample_per_domain` bounds :func:`analyze_store`'s
    classification cost, and for the same reason: an uneven repo-per-domain split shouldn't
    give one domain a disproportionate share of a bounded sample.
    """
    counts: dict[str | None, int] = defaultdict(int)
    sampled = []
    for item in items:
        if item.get("type") != "issue" or item.get("state") != "open":
            continue
        if _candidate_source(item) is None:
            continue
        domain = parity_module._domain_of(item.get("repo") or "")
        if counts[domain] >= limit:
            continue
        counts[domain] += 1
        sampled.append(item)
    return sampled


def discover_candidates(
    items: list[dict],
    gaps: list[parity_module.Gap] | None = None,
    *,
    per_domain_limit: int | None = None,
) -> list[Candidate]:
    """Every parity-gap, good-first-issue, and ROCm-reproducible candidate found in `items`
    (open issues only for the latter two — a PR isn't something to "reproduce and fix"),
    scored and ranked by :attr:`Candidate.priority`, highest first.

    `gaps` (typically :func:`~src.parity.find_gaps_for_all_targets` over `items`' own
    :func:`~src.parity.build_matrix`) is accepted as a parameter rather than recomputed here,
    so a caller that already has it (e.g. :func:`discover_from_store`) doesn't pay for it
    twice.

    `per_domain_limit` caps how many good-first-issue/ROCm-reproducible candidates *per domain*
    get an LLM scoring call this run (via :func:`_sample_matched_per_domain`) — a large KB can
    have hundreds of matching open issues, each a separate ``llm.complete`` call; a small
    per-domain cap still samples every domain for a bounded dry run (T3.19). Parity-gap
    candidates are never capped — one gap per (target, capability) pair is already naturally
    bounded by the taxonomy's own size, not the KB's raw item count. `None` (the default)
    scores every match, as before.
    """
    candidates: list[Candidate] = []
    seen: set[tuple[str | None, int | None]] = set()
    velocity_by_repo = _merge_velocity_by_repo(items)
    velocity_median = _merge_velocity_median(velocity_by_repo)

    for gap in gaps or []:
        if not gap.evidence:
            continue
        title = f"Port to {gap.target_engine}: {gap.capability} (shipped in {gap.source_engine})"
        gap_item = {"repo": gap.target_engine, "title": title, "body": ""}
        _try_score_and_append(
            candidates,
            title=title,
            body="",
            source="parity-gap",
            evidence=gap.evidence,
            boost=_boost_for(gap_item, velocity_by_repo, velocity_median),
        )

    eligible_items = (
        items if per_domain_limit is None else _sample_matched_per_domain(items, per_domain_limit)
    )
    for item in eligible_items:
        if item.get("type") != "issue" or item.get("state") != "open":
            continue
        key = (item.get("repo"), item.get("number"))
        if key in seen:
            continue
        source = _candidate_source(item)
        if source is None:
            continue
        seen.add(key)
        title = item.get("title") or ""
        _try_score_and_append(
            candidates,
            title=title,
            body=item.get("body") or "",
            source=source,
            evidence=evidence_url(item),
            boost=_boost_for(item, velocity_by_repo, velocity_median),
        )

    return sorted(candidates, key=lambda c: c.priority, reverse=True)


def discover_from_store(store: Store, *, per_domain_limit: int | None = None) -> list[Candidate]:
    """:func:`discover_candidates` over every item currently in `store` — see module docstring
    for why a :class:`~src.parity.ParityError` here degrades to "no parity gaps" rather than
    aborting the good-first-issue/ROCm-reproducible sources too.
    """
    items = store.query()
    gaps = parity_module.safe_find_gaps_for_all_targets(
        parity_module.build_matrix(items), caller="scout"
    )
    return discover_candidates(items, gaps, per_domain_limit=per_domain_limit)
