"""Scout agent (T2.5): candidate discovery + risk/effort/impact ranking.

This is M2's "*where can I contribute?*" output — it turns T2.4's parity gaps and the raw
GitHub signal into a single ranked queue a human (or M3's Engineer) can pick up from, top to
bottom. Every candidate is scored on **three independent dimensions** (risk, effort, impact —
each ``"low"``/``"medium"``/``"high"``), not one collapsed tier: a low-risk/low-impact
candidate isn't obviously worth doing before a medium-risk/high-impact one, and effort is what
tells you whether something is schedulable soon or someday.

Three candidate sources, per the DEVPLAN spec:

- **parity gap** (:func:`~src.parity.find_gaps`) — a capability shipped on the ROCm fork but
  missing upstream; the clearest, already-evidenced port candidate. A gap with no resolvable
  evidence (:attr:`~src.parity.Gap.evidence` is `None` — see that module's own docstring for
  when) is skipped, not surfaced with a fabricated-empty citation: the evidence principle means
  a candidate a human can't check a source for isn't a candidate worth ranking.
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
  list every call (:func:`discover_from_store` alone makes three passes over it: the store
  scan, :func:`~src.parity.build_matrix`, and this module's own discovery loop), with no
  persistence of what a human already picked up, and no dedup against a near-duplicate
  candidate seen before (T2.7's job).
- :func:`discover_from_store` computes parity gaps via :func:`~src.parity.find_gaps`'s own
  ``config.REPOS``-derived defaults; if no entry has a ``"fork"``/``"primary"`` role,
  :class:`~src.parity.ParityError` is caught and logged here rather than left to crash the
  whole run — unlike T2.2's policy-missing case, a parity misconfiguration shouldn't take down
  the two GitHub-signal sources that don't depend on it at all.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .. import llm
from .. import parity as parity_module
from ..store.base import Store
from .reporter import categorize, evidence_url

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


class ScoutError(RuntimeError):
    """A candidate was constructed with a risk/effort/impact value outside `_LEVELS`."""


@dataclass(frozen=True)
class Candidate:
    """One ranked contribution candidate.

    `priority` combines all three dimensions into one sort key — impact weighted most heavily,
    risk and effort each a mild penalty. It's a ranking aid, not a claim of precise value.
    """

    title: str
    source: str
    risk: str
    effort: str
    impact: str
    evidence: str

    def __post_init__(self) -> None:
        for name, value in (("risk", self.risk), ("effort", self.effort), ("impact", self.impact)):
            if value not in _LEVEL_SCORE:
                raise ScoutError(f"{name} must be one of {_LEVELS}, got {value!r}")

    @property
    def priority(self) -> int:
        """3×impact − risk − effort — weighted so impact dominates enough to satisfy the
        DEVPLAN's own acceptance criterion (a low-risk/low-impact candidate never outranks a
        medium-risk/high-impact one, for any effort value — see this module's own test), while
        still letting a cheap, low-risk medium-impact candidate outrank an expensive, risky
        high-impact one: a deliberate tiebreak, not a gap, since effort is what makes a
        candidate schedulable soon rather than someday (see module docstring).
        """
        return 3 * _LEVEL_SCORE[self.impact] - _LEVEL_SCORE[self.risk] - _LEVEL_SCORE[self.effort]


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
    candidates: list[Candidate], *, title: str, body: str, source: str, evidence: str
) -> None:
    """Score `(title, body, source)`; append a :class:`Candidate` to `candidates` if it scored
    (silently a no-op otherwise — :func:`_score` already logged why)."""
    scores = _score(title, body, source)
    if scores is not None:
        candidates.append(Candidate(title=title, source=source, evidence=evidence, **scores))


def discover_candidates(
    items: list[dict], gaps: list[parity_module.Gap] | None = None
) -> list[Candidate]:
    """Every parity-gap, good-first-issue, and ROCm-reproducible candidate found in `items`
    (open issues only for the latter two — a PR isn't something to "reproduce and fix"),
    scored and ranked by :attr:`Candidate.priority`, highest first.

    `gaps` (typically :func:`~src.parity.find_gaps` over `items`' own
    :func:`~src.parity.build_matrix`) is accepted as a parameter rather than recomputed here,
    so a caller that already has it (e.g. :func:`discover_from_store`) doesn't pay for it
    twice.
    """
    candidates: list[Candidate] = []
    seen: set[tuple[str | None, int | None]] = set()

    for gap in gaps or []:
        if not gap.evidence:
            continue
        title = f"Port to upstream: {gap.capability}"
        _try_score_and_append(
            candidates, title=title, body="", source="parity-gap", evidence=gap.evidence
        )

    for item in items:
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
        )

    return sorted(candidates, key=lambda c: c.priority, reverse=True)


def discover_from_store(store: Store) -> list[Candidate]:
    """:func:`discover_candidates` over every item currently in `store` — see module docstring
    for why a :class:`~src.parity.ParityError` here degrades to "no parity gaps" rather than
    aborting the good-first-issue/ROCm-reproducible sources too.
    """
    items = store.query()
    try:
        gaps = parity_module.find_gaps(parity_module.build_matrix(items))
    except parity_module.ParityError as exc:
        print(f"scout: skipping parity gaps: {exc}", file=sys.stderr)
        gaps = []
    return discover_candidates(items, gaps)
