"""Scout agent (T2.5): candidate discovery + risk/effort/impact ranking.

This is M2's "*where can I contribute?*" output — it turns T2.4's parity gaps and the raw
GitHub signal into a single ranked queue a human (or M3's Engineer) can pick up from, top to
bottom. Every candidate is scored on **three independent dimensions** (risk, effort, impact —
each ``"low"``/``"medium"``/``"high"``), not one collapsed tier: a low-risk/low-impact
candidate isn't obviously worth doing before a medium-risk/high-impact one, and effort is what
tells you whether something is schedulable soon or someday.

Three candidate sources, per the DEVPLAN spec:

- **parity gap** (:func:`~src.parity.find_gaps`) — a capability shipped on the ROCm fork but
  missing upstream; the clearest, already-evidenced port candidate.
- **good-first-issue** — an open issue labeled ``"good first issue"`` (a real, common GitHub
  label) on any tracked repo.
- **ROCm-reproducible** — an open issue :func:`~src.agents.reporter.categorize` files under
  ``"ROCm / AMD"`` (the same word-boundary-safe keyword match the v0 report already uses) — a
  real bug on the hardware this project cares about, worth reproducing and fixing.

An item matching more than one source (e.g. a "good first issue" that's also ROCm-tagged) is
only scored once, under whichever source is checked first — deliberately not double-counted,
since ranking the same underlying work twice would misrepresent the queue's actual size.

Risk/effort/impact are judged by ``llm.complete`` — like T1.4/T1.5/T2.1/T2.3's own qualitative
calls, these aren't reducible to a deterministic field check. A candidate whose scoring call
fails (``llm.LLMError`` or a malformed reply) is skipped and logged, not allowed to abort the
whole queue — the by-now-established per-item failure isolation every sibling agent applies.

Known limitation, not fixed here: like :mod:`~src.agents.curator`'s proposals, this queue is
recomputed fresh from a full item scan every call, with no persistence of what a human already
picked up, and no dedup against a near-duplicate candidate seen before (T2.7's job).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .. import llm
from .. import parity as parity_module
from ..store.base import Store
from . import reporter
from .reporter import evidence_url

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
        """3×impact − risk − effort.

        Weighting impact at 3× satisfies the DEVPLAN's own acceptance criterion — a
        low-risk/low-impact candidate's best case (score 1) never outranks a
        medium-risk/high-impact candidate's worst case (score 4), regardless of effort — but
        impact does *not* unconditionally dominate in general: a cheap, low-risk medium-impact
        candidate (best case: score 4) can still outrank an expensive, risky high-impact one
        (worst case: score 3). That's deliberate, not a gap — effort is what makes a candidate
        "schedulable soon" (see module docstring), so it should be able to tip a close call.
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
    scores = {key: reply.get(key) for key in ("risk", "effort", "impact")}
    if any(value not in _LEVEL_SCORE for value in scores.values()):
        return None
    return {key: str(value) for key, value in scores.items()}


def _candidate_source(item: dict) -> str | None:
    """Which of T2.5's two GitHub-signal sources `item` qualifies for, or `None` for neither."""
    labels = [str(label).lower() for label in item.get("labels") or []]
    if any("good first issue" in label for label in labels):
        return "good-first-issue"
    if reporter.categorize(item) == "ROCm / AMD":
        return "rocm-reproducible"
    return None


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
    candidates = []
    seen: set[tuple[str | None, int | None]] = set()

    for gap in gaps or []:
        title = f"Port to upstream: {gap.capability}"
        scores = _score(title, "", "parity-gap")
        if scores is None:
            continue
        candidates.append(
            Candidate(title=title, source="parity-gap", evidence=gap.evidence or "", **scores)
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
        scores = _score(title, item.get("body") or "", source)
        if scores is None:
            continue
        candidates.append(
            Candidate(title=title, source=source, evidence=evidence_url(item), **scores)
        )

    return sorted(candidates, key=lambda c: c.priority, reverse=True)


def discover_from_store(store: Store) -> list[Candidate]:
    """:func:`discover_candidates` over every item currently in `store`."""
    items = store.query()
    gaps = parity_module.find_gaps(parity_module.build_matrix(items))
    return discover_candidates(items, gaps)
