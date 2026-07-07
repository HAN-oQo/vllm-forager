"""Analyst agent (T1.4, path-classifying since T1.5.2): classify delta items into the active
taxonomy via ``llm.complete``.

The first LLM-backed stage of the intelligence plane: every collected issue/PR that hasn't
been classified yet ("delta" — no ``path`` field, so a rerun only touches new items and is
safe to repeat) is bucketed into the KB's current taxonomy, and the result is written back to
the item's own KB record — no separate classification table, so a later reporter/grader reads
one record per item, not a join.

Evidence principle: classification never drops or rewrites an item's own source ``url`` — a
category is an *addition* to the record, not a replacement of it. Each classified record also
carries ``taxonomy_version``, the version under which it was classified, so a later audit or
grading pass (T2.2) can ask "what taxonomy was active for this item" without guessing.

T1.5.2: the taxonomy is a tree of **paths** (T1.5.1), not a flat list, so classification now
walks the tree one level at a time — the model returns a whole path in one ``llm.complete``
call (cheaper than one call per level), and :func:`_validated_subpath` then validates it level
by level against :meth:`~src.taxonomy.Taxonomy.children`: the first level whose value isn't
one of that step's controlled options is where drift stops the walk, keeping the valid prefix
classified so far (an item is never rejected outright for going one level too far off-script).

**Domain-rooted classification (T3.15):** the target set went multi-domain (PR #82) — vLLM
(ROCm speech) · vllm-omni · vime, plus the RL/omni ecosystems watched around them
(:data:`~src.config.REPOS`'s ``domain`` field: ``"speech"``/``"rl"``/``"omni"``/``"engine"``).
A ROCm-kernel-only taxonomy can't classify a `verl` PPO PR or a `diffusers` scheduler bug into
anything meaningful. Rather than ask the model to also guess the domain (it's already a known
KB fact — :func:`_domain_of_item` reads it straight from ``config.REPOS`` by the item's own
``repo``), every classified path is now rooted at that domain deterministically: the model
only ever classifies the levels *beneath* it (:func:`_path_prompt`/:func:`_validated_subpath`
are both scoped to ``prefix=(domain,)``), and an item whose repo isn't tracked in
``config.REPOS`` at all (a stale/retired repo — e.g. ``ROCm/vllm``, collected before T3.14
dropped it) falls back to a depth-1 :data:`OTHER` path with no LLM call — mirrors the old
"hallucinated root" fallback, just for a different reason (unknown domain, not an off-taxonomy
guess). This is also what "seeds" the domain-rooted top-level nodes T1.5.4's tree report shows
(:func:`~src.agents.reporter_v1.build_tree` groups purely by each item's own classified
``path``, not by :attr:`~src.taxonomy.Taxonomy.categories` directly) — no separate taxonomy
bootstrap step needed; the first classified item under a domain is what puts that domain on
the tree.

An item whose domain has no registered subcategories yet (``taxonomy.children((domain,))`` is
empty — e.g. right after T3.15 ships, before T2.3's Curator has proposed anything under a new
domain) gets classified as just ``(domain,)``, no LLM call: classifying finer than the domain
root would be an unconstrained guess with nothing to validate the answer against, the same
"don't ask when there's nothing to check the reply against" rule the old whole-taxonomy-empty
short-circuit applied globally (T3.15 makes that check per-domain instead, since a partially
seeded taxonomy can have subcategories under one domain and none under another).

Two fields are written: ``path`` (the list of levels — T1.5.4's tree report reads this) and
``category`` (``path`` flattened via :data:`~src.taxonomy.LEVEL_SEPARATOR` — the same join
:attr:`~src.taxonomy.Taxonomy.labels` uses — kept for T1.6/T1.7's reporter/trends modules,
which only know how to group by one flat string per item and haven't been taught to read
``path`` yet). Known limitation, not fixed here: once the taxonomy grows past depth 1,
:mod:`src.trends`'s ``category_trends`` will bucket by the *full* flattened path (e.g. an
opaque ``"rl > post-training > PPO"`` bucket) rather than rolling up under ``"rl"`` — a
T1.6/T1.7 concern to fix once ``path`` is available to group by, not this module's.

Known limitation, not fixed here: :func:`_path_prompt` lists every known subpath under an
item's domain in one unbounded, comma-joined hint string, and the model never sees
:meth:`~src.taxonomy.Taxonomy.children`'s exact per-level options before answering
(:func:`_validated_subpath` validates its whole guess post-hoc instead) — both fine at
today's taxonomy size (never seeded in production yet), but worth revisiting once the
taxonomy is large/deep enough that prompt length or guess-then-validate accuracy actually
matters (T1.5.4/T2.3).

Known limitation, not fixed here (a code-review finding on this rework, confirmed by 4
independent finder angles): a domain with no registered subcategories yet now classifies as
just ``(domain,)``, not :data:`OTHER`. :mod:`~src.agents.curator`'s ``propose_new_categories``
(T2.3, unchanged by this rework) discovers new-category candidates by clustering items where
``category == OTHER`` — domain-only items no longer satisfy that filter. Since every domain
starts with zero subcategories in production, this silently disables curator's already-built
new-category-discovery pipeline for every domain until a human manually calls
:func:`~src.taxonomy.add_category` at least once per domain to seed a first subcategory — the
exact population (domain-root-only items) that most needs it. A real fix means teaching
``curator.py`` to cluster per-domain (not just :data:`OTHER`) and giving its
``NewCategoryProposal`` a domain field so a human can add a correctly domain-rooted path — a
design change to a different module/milestone (T2.3/M2), not a contained fix within this
rework's own files; left for a dedicated follow-up. Relatedly, :func:`~src.taxonomy.
add_category`/:func:`~src.taxonomy.create_taxonomy` have no concept of the four
:data:`~src.config.REPOS` domain names as reserved level-0 roots — nothing stops
``add_category(store, "not-a-real-domain")`` from creating a permanently unreachable top-level
category, since :func:`classify_item` only ever walks from ``config.REPOS``-derived domains,
never from ``taxonomy.categories``' own roots. Low risk in practice (no crash, just dead data
a human would need to notice by inspection) — the domain-rooted design lives entirely in this
module's read path, with no corresponding write-side enforcement in the shared
:class:`~src.taxonomy.Taxonomy`/``Store`` layer; deferred alongside the curator fix above.

Previously a known limitation, fixed in T1.10: ``Store.upsert_items`` now merges the given
fields onto an existing ``(repo, number)`` record rather than fully replacing it (both
backends). Before that fix, the collector's own ``_normalize()`` — which never carries
classification fields forward — would silently erase this module's classification the next
time it re-fetched an already-classified item (any new comment/label bumps ``updated_at`` back
into the incremental window). Now the collector's re-normalized record simply doesn't mention
``path``/``category``, and the store-level merge leaves the existing value alone — which also
means the "delta" check above no longer treats a re-clobbered item as unclassified again, so a
re-collected item doesn't get repeatedly and wastefully re-sent through ``llm.complete``.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from .. import llm
from ..parity import _domain_of
from ..store.base import Store
from ..taxonomy import LEVEL_SEPARATOR, CategoryPath, Taxonomy, casefold_label
from ..taxonomy import get_active as get_active_taxonomy
from .reporter import OTHER

_PATH_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "array", "items": {"type": "string"}}},
    "required": ["path"],
}

# How much of an item's body to feed the model — enough for context without an unbounded
# prompt on the rare very-long issue.
_BODY_CHARS = 2000


def _domain_of_item(item: dict) -> str | None:
    """The item's repo's :data:`~src.config.REPOS` ``domain``, or `None` if the repo isn't
    tracked there (a stale/retired repo — e.g. ``ROCm/vllm``, collected before T3.14 dropped
    it — or a malformed record). Domain is a deterministic KB fact, not something the model
    should guess: it only ever classifies *beneath* it (see :func:`classify_item`).

    Reuses :func:`~src.parity._domain_of` (the identical ``config.REPOS`` slug->domain lookup
    T3.14 already added) rather than a second copy — a code-review finding on this module."""
    return _domain_of(item.get("repo") or "")


def _known_subpaths(taxonomy: Taxonomy, domain: str) -> tuple[str, ...]:
    """Every known category path under `domain`, flattened with the domain prefix stripped —
    the model is scoped to answering *beneath* `domain` (see :func:`_path_prompt`), so it
    should never see (or need to repeat) the domain level itself."""
    normalized_domain = casefold_label(domain)
    return tuple(
        LEVEL_SEPARATOR.join(path[1:])
        for path in taxonomy.categories
        if len(path) > 1 and casefold_label(path[0]) == normalized_domain
    )


def _path_prompt(item: dict, taxonomy: Taxonomy, domain: str) -> str:
    """The classification prompt for one item: title + a body excerpt + the known subpaths
    under its deterministic `domain` root (T3.15) — the domain itself is never asked of the
    model; see module docstring."""
    title = item.get("title") or ""
    body = (item.get("body") or "")[:_BODY_CHARS]
    known_subpaths = ", ".join(_known_subpaths(taxonomy, domain)) or "(none yet)"
    return (
        f"Classify the following GitHub issue/PR into a subcategory under the {domain!r} "
        f"domain of a taxonomy tree. Known subcategory paths under this domain (root > ... > "
        f"leaf, domain prefix omitted): {known_subpaths}.\n\n"
        "Respond with a path as a JSON array of level names *beneath* the domain, root "
        'first — e.g. ["post-training", "PPO"]. Use an existing path exactly where it fits; '
        "if you're only confident about the higher levels, return a shorter path (just the "
        "levels you're sure of) rather than guessing the rest. Return an empty array if "
        "nothing more specific than the domain itself applies.\n\n"
        f"Title: {title}\n\nBody: {body}"
    )


def _validated_subpath(
    raw: object,
    taxonomy: Taxonomy,
    *,
    prefix: CategoryPath,
    root_options: tuple[str, ...] | None = None,
) -> CategoryPath:
    """The longest prefix of `raw`'s levels that validates against `taxonomy`'s tree, walked
    from `prefix` (T3.15's deterministic domain root) via :meth:`~src.taxonomy.Taxonomy.
    children` — the "controlled per-level label set" T1.5.2 introduced, scoped to whatever
    root the caller already knows rather than the tree's true root.

    `root_options` is `prefix`'s own children, if the caller already computed it (e.g.
    :func:`classify_item` calls :meth:`~src.taxonomy.Taxonomy.children` once to decide whether
    there's anything to validate against at all) — reused for the walk's first iteration
    instead of a second identical, O(len(categories)) scan; computed here if omitted.

    Matches each level case/whitespace-insensitively (:func:`~src.taxonomy.casefold_label`);
    the walk stops at the first level that doesn't match (keeping the valid sub-prefix, not
    discarding the whole classification). Returns ``()`` — never raises, never falls back to
    :data:`OTHER` itself — if `raw` isn't a list of strings, or its first level doesn't match
    `prefix`'s own children: the caller already has `prefix` as a valid classification on its
    own (see :func:`classify_item`).
    """
    if not isinstance(raw, list) or not all(isinstance(level, str) for level in raw):
        return ()
    validated: list[str] = []
    for level in raw:
        options = root_options if not validated and root_options is not None else None
        if options is None:
            options = taxonomy.children((*prefix, *validated))
        normalized = casefold_label(level)
        match = next((opt for opt in options if casefold_label(opt) == normalized), None)
        if match is None:
            break
        validated.append(match)
    return tuple(validated)


def _classified_record(item: dict, path: CategoryPath, taxonomy_version: int) -> dict:
    """`item`'s fields (its ``url`` citation included, unchanged) plus ``path`` (list of
    levels), ``category`` (``path`` flattened via :data:`~src.taxonomy.LEVEL_SEPARATOR`, for
    callers that only read a flat label), and ``taxonomy_version``.

    The one place a classification result — LLM-produced or a fallback — is assembled, so
    :func:`classify_item` can never produce a result that disagrees with the shape of "no
    valid classification" (e.g. if :data:`OTHER`'s own representation ever changes).
    """
    return {
        **item,
        "path": list(path),
        "category": LEVEL_SEPARATOR.join(path),
        "taxonomy_version": taxonomy_version,
    }


def classify_item(item: dict, taxonomy: Taxonomy) -> dict:
    """Classify one item into a path in `taxonomy`'s tree, rooted at its deterministic
    :data:`~src.config.REPOS` domain (T3.15) — see module docstring.

    Returns a **new** dict — see :func:`_classified_record`. An item whose repo isn't tracked
    in :data:`~src.config.REPOS` (no domain known) falls back to a depth-1 :data:`OTHER` path;
    an item whose domain has no registered subcategories yet falls back to just the domain
    itself — neither case calls the LLM (see module docstring).

    Raises:
        llm.LLMError: the completion call failed (transport error, timeout, non-JSON reply) —
            only possible when a domain is known **and** it already has subcategories to
            validate an answer against.
    """
    domain = _domain_of_item(item)
    if domain is None:
        return _classified_record(item, (OTHER,), taxonomy.version)
    root_options = taxonomy.children((domain,))
    if not root_options:
        return _classified_record(item, (domain,), taxonomy.version)
    reply = llm.complete(_path_prompt(item, taxonomy, domain), json_schema=_PATH_SCHEMA)
    raw_path = reply.get("path") if isinstance(reply, dict) else None
    sublevels = _validated_subpath(raw_path, taxonomy, prefix=(domain,), root_options=root_options)
    return _classified_record(item, (domain, *sublevels), taxonomy.version)


def _sample_per_domain(items: list[dict], limit: int) -> list[dict]:
    """At most `limit` items per **domain** (:func:`_domain_of_item`), preserving `items`' own
    relative order.

    Grouped by domain, not repo: `config.REPOS` has wildly uneven repo counts per domain (one
    `"speech"` repo vs. seven `"rl"` repos, as of T3.19) — a per-*repo* cap would silently give
    `"rl"` up to 7x `"speech"`'s sample size, systematically under-representing `"speech"`, the
    one domain `config.py`'s own comments call this project's current top priority. Grouping by
    domain instead guarantees every domain gets an equal look regardless of how many repos
    happen to be tracked under it — the actual property a retarget dry run (T3.19) needs, since
    the retarget itself is about domains, not individual repos. An item whose repo isn't
    tracked in `config.REPOS` (`_domain_of_item` returns `None`) is grouped under `None`, capped
    the same as any other "domain".
    """
    counts: dict[str | None, int] = defaultdict(int)
    sampled = []
    for item in items:
        domain = _domain_of_item(item)
        if counts[domain] >= limit:
            continue
        counts[domain] += 1
        sampled.append(item)
    return sampled


def analyze_store(store: Store, *, per_domain_limit: int | None = None) -> list[dict]:
    """Classify every not-yet-classified item in `store` and write the results back.

    "Delta" = items with no ``path`` key yet, so a rerun only classifies what an earlier run
    (or the most recent collection) hasn't already labeled — this also means an item
    classified by the pre-T1.5.2 flat-``category``-only Analyst gets re-classified once, to
    backfill its ``path``. A failing item (an ``llm.LLMError``) is skipped — logged to stderr
    and left pending for the next run — rather than discarding every other item already
    classified in this batch.

    `per_domain_limit` caps how many pending items *per domain* get classified this run (via
    :func:`_sample_per_domain`) — for a KB with tens of thousands of pending items across many
    repos and domains (e.g. right after a multi-domain retarget, T3.19), classifying everything
    is a multi-hour, one-LLM-call-per-item commitment; a small per-domain cap still gives every
    domain an equal-sized, representative slice for a bounded dry run instead of one domain's
    disproportionate repo count skewing the sample. `None` (the default) classifies everything
    pending, as before. Must be a positive int if given — raises `ValueError` otherwise, rather
    than a `0`-or-negative value silently classifying nothing and printing the exact same
    "classified 0 item(s)" a genuinely empty backlog would (indistinguishable operator footgun,
    caught on T3.19's own code review).

    Returns the newly-classified items (``[]`` if there was nothing to do).

    Raises:
        ValueError: `per_domain_limit` is given and isn't a positive int.
        TaxonomyError: there are pending items but no taxonomy has been created yet.
    """
    if per_domain_limit is not None and per_domain_limit <= 0:
        raise ValueError(f"per_domain_limit must be a positive int, got {per_domain_limit!r}")
    pending = [item for item in store.query() if "path" not in item]
    if per_domain_limit is not None:
        pending = _sample_per_domain(pending, per_domain_limit)
    if not pending:
        return []

    active = get_active_taxonomy(store)
    classified = []
    for item in pending:
        try:
            classified.append(classify_item(item, active))
        except llm.LLMError as exc:
            print(
                f"analyst: skipping {item.get('repo')}#{item.get('number')}: {exc}",
                file=sys.stderr,
            )

    if classified:
        store.upsert_items(classified)
    return classified
