"""Contribution-norms adapter (T3.8): fetch + cache a target repo's contribution conventions.

Part of M3.5 (fork-first, maintainer-grade upstream PRs) — added alongside T3.7 in response to
the same incident: a PR that ignores the target repo's template, skips a required DCO
`Signed-off-by`, or uses the wrong title style gets bounced by a maintainer regardless of code
quality. T3.9's PR-author agent needs this repo-specific context *before* composing a title/body;
this module is the one place that fetches and shapes it into a :class:`RepoProfile`.

Fetches, best-effort, three raw things about `repo` — `contributing`, `pr_template`,
`exemplars` are the only real, *stored* fields; `requires_dco`/`title_pattern`/`title_examples`
are :class:`RepoProfile` properties **derived** from those three, computed fresh every access
rather than stored redundantly — a code-review finding on this module pointed out that storing
all six independently let a hand-edited or partially-migrated cache file drift out of sync (e.g.
`pr_template` edited to add a DCO mention without updating a separately-stored `requires_dco`).
Deriving them as properties makes that inconsistency structurally impossible while keeping the
exact same read-only attribute-access API for callers (`profile.requires_dco` etc. still works
identically; nothing downstream needs to change).

- `CONTRIBUTING.md` (checked at a few conventional paths — repo root, `.github/`, `docs/`).
- A PR template (`.github/pull_request_template.md`/`PULL_REQUEST_TEMPLATE.md`, case variants).
- A handful of recently **merged** PRs (via GitHub's search API), as title/body style exemplars.
  Merged, not just any PR, since a merged PR is proof a maintainer actually accepted that shape;
  an open or closed-unmerged PR proves nothing about acceptance. GitHub's search API has no
  `sort=merged` option, only `sort=updated` — a PR merged long ago but recently touched (a bot
  comment, a label) would otherwise outrank a genuinely recent merge, so
  :func:`_fetch_recent_merged_prs` over-fetches (`n * _OVERFETCH_FACTOR`) by `updated` and
  re-sorts client-side by `closed_at` (the closest proxy the search API's own response actually
  carries) before truncating to `n`.

`requires_dco` (:func:`_mentions_dco`) is a simple substring check on `contributing`/
`pr_template`, not a claim that some other, differently-worded requirement (or an explicit
*non*-requirement, e.g. "we do NOT require a DCO") would never be misread — T3.9 should treat
the raw `contributing`/`pr_template` text as authoritative and this boolean as a cheap hint on
top of it, not a substitute for showing the LLM the real text.

`title_pattern` (:func:`_infer_title_pattern`) reports whether most exemplar titles start with
*some* `[Category] ...` bracket tag and which categories are actually in use — deliberately not
"the one literal prefix most titles share": a repo like `vllm-project/vllm` (this project's own
primary target) uses many different tags (`[Bugfix]`, `[Core]`, `[Doc]`, `[V1]`, ...), not one
repeated string, and an earlier version of this heuristic that looked for a single common literal
returned `None` for exactly that repo — a code-review finding caught it failing on its own
primary use case. Still just a cheap signal for T3.9's prompt, not a validator: a repo whose real
convention doesn't use brackets at all yields `title_pattern=None`, and T3.9 always has the raw
`title_examples` to learn from either way. Requires at least 3 exemplars before inferring
anything — a "pattern" from a sample of 1 isn't a pattern.

`build_profile` always does the real fetch; `get_profile` is what callers should actually use —
it checks a local JSON cache first (see :func:`_profile_path`) and only calls `build_profile` on
a cache miss, a cache holding fewer exemplars than `n_exemplars` now asks for, or `refresh=True`.
The cache write is atomic (write to a `.tmp` sibling, then `Path.replace`) so a crash mid-write
can't leave a truncated, unparseable JSON file behind — matching
:func:`~src.store.jsonl_store.save_state`'s own established tmp+replace convention. A cache read
that fails to parse (corrupted file, or an incompatible schema from an older version of this
module) is treated the same as a cache miss — logged to stderr and rebuilt — rather than
propagating the exception and taking down whatever called `get_profile`.

This cache deliberately lives as local JSON files under `config.DATA_DIR`, not behind the
:mod:`~src.store` abstraction (`Store.get_item`/`record_run`) the KB otherwise uses for
backend-agnostic (jsonl/Firestore) persistence — the same choice `gate.py` already made for its
own PR-draft files. A repo profile isn't KB knowledge accumulated from history the way collected
issues/runs are; it's a cheaply-recomputable-from-GitHub cache (`refresh=True` rebuilds it from
scratch at any time), closer in kind to a local build artifact than to durable, must-replicate
knowledge.

Auth reuses :mod:`src.collector`'s own `_headers()` (adds `Authorization: Bearer $GITHUB_TOKEN`
when set, works unauthenticated otherwise) rather than inventing a second GitHub-auth mechanism —
the same "reuse a private cross-module helper instead of a redundant parallel one" choice
`gate.py` already made for `scout._score`, and `audit.py` already made for this exact same
`collector._headers` import.

Known limitations, not fixed here:
- `_fetch_file`/`_fetch_recent_merged_prs` are single-shot `requests.get` calls with no retry or
  rate-limit handling, unlike `collector.py`'s own `_request` — reasonable for T3.8's on-demand,
  few-calls-per-candidate usage (not `collector.py`'s bulk/scheduled fetch loop), but a real gap
  if this is ever called in a tight loop across many repos without a delay. A transient failure
  is now at least logged to stderr (not silently swallowed), but a rate-limited or errored fetch
  is still indistinguishable, once cached, from "this repo genuinely has no CONTRIBUTING doc" —
  `refresh=True` is the only recovery today; there's no automatic retry or negative-result TTL.
- T3.9's own DEVPLAN description mentions a PR body with "checklist ticked" — this module doesn't
  parse `pr_template`'s raw markdown into a structured list of checkboxes; that parsing (and
  deciding which to tick) is T3.9's own job, operating on the raw `pr_template` text this module
  hands it.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import requests

from . import config
from .collector import _headers

_REQUEST_TIMEOUT_S = 30
_DEFAULT_N_EXEMPLARS = 5
_BODY_EXCERPT_CHARS = 1000
_MIN_TITLES_FOR_PATTERN = 3
# GitHub's search API only supports sort=updated, not sort=merged -- over-fetch by this factor
# and re-sort client-side by `closed_at` (the closest available proxy for merge time) so a PR
# merged long ago but recently touched by a bot/label doesn't outrank a genuinely recent merge.
_OVERFETCH_FACTOR = 3

# Conventional paths GitHub itself recognizes for these two files (checked in this order; the
# first that exists wins) -- not exhaustive (GitHub also allows a bare "CONTRIBUTING" with no
# extension, various cases, etc.), but covers the overwhelming majority of real repos.
_CONTRIBUTING_PATHS = ("CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md")
_PR_TEMPLATE_PATHS = (
    ".github/pull_request_template.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/PULL_REQUEST_TEMPLATE/pull_request_template.md",
)

_PROFILES_SUBDIR = "pr_profiles"
_PROFILES_DIR = config.DATA_DIR / _PROFILES_SUBDIR


def _mentions_dco(*texts: str | None) -> bool:
    """Whether any of `texts` mentions a DCO/`Signed-off-by` requirement — see module docstring
    for why this is a simple substring check, not a claim of completeness."""
    combined = " ".join(t for t in texts if t).lower()
    return "signed-off-by" in combined or "dco" in combined


def _infer_title_pattern(titles: list[str]) -> str | None:
    """Whether most `titles` start with *some* `[Category] ...` bracket tag, and which
    categories are actually in use — see module docstring for why this looks for "a bracket tag,
    any tag" rather than one specific repeated literal."""
    if len(titles) < _MIN_TITLES_FOR_PATTERN:
        return None
    categories = [m.group(1) for t in titles if (m := re.match(r"^\[([^\]]+)\]", t))]
    if len(categories) < len(titles) / 2:
        return None
    top = [c for c, _count in Counter(categories).most_common(5)]
    return f"[Category] ... (observed categories: {', '.join(f'[{c}]' for c in top)})"


@dataclass(frozen=True)
class RepoProfile:
    """One repo's contribution conventions, as T3.9's PR-author agent needs them.

    `requires_dco`, `title_pattern`, and `title_examples` are properties derived from
    `contributing`/`pr_template`/`exemplars` — see module docstring for why they aren't stored
    fields."""

    repo: str
    contributing: str | None
    pr_template: str | None
    exemplars: tuple[dict, ...]

    @property
    def title_examples(self) -> tuple[str, ...]:
        return tuple(e["title"] for e in self.exemplars if e.get("title"))

    @property
    def requires_dco(self) -> bool:
        return _mentions_dco(self.contributing, self.pr_template)

    @property
    def title_pattern(self) -> str | None:
        return _infer_title_pattern(list(self.title_examples))

    def to_dict(self) -> dict:
        """Plain-dict form for JSON caching — only the real, stored fields; the derived
        properties are recomputed on load by :meth:`from_dict`, not persisted."""
        return {
            "repo": self.repo,
            "contributing": self.contributing,
            "pr_template": self.pr_template,
            "exemplars": list(self.exemplars),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RepoProfile:
        """Reconstruct from :meth:`to_dict`'s own output — `exemplars` comes back as a `list`
        of dicts through the `json.dumps`/`json.loads` round trip and is restored to a `tuple`
        here, matching this class's own frozen/hashable field type."""
        return cls(
            repo=data["repo"],
            contributing=data["contributing"],
            pr_template=data["pr_template"],
            exemplars=tuple(data["exemplars"]),
        )


def _fetch_file(repo: str, path: str) -> str | None:
    """Raw text content of `path` in `repo`'s default branch, or `None` if it doesn't exist
    (`404`) or the fetch otherwise failed (logged to stderr for the latter case — a missing
    contribution doc is a real, common case, not worth logging over, but a network/auth/rate
    -limit failure silently collapsing to the same `None` is worth at least a trace)."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        resp = requests.get(
            url,
            headers={**_headers(), "Accept": "application/vnd.github.raw+json"},
            timeout=_REQUEST_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        print(f"pr_profile: failed to fetch {repo}:{path}: {exc}", file=sys.stderr)
        return None
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        print(
            f"pr_profile: unexpected status {resp.status_code} fetching {repo}:{path}",
            file=sys.stderr,
        )
        return None
    # GitHub doesn't always send a charset in Content-Type for raw content; source/doc files are
    # UTF-8 in practice, and `requests` would otherwise guess (historically ISO-8859-1 for
    # text/*), mis-decoding non-ASCII contributor names/quotes in the cached text.
    resp.encoding = "utf-8"
    return resp.text


def _fetch_first_existing(repo: str, paths: tuple[str, ...]) -> str | None:
    """The content of the first path in `paths` that exists in `repo`, or `None` if none do."""
    for path in paths:
        content = _fetch_file(repo, path)
        if content is not None:
            return content
    return None


def _fetch_recent_merged_prs(repo: str, *, n: int) -> list[dict]:
    """The `n` most recently merged pull requests against `repo`'s default branch, each as
    `{"title": ..., "url": ..., "body": ...}` (`body` truncated to
    :data:`_BODY_EXCERPT_CHARS`) — an empty list if the fetch fails or none exist. See module
    docstring for why this over-fetches and re-sorts by `closed_at` rather than trusting the
    search API's own `sort=updated` ordering directly."""
    url = "https://api.github.com/search/issues"
    params = {
        "q": f"repo:{repo} is:pr is:merged",
        "sort": "updated",
        "order": "desc",
        "per_page": str(n * _OVERFETCH_FACTOR),
    }
    try:
        resp = requests.get(url, headers=_headers(), params=params, timeout=_REQUEST_TIMEOUT_S)
    except requests.RequestException as exc:
        print(f"pr_profile: failed to fetch merged PRs for {repo}: {exc}", file=sys.stderr)
        return []
    if resp.status_code != 200:
        print(
            f"pr_profile: unexpected status {resp.status_code} fetching merged PRs for {repo}",
            file=sys.stderr,
        )
        return []
    items = resp.json().get("items", [])
    items.sort(key=lambda item: item.get("closed_at") or "", reverse=True)
    return [
        {
            "title": item.get("title") or "",
            "url": item.get("html_url") or "",
            "body": (item.get("body") or "")[:_BODY_EXCERPT_CHARS],
        }
        for item in items[:n]
    ]


def build_profile(repo: str, *, n_exemplars: int = _DEFAULT_N_EXEMPLARS) -> RepoProfile:
    """Fetch everything about `repo` fresh (no cache) and assemble a :class:`RepoProfile`."""
    return RepoProfile(
        repo=repo,
        contributing=_fetch_first_existing(repo, _CONTRIBUTING_PATHS),
        pr_template=_fetch_first_existing(repo, _PR_TEMPLATE_PATHS),
        exemplars=tuple(_fetch_recent_merged_prs(repo, n=n_exemplars)),
    )


def _profile_path(repo: str, *, profiles_dir: Path | None = None) -> Path:
    """Where :func:`get_profile` caches (and looks for) one repo's profile — mirrors
    `gate.py`'s `_pr_draft_path`/`_PR_DRAFTS_DIR` pattern exactly (a `None` sentinel resolved
    against the module-level default *at call time*, not `def`-time, so tests can monkeypatch
    `_PROFILES_DIR`)."""
    return (profiles_dir or _PROFILES_DIR) / f"{repo.replace('/', '-')}.json"


def _load_cached_profile(path: Path) -> RepoProfile | None:
    """The profile cached at `path`, or `None` if it doesn't exist or fails to parse (a
    corrupted file from an interrupted write, or an incompatible schema from an older version
    of this module) — treated the same as a cache miss, logged, not raised, so a bad cache file
    degrades to a rebuild instead of taking down whatever called :func:`get_profile`."""
    if not path.exists():
        return None
    try:
        return RepoProfile.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"pr_profile: cache at {path} unreadable ({exc}), rebuilding", file=sys.stderr)
        return None


def _write_cached_profile(profile: RepoProfile, path: Path) -> None:
    """Write `profile` to `path` atomically — a `.tmp` sibling written first, then renamed into
    place via `Path.replace` (an atomic rename on the same filesystem), so a crash mid-write
    can't leave a truncated, unparseable cache file — matching
    :func:`~src.store.jsonl_store.save_state`'s own established tmp+replace convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(profile.to_dict(), indent=2))
    tmp_path.replace(path)


def get_profile(
    repo: str,
    *,
    refresh: bool = False,
    profiles_dir: Path | None = None,
    n_exemplars: int = _DEFAULT_N_EXEMPLARS,
) -> RepoProfile:
    """`repo`'s cached profile, or a freshly-built one (written to the cache) if none exists yet,
    the cache is unreadable, the cache holds fewer exemplars than `n_exemplars` now asks for, or
    `refresh=True`. This is what real callers (T3.9) should use — :func:`build_profile` alone
    re-fetches CONTRIBUTING/template/PR-history over the network every time."""
    path = _profile_path(repo, profiles_dir=profiles_dir)
    if not refresh:
        cached = _load_cached_profile(path)
        if cached is not None and len(cached.exemplars) >= n_exemplars:
            return cached
    profile = build_profile(repo, n_exemplars=n_exemplars)
    _write_cached_profile(profile, path)
    return profile
