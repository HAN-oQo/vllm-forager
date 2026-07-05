"""Contribution-norms adapter (T3.8): fetch + cache a target repo's contribution conventions.

Part of M3.5 (fork-first, maintainer-grade upstream PRs) — added alongside T3.7 in response to
the same incident: a PR that ignores the target repo's template, skips a required DCO
`Signed-off-by`, or uses the wrong title style gets bounced by a maintainer regardless of code
quality. T3.9's PR-author agent needs this repo-specific context *before* composing a title/body;
this module is the one place that fetches and shapes it into a :class:`RepoProfile`.

Fetches, best-effort, four things about `repo`:
- `CONTRIBUTING.md` (checked at a few conventional paths — repo root, `.github/`, `docs/`).
- A PR template (`.github/pull_request_template.md`/`PULL_REQUEST_TEMPLATE.md`, case variants).
- Whether either of those mentions a DCO/`Signed-off-by` requirement (:func:`_mentions_dco`) —
  a simple substring check, not a claim that some other, differently-worded DCO requirement
  would never be missed; a false negative here just means `requires_dco=False` when the repo
  actually does want one, which T3.9's own body-composition prompt can still be told to add
  defensively regardless of what this module inferred.
- A handful of recently **merged** PRs (via GitHub's search API), as title/body style exemplars.
  Merged, not just any PR, since a merged PR is proof a maintainer actually accepted that shape;
  an open or closed-unmerged PR proves nothing about acceptance.

`build_profile` always does the real fetch; `get_profile` is what callers should actually use —
it checks a local JSON cache first (see :func:`_profile_path`) and only calls `build_profile` on
a cache miss or `refresh=True`, so T3.9 doesn't re-fetch a whole repo's contribution docs and PR
history on every single candidate.

Auth reuses :mod:`src.collector`'s own `_headers()` (adds `Authorization: Bearer $GITHUB_TOKEN`
when set, works unauthenticated otherwise) rather than inventing a second GitHub-auth mechanism —
the same "reuse a private cross-module helper instead of a redundant parallel one" choice
`gate.py` already made for `scout._score`.

Known limitations, not fixed here:
- `_fetch_file`/`_fetch_recent_merged_prs` are single-shot `requests.get` calls with no retry or
  rate-limit handling, unlike `collector.py`'s own `_request` — reasonable for T3.8's on-demand,
  few-calls-per-candidate usage (not `collector.py`'s bulk/scheduled fetch loop), but a real gap
  if this is ever called in a tight loop across many repos without a delay.
- `_infer_title_pattern`'s bracket-prefix heuristic is a cheap signal for T3.9's prompt, not a
  validator — a repo with a genuinely different title convention (no brackets at all, a
  different delimiter) yields `title_pattern=None`, and T39 still has the raw `title_examples`
  to work from either way.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from . import config
from .collector import _headers

_REQUEST_TIMEOUT_S = 30
_DEFAULT_N_EXEMPLARS = 5
_BODY_EXCERPT_CHARS = 1000

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


@dataclass(frozen=True)
class RepoProfile:
    """One repo's contribution conventions, as T3.9's PR-author agent needs them."""

    repo: str
    contributing: str | None
    pr_template: str | None
    requires_dco: bool
    title_pattern: str | None
    title_examples: tuple[str, ...]
    exemplars: tuple[dict, ...]

    def to_dict(self) -> dict:
        """Plain-dict form for JSON caching — `exemplars`/`title_examples` are already
        JSON-safe (dicts/strings), so `dataclasses.asdict` alone is enough; the tuples become
        lists on the round trip through `json.dumps`/`json.loads`, corrected back in
        :meth:`from_dict`."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RepoProfile:
        return cls(
            repo=data["repo"],
            contributing=data["contributing"],
            pr_template=data["pr_template"],
            requires_dco=data["requires_dco"],
            title_pattern=data["title_pattern"],
            title_examples=tuple(data["title_examples"]),
            exemplars=tuple(data["exemplars"]),
        )


def _fetch_file(repo: str, path: str) -> str | None:
    """Raw text content of `path` in `repo`'s default branch, or `None` if it doesn't exist
    (`404`) or the fetch otherwise failed — a missing contribution doc is a real, common case
    (not every repo has one), not an error worth raising over."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        resp = requests.get(
            url,
            headers={**_headers(), "Accept": "application/vnd.github.raw+json"},
            timeout=_REQUEST_TIMEOUT_S,
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return resp.text


def _fetch_first_existing(repo: str, paths: tuple[str, ...]) -> str | None:
    """The content of the first path in `paths` that exists in `repo`, or `None` if none do."""
    for path in paths:
        content = _fetch_file(repo, path)
        if content is not None:
            return content
    return None


def _mentions_dco(*texts: str | None) -> bool:
    """Whether any of `texts` mentions a DCO/`Signed-off-by` requirement — see module docstring
    for why this is a simple substring check, not a claim of completeness."""
    combined = " ".join(t for t in texts if t).lower()
    return "signed-off-by" in combined or "dco" in combined


def _infer_title_pattern(titles: list[str]) -> str | None:
    """A representative `[Bracket] ...` title prefix, if at least half of `titles` share one —
    `None` if the repo's real convention doesn't use bracket prefixes at all (T3.9 still has
    the raw `titles` to learn from either way; see module docstring)."""
    if not titles:
        return None
    prefixes = [m.group(0) for t in titles if (m := re.match(r"^(\[[^\]]+\])", t))]
    if len(prefixes) < len(titles) / 2:
        return None
    most_common, _count = Counter(prefixes).most_common(1)[0]
    return f"{most_common} ..."


def _fetch_recent_merged_prs(repo: str, *, n: int) -> list[dict]:
    """The `n` most recently merged pull requests against `repo`'s default branch, each as
    `{"title": ..., "url": ..., "body": ...}` (`body` truncated to
    :data:`_BODY_EXCERPT_CHARS`) — an empty list if the fetch fails or none exist. Uses GitHub's
    search API (`is:pr is:merged`, sorted by most-recently-updated) rather than listing+filtering
    every closed PR, since a repo's total closed-PR count can be huge."""
    url = "https://api.github.com/search/issues"
    params = {
        "q": f"repo:{repo} is:pr is:merged",
        "sort": "updated",
        "order": "desc",
        "per_page": str(n),
    }
    try:
        resp = requests.get(url, headers=_headers(), params=params, timeout=_REQUEST_TIMEOUT_S)
    except requests.RequestException:
        return []
    if resp.status_code != 200:
        return []
    items = resp.json().get("items", [])
    return [
        {
            "title": item.get("title") or "",
            "url": item.get("html_url") or "",
            "body": (item.get("body") or "")[:_BODY_EXCERPT_CHARS],
        }
        for item in items
    ]


def build_profile(repo: str, *, n_exemplars: int = _DEFAULT_N_EXEMPLARS) -> RepoProfile:
    """Fetch everything about `repo` fresh (no cache) and assemble a :class:`RepoProfile`."""
    contributing = _fetch_first_existing(repo, _CONTRIBUTING_PATHS)
    pr_template = _fetch_first_existing(repo, _PR_TEMPLATE_PATHS)
    exemplars = _fetch_recent_merged_prs(repo, n=n_exemplars)
    title_examples = tuple(e["title"] for e in exemplars if e["title"])
    return RepoProfile(
        repo=repo,
        contributing=contributing,
        pr_template=pr_template,
        requires_dco=_mentions_dco(contributing, pr_template),
        title_pattern=_infer_title_pattern(list(title_examples)),
        title_examples=title_examples,
        exemplars=tuple(exemplars),
    )


def _profile_path(repo: str, *, profiles_dir: Path | None = None) -> Path:
    """Where :func:`get_profile` caches (and looks for) one repo's profile — mirrors
    `gate.py`'s `_pr_draft_path`/`_PR_DRAFTS_DIR` pattern exactly (a `None` sentinel resolved
    against the module-level default *at call time*, not `def`-time, so tests can monkeypatch
    `_PROFILES_DIR`)."""
    return (profiles_dir or _PROFILES_DIR) / f"{repo.replace('/', '-')}.json"


def get_profile(
    repo: str,
    *,
    refresh: bool = False,
    profiles_dir: Path | None = None,
    n_exemplars: int = _DEFAULT_N_EXEMPLARS,
) -> RepoProfile:
    """`repo`'s cached profile, or a freshly-built one (written to the cache) if none exists yet
    or `refresh=True`. This is what real callers (T3.9) should use — :func:`build_profile` alone
    re-fetches CONTRIBUTING/template/PR-history over the network every time."""
    path = _profile_path(repo, profiles_dir=profiles_dir)
    if not refresh and path.exists():
        return RepoProfile.from_dict(json.loads(path.read_text()))
    profile = build_profile(repo, n_exemplars=n_exemplars)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2))
    return profile
