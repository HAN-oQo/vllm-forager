"""repos-md.py — print or sync `config.REPOS` grouped by domain, as Markdown (T3.18).

`docs/PLAN.md`'s "Tracked repos" section embeds this script's exact output between
``<!-- repos-md:start -->``/``<!-- repos-md:end -->`` markers, so the tracked-repo list stays a
single generated block instead of a hand-copied one that silently drifts from `config.REPOS`
after a retarget.

Two modes:

- ``python scripts/repos-md.py`` — print the grouped list to stdout.
- ``python scripts/repos-md.py --sync`` — rewrite `docs/PLAN.md`'s marker block **in place**
  and exit ``1`` if the file changed, the same convention black/ruff already use in this repo's
  own `.pre-commit-config.yaml` (auto-fix, then fail so the change gets re-staged). Wired in as
  a local pre-commit hook (code-review follow-up on T3.18: a `pytest`-only drift check catches
  a stale doc one commit/CI cycle *after* it happens; this makes the stale state structurally
  unreachable at commit time instead, matching the stronger guarantee `config.py`'s own
  `_validate_repos` and this repo's other embedded-artifact tests already give other content).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402  (path insert above must run first)

_START = "<!-- repos-md:start -->"
_END = "<!-- repos-md:end -->"
_PLAN_MD = Path(__file__).resolve().parent.parent / "docs" / "PLAN.md"


def render(repos: list[dict]) -> str:
    """One heading per domain (first-seen order in `repos`), each repo as `` `slug` — role``.

    Grouped by `domain` (not `role`) since `config.py`'s own docstring already frames domain as
    what "groups a target with the ecosystem watched around it" — the grouping a reader wants
    when asking "what's tracked around vime/vllm-omni/vllm". See `docs/PLAN.md`'s "Targets &
    sources" section for the role-first narrative (which repos are actual contribution targets
    vs. watch-only) this block doesn't repeat.
    """
    by_domain: dict[str, list[dict]] = {}
    for repo in repos:
        by_domain.setdefault(repo["domain"], []).append(repo)

    lines = [f"{len(repos)} tracked repos across {len(by_domain)} domains.", ""]
    for domain, group in by_domain.items():
        lines.append(f"**{domain}**")
        for repo in group:
            lines.append(f"- `{repo['slug']}` — {repo['role']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def sync(plan_md_path: Path) -> bool:
    """Rewrite `plan_md_path`'s `_START`/`_END` block in place with the current `config.REPOS`;
    return whether the file changed.

    Raises:
        SystemExit: `plan_md_path` doesn't have exactly one `_START`/`_END` marker pair — a
            second, later match could otherwise silently become the "real" boundary instead of
            a loud, obvious failure.
    """
    text = plan_md_path.read_text()
    if text.count(_START) != 1 or text.count(_END) != 1:
        raise SystemExit(
            f"repos-md: expected exactly one {_START!r}/{_END!r} marker pair in {plan_md_path}"
        )
    start = text.index(_START) + len(_START)
    end = text.index(_END)
    fresh_block = "\n" + render(config.REPOS).rstrip("\n") + "\n"
    new_text = text[:start] + fresh_block + text[end:]
    if new_text == text:
        return False
    plan_md_path.write_text(new_text)
    return True


def main() -> None:
    if "--sync" in sys.argv[1:]:
        # An optional path argument overrides the default docs/PLAN.md target — lets
        # tests/test_repos_md.py exercise `sync()`'s rewrite/no-op behavior against a
        # throwaway file instead of the real doc.
        rest = [a for a in sys.argv[1:] if a != "--sync"]
        target = Path(rest[0]).resolve() if rest else _PLAN_MD
        if sync(target):
            print(f"repos-md: updated {target} — re-stage and commit again", file=sys.stderr)
            raise SystemExit(1)
        return
    sys.stdout.write(render(config.REPOS))


if __name__ == "__main__":
    main()
