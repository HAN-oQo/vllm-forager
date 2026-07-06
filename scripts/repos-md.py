"""repos-md.py — print `config.REPOS` grouped by domain, as Markdown (T3.18).

`docs/PLAN.md`'s "Tracked repos" section embeds this script's exact output between
``<!-- repos-md:start -->``/``<!-- repos-md:end -->`` markers, so the tracked-repo list stays a
single generated block instead of a hand-copied one that silently drifts from `config.REPOS`
after a retarget. `tests/test_repos_md.py` asserts the two stay byte-identical — regenerate and
paste this script's output between those markers whenever `config.REPOS` changes.

Usage:
    python scripts/repos-md.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402  (path insert above must run first)


def render(repos: list[dict]) -> str:
    """One heading per domain (first-seen order in `repos`), each repo as `` `slug` — role``.

    Grouped by `domain` (not `role`) since `config.py`'s own docstring already frames domain as
    what "groups a target with the ecosystem watched around it" — the grouping a reader wants
    when asking "what's tracked around vime/vllm-omni/vllm".
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


def main() -> None:
    sys.stdout.write(render(config.REPOS))


if __name__ == "__main__":
    main()
