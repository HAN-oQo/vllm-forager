"""PR-author agent (T3.9): compose a maintainer-grade PR title + body from a gated candidate's
evidence bundle.

Part of M3.5, the same incident this whole milestone responds to: PR #47645 (withdrawn) didn't
just open against the wrong repo (T3.7's fix) — its body was also thin, a `# Candidate … Risk
badge` dump with an empty Repro block and a raw diff pasted in, exactly the shape a maintainer
bounces on sight regardless of code quality. `gate.py`'s :meth:`~src.gate.EvidenceBundle.format`
is still what the human sees when reviewing/approving (a faithful, mechanical rendering of every
field); this module composes the *separate*, narrative document meant for the upstream
maintainer once a human has approved submission — problem, why it happens, why the fix is
correct, a copy-pasteable before/after repro, and what's still unaddressed.

`compose_pr_body` asks the LLM for five narrative fields only — `title`, `problem`,
`root_cause`, `fix_rationale`, `limitations` — never for the repro command, log, or MI250
verification output. Those three are spliced into the final body verbatim from `bundle` itself
(see :func:`_render_body`), never paraphrased by the model: CLAUDE.md's evidence principle
("every KB record and every report claim carries a source ... link") only holds for a repro/perf
claim if the text making that claim is character-for-character what MI250 actually printed, not
an LLM's summary of it. The `Fixes #{number}` line and the reproduction/verification sections are
therefore guaranteed correct regardless of what the model does; only the connective prose around
them is generated.

`RepoProfile` (T3.8) is advisory context in the prompt (contribution norms, DCO requirement,
observed title style/examples) — this module does not itself enforce that the composed title
matches `profile.title_pattern` or that a DCO line is present when `profile.requires_dco`; that
"would a maintainer accept this" judgment is T3.10's job (a separate adversarial gate that can
send a draft back here). `profile=None` (e.g. `pr_profile.get_profile` itself failed, or the
caller has none yet) is handled the same as an empty profile — the prompt just omits that
section; composition never blocks on it.

DCO `Signed-off-by` uses the local git identity (`git config user.name`/`user.email` —
:func:`_default_signoff`) rather than a separate identity config: this is already the identity
every commit in this repo carries, so a second config knob for the same fact would just be
another place for it to drift. If neither is set, the composed body omits the trailer and its
checklist item is left unticked rather than fabricated — a real gap for a candidate that reaches
submission with no git identity configured, but one T3.10 (or the human, reading the draft before
`--submit`) will see, not one this module should paper over with a fake name/email.

Known limitations, not fixed here:
- Like `self_review.py`'s critiques, a single failed/malformed LLM reply is `None`, logged —
  same per-candidate failure isolation as every other M3 stage; there's no retry.
- `run_pr_author` persists its result (`stage="pr_author"`) but nothing downstream reads it yet
  — `gate.py`'s draft-writing path still uses `bundle.format()`. Wiring a composed body into the
  actual submitted PR draft is left to T3.10 (which needs to score/gate this module's output
  first) or M4's orchestrator, matching the same "not wired into the pipeline yet" note T3.7 left
  for `engineer.push_branch`.
- `run_pr_author` recomputes and re-persists a fresh (non-deterministic) run on every call, with
  no idempotency check against an already-composed body for the same candidate — the same gap
  `gate.py`'s own docstring documents for its own re-submission case, not newly introduced here.
- The composed `title`/prose don't themselves enforce `profile.title_pattern`/`requires_dco`
  compliance (see above: that's T3.10's job) — `_escape_markdown_structure` only guards against
  the model's prose corrupting this module's own template structure, not against it ignoring the
  target repo's conventions.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import gate
from .agents.forecaster import TS_FORMAT
from .pr_profile import RepoProfile, get_profile
from .stages import complete_or_none, record_run_best_effort
from .store import resolve_store
from .store.base import Store

_BODY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "problem": {"type": "string"},
        "root_cause": {"type": "string"},
        "fix_rationale": {"type": "string"},
        "limitations": {"type": "string"},
    },
    "required": ["title", "problem", "root_cause", "fix_rationale", "limitations"],
}

_NARRATIVE_FIELDS = ("title", "problem", "root_cause", "fix_rationale", "limitations")

# Truncation windows matching self_review.py's own conventions for the same two kinds of
# context (a full patch/body vs. a log tail, where the signal is conventionally at the end).
_DIFF_CONTEXT_CHARS = 4000
_LOG_CONTEXT_CHARS = 2000
_PROFILE_EXCERPT_CHARS = 1000

_GIT_CONFIG_TIMEOUT_S = 5.0

# Matches a line that would render as a Markdown heading (`#`..`######`) or list/checklist item
# (`- `) once spliced into the body -- see `_escape_markdown_structure`.
_STRUCTURAL_LINE = re.compile(r"^(\s*)(#{1,6}\s|-\s)")


@dataclasses.dataclass(frozen=True)
class PRBody:
    """A composed title + body, ready to hand to `gate.py` as a PR draft's content."""

    title: str
    body: str


@dataclasses.dataclass(frozen=True)
class PRAuthorResult:
    """The outcome of one :func:`run_pr_author` call, persisted as `stage="pr_author"`."""

    repo: str
    number: int
    title: str
    body: str
    recorded_at: str

    def to_run_record(self) -> dict:
        record = dataclasses.asdict(self)
        record["stage"] = "pr_author"
        return record


def _git_config(key: str) -> str | None:
    """`git config {key}` in the current working tree, or `None` if unset/unavailable."""
    try:
        result = subprocess.run(
            ["git", "config", key],
            capture_output=True,
            text=True,
            timeout=_GIT_CONFIG_TIMEOUT_S,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _default_signoff() -> str | None:
    """`Signed-off-by: {name} <{email}>` from the local git identity, or `None` if either half
    is unset — see module docstring for why this isn't a separate identity config."""
    name = _git_config("user.name")
    email = _git_config("user.email")
    if not name or not email:
        return None
    return f"Signed-off-by: {name} <{email}>"


def _profile_context(profile: RepoProfile | None) -> str:
    """The prompt section describing `profile`'s contribution norms, or a note that none are
    known if `profile` is `None` — never omitted outright, so the model isn't left guessing
    whether norms were checked and found empty vs. never checked at all."""
    if profile is None:
        return "No contribution-norms profile is available for this repo."
    lines = [
        f"DCO/Signed-off-by required: {profile.requires_dco}",
        f"Observed title convention: {profile.title_pattern or '(no consistent pattern detected)'}",
    ]
    if profile.title_examples:
        lines.append("Recent merged PR titles: " + "; ".join(profile.title_examples))
    if profile.contributing:
        lines.append(f"CONTRIBUTING excerpt: {profile.contributing[:_PROFILE_EXCERPT_CHARS]}")
    if profile.pr_template:
        lines.append(f"PR template excerpt: {profile.pr_template[:_PROFILE_EXCERPT_CHARS]}")
    return "\n".join(lines)


def _body_prompt(bundle: gate.EvidenceBundle, profile: RepoProfile | None) -> str:
    votes = "; ".join(
        f"{'approve' if v.get('looks_correct') else 'reject'}: {v.get('reason', '')}"
        for v in bundle.self_review_votes
    )
    return (
        "Compose a maintainer-grade pull request for the following verified, self-reviewed "
        "vLLM/ROCm fix. Write for the repo's own maintainers, not for an internal audience: be "
        "specific and evidence-based, not generic or boilerplate. Reply with exactly five "
        "fields: `title` (following the repo's observed title convention if one is given), "
        "`problem` (the user-facing symptom, from the maintainer's point of view), "
        "`root_cause` (why the bug actually happens), `fix_rationale` (why this specific patch "
        "correctly addresses that root cause, referencing the diff), and `limitations` (what "
        "this patch does not address, or edge cases still open). Do not include a reproduction "
        "command, log output, or a `Fixes #` line yourself — those are added separately from "
        "captured evidence, verbatim.\n\n"
        f"Repo: {bundle.repo}\nIssue/PR: #{bundle.number}\nOriginal title: {bundle.title[:500]}\n"
        f"Risk badge: risk={bundle.risk} effort={bundle.effort} impact={bundle.impact}\n\n"
        f"Diff (branch `{bundle.branch}`):\n{bundle.diff[:_DIFF_CONTEXT_CHARS]}\n\n"
        f"Self-review ({bundle.approve_count}/{bundle.total_votes} approve): {votes}\n\n"
        f"Repo contribution norms:\n{_profile_context(profile)}"
    )


def _escape_markdown_structure(text: str) -> str:
    """Escape a leading heading (`#`) or list-item (`- `) marker on any line of `text` (the
    LLM's own free-form prose) so it can never render as a spoofed section heading or checklist
    item once spliced next to the real, bundle-derived ones in :func:`_render_body` — the model
    is untrusted content here the same way any other external input would be."""
    return "\n".join(
        _STRUCTURAL_LINE.sub(lambda m: f"{m.group(1)}\\{m.group(2)}", line)
        for line in text.split("\n")
    )


def _render_repro_section(bundle: gate.EvidenceBundle) -> str:
    if not bundle.repro_command and not bundle.repro_log:
        return "(no captured pre-fix reproduction for this candidate)"
    command_line = (
        f"$ {bundle.repro_command}" if bundle.repro_command else "$ (command not captured)"
    )
    log_text = (
        bundle.repro_log[-_LOG_CONTEXT_CHARS:] if bundle.repro_log else "(no output captured)"
    )
    return f"```\n{command_line}\n{log_text}\n```"


def _render_verify_section(bundle: gate.EvidenceBundle) -> str:
    if not bundle.verify_log:
        return "(no captured MI250 verification log for this candidate)"
    return f"```\n{bundle.verify_log[-_LOG_CONTEXT_CHARS:]}\n```"


def _render_body(bundle: gate.EvidenceBundle, fields: dict[str, str], signoff: str | None) -> str:
    """Assemble the final markdown body: `fields`' narrative prose around the repro/verify/
    checklist/sign-off sections, which are built here from `bundle`/`signoff` directly rather
    than from anything the model returned — see module docstring."""
    has_repro = bool(bundle.repro_command or bundle.repro_log)
    has_verify_log = bool(bundle.verify_log)
    checklist = [
        f"- [{'x' if has_repro else ' '}] Reproduced the reported failure before the fix "
        f"(see Reproduction)",
        f"- [{'x' if has_verify_log else ' '}] Verified the fix on MI250 "
        f"(see MI250 verification)",
        f"- [x] Self-reviewed by an independent ensemble "
        f"({bundle.approve_count}/{bundle.total_votes} approve)",
        f"- [{'x' if signoff else ' '}] Includes a DCO sign-off below",
    ]
    sections = [
        f"## Problem\n{fields['problem']}\n\nFixes #{bundle.number}",
        f"## Root cause\n{fields['root_cause']}",
        f"## Fix rationale\n{fields['fix_rationale']}",
        f"## Reproduction\nBefore the fix:\n{_render_repro_section(bundle)}",
        f"## MI250 verification\nAfter the fix, on MI250:\n{_render_verify_section(bundle)}",
        f"## Limitations\n{fields['limitations']}",
        "## Checklist\n" + "\n".join(checklist),
    ]
    body = "\n\n".join(sections)
    if signoff:
        body += f"\n\n{signoff}"
    return body


def compose_pr_body(
    bundle: gate.EvidenceBundle,
    profile: RepoProfile | None = None,
    *,
    signoff: str | None = None,
) -> PRBody | None:
    """Compose a title + body for `bundle`, or `None` if the LLM call failed or its reply didn't
    shape into five non-empty narrative fields (logged to stderr, not raised — see module
    docstring's per-candidate failure isolation). `signoff` defaults to the local git identity
    (:func:`_default_signoff`) when not given."""
    subject = f"{bundle.repo}#{bundle.number}"
    reply = complete_or_none(
        _body_prompt(bundle, profile), _BODY_SCHEMA, stage="pr_author", subject=subject
    )
    if reply is None:
        return None
    fields: dict[str, str] = {}
    for key in _NARRATIVE_FIELDS:
        value = reply.get(key)
        if not isinstance(value, str) or not value.strip():
            print(f"pr_author: reply missing/empty {key!r} for {subject}", file=sys.stderr)
            return None
        fields[key] = value.strip()
    # `title` becomes a PR title, not body prose -- collapse it to one line so a stray
    # newline in the reply (or an embedded `Fixes #`/trailer line) can't smuggle extra
    # lines into whatever eventually calls `gh pr create --title`.
    fields["title"] = " ".join(fields["title"].split())
    for key in ("problem", "root_cause", "fix_rationale", "limitations"):
        fields[key] = _escape_markdown_structure(fields[key])
    resolved_signoff = signoff if signoff is not None else _default_signoff()
    body = _render_body(bundle, fields, resolved_signoff)
    return PRBody(title=fields["title"], body=body)


def run_pr_author(
    store: Store,
    repo: str,
    number: int,
    *,
    profile: RepoProfile | None = None,
    signoff: str | None = None,
    now: datetime | None = None,
) -> PRAuthorResult | None:
    """Assemble `repo`#`number`'s evidence bundle (`gate.assemble_bundle` — skips, logged, the
    same way if the candidate isn't gate-ready yet), compose its PR body, and persist the result
    as `stage="pr_author"`. `profile` defaults to `pr_profile.get_profile(repo)` when not given."""
    bundle = gate.assemble_bundle(store, repo, number)
    if bundle is None:
        return None
    resolved_profile = profile if profile is not None else get_profile(repo)
    composed = compose_pr_body(bundle, resolved_profile, signoff=signoff)
    if composed is None:
        return None
    recorded_at = (now or datetime.now(timezone.utc)).strftime(TS_FORMAT)
    result = PRAuthorResult(
        repo=repo, number=number, title=composed.title, body=composed.body, recorded_at=recorded_at
    )
    record_run_best_effort(
        store, result.to_run_record(), stage="pr_author", repo=repo, number=number
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: compose and print `--candidate`'s PR title + body."""
    ap = argparse.ArgumentParser(
        prog="python -m src.pr_author",
        description="Compose a maintainer-grade PR title + body for a gate-ready candidate.",
    )
    ap.add_argument(
        "--candidate", required=True, type=gate._parse_candidate, help="owner/repo#number"
    )
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    repo, number = args.candidate

    store, _ = resolve_store(args.data_dir)
    result = run_pr_author(store, repo, number)
    if result is None:
        print(f"{repo}#{number} is not ready for PR authoring (see stderr for why).")
        return 1
    print(f"# {result.title}\n\n{result.body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
