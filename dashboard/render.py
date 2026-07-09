"""Pure render functions: KB (store) in, HTML/data out — no server involved.

Kept separate from :mod:`dashboard.server` so every render can be unit-tested against a
seeded store fixture without opening a socket (per T1.9's own test plan).

T1.5.5: the dashboard's centerpiece is now the same collapsible taxonomy tree T1.5.4's
``report.py --tree`` writes to disk — :func:`~src.agents.reporter_v1.tree_from_store` is
called **live** here (not read from a ``.tree.json`` file), matching how Trends/the forecast
log already work. This also retires T1.9's own "known limitation" (the old flat-report section
read a local file that could be stale/absent on a different host than whatever last ran
``python -m src.report``) — the tree section has no such gap, since it's computed the same way
Trends/forecasts always were: straight from the `Store`.

Node summaries (T1.5.3), counts/gaps (T1.5.4, rolled up), and cited PR/issue rows render as
collapsible ``<details>`` per node, matching the approved ``docs/design/report-tree-mockup.
html`` — the design spec, not a throwaway: CSS variables, the expand/collapse + filter
``<script>``, and the light/dark theming below are adapted from it directly, not reinvented.

Known limitation (not fixed here): the collector normalizes only ``state`` (open/closed), not
a ``merged``/``merged_at`` field, so a PR's chip treats "closed" as "merged" — a closed-without
-merging PR would be mislabeled. Distinguishing the two needs a collector change (capturing
GitHub's own ``merged``/``merged_at`` fields), out of scope for a dashboard-rendering todo.

T5.16: :func:`render_page` now builds one :func:`~dashboard.snapshot.build_snapshot` (still one
full store scan — that part isn't fixed here, see that module's own docstring for what would
be) and renders from it via :func:`render_snapshot_page`, a pure ``dict -> str`` function with
no `Store` import at all. The rendering step itself is now bounded regardless of KB size: each
tree node's own leaf rows are capped at :data:`~dashboard.snapshot.DEFAULT_MAX_PRS_PER_NODE` in
the snapshot already, with a "show N more" control per truncated node fetching the rest from
:mod:`dashboard.server`'s ``/api/node-prs`` endpoint on demand — never embedded in the page.
"""

from __future__ import annotations

from dataclasses import asdict
from html import escape
from urllib.parse import quote

from src.agents.forecaster import Prediction, list_predictions
from src.agents.grader import parse_repo_number
from src.agents.reporter import is_merged, repo_number_label
from src.selection import decisions_by_key
from src.store.base import Store
from src.taxonomy import LEVEL_SEPARATOR

from . import api, archive, attempts, cost, guardrails
from .health import health_panel
from .snapshot import build_snapshot

# Only these schemes are ever rendered as a clickable href — GitHub-sourced items always carry
# an http(s) url (see reporter.evidence_url), but this dashboard has no auth and renders
# whatever the KB contains, so a stray `javascript:`/`data:` value in a malformed or
# non-GitHub-sourced record must never become a clickable link.
_SAFE_URL_SCHEMES = ("http://", "https://")

# Depth 0 -> "cat" (top-level taxonomy category), depth 1 -> "sub", depth 2+ -> "leaf" —
# purely a styling hook (nesting depth), independent of whether a node actually has children.
_NODE_CLASS_BY_DEPTH = ("cat", "sub")


def render_trends(store: Store) -> dict[str, dict[str, int]]:
    """Per-category, per-week activity counts. Delegates to `dashboard.api.trends` (T5.1) — an
    earlier version called `trends.trends_from_store` directly, an independent byte-for-byte
    duplicate of `api.trends`'s own body that a code-review finding flagged as exactly the
    "two clean read APIs, not one" problem T5.1's own stated Why exists to avoid."""
    return api.trends(store)


def render_forecasts(store: Store) -> list[Prediction]:
    """Every recorded prediction, oldest first (thin re-export of `forecaster.list_predictions`)."""
    return list_predictions(store)


def _section(title: str, body_html: str) -> str:
    """Wrap `body_html` in the page's common ``<section><h2>title</h2>...</section>`` shell."""
    return f"<section><h2>{title}</h2>{body_html}</section>"


def _node_class(depth: int) -> str:
    return _NODE_CLASS_BY_DEPTH[depth] if depth < len(_NODE_CLASS_BY_DEPTH) else "leaf"


def _state_chip(pr: dict) -> tuple[str, str]:
    """``(css-class, label)`` for a PR/issue row's state chip — merged / open pr / issue.

    See this module's docstring: a closed PR is assumed merged, since the collector doesn't
    capture a real ``merged`` flag. A PR record with no/unrecognized ``state`` (only reachable
    from a malformed/legacy record — the collector always sets ``open``/``closed``) renders as
    "open pr" rather than "merged": claiming an unknown state is open is the less misleading
    of the two guesses, since "merged" implies a completed, verifiable action that may not
    have happened.
    """
    if pr.get("type") != "pr":
        return "issue", "issue"
    if is_merged(pr):
        return "merged", "merged"
    return "open", "open pr"


def _safe_href(url: str) -> str:
    """`url` if it's http(s), else ``""`` — see this module's ``_SAFE_URL_SCHEMES`` docstring."""
    return url if url.startswith(_SAFE_URL_SCHEMES) else ""


def _pr_row_html(pr: dict) -> str:
    """One cited PR/issue row: ``repo#number``, a linked title, and a state chip."""
    chip_class, chip_label = _state_chip(pr)
    url = escape(_safe_href(pr.get("url") or ""))
    title = escape(pr.get("title") or "")
    return (
        '<div class="pr">'
        f'<span class="id">{escape(repo_number_label(pr))}</span>'
        f'<span class="t"><a href="{url}">{title}</a></span>'
        f'<span class="chip {chip_class}">{escape(chip_label)}</span>'
        "</div>"
    )


def _gap_chip_html(gaps: int) -> str:
    if not gaps:
        return ""
    label = f"{gaps} gap" if gaps == 1 else f"{gaps} gaps"
    return f'<span class="tag-gap">{escape(label)}</span>'


def _more_prs_html(node: dict) -> str:
    """T5.16: a "show N more" control for a node whose own leaf rows were capped by
    :func:`~dashboard.snapshot.build_snapshot` — clicking it fetches the rest, page by page,
    from :mod:`dashboard.server`'s ``/api/node-prs`` endpoint (see ``_TREE_SCRIPT``'s own
    handler) rather than ever embedding them in the page. Absent entirely for a node that
    wasn't truncated — no dead control for the common (small) case."""
    if not node["prs_truncated"]:
        return ""
    remaining = node["prs_total"] - len(node["prs"])
    path = escape(LEVEL_SEPARATOR.join(node["path"]))
    # `data-limit` (not a hardcoded page size in the JS fetch) reuses the exact page size
    # already implied by a truncated node's own capped `prs` length -- so a future change to
    # `DEFAULT_MAX_PRS_PER_NODE` only needs updating in one place, not also in `_TREE_SCRIPT`.
    page_size = len(node["prs"])
    return (
        f'<button class="btn more" data-path="{path}" data-offset="{page_size}" '
        f'data-limit="{page_size}">'
        f"Show {remaining} more"
        "</button>"
    )


def _tree_node_html(node: dict, depth: int) -> str:
    """One collapsible ``<details>`` node: name, count, an optional gap chip and summary
    line, its own cited PR rows (capped — see :func:`_more_prs_html`), then its children —
    recursively, matching the mockup's 대(大) → 소(summary) → 소소 → PRs nesting.

    `node` is the snapshot's own dict shape (:func:`~dashboard.snapshot.build_snapshot`), not
    a :class:`~src.agents.reporter_v1.TreeNode` — this function never touches a `Store`.
    """
    summary_html = (
        f'<div class="summary-line">{escape(node["summary"])}</div>' if node["summary"] else ""
    )
    prs_html = "".join(_pr_row_html(pr) for pr in node["prs"]) + _more_prs_html(node)
    kids_html = "".join(_tree_node_html(child, depth + 1) for child in node["children"])
    body = prs_html + kids_html
    kids_wrapped = f'<div class="kids">{body}</div>' if body else ""
    open_attr = " open" if depth == 0 else ""
    return (
        f'<details class="{_node_class(depth)}"{open_attr}>'
        '<summary><span class="chev">▶</span>'
        f'<span class="name">{escape(node["name"])}</span>'
        f'<span class="count">{node["count"]}</span>{_gap_chip_html(node["gaps"])}</summary>'
        f"{summary_html}{kids_wrapped}"
        "</details>"
    )


_TREE_CONTROLS = (
    '<div class="controls">'
    '<input id="q" type="search" '
    'placeholder="Filter — e.g. attention, DeepSeek, fp8, sglang#…" aria-label="Filter report">'
    '<button class="btn" data-all="1">Expand all</button>'
    '<button class="btn" data-all="0">Collapse all</button>'
    "</div>"
    '<div class="legend">'
    '<span><span class="dot" style="background:var(--merged)"></span>merged PR</span>'
    '<span><span class="dot" style="background:var(--open)"></span>open PR</span>'
    '<span><span class="dot" style="background:var(--issue)"></span>issue</span>'
    '<span><span class="tag-gap" style="border:none;padding:.02rem .3rem">gap</span>'
    "present on another tracked engine, missing on a contribution target</span>"
    "</div>"
)


def _tree_section_html(tree: list[dict]) -> str:
    """The collapsible report tree, or a placeholder when the store has no items at all.

    `tree` is :func:`~dashboard.snapshot.build_snapshot`'s own ``"tree"`` list of node dicts —
    this function never touches a `Store`.

    Known limitation (inherited from T1.5.4's `build_tree`, not fixed here): an item with no
    classified ``path`` yet is folded into a real ``Other`` root node, not treated as "nothing
    to show" — so this placeholder can only ever fire for a literally empty store, never for
    the "items exist but `python -m src.analyze` hasn't run yet" case its own copy describes.
    Distinguishing those two would mean `build_tree` (or this function) telling "never
    classified" apart from "classified into Other" — a `build_tree` change, out of scope here.
    """
    if not tree:
        return _section(
            "Report tree",
            "<p>No classified items yet — run <code>python -m src.analyze</code>.</p>",
        )
    tree_html = "".join(_tree_node_html(node, 0) for node in tree)
    return _section("Report tree", f'{_TREE_CONTROLS}<div id="tree">{tree_html}</div>')


def _trends_section_html(series: dict[str, dict[str, int]]) -> str:
    """`series` is :func:`~dashboard.snapshot.build_snapshot`'s own ``"trends"`` dict — this
    function never touches a `Store`."""
    if not series:
        return _section("Trends", "<p>No classified items yet.</p>")

    rows = []
    for category in sorted(series):
        weeks = series[category]
        max_count = max(weeks.values())
        bars = "".join(_bar_html(week, count, max_count) for week, count in sorted(weeks.items()))
        rows.append(f"<div class='category'><h3>{escape(category)}</h3>{bars}</div>")
    return _section("Trends", "".join(rows))


def _bar_html(week: str, count: int, max_count: int) -> str:
    width = count / max_count * 100
    return (
        f'<div class="week"><span class="label">{escape(week)}</span>'
        f'<div class="bar" style="width:{width:.0f}%">{count}</div></div>'
    )


def _forecasts_section_html(predictions: list[dict]) -> str:
    """`predictions` is :func:`~dashboard.snapshot.build_snapshot`'s own ``"forecasts"`` list
    (each a ``dataclasses.asdict(Prediction)`` dict) — this function never touches a `Store`."""
    if not predictions:
        return _section("Forecast log", "<p>No predictions recorded yet.</p>")

    rows = "".join(
        "<tr>"
        f"<td>{escape(p['claim'])}</td>"
        f"<td>{p['prob']:.2f}</td>"
        f"<td>{escape(p['due_date'])}</td>"
        f"<td>{escape(', '.join(p['evidence']))}</td>"
        "</tr>"
        for p in predictions
    )
    table = (
        "<table><thead><tr><th>Claim</th><th>Prob</th><th>Due</th><th>Evidence</th></tr>"
        f"</thead><tbody>{rows}</tbody></table>"
    )
    return _section("Forecast log", table)


# The expand/collapse + filter behavior, adapted directly from the approved
# docs/design/report-tree-mockup.html (same element shapes: #tree, .pr, .btn[data-all], #q).
# T5.16 adds one more control this script drives: `.btn.more` (see `_more_prs_html`) fetches
# the next page of a truncated node's own rows from `/api/node-prs` instead of ever having
# them embedded in the page — `prRowHtml`/`escHtml`/`safeHref` mirror `_pr_row_html`/`escape`/
# `_safe_href`'s templates client-side since the endpoint returns JSON rows
# (`src.agents.reporter_v1.pr_entry`'s shape), not pre-rendered HTML, matching the same "read
# layer stays plain data, rendering is a client concern" split T5.1 aims for. Known limitation,
# not fixed here (code-review finding): this is a real, disclosed duplication, not just an
# inherent language-boundary cost — the endpoint *could* return pre-rendered HTML fragments
# instead of JSON to avoid it, at the cost of coupling the read layer to HTML rendering. If
# `_pr_row_html`/`_state_chip`'s chip-class rules ever change, this JS copy must be updated by
# hand or paginated rows will render different chips/labels than the initially-embedded ones.
_TREE_SCRIPT = """
document.querySelectorAll('.btn[data-all]').forEach(function(b){
  b.addEventListener('click',function(){
    var open=b.dataset.all==='1';
    document.querySelectorAll('#tree details').forEach(function(d){d.open=open;});
  });
});
var q=document.getElementById('q');
if(q){
  q.addEventListener('input',function(){
    var term=q.value.trim().toLowerCase();
    var details=document.querySelectorAll('#tree details');
    if(!term){
      document.querySelectorAll('#tree .hidden').forEach(function(e){
        e.classList.remove('hidden');
      });
      details.forEach(function(d){
        d.open=(d.classList.contains('cat')||d.classList.contains('sub'));
      });
      return;
    }
    document.querySelectorAll('#tree .pr').forEach(function(pr){
      pr.classList.toggle('hidden', pr.textContent.toLowerCase().indexOf(term)===-1);
    });
    details.forEach(function(d){
      var self=d.querySelector('summary').textContent.toLowerCase().indexOf(term)!==-1;
      var hasPr=d.querySelector('.pr:not(.hidden)');
      var vis=self||hasPr;
      d.classList.toggle('hidden',!vis);
      d.open=vis;
      if(self){ d.querySelectorAll('.pr').forEach(function(pr){pr.classList.remove('hidden');}); }
    });
  });
}
function escHtml(s){
  return String(s).replace(/[&<>"']/g, function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}
function safeHref(u){
  return (u && (u.indexOf('http://')===0 || u.indexOf('https://')===0)) ? u : '';
}
function prRowHtml(pr){
  var chipClass='issue', chipLabel='issue';
  if(pr.type==='pr'){
    if(pr.state==='closed'){ chipClass='merged'; chipLabel='merged'; }
    else { chipClass='open'; chipLabel='open pr'; }
  }
  var id=(pr.repo||'?')+'#'+(pr.number!==null&&pr.number!==undefined?pr.number:'?');
  return '<div class="pr"><span class="id">'+escHtml(id)+'</span>'+
    '<span class="t"><a href="'+escHtml(safeHref(pr.url))+'">'+escHtml(pr.title||'')+'</a></span>'+
    '<span class="chip '+chipClass+'">'+escHtml(chipLabel)+'</span></div>';
}
document.querySelectorAll('.btn.more').forEach(function(b){
  b.addEventListener('click', function(){
    var path=b.dataset.path, offset=parseInt(b.dataset.offset, 10), limit=b.dataset.limit;
    var url='/api/node-prs?path='+encodeURIComponent(path)+'&offset='+offset+'&limit='+limit;
    fetch(url).then(function(r){ return r.json(); }).then(function(data){
      var html=data.prs.map(prRowHtml).join('');
      b.insertAdjacentHTML('beforebegin', html);
      var next=offset+data.prs.length;
      if(next>=data.total){ b.remove(); }
      else{
        b.dataset.offset=String(next);
        b.textContent='Show '+(data.total-next)+' more';
      }
    });
  });
});
"""

# CSS variables + tree rules adapted directly from the approved mockup — the design spec.
# Only two palettes exist (light/dark); `:root` and `:root[data-theme="light"]` share one rule
# since they're identical — `:root` always matches the root element regardless of its
# `data-theme` attribute, so this is equivalent to (not a behavior change from) writing them
# out separately, and the explicit `[data-theme="dark"]` override still wins on specificity
# over both the bare `:root` default and the `prefers-color-scheme` media query.
_STYLE = """
:root, :root[data-theme="light"]{
  --bg:#f6f8fb; --surface:#ffffff; --surface-2:#eef2f7; --border:#d7dee8;
  --ink:#182231; --ink-2:#586576; --ink-3:#8b96a6;
  --accent:#2f6db3; --merged:#7a5bd0; --merged-soft:#efe9fb;
  --open:#2e9e6b; --open-soft:#e4f4ec; --issue:#c0872a; --issue-soft:#f7eddb;
  --gap:#c74b45; --gap-soft:#f8e7e6; --guide:#e2e8f1;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0f141b; --surface:#151c26; --surface-2:#1b2430; --border:#2a3543;
    --ink:#e7ecf3; --ink-2:#9dabbd; --ink-3:#6c798b;
    --accent:#5c9cd9; --merged:#a488e8; --merged-soft:#241d38;
    --open:#4cbd8a; --open-soft:#12271d; --issue:#d8a24e; --issue-soft:#2b2413;
    --gap:#e0655e; --gap-soft:#2e1817; --guide:#26313f;
  }
}
:root[data-theme="dark"]{
  --bg:#0f141b; --surface:#151c26; --surface-2:#1b2430; --border:#2a3543;
  --ink:#e7ecf3; --ink-2:#9dabbd; --ink-3:#6c798b;
  --accent:#5c9cd9; --merged:#a488e8; --merged-soft:#241d38;
  --open:#4cbd8a; --open-soft:#12271d; --issue:#d8a24e; --issue-soft:#2b2413;
  --gap:#e0655e; --gap-soft:#2e1817; --guide:#26313f;
}
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  max-width:900px;margin:2rem auto;padding:0 1rem;background:var(--bg);color:var(--ink)}
h1{color:var(--ink)}
section{margin-bottom:2rem}
pre{white-space:pre-wrap;background:var(--surface-2);padding:1rem;border-radius:4px;color:var(--ink)}
table{border-collapse:collapse;width:100%}
td,th{border:1px solid var(--border);padding:.3rem .6rem;text-align:left;color:var(--ink)}
.week{display:flex;align-items:center;gap:.5rem;margin:.2rem 0}
.label{width:6rem;flex-shrink:0}
.bar{background:var(--accent);color:#fff;padding:.1rem .4rem;border-radius:2px;min-width:1.5rem}

.controls{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:1rem 0}
.controls input{flex:1;min-width:200px;background:var(--surface);border:1px solid var(--border);
  color:var(--ink);border-radius:7px;padding:.5rem .7rem;font-size:.9rem}
.btn{background:var(--surface);border:1px solid var(--border);color:var(--ink-2);
  border-radius:7px;padding:.5rem .75rem;font-size:.82rem;cursor:pointer}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.legend{display:flex;gap:.9rem;flex-wrap:wrap;font-size:.75rem;color:var(--ink-3);margin-bottom:1rem}
.legend span{display:inline-flex;align-items:center;gap:.35rem}
.dot{width:.6rem;height:.6rem;border-radius:50%}

details{margin:0}
summary{list-style:none;cursor:pointer;display:flex;align-items:baseline;gap:.5rem;
  padding:.4rem .5rem;border-radius:7px}
summary::-webkit-details-marker{display:none}
summary:hover{background:var(--surface-2)}
.chev{color:var(--ink-3);font-size:.7rem;width:.8rem;flex-shrink:0;transition:transform .15s}
details[open]>summary>.chev{transform:rotate(90deg)}
.name{font-weight:600}
.count{font-family:ui-monospace,monospace;font-size:.72rem;color:var(--ink-2);
  background:var(--surface-2);border:1px solid var(--border);border-radius:999px;
  padding:.05rem .45rem}
.cat{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:.35rem;margin-bottom:.6rem}
.cat>summary .name{font-size:1.05rem}
.sub>summary .name{color:var(--ink)}
.leaf>summary .name{font-weight:500;font-size:.92rem}
.kids{padding-left:1.15rem;margin-left:.55rem;border-left:1.5px solid var(--guide)}
.summary-line{color:var(--ink-2);font-size:.83rem;padding:.1rem .5rem .5rem 1.8rem;max-width:66ch}
.pr{display:flex;align-items:baseline;gap:.6rem;padding:.32rem .5rem .32rem 1.8rem;
  border-radius:6px;font-size:.9rem}
.pr:hover{background:var(--surface-2)}
.pr .id{font-family:ui-monospace,monospace;font-size:.78rem;color:var(--ink-3);
  flex-shrink:0;min-width:9.5rem}
.pr .t{color:var(--ink);flex:1}
.pr .t a{color:inherit}
.pr .t a:hover{color:var(--accent)}
.chip{font-size:.66rem;font-weight:600;letter-spacing:.02em;text-transform:uppercase;
  padding:.08rem .4rem;border-radius:5px;flex-shrink:0}
.chip.merged{color:var(--merged);background:var(--merged-soft)}
.chip.open{color:var(--open);background:var(--open-soft)}
.chip.issue{color:var(--issue);background:var(--issue-soft)}
.tag-gap{font-size:.66rem;font-weight:600;color:var(--gap);background:var(--gap-soft);
  border:1px solid var(--gap);border-radius:5px;padding:.02rem .38rem;margin-left:.4rem}
.btn.more{margin:.32rem 0 .32rem 1.8rem;font-size:.78rem}
.hidden{display:none !important}
"""


def render_snapshot_page(snapshot: dict) -> str:
    """The full dashboard page rendered from an already-built
    :func:`~dashboard.snapshot.build_snapshot` dict — the collapsible report tree + per-category
    trends + the forecast log. Pure ``dict -> str``: no `Store` import, no KB access at all
    (T5.16) — a page can be rendered from a snapshot loaded straight off disk
    (:func:`~dashboard.snapshot.load_snapshot`) with zero further store reads.
    """
    body = (
        _tree_section_html(snapshot["tree"])
        + _trends_section_html(snapshot["trends"])
        + _forecasts_section_html(snapshot["forecasts"])
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>vllm-forager dashboard</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<h1>vllm-forager dashboard</h1>" + body + f"<script>{_TREE_SCRIPT}</script>"
        "</body></html>"
    )


def render_page(store: Store) -> str:
    """The full dashboard page, built straight from `store` — one
    :func:`~dashboard.snapshot.build_snapshot` (still one full store scan; see that module's
    own docstring) then :func:`render_snapshot_page`. Unchanged signature/behavior for existing
    callers; the *rendering* step itself is now bounded regardless of KB size (T5.16).

    Superseded as the server's own default view by the T5.12 tabbed shell below (see
    :func:`render_tab_page`) — kept as-is, not deleted: it's still a complete, correct,
    independently-tested (``tests/test_dashboard.py``) pure `Store` -> HTML pipeline a future
    caller (e.g. a static-export CLI) could still reuse; rewriting those tests' own already-
    passing assertions to match a page shape they were never about is a distinct cleanup this
    integration todo doesn't need to also do.
    """
    return render_snapshot_page(build_snapshot(store))


# ============================================================================================
# T5.12 — tabbed console shell
#
# Unifies the panels already built (data-layer only, until now) across T5.2-T5.11 into one
# navigable, deep-linkable console: Issues (the tree section above, unchanged) · Reports/Trends
# (the trends/forecasts sections above, plus the T5.9 archive) · Candidates (T5.10) · Attempts
# (T5.11) · Agents/Ops (T5.8 live health + T5.13 cost + T5.7 guardrails). Each tab is its own
# full page (``GET /tab/<name>``, see dashboard/server.py) rather than one client-side-routed
# app — no JS framework, matching this whole dashboard's established "plain http.server, no
# build step" convention (server.py's own module docstring).
#
# Deliberately excludes one nav item T5.12's own DEVPLAN line names but that has no backing
# module yet: **Upstream PRs** (T5.14, not yet built at all) — matching this codebase's own
# repeated "wire what's real, disclose the rest as a known gap" convention (e.g. T5.1's own
# `candidates()` docstring on the identical kind of forward reference). Slots in as a
# straightforward addition once its own todo lands: a new tab_labels entry + render function.
# (T5.13's own cost sub-panel, also named on this same line, landed in a follow-up PR — see
# `render_ops_tab`/`dashboard.cost`.)
#
# T5.4 (parity heatmap), T5.5 (pipeline diagram), T5.2/T5.3 (richer monitoring/trend panels)
# are NOT wired here either — T5.12's own checklist line cites only T5.9/T5.10/T5.11/T5.8/T5.7,
# not those four; wiring them in is a further polish pass, not this todo's own scope (the same
# "the DEVPLAN line is the contract, its 'e.g.' and the milestone header's broader wish-list
# aren't" reading T5.5/T5.8's own already-merged notes established for their own per-agent
# nodes).
# ============================================================================================

TAB_LABELS = (
    ("issues", "Issues"),
    ("reports", "Reports/Trends"),
    ("candidates", "Candidates"),
    ("attempts", "Attempts"),
    ("ops", "Agents/Ops"),
)

_DEFAULT_TAB = "issues"

# 10 minutes — matches the value every existing health/liveness test already treats as a
# realistic default (no project-wide constant exists to import instead; see test_dashboard_
# health.py/test_liveness.py's own `stale_after_s=600`).
_STALE_AFTER_S = 600.0

_TAB_STYLE = """
.tabs{display:flex;gap:.3rem;flex-wrap:wrap;margin-bottom:1.5rem;
  border-bottom:1px solid var(--border)}
.tabs a{padding:.5rem .9rem;color:var(--ink-2);text-decoration:none;font-size:.88rem;
  border-bottom:2px solid transparent}
.tabs a:hover{color:var(--accent)}
.tabs a.active{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
.inline-form{display:inline-flex;gap:.4rem;margin:.3rem 0}
.chip.risk-low,.chip.decision-selected{color:var(--open);background:var(--open-soft)}
.chip.risk-medium{color:var(--issue);background:var(--issue-soft)}
.chip.risk-high,.chip.decision-skip{color:var(--gap);background:var(--gap-soft)}
.candidate,.attempt-row,.health-row{display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;
  padding:.45rem .5rem;border-radius:6px}
.candidate:hover,.attempt-row:hover,.health-row:hover{background:var(--surface-2)}
.candidate{flex-direction:column;align-items:flex-start;border:1px solid var(--border);
  margin-bottom:.5rem}
.candidate .badges{display:flex;gap:.4rem;flex-wrap:wrap}
.attempt-detail{border:1px solid var(--border);border-radius:8px;padding:1rem;margin-top:1rem}
.archive-list{display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:1rem}
.health-row{flex-direction:column;align-items:flex-start;border:1px solid var(--border);
  margin-bottom:.5rem}
.health-row pre{width:100%;box-sizing:border-box;font-size:.78rem;margin:.3rem 0 0}
.health-error{color:var(--gap);background:var(--gap-soft)}
.cost-row{display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;padding:.45rem .5rem;
  border-radius:6px;border:1px solid var(--border);margin-bottom:.5rem}
.cost-models{color:var(--ink-2);font-size:.82rem}
"""


def _nav_html(active: str) -> str:
    """The top nav bar shared by every tab page — plain links, each ``GET /tab/<name>``, so
    every tab is a real, deep-linkable URL (this todo's own "e.g."), not a client-side route."""
    links = "".join(
        f'<a href="/tab/{slug}" class="{"active" if slug == active else ""}">{escape(label)}</a>'
        for slug, label in TAB_LABELS
    )
    return f'<nav class="tabs">{links}</nav>'


def _page_shell(active_tab: str, body_html: str) -> str:
    """The full HTML document for one tab — nav + `body_html`, sharing the same style/tree
    script as the legacy single page (:data:`_STYLE`/:data:`_TREE_SCRIPT`) plus this shell's own
    :data:`_TAB_STYLE`. :data:`_TREE_SCRIPT` is embedded on every tab (not just Issues/Reports)
    -- its selectors (``#tree``, ``.btn[data-all]``, ``#q``) simply match nothing on a tab that
    has none of those elements, the same "harmless no-op elsewhere" property a page-wide
    ``<script>`` already has today.
    """
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>vllm-forager dashboard</title>"
        f"<style>{_STYLE}{_TAB_STYLE}</style></head><body>"
        "<h1>vllm-forager dashboard</h1>"
        + _nav_html(active_tab)
        + body_html
        + f"<script>{_TREE_SCRIPT}</script>"
        "</body></html>"
    )


def render_issues_tab(store: Store) -> str:
    """The Issues tab: the collapsible report tree alone (unchanged from the legacy single
    page's own tree section) — Trends/Forecasts move to the Reports/Trends tab below.

    Uses :func:`~dashboard.api.tree`, not ``build_snapshot(store)["tree"]`` — a code-review
    finding: `build_snapshot` unconditionally also computes trends (a second full store scan)
    and forecasts (a full prediction-log read) only to have both discarded here, the exact
    "wasted work, and a tree-only read could crash on an unrelated corrupt prediction record"
    problem `api.tree`'s own docstring already fixed this identical shape for once before."""
    return _tree_section_html(api.tree(store))


def _archive_tree_node_html(node: dict, depth: int) -> str:
    """One archived report's own tree node — like :func:`_tree_node_html`, but an archived
    node (:meth:`~src.agents.reporter_v1.TreeNode.to_dict`'s own ``{name, summary, count, gaps,
    children, prs}`` shape) carries no ``prs_truncated``/``prs_total`` fields: those are a
    live-snapshot-only concept :func:`~dashboard.snapshot.build_snapshot` adds for its own
    "show N more" pagination, which has no ``/api/node-prs`` counterpart for an old, static
    snapshot to page through. Every PR row renders directly here, uncapped."""
    summary_html = (
        f'<div class="summary-line">{escape(node["summary"])}</div>' if node["summary"] else ""
    )
    prs_html = "".join(_pr_row_html(pr) for pr in node["prs"])
    kids_html = "".join(_archive_tree_node_html(child, depth + 1) for child in node["children"])
    body = prs_html + kids_html
    kids_wrapped = f'<div class="kids">{body}</div>' if body else ""
    open_attr = " open" if depth == 0 else ""
    return (
        f'<details class="{_node_class(depth)}"{open_attr}>'
        '<summary><span class="chev">▶</span>'
        f'<span class="name">{escape(node["name"])}</span>'
        f'<span class="count">{node["count"]}</span>{_gap_chip_html(node["gaps"])}</summary>'
        f"{summary_html}{kids_wrapped}"
        "</details>"
    )


def _archive_section_html(store: Store, *, report_stamp: str | None) -> str:
    """The T5.9 reports archive: a row of links to every archived weekly report, plus (when
    `report_stamp` names one) that report's own tree, rendered via :func:`_archive_tree_node_html`
    (not :func:`_tree_node_html` — see that function's own docstring for why the two node
    shapes aren't interchangeable)."""
    reports = archive.list_archived_reports(store)
    if not reports:
        return _section(
            "Reports archive",
            "<p>No archived weekly reports yet — run <code>python -m src.report --tree</code>.</p>",
        )
    links = "".join(
        f'<a class="btn" href="/tab/reports?report={quote(r["stamp"])}">{escape(r["stamp"])}</a>'
        for r in reports
    )
    viewer_html = ""
    if report_stamp:
        tree = archive.open_archived_report(store, report_stamp)
        viewer_html = (
            f'<div class="archive-viewer">'
            f'{"".join(_archive_tree_node_html(n, 0) for n in tree)}</div>'
            if tree is not None
            else "<p>Unknown report.</p>"
        )
    return _section("Reports archive", f'<div class="archive-list">{links}</div>{viewer_html}')


def render_reports_tab(store: Store, *, report_stamp: str | None = None) -> str:
    """The Reports/Trends tab: the legacy Trends + Forecast-log sections (unchanged), plus the
    T5.9 archive (list + an optional opened report, via `report_stamp`).

    Uses :func:`render_trends`/:func:`render_forecasts` directly, not `build_snapshot` — a
    code-review finding: this tab never needs the classified tree (the most expensive part of
    a snapshot: a full scan plus one summary read per node), but `build_snapshot` computed and
    discarded it anyway on every load."""
    return (
        _trends_section_html(render_trends(store))
        + _forecasts_section_html([asdict(p) for p in render_forecasts(store)])
        + _archive_section_html(store, report_stamp=report_stamp)
    )


def _candidate_row_html(candidate: dict, decision: str | None) -> str:
    """One Candidates-tab row: title/evidence link, risk/effort/impact badges, the current
    decision (if any), and -- only when `candidate`'s own evidence resolves to a real
    ``(repo, number)`` -- the "Work this"/"Skip" form (T5.10). A candidate whose evidence
    doesn't parse can't be selected (mirrors :func:`~src.selection.filter_selected`'s own
    "no (repo, number), no decision" contract) -- it still lists, just without the actions.
    """
    parsed = parse_repo_number(candidate.get("evidence") or "")
    url = escape(_safe_href(candidate.get("evidence") or ""))
    title = escape(candidate.get("title") or "")
    risk = escape(candidate["risk"])
    badges = (
        f'<span class="chip risk-{risk}">{risk} risk</span>'
        f'<span class="chip">{escape(candidate["effort"])} effort</span>'
        f'<span class="chip">{escape(candidate["impact"])} impact</span>'
    )
    if decision:
        badges += f'<span class="chip decision-{escape(decision)}">{escape(decision)}</span>'
    actions_html = ""
    if parsed is not None:
        repo, number = parsed
        actions_html = (
            '<form method="post" action="/tab/candidates/decide" class="inline-form">'
            f'<input type="hidden" name="repo" value="{escape(repo)}">'
            f'<input type="hidden" name="number" value="{number}">'
            '<button name="decision" value="selected" class="btn">Work this</button>'
            '<button name="decision" value="skip" class="btn">Skip</button>'
            "</form>"
        )
    return (
        '<div class="candidate">'
        f'<div><a href="{url}">{title}</a></div>'
        f'<div class="badges">{badges}</div>'
        f"{actions_html}"
        "</div>"
    )


def render_candidates_tab(store: Store, *, compute: bool = True) -> str:
    """The Candidates tab (T5.10): the risk-ranked queue with why-selected (score breakdown +
    evidence) and a "Work this"/"Skip" control per row.

    Calls :func:`~dashboard.api.candidates` with ``compute=True`` by default -- unlike a read
    endpoint a page-view hazard would silently re-score on every load, a human *navigating to
    this tab* is exactly the explicit, bounded action T5.1's own ``compute=False`` default was
    written to require (the same "explicitly opened, right now" cost justification
    :mod:`dashboard.review`'s own docstring already established for an identical one-call-per-
    explicit-view shape).

    `compute=False` -- a code-review finding: `_serve_candidate_decision`'s own 303 redirect
    back to this same tab is *not* that explicit navigation, it's an automatic consequence of
    clicking "Work this"/"Skip" -- so the original always-`compute=True` version re-paid the
    full N-candidate LLM-scoring pass after every single decision (O(N) work per click, O(N^2)
    to clear a queue of N). The server now redirects with `?after_decision=1`
    (`dashboard.server._serve_candidate_decision`), which this function's caller
    (:func:`render_tab_page`) turns into `compute=False`: skip the recompute, show a
    confirmation instead, and let a human who actually wants the refreshed queue pay for it via
    one more explicit click -- mirroring `dashboard.attempts.open_attempt`'s own
    `include_review_bundle` opt-in shape for the identical "browsing != an explicit costly
    action" distinction.
    """
    if not compute:
        return _section(
            "Candidates",
            '<p>Decision recorded. <a class="btn" href="/tab/candidates">Refresh the queue</a> '
            "to re-score and see the current list (this re-runs LLM scoring for every "
            "candidate).</p>",
        )
    candidates = api.candidates(store, compute=True)
    if not candidates:
        return _section("Candidates", "<p>No candidates discovered.</p>")
    decisions = decisions_by_key(store)
    rows = "".join(_candidate_row_html(c, _decision_for(c, decisions)) for c in candidates)
    return _section("Candidates", rows)


def _decision_for(candidate: dict, decisions: dict[tuple[str, int], str]) -> str | None:
    parsed = parse_repo_number(candidate.get("evidence") or "")
    return decisions.get(parsed) if parsed is not None else None


def _attempt_row_html(row: dict) -> str:
    badge_class, badge_label = ("open", "🟢 verified") if row["verified"] else ("gap", "🔴 failed")
    label = escape(row.get("title") or f'{row["repo"]}#{row["number"]}')
    link = f'/tab/attempts?repo={quote(row["repo"])}&number={row["number"]}'
    return (
        '<div class="attempt-row">'
        f'<span class="chip {badge_class}">{badge_label}</span>'
        f'<a href="{escape(link)}">{escape(row["repo"])}#{row["number"]} — {label}</a>'
        "</div>"
    )


def _approve_hold_form_html(repo: str, number: int) -> str:
    return (
        '<form method="post" action="/tab/attempts/decide" class="inline-form">'
        f'<input type="hidden" name="repo" value="{escape(repo)}">'
        f'<input type="hidden" name="number" value="{number}">'
        '<button name="approve" value="true" class="btn">Approve</button>'
        '<button name="approve" value="false" class="btn">Hold</button>'
        "</form>"
    )


def _attempt_detail_html(
    repo: str, number: int, detail: dict, *, just_decided: bool = False
) -> str:
    """One opened attempt's full report (T3.12's 3 sections + outcome), plus -- only when
    `detail["review_bundle"]` is present (a verified, gate-ready candidate) -- the T5.6
    approve/hold form, satisfying T5.11's own DEVPLAN note ("go straight from browsing to
    approving without leaving the console").

    `just_decided=True` (a code-review finding) shows a confirmation banner instead -- the
    caller (:func:`render_attempts_tab`) only passes this right after the T5.6 POST redirect,
    when `detail` was built with `include_review_bundle=False` specifically to avoid re-paying
    `review_bundle`'s real LLM cost purely to redisplay a decision the human just made."""
    sections = (
        f"<h3>{escape(repo)}#{number}</h3>"
        f"<h4>Issue overview</h4><pre>{escape(detail['issue_overview'])}</pre>"
        f"<h4>Approach</h4><pre>{escape(detail['approach'])}</pre>"
        f"<h4>Reproduce</h4><pre>{escape(detail['reproduce'])}</pre>"
        f"<h4>Outcome</h4><pre>{escape(detail['outcome'])}</pre>"
    )
    if just_decided:
        sections += "<p>Decision recorded.</p>"
    elif detail.get("review_bundle") is not None:
        sections += _approve_hold_form_html(repo, number)
    return f'<div class="attempt-detail">{sections}</div>'


def render_attempts_tab(
    store: Store,
    *,
    open_candidate: tuple[str, int] | None = None,
    just_decided: bool = False,
) -> str:
    """The Attempts tab (T5.11): every worked candidate's outcome badge, newest first; opening
    one (`open_candidate`) renders its full report, with the T5.6 approve/hold form folded in
    for a verified, gate-ready candidate -- `include_review_bundle=True` is safe to pay here
    unconditionally (unlike :func:`~dashboard.attempts.open_attempt`'s own opt-in default),
    since opening one specific attempt from this list *is* the explicit, bounded action that
    default exists to gate -- the same distinction :func:`render_candidates_tab` draws for
    `api.candidates`'s own `compute` flag.

    `just_decided=True` (a code-review finding) skips `include_review_bundle` -- the server's
    own 303 redirect after recording a T5.6 approve/hold decision (`dashboard.server.
    _serve_attempt_decision`) landed back on this exact detail view and would otherwise
    re-trigger `review_bundle`'s real LLM call purely to redisplay content the human already
    saw right before deciding.
    """
    rows = attempts.list_attempts(store)
    list_html = (
        "".join(_attempt_row_html(r) for r in rows) if rows else "<p>No worked candidates yet.</p>"
    )
    detail_html = ""
    if open_candidate is not None:
        repo, number = open_candidate
        detail = attempts.open_attempt(store, repo, number, include_review_bundle=not just_decided)
        detail_html = (
            _attempt_detail_html(repo, number, detail, just_decided=just_decided)
            if detail is not None
            else "<p>Unknown attempt.</p>"
        )
    return _section("Attempts", f'<div class="attempt-list">{list_html}</div>{detail_html}')


def _health_row_html(stage: dict) -> str:
    """One stage's row: status chip, current step, elapsed time, and — a code-review finding —
    its intermediate ``output_tail``/``error``, both of which `_health_from_runs`
    (:mod:`dashboard.health`) computes specifically so a stall/failure is never just a bare
    status word (CLAUDE.md's own T5.8 guardrail: "the dashboard always shows what is running
    now... and its partial output. A crash must never leave a stage silently 'running'.")."""
    chip_class = {"ok": "open", "running": "open", "stalled": "gap", "failed": "gap"}.get(
        stage["status"], "issue"
    )
    elapsed = stage.get("elapsed_s")
    elapsed_label = f"{elapsed:.0f}s ago" if elapsed is not None else "—"
    detail_html = ""
    error = stage.get("error")
    output_tail = stage.get("output_tail")
    if error:
        detail_html += f'<pre class="health-error">{escape(str(error))}</pre>'
    if output_tail:
        detail_html += f"<pre>{escape(str(output_tail))}</pre>"
    return (
        '<div class="health-row">'
        f'<span class="id">{escape(stage["stage"])}</span>'
        f'<span class="chip {chip_class}">{escape(stage["status"])}</span>'
        f'<span>{escape(stage.get("step") or "")}</span>'
        f"<span>{escape(elapsed_label)}</span>"
        f"{detail_html}"
        "</div>"
    )


def _health_section_html(panel: dict) -> str:
    badge_class = "open" if panel["overall"] == "green" else "gap"
    rows = "".join(_health_row_html(s) for s in panel["stages"])
    return _section(
        "Live health",
        f'<span class="chip {badge_class}">{escape(panel["overall"])}</span>'
        f'<div class="health-rows">{rows}</div>',
    )


def _data_quality_section_html(records: list[dict]) -> str:
    if not records:
        return _section("Guardrails — data quality", "<p>No checks recorded yet.</p>")
    rows = "".join(
        "<tr>"
        f"<td>{escape(r.get('checked_at') or '')}</td>"
        f"<td>{escape(r.get('repo') or '')}</td>"
        f"<td>{escape(r.get('reason') or '')}</td>"
        f"<td>{r.get('count_delta', '')}</td>"
        f"<td>{'⚠️ flagged' if r.get('flagged') else 'ok'}</td>"
        "</tr>"
        for r in records
    )
    table = (
        "<table><thead><tr><th>Checked</th><th>Repo</th><th>Reason</th>"
        "<th>Δcount</th><th>Status</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    return _section("Guardrails — data quality", table)


def _rag_eval_section_html(scores: list[dict]) -> str:
    if not scores:
        return _section("Guardrails — RAG eval", "<p>No RAG-eval runs recorded yet.</p>")
    rows = "".join(
        "<tr>"
        f"<td>{escape(s.get('created_at') or '')}</td>"
        f"<td>{s.get('recall_at_k', 0):.2f}</td>"
        f"<td>{s.get('mrr', 0):.2f}</td>"
        f"<td>{s.get('faithfulness', 0):.2f}</td>"
        f"<td>{'✅ pass' if s.get('passed') else '❌ fail'}</td>"
        "</tr>"
        for s in scores
    )
    table = (
        "<table><thead><tr><th>Run</th><th>Recall@k</th><th>MRR</th>"
        "<th>Faithfulness</th><th>Status</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    return _section("Guardrails — RAG eval", table)


def _cost_agent_row_html(agent: dict) -> str:
    models_label = ", ".join(
        f"{escape(m['model'])} ({escape(m['provider'])}): ${m['cost_usd']:.2f}"
        for m in agent["by_model"]
    )
    return (
        '<div class="cost-row">'
        f'<span class="id">{escape(agent["agent"])}</span>'
        f'<span>today ${agent["today_usd"]:.2f}</span>'
        f'<span>7d ${agent["week_usd"]:.2f}</span>'
        f'<span>total ${agent["total_usd"]:.2f}</span>'
        f'<span class="cost-models">{models_label}</span>'
        "</div>"
    )


def _cost_section_html(panel: dict) -> str:
    """T5.13's own "e.g.": a bar per agent (today/7d/total, broken down by model & provider —
    :func:`~dashboard.cost.cost_panel`'s own `by_model`) plus a total-vs-budget badge that
    turns red once `panel["budget_exceeded"]` (today's spend crossed :data:`~src.config.
    DAILY_COST_BUDGET_USD`)."""
    badge_class = "gap" if panel["budget_exceeded"] else "open"
    budget_label = f'${panel["today_total_usd"]:.2f} / ${panel["budget_usd"]:.2f} today'
    rows = (
        "".join(_cost_agent_row_html(a) for a in panel["agents"])
        if panel["agents"]
        else "<p>No cost recorded yet.</p>"
    )
    return _section(
        "Cost",
        f'<span class="chip {badge_class}">{escape(budget_label)}</span> '
        f'<span>burn rate: ${panel["burn_rate_usd_per_day"]:.2f}/day</span>'
        f'<div class="cost-rows">{rows}</div>',
    )


def render_ops_tab(store: Store) -> str:
    """The Agents/Ops tab: T5.8's live health panel, T5.13's cost panel (placed right after
    health -- this todo's own "cost belongs with health, not with the reports" framing), and
    T5.7's guardrail panels (data quality, RAG eval)."""
    panel = health_panel(store, stale_after_s=_STALE_AFTER_S)
    return (
        _health_section_html(panel)
        + _cost_section_html(cost.cost_panel(store))
        + _data_quality_section_html(guardrails.data_quality_series(store))
        + _rag_eval_section_html(guardrails.rag_eval_series(store))
    )


def render_tab_page(
    store: Store,
    tab: str,
    *,
    report_stamp: str | None = None,
    open_candidate: tuple[str, int] | None = None,
    after_decision: bool = False,
) -> str:
    """The full page (nav + body) for `tab` — falls back to the Issues tab for an unrecognized
    name (a stale bookmark/typo'd URL degrades to the default view, not a 404 or a crash).

    `after_decision=True` (a code-review finding — see `render_candidates_tab`'s/
    `render_attempts_tab`'s own docstrings): set only when the caller (`dashboard.server`'s two
    POST handlers) is redirecting straight back from recording a T5.10/T5.6 decision, so the
    landing page doesn't silently re-pay the LLM cost that decision's own tab guards behind an
    explicit-navigation default.
    """
    if tab == "reports":
        body = render_reports_tab(store, report_stamp=report_stamp)
    elif tab == "candidates":
        body = render_candidates_tab(store, compute=not after_decision)
    elif tab == "attempts":
        body = render_attempts_tab(
            store, open_candidate=open_candidate, just_decided=after_decision
        )
    elif tab == "ops":
        body = render_ops_tab(store)
    else:
        tab = _DEFAULT_TAB
        body = render_issues_tab(store)
    return _page_shell(tab, body)
