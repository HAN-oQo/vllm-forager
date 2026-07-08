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

from html import escape

from src.agents.forecaster import Prediction, list_predictions
from src.agents.reporter import is_merged, repo_number_label
from src.store.base import Store
from src.taxonomy import LEVEL_SEPARATOR

from . import api
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
    callers; the *rendering* step itself is now bounded regardless of KB size (T5.16)."""
    return render_snapshot_page(build_snapshot(store))
