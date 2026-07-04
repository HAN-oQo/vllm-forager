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

Known limitation (not fixed here): every request re-derives the tree (one `Store.query()`) and
trends (a second, independent `Store.query()`), plus one `get_state()` per taxonomy tree node
for its summary and one per historical prediction — no caching, batching, or cross-section
sharing anywhere. See `dashboard/server.py`'s own docstring; fixing this needs a caching layer
this thin M1 slice doesn't have.
"""

from __future__ import annotations

from html import escape

from src.agents.forecaster import Prediction, list_predictions
from src.agents.reporter import repo_number_label
from src.agents.reporter_v1 import TreeNode, tree_from_store
from src.store.base import Store
from src.trends import trends_from_store

# Only these schemes are ever rendered as a clickable href — GitHub-sourced items always carry
# an http(s) url (see reporter.evidence_url), but this dashboard has no auth and renders
# whatever the KB contains, so a stray `javascript:`/`data:` value in a malformed or
# non-GitHub-sourced record must never become a clickable link.
_SAFE_URL_SCHEMES = ("http://", "https://")

# Depth 0 -> "cat" (top-level taxonomy category), depth 1 -> "sub", depth 2+ -> "leaf" —
# purely a styling hook (nesting depth), independent of whether a node actually has children.
_NODE_CLASS_BY_DEPTH = ("cat", "sub")


def render_trends(store: Store) -> dict[str, dict[str, int]]:
    """Per-category, per-week activity counts (thin re-export of `trends.trends_from_store`)."""
    return trends_from_store(store)


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
    if pr.get("state") == "closed":
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


def _tree_node_html(node: TreeNode, depth: int) -> str:
    """One collapsible ``<details>`` node: name, count, an optional gap chip and summary
    line, its own cited PR rows, then its children — recursively, matching the mockup's
    대(大) → 소(summary) → 소소 → PRs nesting."""
    summary_html = f'<div class="summary-line">{escape(node.summary)}</div>' if node.summary else ""
    prs_html = "".join(_pr_row_html(pr) for pr in node.prs)
    kids_html = "".join(_tree_node_html(child, depth + 1) for child in node.children)
    body = prs_html + kids_html
    kids_wrapped = f'<div class="kids">{body}</div>' if body else ""
    open_attr = " open" if depth == 0 else ""
    return (
        f'<details class="{_node_class(depth)}"{open_attr}>'
        '<summary><span class="chev">▶</span>'
        f'<span class="name">{escape(node.name)}</span>'
        f'<span class="count">{node.count}</span>{_gap_chip_html(node.gaps)}</summary>'
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
    "present in fork, missing upstream</span>"
    "</div>"
)


def _tree_section_html(store: Store) -> str:
    """The collapsible report tree, or a placeholder when the store has no items at all.

    Known limitation (inherited from T1.5.4's `build_tree`, not fixed here): an item with no
    classified ``path`` yet is folded into a real ``Other`` root node, not treated as "nothing
    to show" — so this placeholder can only ever fire for a literally empty store, never for
    the "items exist but `python -m src.analyze` hasn't run yet" case its own copy describes.
    Distinguishing those two would mean `build_tree` (or this function) telling "never
    classified" apart from "classified into Other" — a `build_tree` change, out of scope here.
    """
    nodes = tree_from_store(store)
    if not nodes:
        return _section(
            "Report tree",
            "<p>No classified items yet — run <code>python -m src.analyze</code>.</p>",
        )
    tree_html = "".join(_tree_node_html(node, 0) for node in nodes)
    return _section("Report tree", f'{_TREE_CONTROLS}<div id="tree">{tree_html}</div>')


def _trends_section_html(store: Store) -> str:
    series = render_trends(store)
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


def _forecasts_section_html(store: Store) -> str:
    predictions = render_forecasts(store)
    if not predictions:
        return _section("Forecast log", "<p>No predictions recorded yet.</p>")

    rows = "".join(
        "<tr>"
        f"<td>{escape(p.claim)}</td>"
        f"<td>{p.prob:.2f}</td>"
        f"<td>{escape(p.due_date)}</td>"
        f"<td>{escape(', '.join(p.evidence))}</td>"
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
.hidden{display:none !important}
"""


def render_page(store: Store) -> str:
    """The full dashboard page: the collapsible report tree + per-category trends + the
    forecast log."""
    body = _tree_section_html(store) + _trends_section_html(store) + _forecasts_section_html(store)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>vllm-forager dashboard</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<h1>vllm-forager dashboard</h1>" + body + f"<script>{_TREE_SCRIPT}</script>"
        "</body></html>"
    )
