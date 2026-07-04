"""Pure render functions: KB (store + reports dir) in, HTML/data out — no server involved.

Kept separate from :mod:`dashboard.server` so every render can be unit-tested against a
seeded store/tmp_path fixture without opening a socket (per T1.9's own test plan).

The report body is read from the **latest already-generated** file under `reports_dir`
(written by ``python -m src.report``), not regenerated live from the store on every page
load — :mod:`src.agents.reporter_v1` calls the LLM once per category, so re-rendering it on
every HTTP request would mean an LLM call per dashboard refresh. Trends and the forecast log
are cheap, pure store reads (:mod:`src.trends`, :mod:`src.agents.forecaster`), so those ARE
computed live — the dashboard reflects new items/predictions immediately; only the narrative
report itself lags to its last scheduled ``python -m src.report`` run.

Known limitation: unlike trends/forecasts, the report section is NOT read through the `Store`
interface — ``src.report``'s ``generate()`` always writes its Markdown to a **local**
`reports_dir` file, never into the KB, regardless of the `STORE` backend. If ``STORE=firestore``
and this dashboard runs on a different host/filesystem than whatever last ran
``python -m src.report``, `render_report` will return the "no report yet" placeholder
indefinitely — that's a "wrong host", not "no report exists" — with no way to tell them apart
from here. A real fix means teaching ``src.report`` to persist report bodies into the KB
itself; out of scope for this thin M1 slice, where the assumption is a single shared
`FORAGER_DATA_DIR` (docs/PLAN.md) local to wherever the dashboard runs.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from src.agents.forecaster import Prediction, list_predictions
from src.store.base import Store
from src.trends import trends_from_store

_NO_REPORT_YET = "No report has been generated yet — run `python -m src.report`."


def latest_report_path(reports_dir: Path) -> Path | None:
    """The most recently written ``*.md`` report under `reports_dir`, or None if there are none.

    Report filenames are ISO year-week stamps (``2026-W27.md``, T1.7's ``week_stamp``), which
    sort lexicographically in chronological order — so the max filename IS the latest report,
    no filesystem mtime (unreliable across copies/checkouts) needed.
    """
    if not reports_dir.is_dir():
        return None
    reports = sorted(reports_dir.glob("*.md"))
    return reports[-1] if reports else None


def render_report(reports_dir: Path) -> str:
    """The latest report's raw Markdown body, or a placeholder if none exists yet."""
    path = latest_report_path(reports_dir)
    if path is None:
        return _NO_REPORT_YET
    return path.read_text(encoding="utf-8")


def render_trends(store: Store) -> dict[str, dict[str, int]]:
    """Per-category, per-week activity counts (thin re-export of `trends.trends_from_store`)."""
    return trends_from_store(store)


def render_forecasts(store: Store) -> list[Prediction]:
    """Every recorded prediction, oldest first (thin re-export of `forecaster.list_predictions`)."""
    return list_predictions(store)


def _section(title: str, body_html: str) -> str:
    """Wrap `body_html` in the page's common ``<section><h2>title</h2>...</section>`` shell."""
    return f"<section><h2>{title}</h2>{body_html}</section>"


def _report_section_html(reports_dir: Path) -> str:
    body = render_report(reports_dir)
    return _section("Latest report", f"<pre>{escape(body)}</pre>")


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


def render_page(store: Store, reports_dir: Path) -> str:
    """The full dashboard page: latest report + per-category trends + the forecast log."""
    body = (
        _report_section_html(reports_dir)
        + _trends_section_html(store)
        + _forecasts_section_html(store)
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>vllm-forager dashboard</title>"
        "<style>"
        "body{font-family:sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}"
        "pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem;border-radius:4px}"
        "table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccc;padding:.3rem .6rem;text-align:left}"
        ".week{display:flex;align-items:center;gap:.5rem;margin:.2rem 0}"
        ".label{width:6rem;flex-shrink:0}"
        ".bar{background:#4a7ebb;color:#fff;padding:.1rem .4rem;border-radius:2px;min-width:1.5rem}"
        "</style></head><body>"
        "<h1>vllm-forager dashboard</h1>" + body + "</body></html>"
    )
