"""Thin read-only dashboard (T1.9): a local web view over M1's outputs.

Pulled forward from M5 (docs/PLAN.md sanctions an early thin version as soon as there's a
loop to watch) — a "follow-along" tool so M1's report/trends/forecasts are visible on one
screen instead of running three separate CLI commands.

:mod:`dashboard.render` holds the pure, store-in/HTML-out render functions (no server, no
sockets — trivially unit-testable). :mod:`dashboard.server` wires those into a stdlib
``http.server`` (no new web-framework dependency for this thin slice). Read-only, no auth: it
never writes to the store or to disk, matching a local monitoring tool's threat model.
"""
