"""Versioned policy object (T1.3).

The M2 outer loop grades past predictions and proposes an updated policy from the results
(T2.2); the current policy — scoring weights, prompt templates, and which taxonomy version
is active — is what every scheduled agent run reads to decide *how* to score/classify/prompt.
Same append-only-version + active-pointer shape as :mod:`src.taxonomy` (T1.2), so grading can
always ask "what policy was active when this prediction was made?" — never just the latest.

Storage: same as :mod:`src.taxonomy` — the KB's generic state map (``get_state``/
``set_state``), keyed ``policy@1``, ``policy@2``, ... plus one ``policy_active_version``
pointer. Known limitations (race on concurrent writers; redundant full-state.json I/O per
call) are the same as documented in :mod:`src.taxonomy` and not repeated here.

Known duplication (not fixed here): this module's version/active-pointer machinery is
structurally near-identical to :mod:`src.taxonomy`'s — the two evolved independently rather
than sharing a base, because their "evolve" semantics genuinely differ (taxonomy appends one
category; policy merges partial field updates). A shared read/version-bookkeeping helper is a
reasonable future refactor once a third versioned-object need clarifies the right boundary,
not done here to avoid re-touching the already-merged, already-tested ``taxonomy.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TypeVar

from . import taxonomy
from .store.base import Store

_VERSION_KEY_PREFIX = "policy@"
_ACTIVE_KEY = "policy_active_version"

_K = TypeVar("_K")
_V = TypeVar("_V")


class PolicyError(RuntimeError):
    """Any policy failure: no policy exists yet, an unknown/corrupt version, or a dangling
    ``active_taxonomy_version`` reference."""


def _freeze(mapping: dict[_K, _V]) -> MappingProxyType[_K, _V]:
    """Defensively copy `mapping` and wrap it read-only."""
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class Policy:
    """One immutable policy version: scoring weights, prompt templates, and a taxonomy ref.

    ``scoring_weights``/``prompt_templates`` are ``MappingProxyType`` (not plain ``dict``) so
    ``policy.scoring_weights["x"] = 1`` fails loudly instead of silently mutating a version
    that's supposed to be immutable — the same lesson learned from T1.2's ``categories``
    needing a ``tuple`` instead of a ``list``. Unlike :class:`~src.taxonomy.Taxonomy`,
    ``Policy`` is intentionally **not hashable** — a mapping value can never be hashable, and
    nothing in this codebase needs a ``Policy`` as a set/dict key.
    """

    version: int
    scoring_weights: MappingProxyType[str, float]
    prompt_templates: MappingProxyType[str, str]
    active_taxonomy_version: int

    def to_json(self) -> str:
        """Serialize for storage in the KB's state map."""
        return json.dumps(
            {
                "version": self.version,
                "scoring_weights": dict(self.scoring_weights),
                "prompt_templates": dict(self.prompt_templates),
                "active_taxonomy_version": self.active_taxonomy_version,
            }
        )

    @staticmethod
    def from_json(raw: str) -> Policy:
        """Deserialize a value previously produced by :meth:`to_json`.

        Raises:
            PolicyError: `raw` isn't valid JSON, or isn't shaped like a policy record.
        """
        try:
            data = json.loads(raw)
            return Policy(
                version=data["version"],
                scoring_weights=_freeze({k: float(v) for k, v in data["scoring_weights"].items()}),
                prompt_templates=_freeze(data["prompt_templates"]),
                active_taxonomy_version=data["active_taxonomy_version"],
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise PolicyError(f"corrupt policy record: {exc}") from exc


def _version_key(version: int) -> str:
    return f"{_VERSION_KEY_PREFIX}{version}"


def _parse_version(raw: str) -> int:
    """Parse the active-version pointer; raise :class:`PolicyError` (not ValueError)."""
    try:
        return int(raw)
    except ValueError as exc:
        raise PolicyError(f"corrupt active-version pointer: {raw!r}") from exc


def _require_taxonomy_version(store: Store, version: int) -> None:
    """Confirm `version` is a real taxonomy version before letting a policy reference it.

    Raises:
        PolicyError: no such taxonomy version exists.
    """
    try:
        taxonomy.get_taxonomy(store, version)
    except taxonomy.TaxonomyError as exc:
        raise PolicyError(f"active_taxonomy_version {version} is invalid: {exc}") from exc


def create_policy(
    store: Store,
    *,
    scoring_weights: dict[str, float],
    prompt_templates: dict[str, str],
    active_taxonomy_version: int,
) -> Policy:
    """Create version 1 of the policy and make it active.

    Raises:
        PolicyError: a policy already exists — evolve it with :func:`update_policy` instead
            of creating a second v1 — or `active_taxonomy_version` doesn't exist.
    """
    if store.get_state(_ACTIVE_KEY) is not None:
        raise PolicyError("a policy already exists; use update_policy() to evolve it")
    _require_taxonomy_version(store, active_taxonomy_version)
    policy = Policy(
        version=1,
        scoring_weights=_freeze({k: float(v) for k, v in scoring_weights.items()}),
        prompt_templates=_freeze(prompt_templates),
        active_taxonomy_version=active_taxonomy_version,
    )
    store.set_state(_version_key(1), policy.to_json())
    store.set_state(_ACTIVE_KEY, str(policy.version))
    return policy


def get_policy(store: Store, version: int) -> Policy:
    """Return a specific, immutable policy version — older versions stay retrievable.

    Raises:
        PolicyError: `version` doesn't exist, or the stored record's own version field
            doesn't match `version` (KB corruption — the two are stored redundantly).
    """
    raw = store.get_state(_version_key(version))
    if raw is None:
        raise PolicyError(f"policy version {version} not found")
    policy = Policy.from_json(raw)
    if policy.version != version:
        raise PolicyError(
            f"policy@{version} record claims version {policy.version} — KB corruption"
        )
    return policy


def get_active(store: Store) -> Policy:
    """Return the currently active (latest) policy version.

    Raises:
        PolicyError: no policy has been created yet, or the active pointer is corrupt.
    """
    raw = store.get_state(_ACTIVE_KEY)
    if raw is None:
        raise PolicyError("no policy exists yet; call create_policy() first")
    return get_policy(store, _parse_version(raw))


def update_policy(
    store: Store,
    *,
    scoring_weights: dict[str, float] | None = None,
    prompt_templates: dict[str, str] | None = None,
    active_taxonomy_version: int | None = None,
) -> Policy:
    """Evolve the policy: create version N+1 and activate it.

    Each of ``scoring_weights``/``prompt_templates`` is **merged** key-by-key into the
    current version's mapping (only the keys you pass are added/overwritten; every other
    key carries over unchanged) — a caller adjusting one category's weight doesn't need to
    restate every other category. ``active_taxonomy_version`` left ``None`` carries over
    unchanged. The prior version is left untouched (append-only), so a grading record that
    names "graded against policy v2" can still look v2 up after v3 becomes active.

    Raises:
        PolicyError: no policy exists yet, or `active_taxonomy_version` doesn't exist.
    """
    current = get_active(store)
    new_taxonomy_version = (
        active_taxonomy_version
        if active_taxonomy_version is not None
        else current.active_taxonomy_version
    )
    _require_taxonomy_version(store, new_taxonomy_version)
    merged_weights = {**current.scoring_weights, **(scoring_weights or {})}
    merged_templates = {**current.prompt_templates, **(prompt_templates or {})}
    updated = replace(
        current,
        version=current.version + 1,
        scoring_weights=_freeze({k: float(v) for k, v in merged_weights.items()}),
        prompt_templates=_freeze(merged_templates),
        active_taxonomy_version=new_taxonomy_version,
    )
    store.set_state(_version_key(updated.version), updated.to_json())
    store.set_state(_ACTIVE_KEY, str(updated.version))
    return updated
