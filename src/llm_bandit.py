"""Cost-aware LLM provider selection (T2.6): a UCB1 bandit over ``LLM_PROVIDER``.

Different tasks warrant different models — classification runs fine on a cheap local vLLM
server, while patch-writing wants a stronger (pricier) provider — but nothing in this codebase
decides that automatically; :mod:`src.llm` dispatches on one hardcoded env var for every call.
This module is the *policy* an agent could ask "which provider should I use for this kind of
call," learned from measured reward-per-cost rather than a hardcoded preference — one arm per
provider (:class:`~src.llm.CallMeta`'s own ``provider`` field is the natural arm label; its
``model`` field is *not* part of the label — two models under the same provider are pooled
into one arm, matching the DEVPLAN's own explicit three-arm framing: ``claude_cli`` /
``claude_api`` / ``local``-vLLM, not per-model).

**Value = reward / (cost + latency-as-cost)**, not raw reward: a provider that succeeds
cheaply and quickly should outrank one that succeeds just as often but expensively or slowly —
raw-reward UCB can't express either tradeoff. Latency is folded into cost via
:data:`_LATENCY_WEIGHT_USD_PER_S` (an illustrative, tunable USD-per-second exchange rate,
following the DEVPLAN's own framing of weighing reward against "cost/latency" together rather
than as two independently-scored dimensions). A provider whose combined cost is ``0.0`` (a
self-hosted ``local`` endpoint with negligible latency, or any real provider before its first
billed call resolves) would divide by exactly zero — floored at :data:`_MIN_COST` instead, so a
free/instant provider's value is very large (favored) but never literally infinite or
undefined.

Known simplification, not fixed here: a *failed* call (``reward=0.0``) always has ``value ==
0.0`` regardless of cost — reward-per-cost is genuinely zero for zero reward, so a cheap
failure and an expensive failure currently score identically. Penalizing cost even on failure
would need an additive (not ratio) value model, a bigger redesign than this first cut's
"reward-per-cost" framing calls for.

Classic UCB1 "optimism under uncertainty": an arm with zero observations always scores
``+inf`` and is picked first, before the bandit ever exploits its current best guess — this is
what "an unseen provider still gets explored" (the DEVPLAN's own acceptance line) means
mechanically, not a separate exploration mode bolted on top.

:class:`Bandit` is immutable, updated functionally — :meth:`Bandit.record_outcome` returns a
*new* `Bandit` rather than mutating in place (**a caller must reassign**, e.g. ``bandit =
bandit.record_outcome(...)`` — discarding the return value silently loses the update, a real
footgun once a real per-call-site integration exists). This is similar in spirit to
:class:`~src.taxonomy.Taxonomy`/:class:`~src.policy.Policy`'s own immutable updates — a caller
that wants to persist a `Bandit` in the KB can serialize it the same way (:meth:`Bandit.to_json`
/:meth:`Bandit.from_json`), without this module needing to know about
:class:`~src.store.base.Store` itself (mirrors :mod:`src.parity`/:mod:`~src.agents.curator`'s
own pure-core, store-agnostic design) — but unlike Taxonomy/Policy, `Bandit` keeps only its
*latest* state, with no version counter or history of earlier states to retrieve; and unlike
:class:`~src.policy.PolicyError`/:class:`~src.taxonomy.TaxonomyError`, :class:`BanditError` has
no "nothing created yet" case, because a fresh ``Bandit()`` (no arms recorded) is always a
cheap, valid starting point — there's no missing-prerequisite state to guard against.

Known limitations, not fixed here:
- No agent in this codebase actually calls :func:`~src.llm.complete_detailed` (the variant
  that returns :class:`~src.llm.CallMeta`) or feeds its result into
  :meth:`Bandit.record_outcome` yet, nor is there a standard convention for how a caller should
  derive `reward` (task success) from a completion — that's inherently task-specific (only the
  calling agent knows if its own output was actually useful), not something this module or
  `CallMeta` itself could supply. This module is the learning *algorithm*, tested against
  synthetic history per the DEVPLAN's own Test bullet; wiring a real agent's calls through it
  (and persisting the running `Bandit` in the KB across runs) is follow-up integration work.
- Nothing here validates a `provider` string against :mod:`src.llm`'s own real dispatch table
  (which has no public canonical list to check against) — a caller passing a typo'd or retired
  provider name gets a permanently-"unseen" arm that keeps winning `select()` (``+inf`` never
  decays), silently short-circuiting real exploration. Fixing this needs a public provider
  registry in :mod:`src.llm` itself, out of scope for this module.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType

# Floor for the combined (cost + latency-as-cost) value in the reward/cost calculation —
# without it, a provider with cost_usd == 0.0 and negligible latency (a self-hosted `local`
# endpoint) would divide by exactly zero. Small enough that any real (non-zero) cost still
# dominates the comparison; just large enough that `reward / _MIN_COST` is a large, finite
# number rather than `inf`/`nan`.
_MIN_COST = 1e-4

# Illustrative USD-per-second exchange rate for folding CallMeta.latency_s into the same
# combined "cost" the value calculation already uses for cost_usd — see module docstring for
# why cost and latency are weighed together, not as two independently-scored dimensions.
# Tunable; not derived from any real pricing data.
_LATENCY_WEIGHT_USD_PER_S = 0.001

_DEFAULT_EXPLORATION = math.sqrt(2)  # the standard UCB1 constant


class BanditError(RuntimeError):
    """`select()` was called with no candidate providers, an outcome had an out-of-range
    `reward`/negative-or-NaN `cost`/`latency_s`, or a corrupt serialized `Bandit` record."""


def _value(reward: float, cost: float, latency_s: float) -> float:
    """reward / combined-cost, with the combined cost floored at :data:`_MIN_COST` (see module
    docstring for both the floor and the cost+latency weighting)."""
    combined_cost = cost + _LATENCY_WEIGHT_USD_PER_S * latency_s
    return reward / max(combined_cost, _MIN_COST)


@dataclass(frozen=True)
class ArmStats:
    """One provider's accumulated bandit statistics.

    Raises:
        BanditError: `pulls` is negative, or `pulls == 0` with a nonzero `total_value` (an
            arm with no recorded observations can't have accumulated any value) — enforced
            here, not just in :meth:`Bandit.record_outcome`, so :meth:`Bandit.replace_arm` and
            :meth:`Bandit.from_json` can't construct an inconsistent arm either.
    """

    pulls: int = 0
    total_value: float = 0.0  # running sum of reward/cost across every recorded outcome

    def __post_init__(self) -> None:
        if self.pulls < 0:
            raise BanditError(f"pulls must be >= 0, got {self.pulls}")
        if self.pulls == 0 and self.total_value != 0.0:
            raise BanditError("total_value must be 0.0 when pulls == 0")

    @property
    def mean_value(self) -> float:
        """Average reward-per-cost so far, or ``0.0`` for a never-pulled arm (never consulted
        when `pulls == 0` — :meth:`Bandit.ucb_score` short-circuits to ``+inf`` first)."""
        return self.total_value / self.pulls if self.pulls else 0.0


@dataclass(frozen=True)
class Bandit:
    """A UCB1 bandit over named arms (LLM providers) — see module docstring.

    `stats` is a read-only :class:`~types.MappingProxyType` (not a plain ``dict``) — the same
    "frozen dataclass alone doesn't stop mutating a mapping it points to" lesson
    :func:`~src.policy._freeze` was written for — so ``bandit.stats["x"] = ...`` fails loudly
    instead of silently mutating a `Bandit` that's supposed to be immutable.
    """

    stats: MappingProxyType[str, ArmStats] = field(default_factory=lambda: MappingProxyType({}))

    def record_outcome(
        self, provider: str, *, reward: float, cost: float, latency_s: float = 0.0
    ) -> Bandit:
        """A new `Bandit` with `provider`'s statistics updated by one more observation.

        Raises:
            BanditError: `reward` isn't in ``[0, 1]``, or `cost`/`latency_s` is negative or NaN.
        """
        if not 0.0 <= reward <= 1.0:
            raise BanditError(f"reward must be in [0, 1], got {reward}")
        if not cost >= 0.0:  # also rejects NaN: any comparison with NaN is False
            raise BanditError(f"cost must be >= 0, got {cost}")
        if not latency_s >= 0.0:
            raise BanditError(f"latency_s must be >= 0, got {latency_s}")
        current = self.stats.get(provider, ArmStats())
        updated = ArmStats(
            pulls=current.pulls + 1,
            total_value=current.total_value + _value(reward, cost, latency_s),
        )
        return replace(self, stats=MappingProxyType({**self.stats, provider: updated}))

    def ucb_score(
        self, provider: str, *, total_pulls: int, exploration: float = _DEFAULT_EXPLORATION
    ) -> float:
        """`provider`'s UCB1 score given `total_pulls` (the standard UCB1 "elapsed rounds"
        term — see :meth:`select` for how this is scoped). ``+inf`` if `provider` has never
        been pulled.
        """
        arm = self.stats.get(provider, ArmStats())
        if arm.pulls == 0:
            return float("inf")
        bonus = exploration * math.sqrt(math.log(max(total_pulls, 1)) / arm.pulls)
        return arm.mean_value + bonus

    def select(self, providers: list[str], *, exploration: float = _DEFAULT_EXPLORATION) -> str:
        """The provider among `providers` with the highest UCB1 score — an arm with zero
        recorded outcomes always wins first (see module docstring), so every provider is tried
        at least once before the bandit commits to exploiting its current best guess.

        `total_pulls` (the UCB1 "elapsed rounds" term) is scoped to just `providers`'s own
        combined pull count, not every arm this `Bandit` has ever recorded — a stale/retired
        provider's history shouldn't inflate the exploration bonus for arms actually being
        compared here.

        Raises:
            BanditError: `providers` is empty.
        """
        if not providers:
            raise BanditError("select() needs at least one candidate provider")
        total_pulls = sum(self.stats.get(p, ArmStats()).pulls for p in providers)
        return max(
            providers,
            key=lambda p: self.ucb_score(p, total_pulls=total_pulls, exploration=exploration),
        )

    def to_json(self) -> str:
        """Serialize for storage (e.g. in the KB's state map, by a future caller)."""
        return json.dumps(
            {
                provider: {"pulls": arm.pulls, "total_value": arm.total_value}
                for provider, arm in self.stats.items()
            }
        )

    @staticmethod
    def from_json(raw: str) -> Bandit:
        """Deserialize a value previously produced by :meth:`to_json`.

        Raises:
            BanditError: `raw` isn't valid JSON, isn't shaped like a `Bandit` record, or an
                arm's `pulls`/`total_value` has the wrong type or fails :class:`ArmStats`'s
                own validation.
        """
        try:
            data = json.loads(raw)
            stats = {}
            for provider, arm in data.items():
                pulls, total_value = arm["pulls"], arm["total_value"]
                if not isinstance(pulls, int) or not isinstance(total_value, (int, float)):
                    raise TypeError(f"{provider!r} has non-numeric pulls/total_value: {arm!r}")
                stats[provider] = ArmStats(pulls=pulls, total_value=float(total_value))
            return Bandit(stats=MappingProxyType(stats))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise BanditError(f"corrupt bandit record: {exc}") from exc

    def replace_arm(self, provider: str, stats: ArmStats) -> Bandit:
        """A new `Bandit` with `provider`'s stats set directly — mainly for tests that need to
        seed a specific pull count/value without replaying :meth:`record_outcome` calls (real
        callers should use that instead: this bypasses reward/cost validation, only
        :class:`ArmStats`'s own internal-consistency check still applies)."""
        return replace(self, stats=MappingProxyType({**self.stats, provider: stats}))
