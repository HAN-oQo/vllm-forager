"""Cost-aware LLM provider selection (T2.6): a UCB1 bandit over ``LLM_PROVIDER``.

Different tasks warrant different models — classification runs fine on a cheap local vLLM
server, while patch-writing wants a stronger (pricier) provider — but nothing in this codebase
decides that automatically; :mod:`src.llm` dispatches on one hardcoded env var for every call.
This module is the *policy* an agent could ask "which provider should I use for this kind of
call," learned from measured reward-per-cost rather than a hardcoded preference — one arm per
provider (:class:`~src.llm.CallMeta`'s own ``provider`` field is the natural arm label).

**Value = reward / cost**, not raw reward: a provider that succeeds cheaply should outrank one
that succeeds just as often but expensively, which raw-reward UCB can't express. A provider
with ``cost_usd == 0.0`` (a self-hosted ``local`` endpoint, or any real provider before its
first billed call resolves) would divide by exactly zero — floored at :data:`_MIN_COST`
instead, so a free provider's value is very large (favored) but never literally infinite or
undefined.

Classic UCB1 "optimism under uncertainty": an arm with zero observations always scores
``+inf`` and is picked first, before the bandit ever exploits its current best guess — this is
what "an unseen provider still gets explored" (the DEVPLAN's own acceptance line) means
mechanically, not a separate exploration mode bolted on top.

:class:`Bandit` is immutable, updated functionally (:meth:`Bandit.record_outcome` returns a
*new* `Bandit`) — the same append-only-evolution shape as :class:`~src.taxonomy.Taxonomy`/
:class:`~src.policy.Policy`, so a caller that wants to persist this in the KB can serialize the
result the same way, without this module needing to know about :class:`~src.store.base.Store`
itself (mirrors :mod:`src.parity`/:mod:`~src.agents.curator`'s own pure-core, store-agnostic
design).

Known limitation, not fixed here: no agent in this codebase actually calls
:func:`~src.llm.complete_detailed` (the variant that returns :class:`~src.llm.CallMeta`) or
feeds its result into :meth:`Bandit.record_outcome` yet — this module is the learning
*algorithm*, tested against synthetic history per the DEVPLAN's own Test bullet; wiring a real
agent's calls through it (and persisting the running `Bandit` in the KB across runs) is
follow-up integration work, not this todo's own scope.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace

# Floor for `cost` in the reward/cost value calculation — without it, a provider with
# cost_usd == 0.0 (a self-hosted `local` endpoint) would divide by exactly zero. Small enough
# that any real (non-zero) cost still dominates the comparison; just large enough that
# `reward / _MIN_COST` is a large, finite number rather than `inf`/`nan`.
_MIN_COST = 1e-4

_DEFAULT_EXPLORATION = math.sqrt(2)  # the standard UCB1 constant


class BanditError(RuntimeError):
    """`select()` was called with no candidate providers, or an outcome had an out-of-range
    `reward`/negative `cost`, or a corrupt serialized `Bandit` record."""


def _value(reward: float, cost: float) -> float:
    """reward-per-cost, with `cost` floored at :data:`_MIN_COST` (see module docstring)."""
    return reward / max(cost, _MIN_COST)


@dataclass(frozen=True)
class ArmStats:
    """One provider's accumulated bandit statistics."""

    pulls: int = 0
    total_value: float = 0.0  # running sum of reward/cost across every recorded outcome

    @property
    def mean_value(self) -> float:
        """Average reward-per-cost so far, or ``0.0`` for a never-pulled arm (never consulted
        when `pulls == 0` — :meth:`Bandit.ucb_score` short-circuits to ``+inf`` first)."""
        return self.total_value / self.pulls if self.pulls else 0.0


@dataclass(frozen=True)
class Bandit:
    """A UCB1 bandit over named arms (LLM providers) — see module docstring."""

    stats: dict[str, ArmStats] = field(default_factory=dict)

    def record_outcome(self, provider: str, *, reward: float, cost: float) -> Bandit:
        """A new `Bandit` with `provider`'s statistics updated by one more observation.

        Raises:
            BanditError: `reward` isn't in ``[0, 1]``, or `cost` is negative.
        """
        if not 0.0 <= reward <= 1.0:
            raise BanditError(f"reward must be in [0, 1], got {reward}")
        if cost < 0.0:
            raise BanditError(f"cost must be >= 0, got {cost}")
        current = self.stats.get(provider, ArmStats())
        updated = ArmStats(
            pulls=current.pulls + 1, total_value=current.total_value + _value(reward, cost)
        )
        return Bandit(stats={**self.stats, provider: updated})

    def ucb_score(
        self, provider: str, *, total_pulls: int, exploration: float = _DEFAULT_EXPLORATION
    ) -> float:
        """`provider`'s UCB1 score given `total_pulls` (across every arm, not just this one —
        the standard UCB1 "elapsed rounds" term). ``+inf`` if `provider` has never been pulled.
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

        Raises:
            BanditError: `providers` is empty.
        """
        if not providers:
            raise BanditError("select() needs at least one candidate provider")
        total_pulls = sum(arm.pulls for arm in self.stats.values())
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
            BanditError: `raw` isn't valid JSON, or isn't shaped like a `Bandit` record.
        """
        try:
            data = json.loads(raw)
            return Bandit(
                stats={
                    provider: ArmStats(pulls=arm["pulls"], total_value=arm["total_value"])
                    for provider, arm in data.items()
                }
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise BanditError(f"corrupt bandit record: {exc}") from exc

    def replace_arm(self, provider: str, stats: ArmStats) -> Bandit:
        """A new `Bandit` with `provider`'s stats set directly — mainly for tests that need to
        seed a specific pull count/value without replaying :meth:`record_outcome` calls."""
        return replace(self, stats={**self.stats, provider: stats})
