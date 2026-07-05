"""Tests for the cost-aware LLM provider bandit (T2.6) — offline & deterministic.

Per the DEVPLAN todo: synthetic reward/cost history → bandit prefers the best reward-per-cost
provider; an unseen provider still gets explored.
"""

import pytest

from src import llm_bandit
from src.llm_bandit import Bandit

pytestmark = pytest.mark.m2


# --------------------------------------------------------------------- ArmStats / record_outcome


def test_arm_stats_mean_value_zero_when_never_pulled() -> None:
    assert llm_bandit.ArmStats().mean_value == 0.0


def test_record_outcome_accumulates() -> None:
    bandit = Bandit()
    bandit = bandit.record_outcome("claude_api", reward=1.0, cost=1.0)
    bandit = bandit.record_outcome("claude_api", reward=0.0, cost=1.0)

    stats = bandit.stats["claude_api"]
    assert stats.pulls == 2
    assert stats.mean_value == pytest.approx(0.5)  # (1.0 + 0.0) / 2


def test_record_outcome_rejects_reward_out_of_range() -> None:
    with pytest.raises(llm_bandit.BanditError, match=r"reward must be in \[0, 1\]"):
        Bandit().record_outcome("claude_api", reward=1.5, cost=1.0)


def test_record_outcome_rejects_negative_cost() -> None:
    with pytest.raises(llm_bandit.BanditError, match="cost must be >= 0"):
        Bandit().record_outcome("claude_api", reward=1.0, cost=-1.0)


def test_record_outcome_is_immutable() -> None:
    """record_outcome returns a NEW Bandit -- the original is untouched."""
    original = Bandit()
    updated = original.record_outcome("claude_api", reward=1.0, cost=1.0)

    assert original.stats == {}
    assert updated.stats["claude_api"].pulls == 1


def test_value_floors_zero_cost_instead_of_dividing_by_zero() -> None:
    """A free/local provider's cost_usd is 0.0 -- must not produce inf/nan."""
    bandit = Bandit().record_outcome("local", reward=1.0, cost=0.0)
    value = bandit.stats["local"].mean_value
    assert value > 0
    assert value == value  # not NaN
    assert value != float("inf")


# --------------------------------------------------------------------- ucb_score / select


def test_ucb_score_never_pulled_is_infinite() -> None:
    bandit = Bandit()
    assert bandit.ucb_score("claude_api", total_pulls=10) == float("inf")


def test_select_raises_on_empty_providers() -> None:
    with pytest.raises(llm_bandit.BanditError, match="at least one candidate"):
        Bandit().select([])


def test_select_prefers_unseen_provider_over_a_seen_one() -> None:
    """DEVPLAN's named scenario: an unseen provider still gets explored."""
    bandit = Bandit()
    for _ in range(20):
        bandit = bandit.record_outcome("claude_api", reward=1.0, cost=1.0)  # strong track record

    assert bandit.select(["claude_api", "local"]) == "local"  # never tried -> explored first


def test_select_prefers_best_reward_per_cost_once_all_explored() -> None:
    """DEVPLAN's named scenario: the bandit prefers the best reward-per-cost provider -- once
    the "unseen" exploration bonus stops dominating (many pulls on both arms), the one with
    genuinely higher reward-per-cost wins."""
    bandit = Bandit()
    for _ in range(200):
        bandit = bandit.record_outcome("claude_api", reward=1.0, cost=10.0)  # succeeds, pricey
        bandit = bandit.record_outcome("local", reward=1.0, cost=0.01)  # succeeds, cheap

    assert bandit.select(["claude_api", "local"]) == "local"


def test_select_prefers_higher_success_rate_at_equal_cost() -> None:
    bandit = Bandit()
    for _ in range(200):
        bandit = bandit.record_outcome("reliable", reward=1.0, cost=1.0)
        bandit = bandit.record_outcome("flaky", reward=0.1, cost=1.0)

    assert bandit.select(["reliable", "flaky"]) == "reliable"


def test_ucb_score_increases_with_more_total_pulls_for_same_arm() -> None:
    """The exploration bonus grows with elapsed rounds -- an arm not pulled recently becomes
    relatively more attractive as other arms accumulate pulls."""
    bandit = Bandit().record_outcome("claude_api", reward=0.5, cost=1.0)
    low = bandit.ucb_score("claude_api", total_pulls=1)
    high = bandit.ucb_score("claude_api", total_pulls=1000)
    assert high > low


# --------------------------------------------------------------------- to_json / from_json


def test_bandit_json_roundtrip() -> None:
    bandit = Bandit().record_outcome("claude_api", reward=1.0, cost=2.0)
    bandit = bandit.record_outcome("local", reward=0.5, cost=0.0)

    restored = Bandit.from_json(bandit.to_json())

    assert restored.stats.keys() == bandit.stats.keys()
    for provider in bandit.stats:
        assert restored.stats[provider] == bandit.stats[provider]


def test_bandit_from_json_malformed_raises() -> None:
    with pytest.raises(llm_bandit.BanditError, match="corrupt bandit record"):
        Bandit.from_json("not valid json")


def test_bandit_from_json_missing_field_raises() -> None:
    with pytest.raises(llm_bandit.BanditError, match="corrupt bandit record"):
        Bandit.from_json('{"claude_api": {"pulls": 1}}')  # missing total_value


def test_bandit_empty_json_roundtrip() -> None:
    assert Bandit.from_json(Bandit().to_json()).stats == {}
