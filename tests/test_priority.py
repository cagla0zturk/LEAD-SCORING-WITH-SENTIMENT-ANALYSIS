"""Tests for the combined priority score logic."""

from __future__ import annotations

from lead_priority.priority.combine import combine_priority, is_cooling, priority_tier


def test_weighted_blend_math():
    result = combine_priority(0.8, 0.5, conversion_weight=0.7)
    assert abs(result.priority_score - (0.7 * 0.8 + 0.3 * 0.5)) < 1e-6


def test_unreachable_is_damped():
    reachable = combine_priority(0.9, 0.9, reachable=True)
    unreachable = combine_priority(0.9, 0.9, reachable=False)
    assert unreachable.priority_score < reachable.priority_score


def test_weight_extremes():
    only_conversion = combine_priority(0.9, 0.1, conversion_weight=1.0)
    only_sentiment = combine_priority(0.9, 0.1, conversion_weight=0.0)
    assert abs(only_conversion.priority_score - 0.9) < 1e-6
    assert abs(only_sentiment.priority_score - 0.1) < 1e-6


def test_inputs_are_clamped():
    result = combine_priority(1.5, -0.2, conversion_weight=0.5)
    assert 0.0 <= result.priority_score <= 1.0


def test_tier_boundaries():
    assert priority_tier(0.9) == "hot"
    assert priority_tier(0.5) == "warm"
    assert priority_tier(0.3) == "cooling"
    assert priority_tier(0.1) == "cold"


def test_cooling_flag():
    # Promising lead that just went disengaged -> at risk.
    assert is_cooling(0.8, "disengaged") is True
    assert is_cooling(0.7, "objection") is True
    # Low-value lead going quiet is not "cooling" (it was never hot).
    assert is_cooling(0.2, "disengaged") is False
    # Promising + positive is not cooling.
    assert is_cooling(0.9, "positive_engagement") is False
