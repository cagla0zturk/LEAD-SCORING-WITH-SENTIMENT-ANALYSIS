"""Combine the two signals into a single, explainable priority score.

Why a transparent weighted blend (and not a meta-model)?
--------------------------------------------------------
A sales rep has to *trust and act on* this number every morning. A weighted blend is:

* **Explainable** - "high because conversion prob is 0.82 and the last email was
  positive" beats "the meta-model said so".
* **Robust with synthetic text** - a stacked meta-model trained on synthetic sentiment +
  real conversion labels would mostly learn artefacts of our note generator and risk
  leaking the synthetic correlation into the conversion signal.
* **Tunable by the business** - product can move ``conversion_weight`` without retraining.

The score is:

    priority = w * P(convert) + (1 - w) * sentiment_score      (both in [0, 1])

with a hard **reachability gate**: if the lead opted out of all contact, the priority is
damped, because a high-probability lead you are not allowed to call is not actionable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lead_priority.config import (
    COOLING_MIN_CONVERSION_PROB,
    COOLING_SENTIMENT_LABELS,
    DEFAULT_CONVERSION_WEIGHT,
)

# Priority tiers for the dashboard ("call today" vs "let it cool").
_TIER_BOUNDS: tuple[tuple[float, str], ...] = (
    (0.65, "hot"),
    (0.45, "warm"),
    (0.25, "cooling"),
    (0.0, "cold"),
)


@dataclass(frozen=True)
class PriorityBreakdown:
    """Explainable decomposition of a priority score."""

    priority_score: float
    tier: str
    conversion_probability: float
    sentiment_score: float
    conversion_weight: float
    reachable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "priority_score": self.priority_score,
            "tier": self.tier,
            "conversion_probability": self.conversion_probability,
            "sentiment_score": self.sentiment_score,
            "conversion_weight": self.conversion_weight,
            "reachable": self.reachable,
        }


def priority_tier(score: float) -> str:
    """Map a [0, 1] priority score to a human-readable tier."""
    for bound, name in _TIER_BOUNDS:
        if score >= bound:
            return name
    return "cold"


def is_cooling(conversion_probability: float, sentiment_label: str) -> bool:
    """Flag an *at-risk* ("soğuyan") lead: was promising, now disengaged/objecting.

    This is the "bu üçü soğuyor" half of the dashboard. We deliberately key off a
    high conversion probability (the lead has real value) combined with a negative
    recent-interaction signal (it is slipping away), so reps rescue deals worth saving
    rather than chasing leads that were never hot.
    """
    return (
        conversion_probability >= COOLING_MIN_CONVERSION_PROB
        and sentiment_label in COOLING_SENTIMENT_LABELS
    )


def combine_priority(
    conversion_probability: float,
    sentiment_score: float,
    *,
    conversion_weight: float = DEFAULT_CONVERSION_WEIGHT,
    reachable: bool = True,
    unreachable_damping: float = 0.5,
) -> PriorityBreakdown:
    """Blend the conversion probability and sentiment score into a priority score.

    Parameters
    ----------
    conversion_probability:
        Calibrated P(convert) in [0, 1] from the lead-scoring model.
    sentiment_score:
        Scalar engagement score in [0, 1] from the sentiment model.
    conversion_weight:
        Weight ``w`` on conversion probability; ``1 - w`` goes to sentiment.
    reachable:
        Whether the lead can still be contacted (not opted out of email *and* phone).
    unreachable_damping:
        Multiplier applied when ``reachable`` is False.
    """
    w = min(max(conversion_weight, 0.0), 1.0)
    p = min(max(conversion_probability, 0.0), 1.0)
    s = min(max(sentiment_score, 0.0), 1.0)

    score = w * p + (1.0 - w) * s
    if not reachable:
        score *= unreachable_damping

    score = round(float(score), 4)
    return PriorityBreakdown(
        priority_score=score,
        tier=priority_tier(score),
        conversion_probability=round(p, 4),
        sentiment_score=round(s, 4),
        conversion_weight=w,
        reachable=reachable,
    )
