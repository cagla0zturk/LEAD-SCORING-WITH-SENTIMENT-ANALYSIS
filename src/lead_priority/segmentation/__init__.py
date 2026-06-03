"""Lead segmentation: action buckets, personas, approach insights and playbooks.

This layer sits on top of the scoring + sentiment models and turns two raw numbers
(conversion probability, sentiment) into something a sales rep can *act on*:

* an **action bucket** — call today / rescue by email / nurture / lost,
* a **lead category (persona)** — ready-to-buy, price-objection, going-cold, ...,
* **approach insights** — what interests this lead, what to avoid, which channel,
* a **playbook** — a ready email draft + call talk-track tailored to the persona,
* an optional **behavioral segment** from KMeans clustering.
"""

from __future__ import annotations

from lead_priority.segmentation.rules import (
    ACTION_BUCKETS,
    LEAD_CATEGORIES,
    ApproachInsights,
    action_bucket,
    approach_insights,
    lead_category,
    objection_reason,
)
from lead_priority.segmentation.playbooks import build_playbook, Playbook

__all__ = [
    "ACTION_BUCKETS",
    "LEAD_CATEGORIES",
    "ApproachInsights",
    "action_bucket",
    "approach_insights",
    "lead_category",
    "objection_reason",
    "build_playbook",
    "Playbook",
]
