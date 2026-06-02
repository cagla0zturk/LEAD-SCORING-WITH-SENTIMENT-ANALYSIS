"""Orchestration layer that ties the three models together for serving.

The :class:`PriorityService` is intentionally framework-agnostic (no FastAPI imports)
so it can be unit-tested directly and reused from a batch job or notebook.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from lead_priority.config import DEFAULT_CONVERSION_WEIGHT, DEMO_LEADS_JSON
from lead_priority.priority.combine import combine_priority
from lead_priority.scoring.model import LeadScorer
from lead_priority.sentiment.model import SentimentClassifier

logger = logging.getLogger(__name__)


def _is_reachable(features: dict[str, Any]) -> bool:
    """A lead is reachable unless it opted out of *both* email and phone."""
    dne = str(features.get("Do Not Email", "No")).strip().lower() == "yes"
    dnc = str(features.get("Do Not Call", "No")).strip().lower() == "yes"
    return not (dne and dnc)


class PriorityService:
    """Loads the models + demo leads and produces scored, prioritised output."""

    def __init__(
        self,
        scorer: LeadScorer,
        sentiment: SentimentClassifier,
        demo_leads: list[dict[str, Any]],
    ) -> None:
        self.scorer = scorer
        self.sentiment = sentiment
        self.demo_leads = demo_leads

    @classmethod
    def load(cls, *, demo_leads_path: Path = DEMO_LEADS_JSON) -> "PriorityService":
        scorer = LeadScorer.load()
        sentiment = SentimentClassifier.load()
        demo_leads = _load_demo_leads(demo_leads_path)
        logger.info("PriorityService ready (%d demo leads)", len(demo_leads))
        return cls(scorer=scorer, sentiment=sentiment, demo_leads=demo_leads)

    # -- core scoring ------------------------------------------------------------------
    def score_lead(
        self,
        features: dict[str, Any],
        interaction_text: str | None = None,
        *,
        conversion_weight: float | None = None,
    ) -> dict[str, Any]:
        """Score a single lead end-to-end (conversion + sentiment + priority)."""
        weight = DEFAULT_CONVERSION_WEIGHT if conversion_weight is None else conversion_weight

        conversion_probability = self.scorer.predict_proba(features)[0]
        conversion_prediction = int(conversion_probability >= self.scorer.threshold)

        sentiment_pred = self.sentiment.predict(interaction_text or "")

        priority = combine_priority(
            conversion_probability=conversion_probability,
            sentiment_score=sentiment_pred.sentiment_score,
            conversion_weight=weight,
            reachable=_is_reachable(features),
        )

        return {
            "conversion_probability": round(conversion_probability, 4),
            "conversion_prediction": conversion_prediction,
            "sentiment": sentiment_pred.as_dict(),
            "priority": priority.as_dict(),
        }

    # -- dashboard ---------------------------------------------------------------------
    def top_leads(
        self, n: int = 5, *, conversion_weight: float | None = None
    ) -> list[dict[str, Any]]:
        """Score the demo lead list and return the top ``n`` by priority score."""
        scored: list[dict[str, Any]] = []
        for lead in self.demo_leads:
            features = lead.get("features", {})
            text = lead.get("interaction_text")
            result = self.score_lead(features, text, conversion_weight=conversion_weight)
            scored.append(
                {
                    "lead_id": str(lead.get("lead_id", "unknown")),
                    "conversion_probability": result["conversion_probability"],
                    "sentiment_label": result["sentiment"]["label"],
                    "priority_score": result["priority"]["priority_score"],
                    "tier": result["priority"]["tier"],
                    "last_interaction": text,
                }
            )
        scored.sort(key=lambda r: r["priority_score"], reverse=True)
        return scored[: max(n, 0)]


def _load_demo_leads(path: Path) -> list[dict[str, Any]]:
    """Load the demo lead list, falling back to a tiny in-memory sample if absent."""
    if path.exists():
        return json.loads(path.read_text())
    logger.warning("Demo leads file %s missing; using a small built-in fallback.", path)
    return _fallback_demo_leads()


def _fallback_demo_leads() -> list[dict[str, Any]]:
    return [
        {
            "lead_id": "demo-1",
            "features": {
                "Lead Origin": "Landing Page Submission",
                "Lead Source": "Google",
                "TotalVisits": 8,
                "Total Time Spent on Website": 1800,
                "Page Views Per Visit": 4.0,
                "Last Activity": "SMS Sent",
                "What is your current occupation": "Working Professional",
                "Do Not Email": "No",
                "Do Not Call": "No",
            },
            "interaction_text": "Müşteri çok ilgili, demo talep etti, hemen başlamak istiyor.",
        },
        {
            "lead_id": "demo-2",
            "features": {
                "Lead Origin": "API",
                "Lead Source": "Olark Chat",
                "TotalVisits": 1,
                "Total Time Spent on Website": 30,
                "Page Views Per Visit": 1.0,
                "Last Activity": "Olark Chat Conversation",
                "What is your current occupation": "Unemployed",
                "Do Not Email": "No",
                "Do Not Call": "No",
            },
            "interaction_text": "Telefonu açmadı, iki haftadır sessiz, ilgisiz görünüyor.",
        },
        {
            "lead_id": "demo-3",
            "features": {
                "Lead Origin": "Landing Page Submission",
                "Lead Source": "Reference",
                "TotalVisits": 4,
                "Total Time Spent on Website": 900,
                "Page Views Per Visit": 3.0,
                "Last Activity": "Email Opened",
                "What is your current occupation": "Working Professional",
                "Do Not Email": "No",
                "Do Not Call": "No",
            },
            "interaction_text": "Fiyatı yüksek buldu ama ilgili, indirim olup olmadığını sordu.",
        },
    ]
