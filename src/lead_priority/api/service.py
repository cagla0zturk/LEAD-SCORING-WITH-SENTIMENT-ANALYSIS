"""Orchestration layer that ties scoring + sentiment + segmentation together.

The :class:`PriorityService` is framework-agnostic (no FastAPI imports) so it can be
unit-tested directly and reused from a batch job or notebook.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from lead_priority.config import DEFAULT_CONVERSION_WEIGHT, DEMO_LEADS_JSON
from lead_priority.priority.combine import combine_priority, is_cooling
from lead_priority.scoring.explain import ScoreExplainer
from lead_priority.scoring.model import LeadScorer
from lead_priority.segmentation import (
    ACTION_BUCKETS,
    LEAD_CATEGORIES,
    action_bucket,
    approach_insights,
    build_playbook,
    lead_category,
    objection_reason,
)
from lead_priority.segmentation.cluster import BehavioralSegmenter
from lead_priority.sentiment.model import SentimentClassifier

logger = logging.getLogger(__name__)


def _is_reachable(features: dict[str, Any]) -> bool:
    """A lead is reachable unless it opted out of *both* email and phone."""
    dne = str(features.get("Do Not Email", "No")).strip().lower() == "yes"
    dnc = str(features.get("Do Not Call", "No")).strip().lower() == "yes"
    return not (dne and dnc)


def _lead_context(features: dict[str, Any]) -> str:
    """Short human-readable context line from real lead fields (no fake names)."""
    parts: list[str] = []
    for key in ("What is your current occupation", "Specialization", "City", "Lead Source"):
        val = features.get(key)
        if val is not None and str(val).strip() and str(val).strip().lower() != "nan":
            parts.append(str(val).strip())
    return " · ".join(parts)


class PriorityService:
    """Loads the models + demo leads and produces scored, segmented output."""

    def __init__(
        self,
        scorer: LeadScorer,
        sentiment: SentimentClassifier,
        demo_leads: list[dict[str, Any]],
        segmenter: BehavioralSegmenter | None = None,
        explainer: ScoreExplainer | None = None,
    ) -> None:
        self.scorer = scorer
        self.sentiment = sentiment
        self.demo_leads = demo_leads
        self.segmenter = segmenter
        self.explainer = explainer

    @classmethod
    def load(cls, *, demo_leads_path: Path = DEMO_LEADS_JSON) -> "PriorityService":
        scorer = LeadScorer.load()
        sentiment = SentimentClassifier.load()
        try:
            segmenter: BehavioralSegmenter | None = BehavioralSegmenter.load()
        except FileNotFoundError:
            segmenter = None
            logger.warning("Behavioral segmentation model missing; segment labels disabled.")
        try:
            explainer: ScoreExplainer | None = ScoreExplainer.load()
        except FileNotFoundError:
            explainer = None
            logger.warning("Score explainer missing; per-lead explanations disabled.")
        demo_leads = _load_demo_leads(demo_leads_path)
        logger.info("PriorityService ready (%d demo leads)", len(demo_leads))
        return cls(
            scorer=scorer,
            sentiment=sentiment,
            demo_leads=demo_leads,
            segmenter=segmenter,
            explainer=explainer,
        )

    # -- enrichment --------------------------------------------------------------------
    def _enrich(
        self,
        features: dict[str, Any],
        interaction_text: str | None,
        conversion_probability: float,
        sentiment_pred: Any,
        weight: float,
    ) -> dict[str, Any]:
        reachable = _is_reachable(features)
        priority = combine_priority(
            conversion_probability=conversion_probability,
            sentiment_score=sentiment_pred.sentiment_score,
            conversion_weight=weight,
            reachable=reachable,
        )
        obj_reason = objection_reason(interaction_text) if sentiment_pred.label == "objection" else "none"
        category = lead_category(
            conversion_probability, sentiment_pred.label, features=features, objection=obj_reason
        )
        bucket = action_bucket(
            conversion_probability, sentiment_pred.label, reachable=reachable, category=category
        )
        insights = approach_insights(features, sentiment_pred.label, objection=obj_reason)
        playbook = build_playbook(category, features, insights)
        segment = self.segmenter.assign_one(features) if self.segmenter else None
        explanation = self.explainer.explain(features) if self.explainer else []

        return {
            "conversion_probability": round(conversion_probability, 4),
            "conversion_prediction": int(conversion_probability >= self.scorer.threshold),
            "sentiment": sentiment_pred.as_dict(),
            "priority": priority.as_dict(),
            "is_cooling": is_cooling(conversion_probability, sentiment_pred.label),
            "explanation": explanation,
            "context": _lead_context(features),
            "segmentation": {
                "action_bucket": bucket,
                "action_bucket_label": ACTION_BUCKETS[bucket],
                "category": category,
                "category_label": LEAD_CATEGORIES.get(category, category),
                "behavioral_segment": segment,
                "objection_reason": obj_reason,
                "insights": insights.as_dict(),
                "playbook": playbook.as_dict(),
            },
        }

    # -- single lead -------------------------------------------------------------------
    def score_lead(
        self,
        features: dict[str, Any],
        interaction_text: str | None = None,
        *,
        conversion_weight: float | None = None,
    ) -> dict[str, Any]:
        """Score + segment a single lead end-to-end."""
        weight = DEFAULT_CONVERSION_WEIGHT if conversion_weight is None else conversion_weight
        p = self.scorer.predict_proba(features)[0]
        sentiment_pred = self.sentiment.predict(interaction_text or "")
        return self._enrich(features, interaction_text, p, sentiment_pred, weight)

    # -- batch / dashboard -------------------------------------------------------------
    def _score_demo_leads(self, *, conversion_weight: float | None = None) -> list[dict[str, Any]]:
        """Batch-score + segment every demo lead (one transformer pass, one model pass)."""
        if not self.demo_leads:
            return []
        weight = DEFAULT_CONVERSION_WEIGHT if conversion_weight is None else conversion_weight
        features_list = [lead.get("features", {}) for lead in self.demo_leads]
        texts = [lead.get("interaction_text") for lead in self.demo_leads]

        conv_probs = self.scorer.predict_proba(features_list)
        sentiments = self.sentiment.predict_batch(texts)

        scored: list[dict[str, Any]] = []
        for lead, feats, prob, sent, text in zip(
            self.demo_leads, features_list, conv_probs, sentiments, texts
        ):
            enriched = self._enrich(feats, text, prob, sent, weight)
            seg = enriched["segmentation"]
            scored.append(
                {
                    "lead_id": str(lead.get("lead_id", "unknown")),
                    "conversion_probability": enriched["conversion_probability"],
                    "sentiment_label": sent.label,
                    "priority_score": enriched["priority"]["priority_score"],
                    "tier": enriched["priority"]["tier"],
                    "reachable": enriched["priority"]["reachable"],
                    "is_cooling": enriched["is_cooling"],
                    "action_bucket": seg["action_bucket"],
                    "action_bucket_label": seg["action_bucket_label"],
                    "category": seg["category"],
                    "category_label": seg["category_label"],
                    "behavioral_segment": seg["behavioral_segment"],
                    "recommended_channel": seg["insights"]["recommended_channel"],
                    "insights": seg["insights"],
                    "playbook": seg["playbook"],
                    "explanation": enriched["explanation"],
                    "context": enriched["context"],
                    "last_interaction": text,
                }
            )
        return scored

    def top_leads(
        self, n: int = 5, *, conversion_weight: float | None = None
    ) -> list[dict[str, Any]]:
        scored = self._score_demo_leads(conversion_weight=conversion_weight)
        scored.sort(key=lambda r: r["priority_score"], reverse=True)
        return scored[: max(n, 0)]

    def morning_brief(
        self, *, n_call: int = 5, n_cooling: int = 5, conversion_weight: float | None = None
    ) -> dict[str, Any]:
        scored = self._score_demo_leads(conversion_weight=conversion_weight)
        call_today = sorted(
            (r for r in scored if r["reachable"]),
            key=lambda r: r["priority_score"],
            reverse=True,
        )[: max(n_call, 0)]
        cooling = sorted(
            (r for r in scored if r["is_cooling"]),
            key=lambda r: r["conversion_probability"],
            reverse=True,
        )[: max(n_cooling, 0)]
        return {"call_today": call_today, "cooling": cooling, "total_leads": len(scored)}

    def dashboard(
        self, *, conversion_weight: float | None = None, per_bucket: int | None = 5
    ) -> dict[str, Any]:
        """Segmented dashboard: leads grouped into the four action buckets.

        ``per_bucket`` caps how many leads are returned per bucket (default 5, for a focused
        view); ``counts`` always reflects the *full* bucket size so the UI can show "+N daha".
        Pass ``per_bucket=None`` to return every lead (used by the JSON endpoint).
        """
        scored = self._score_demo_leads(conversion_weight=conversion_weight)
        # Sort within each bucket by priority (call/nurture) or value-at-risk (rescue/lost).
        buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in ACTION_BUCKETS}
        for lead in scored:
            buckets[lead["action_bucket"]].append(lead)
        for key in ("call_today", "nurture"):
            buckets[key].sort(key=lambda r: r["priority_score"], reverse=True)
        for key in ("rescue_email", "lost"):
            buckets[key].sort(key=lambda r: r["conversion_probability"], reverse=True)

        counts = {k: len(v) for k, v in buckets.items()}  # full sizes (before truncation)
        if per_bucket is not None:
            buckets = {k: v[: max(per_bucket, 0)] for k, v in buckets.items()}

        category_counts: dict[str, int] = {}
        segment_counts: dict[str, int] = {}
        for lead in scored:
            category_counts[lead["category_label"]] = category_counts.get(lead["category_label"], 0) + 1
            if lead["behavioral_segment"]:
                segment_counts[lead["behavioral_segment"]] = (
                    segment_counts.get(lead["behavioral_segment"], 0) + 1
                )

        return {
            "total_leads": len(scored),
            "bucket_labels": ACTION_BUCKETS,
            "buckets": buckets,
            "counts": counts,
            "shown": {k: len(v) for k, v in buckets.items()},
            "per_bucket": per_bucket,
            "category_counts": category_counts,
            "segment_counts": segment_counts,
        }


def _load_demo_leads(path: Path) -> list[dict[str, Any]]:
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
