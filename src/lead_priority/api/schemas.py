"""Pydantic request/response schemas for the API.

Lead features are accepted as a free-form mapping because the underlying X Education
columns contain spaces (e.g. ``"Total Time Spent on Website"``) which are not valid
Python identifiers. The mapping is validated downstream by the model pipeline, which
tolerates missing fields.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    """Input for ``POST /score``."""

    features: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw lead features keyed by the original dataset column names.",
        examples=[
            {
                "Lead Origin": "Landing Page Submission",
                "Lead Source": "Google",
                "TotalVisits": 5,
                "Total Time Spent on Website": 1200,
                "Page Views Per Visit": 3.0,
                "Last Activity": "SMS Sent",
                "What is your current occupation": "Working Professional",
                "Do Not Email": "No",
                "Do Not Call": "No",
            }
        ],
    )
    interaction_text: str | None = Field(
        default=None,
        description="Most recent free-text interaction (email/note), TR or EN.",
        examples=["Müşteri çok ilgili, demo talep etti ve fiyatları sordu."],
    )
    conversion_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional override for the conversion-vs-sentiment blend weight.",
    )


class SentimentResult(BaseModel):
    label: str
    scores: dict[str, float]
    sentiment_score: float


class PriorityResult(BaseModel):
    priority_score: float
    tier: str
    conversion_probability: float
    sentiment_score: float
    conversion_weight: float
    reachable: bool


class ScoreResponse(BaseModel):
    """Output for ``POST /score``."""

    conversion_probability: float
    conversion_prediction: int
    sentiment: SentimentResult
    priority: PriorityResult
    is_cooling: bool
    segmentation: dict[str, Any]


class TopLead(BaseModel):
    lead_id: str
    conversion_probability: float
    sentiment_label: str
    priority_score: float
    tier: str
    reachable: bool = True
    is_cooling: bool = False
    action_bucket: str | None = None
    action_bucket_label: str | None = None
    category: str | None = None
    category_label: str | None = None
    behavioral_segment: str | None = None
    recommended_channel: str | None = None
    last_interaction: str | None = None


class TopLeadsResponse(BaseModel):
    count: int
    leads: list[TopLead]


class MorningBriefResponse(BaseModel):
    """Output for ``GET /dashboard/brief`` — the sales-rep morning view."""

    total_leads: int
    call_today: list[TopLead]
    cooling: list[TopLead]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    n_demo_leads: int
