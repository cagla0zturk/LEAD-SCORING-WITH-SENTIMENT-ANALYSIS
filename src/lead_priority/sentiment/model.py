"""Serving-time wrapper around the persisted sentiment/intent classifier."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from lead_priority.config import SENTIMENT_MODEL_PATH, SENTIMENT_PRIORITY_WEIGHT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentimentPrediction:
    """Structured output of the sentiment classifier for a single text."""

    label: str
    scores: dict[str, float]
    sentiment_score: float  # in [0, 1]; higher = more "call this lead today"

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "scores": self.scores,
            "sentiment_score": self.sentiment_score,
        }


class SentimentClassifier:
    """Predict engagement intent and a scalar ``sentiment_score`` from free text."""

    def __init__(self, model: Any, labels: list[str]) -> None:
        self._model = model
        self.labels = labels

    @classmethod
    def load(cls, path: Path = SENTIMENT_MODEL_PATH) -> "SentimentClassifier":
        if not path.exists():
            raise FileNotFoundError(
                f"Sentiment model not found at {path}. "
                "Train it first with `python -m scripts.train_all`."
            )
        artifact = joblib.load(path)
        logger.info("Loaded sentiment model from %s", path)
        return cls(model=artifact["model"], labels=list(artifact["labels"]))

    def _sentiment_score(self, scores: dict[str, float]) -> float:
        """Expected priority contribution = sum_c P(class=c) * weight(c)."""
        return float(sum(scores.get(c, 0.0) * w for c, w in SENTIMENT_PRIORITY_WEIGHT.items()))

    def predict(self, text: str) -> SentimentPrediction:
        text = (text or "").strip()
        if not text:
            # No interaction yet -> neutral with an even-ish prior.
            scores = {c: 1.0 / len(self.labels) for c in self.labels}
            return SentimentPrediction(
                label="neutral",
                scores=scores,
                sentiment_score=self._sentiment_score(scores),
            )
        proba = self._model.predict_proba([text])[0]
        classes = list(self._model.classes_)
        scores = {cls: float(p) for cls, p in zip(classes, proba)}
        label = max(scores, key=scores.get)
        return SentimentPrediction(
            label=label,
            scores=scores,
            sentiment_score=self._sentiment_score(scores),
        )
