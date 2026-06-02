"""Serving-time wrapper around the fine-tuned multilingual transformer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lead_priority.config import (
    SENTIMENT_LABELS,
    SENTIMENT_MAX_LENGTH,
    SENTIMENT_MODEL_DIR,
    SENTIMENT_PRIORITY_WEIGHT,
)

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
    """Predict engagement intent + a scalar ``sentiment_score`` with a fine-tuned transformer."""

    def __init__(self, model: Any, tokenizer: Any, labels: list[str], max_length: int) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length
        self._model.eval()

    @classmethod
    def load(
        cls, path: Path = SENTIMENT_MODEL_DIR, *, max_length: int = SENTIMENT_MAX_LENGTH
    ) -> "SentimentClassifier":
        if not path.exists():
            raise FileNotFoundError(
                f"Sentiment model not found at {path}. "
                "Train it first with `python -m scripts.train_all`."
            )
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        id2label = model.config.id2label
        labels = [id2label[i] for i in range(len(id2label))]
        logger.info("Loaded fine-tuned sentiment model from %s", path)
        return cls(model=model, tokenizer=tokenizer, labels=labels, max_length=max_length)

    def _sentiment_score(self, scores: dict[str, float]) -> float:
        """Expected priority contribution = sum_c P(class=c) * weight(c)."""
        return float(sum(scores.get(c, 0.0) * w for c, w in SENTIMENT_PRIORITY_WEIGHT.items()))

    def _neutral(self) -> SentimentPrediction:
        scores = {c: 1.0 / len(self.labels) for c in self.labels}
        return SentimentPrediction("neutral", scores, self._sentiment_score(scores))

    @torch.no_grad()
    def _forward(self, texts: list[str]) -> list[dict[str, float]]:
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        probs = F.softmax(self._model(**enc).logits, dim=-1).cpu().numpy()
        return [{self.labels[i]: float(p) for i, p in enumerate(row)} for row in probs]

    def predict(self, text: str) -> SentimentPrediction:
        text = (text or "").strip()
        if not text:
            return self._neutral()
        scores = self._forward([text])[0]
        label = max(scores, key=scores.get)
        return SentimentPrediction(label, scores, self._sentiment_score(scores))

    def predict_batch(self, texts: list[str | None]) -> list[SentimentPrediction]:
        """Vectorised prediction; empty/None texts map to a neutral prior.

        Used by the dashboard so scoring N leads is a single transformer forward pass.
        """
        cleaned = [(t or "").strip() for t in texts]
        non_empty_idx = [i for i, t in enumerate(cleaned) if t]
        results: list[SentimentPrediction | None] = [None] * len(cleaned)
        if non_empty_idx:
            forwarded = self._forward([cleaned[i] for i in non_empty_idx])
            for i, scores in zip(non_empty_idx, forwarded):
                label = max(scores, key=scores.get)
                results[i] = SentimentPrediction(label, scores, self._sentiment_score(scores))
        return [r if r is not None else self._neutral() for r in results]
