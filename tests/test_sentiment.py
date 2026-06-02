"""Tests for the synthetic interaction generator and the sentiment classifier."""

from __future__ import annotations

from lead_priority.config import SENTIMENT_LABELS
from lead_priority.data.synthetic_interactions import generate_interaction_dataset
from lead_priority.sentiment.train import build_sentiment_pipeline


def test_generator_is_balanced_and_bilingual():
    ds = generate_interaction_dataset(n_per_class=50)
    counts = ds.frame["label"].value_counts()
    assert set(counts.index) == set(SENTIMENT_LABELS)
    assert counts.min() == counts.max() == 50
    languages = set(ds.frame["language"].unique())
    assert {"tr", "en"}.issubset(languages)


def test_classifier_learns_clear_intents():
    ds = generate_interaction_dataset(n_per_class=300)
    pipeline = build_sentiment_pipeline()
    pipeline.fit(ds.texts, ds.labels)

    cases = {
        "Müşteri çok ilgili, demo talep etti ve hemen başlamak istiyor.": "positive_engagement",
        "Really excited, wants to enroll today.": "positive_engagement",
        "Fiyatı çok yüksek buldu, indirim istedi.": "objection",
        "Telefonu açmadı, iki haftadır sessiz ve ilgisiz.": "disengaged",
        "No response to the last three emails, gone cold.": "disengaged",
    }
    for text, expected in cases.items():
        assert pipeline.predict([text])[0] == expected
