"""Tests for the synthetic interaction generator and the fine-tuned sentiment model."""

from __future__ import annotations

import pytest

from lead_priority.config import SENTIMENT_LABELS
from lead_priority.data.synthetic_interactions import generate_interaction_dataset


def test_generator_is_balanced_and_bilingual():
    ds = generate_interaction_dataset(n_per_class=50)
    counts = ds.frame["label"].value_counts()
    assert set(counts.index) == set(SENTIMENT_LABELS)
    assert counts.min() == counts.max() == 50
    languages = set(ds.frame["language"].unique())
    assert {"tr", "en"}.issubset(languages)


@pytest.fixture(scope="module")
def classifier(trained_artifacts):
    from lead_priority.sentiment.model import SentimentClassifier

    return SentimentClassifier.load()


def test_prediction_structure_is_valid(classifier):
    pred = classifier.predict("Müşteri çok ilgili, demo talep etti.")
    assert pred.label in set(SENTIMENT_LABELS)
    assert set(pred.scores) == set(SENTIMENT_LABELS)
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-4  # softmax sums to 1
    assert 0.0 <= pred.sentiment_score <= 1.0


def test_empty_text_is_neutral(classifier):
    pred = classifier.predict("   ")
    assert pred.label == "neutral"


def test_classifier_learns_clear_intents(classifier):
    cases = {
        "Müşteri çok ilgili, demo talep etti ve hemen başlamak istiyor.": "positive_engagement",
        "Really excited, wants to enroll today.": "positive_engagement",
        "Telefonu açmadı, iki haftadır sessiz ve ilgisiz.": "disengaged",
        "No response to the last three emails, gone cold.": "disengaged",
    }
    for text, expected in cases.items():
        assert classifier.predict(text).label == expected


def test_batch_matches_single(classifier):
    texts = ["Fiyatı çok yüksek buldu, indirim istedi.", "", "Broşürü gönderdim."]
    batch = classifier.predict_batch(texts)
    assert len(batch) == 3
    assert batch[1].label == "neutral"  # empty -> neutral
    assert batch[0].label == classifier.predict(texts[0]).label