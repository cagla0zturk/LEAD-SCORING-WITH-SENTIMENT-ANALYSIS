"""Train the engagement intent/sentiment classifier.

Model choice (and why it's deliberately lightweight)
----------------------------------------------------
We use **TF-IDF (word 1-2 grams + character 3-5 grams) -> Logistic Regression**.

* Character n-grams make it robust to the **mixed Turkish/English** notes and to the
  morphological richness of Turkish (suffixes) without language detection or tokenisers.
* It is CPU-only, trains in seconds, is deterministic, and ships in a ~MB artifact -
  appropriate given there is **no GPU** and the data is synthetic anyway.
* A multilingual transformer (XLM-R / DistilBERT) is the natural "modern" upgrade and is
  discussed in the README under future work; it is overkill for templated synthetic text
  and would mostly memorise the templates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from lead_priority.config import (
    MODELS_DIR,
    RANDOM_SEED,
    SENTIMENT_LABELS,
    SENTIMENT_METRICS_PATH,
    SENTIMENT_MODEL_PATH,
    TEST_SIZE,
)
from lead_priority.data.synthetic_interactions import generate_interaction_dataset

logger = logging.getLogger(__name__)


@dataclass
class SentimentTrainResult:
    accuracy: float
    macro_f1: float
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_path: Path | None = None


def build_sentiment_pipeline() -> Pipeline:
    """TF-IDF (word + char) features feeding a multinomial Logistic Regression."""
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        lowercase=True,
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        lowercase=True,
        sublinear_tf=True,
    )
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])
    clf = LogisticRegression(
        max_iter=2000,
        C=4.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    return Pipeline([("features", features), ("classifier", clf)])


def train_sentiment_model(
    *,
    n_per_class: int = 600,
    models_dir: Path = MODELS_DIR,
) -> SentimentTrainResult:
    """Generate synthetic notes, train the classifier, evaluate, and persist it."""
    dataset = generate_interaction_dataset(n_per_class=n_per_class)
    x = dataset.texts
    y = dataset.labels

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    pipeline = build_sentiment_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    report = classification_report(
        y_test, y_pred, labels=list(SENTIMENT_LABELS), output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(SENTIMENT_LABELS))

    accuracy = float(report["accuracy"])
    macro_f1 = float(report["macro avg"]["f1-score"])
    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": list(SENTIMENT_LABELS),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "language_distribution": dataset.frame["language"].value_counts().to_dict(),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": pipeline,
        "labels": list(SENTIMENT_LABELS),
        "metadata": {"model_type": "tfidf_word_char_logreg", "random_seed": RANDOM_SEED},
    }
    joblib.dump(artifact, SENTIMENT_MODEL_PATH)
    SENTIMENT_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info(
        "Saved sentiment model to %s (acc=%.3f macro_f1=%.3f)",
        SENTIMENT_MODEL_PATH,
        accuracy,
        macro_f1,
    )

    return SentimentTrainResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        metrics=metrics,
        artifact_path=SENTIMENT_MODEL_PATH,
    )
