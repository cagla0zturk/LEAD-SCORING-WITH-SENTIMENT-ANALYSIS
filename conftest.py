"""Pytest bootstrap: make ``src/`` importable and provide trained-model fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory):
    """Ensure lead-scoring + sentiment models and demo leads exist (quick training).

    Trains once per test session only if the artifacts are not already on disk, so
    repeated local runs stay fast.
    """
    import json
    import random

    import pandas as pd

    from lead_priority.config import (
        DEMO_LEADS_JSON,
        SCORING_MODEL_PATH,
        SEGMENTATION_MODEL_PATH,
        SENTIMENT_LABELS,
        SENTIMENT_MODEL_DIR,
        PROCESSED_DATA_DIR,
    )
    from lead_priority.data.loaders import load_raw_leads
    from lead_priority.data.synthetic_interactions import sample_note
    from lead_priority.features.engineering import model_input_columns
    from lead_priority.scoring.train import train_lead_scoring_model
    from lead_priority.segmentation.cluster import train_segmentation_model
    from lead_priority.sentiment.train import train_sentiment_model

    if not SENTIMENT_MODEL_DIR.exists():
        # Small but real fine-tune so the transformer is usable in tests yet fast.
        train_sentiment_model(n_per_class=150, epochs=2)
    if not SCORING_MODEL_PATH.exists():
        train_lead_scoring_model(quick=True, make_plots=False)
    if not SEGMENTATION_MODEL_PATH.exists():
        train_segmentation_model(df=load_raw_leads())

    if not DEMO_LEADS_JSON.exists():
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df = load_raw_leads()
        cols = [c for c in model_input_columns() if c in df.columns]
        sample = df.sample(n=12, random_state=0).reset_index(drop=True)
        rng = random.Random(0)
        leads = []
        for i, row in sample.iterrows():
            features = {c: (None if pd.isna(row[c]) else _coerce(row[c])) for c in cols}
            note = sample_note(rng.choice(SENTIMENT_LABELS), seed=rng.randint(0, 10**6))
            leads.append(
                {"lead_id": f"lead-{i:03d}", "features": features, "interaction_text": note}
            )
        DEMO_LEADS_JSON.write_text(json.dumps(leads, ensure_ascii=False))

    return {"scoring": SCORING_MODEL_PATH, "sentiment": SENTIMENT_MODEL_DIR}


def _coerce(value):
    return value.item() if hasattr(value, "item") else value
