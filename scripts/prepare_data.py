"""Prepare all data artifacts needed for training and serving.

Produces:
    * ``data/raw/Leads.csv``            - downloaded if missing.
    * ``data/processed/interactions.csv`` - synthetic labelled interaction notes.
    * ``data/processed/demo_leads.json``  - sample leads + notes for ``GET /leads/top``.
"""

from __future__ import annotations

import json
import logging
import random

import pandas as pd

from scripts import _bootstrap  # noqa: F401  (adds src/ to sys.path when not pip-installed)

from lead_priority.api.logging_config import configure_logging
from lead_priority.config import (
    DEMO_LEADS_JSON,
    INTERACTIONS_CSV,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SENTIMENT_LABELS,
)
from lead_priority.data.loaders import load_raw_leads
from lead_priority.data.synthetic_interactions import generate_interaction_dataset, sample_note
from lead_priority.features.engineering import model_input_columns

logger = logging.getLogger("prepare_data")

N_DEMO_LEADS = 40


def _build_demo_leads(n: int = N_DEMO_LEADS, seed: int = RANDOM_SEED) -> list[dict]:
    """Sample real leads and attach a synthetic note (independent of ``Converted``).

    Picking the note independently of the label keeps the demo honest: the combined
    priority score is *not* secretly handed the conversion answer via sentiment.
    """
    df = load_raw_leads()
    cols = [c for c in model_input_columns() if c in df.columns]
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    rng = random.Random(seed)
    leads: list[dict] = []
    for i, row in sample.iterrows():
        features = {
            col: (None if pd.isna(row[col]) else _coerce(row[col])) for col in cols
        }
        intent = rng.choice(SENTIMENT_LABELS)
        note = sample_note(intent, seed=rng.randint(0, 1_000_000))
        leads.append(
            {
                "lead_id": f"lead-{i + 1:03d}",
                "features": features,
                "interaction_text": note,
            }
        )
    return leads


def _coerce(value: object) -> object:
    """Make numpy scalars JSON-serialisable."""
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    configure_logging()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic interaction dataset...")
    interactions = generate_interaction_dataset()
    interactions.frame.to_csv(INTERACTIONS_CSV, index=False)
    logger.info("Wrote %d interaction notes to %s", len(interactions.frame), INTERACTIONS_CSV)

    logger.info("Building demo lead list...")
    demo_leads = _build_demo_leads()
    DEMO_LEADS_JSON.write_text(json.dumps(demo_leads, indent=2, ensure_ascii=False))
    logger.info("Wrote %d demo leads to %s", len(demo_leads), DEMO_LEADS_JSON)


if __name__ == "__main__":
    main()
