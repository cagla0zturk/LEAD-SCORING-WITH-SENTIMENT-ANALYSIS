"""Central configuration: paths, column groups, label definitions and constants.

Keeping these in one typed module means the data, training, and serving code all
agree on the same column names, leakage policy and artifact locations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = PACKAGE_ROOT.parent.parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"

RAW_LEADS_CSV: Final[Path] = RAW_DATA_DIR / "Leads.csv"
INTERACTIONS_CSV: Final[Path] = PROCESSED_DATA_DIR / "interactions.csv"
DEMO_LEADS_JSON: Final[Path] = PROCESSED_DATA_DIR / "demo_leads.json"

# Trained artifacts
SCORING_MODEL_PATH: Final[Path] = MODELS_DIR / "lead_scoring_model.joblib"
SENTIMENT_MODEL_PATH: Final[Path] = MODELS_DIR / "sentiment_model.joblib"
SCORING_METRICS_PATH: Final[Path] = MODELS_DIR / "lead_scoring_metrics.json"
SENTIMENT_METRICS_PATH: Final[Path] = MODELS_DIR / "sentiment_metrics.json"

# Public mirror of the X Education "Leads.csv" (Kaggle: lead-scoring-x-online-education).
# Used only as a convenience fallback when the file is not already present locally.
LEADS_CSV_URL: Final[str] = (
    "https://raw.githubusercontent.com/Likitesh/X_Education/master/Leads.csv"
)

# --------------------------------------------------------------------------------------
# Tabular dataset column policy
# --------------------------------------------------------------------------------------
TARGET_COLUMN: Final[str] = "Converted"

ID_COLUMNS: Final[tuple[str, ...]] = ("Prospect ID", "Lead Number")

# Columns that LEAK the target. These are populated *by the sales process itself*
# (e.g. a rep tags a lead "Closed by Horizzon" only once it is essentially won) or
# are post-hoc quality judgements. Training on them produces a model that looks great
# offline but is useless at the only moment that matters: scoring a *fresh* lead.
# See README "Veri sızıntısı (leakage)" for the full discussion.
LEAKY_COLUMNS: Final[tuple[str, ...]] = (
    "Tags",
    "Lead Quality",
    "Lead Profile",
    "Last Notable Activity",
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
)

# Numeric features used by the tabular model (raw columns; engineered ones are added later).
RAW_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
)

# Categorical features kept for modelling (low-leakage, available at lead-creation time).
RAW_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "Lead Origin",
    "Lead Source",
    "Do Not Email",
    "Do Not Call",
    "Last Activity",
    "Country",
    "Specialization",
    "What is your current occupation",
    "What matters most to you in choosing a course",
    "City",
    "A free copy of Mastering The Interview",
)

# Sentinel value used across the X Education categorical columns; equivalent to NULL.
SELECT_SENTINEL: Final[str] = "Select"

# --------------------------------------------------------------------------------------
# Sentiment / intent labels
# --------------------------------------------------------------------------------------
SENTIMENT_LABELS: Final[tuple[str, ...]] = (
    "positive_engagement",
    "objection",
    "disengaged",
    "neutral",
)

# How each intent class nudges the combined priority score. Positive engagement is a
# strong "call today" signal; disengagement cools a lead down; an objection is actually
# a buying signal that needs a rep (slightly positive); neutral is, well, neutral.
SENTIMENT_PRIORITY_WEIGHT: Final[dict[str, float]] = {
    "positive_engagement": 1.0,
    "objection": 0.55,
    "neutral": 0.40,
    "disengaged": 0.10,
}

# --------------------------------------------------------------------------------------
# Modelling constants
# --------------------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 42
TEST_SIZE: Final[float] = 0.2

# Default blend weight for the combined priority score (see lead_priority.priority).
DEFAULT_CONVERSION_WEIGHT: Final[float] = 0.7
