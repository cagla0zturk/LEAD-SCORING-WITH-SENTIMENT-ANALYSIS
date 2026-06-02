"""Feature engineering for lead scoring.

Everything here is implemented as scikit-learn compatible transformers so that the
*exact same* logic runs at training time (on a DataFrame of 9k rows) and at serving
time (on a single-lead DataFrame inside the API). This is what avoids the classic
"works in the notebook, breaks in prod" feature-skew bug.

Design choices
--------------
* ``FeatureEngineer`` only derives new columns from inputs that exist *the moment a
  lead is created or browses the site* - no post-sale signals (those live in
  ``config.LEAKY_COLUMNS`` and are dropped).
* Imputation and one-hot encoding are fitted **inside** the pipeline, hence on the
  training split only, so no test statistics leak in.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lead_priority.config import RAW_CATEGORICAL_FEATURES, RAW_NUMERIC_FEATURES

# Binary "did the lead come through this channel / interest" flags (Yes/No in the raw data).
# Their *count* is a cheap proxy for channel diversity / marketing touch breadth.
_INTEREST_FLAG_COLUMNS: Final[tuple[str, ...]] = (
    "Search",
    "Magazine",
    "Newspaper Article",
    "X Education Forums",
    "Newspaper",
    "Digital Advertisement",
    "Through Recommendations",
)

# Lead sources that empirically convert well (referrals / partner sites).
_HIGH_INTENT_SOURCES: Final[frozenset[str]] = frozenset(
    {"Reference", "Welingak Website", "Referral Sites"}
)

ENGINEERED_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "time_per_visit",
    "total_page_views",
    "log_time_spent",
    "channel_diversity",
    "missing_field_count",
    "do_not_contact",
    "is_working_professional",
    "is_high_intent_source",
)


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric Series for ``col`` (NaN-filled Series if the column is absent)."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _string_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    """String Series for ``col`` (``default``-filled Series if the column is absent)."""
    if col in df.columns:
        return df[col].astype("string").fillna(default).astype(str).str.strip()
    return pd.Series(default, index=df.index, dtype=str)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with derived features appended.

    The function is defensive: it tolerates missing columns (relevant when the API
    receives a partial lead payload) by treating absent inputs as missing/zero, and it
    normalises missing markers (``None`` / ``pd.NA``) to ``np.nan`` so the downstream
    scikit-learn imputers detect them reliably.
    """
    out = df.copy()
    # Normalise all missing markers to np.nan (sklearn imputers can't handle pd.NA).
    out = out.where(out.notna(), other=np.nan)

    visits = _numeric_series(out, "TotalVisits")
    time_spent = _numeric_series(out, "Total Time Spent on Website")
    pages = _numeric_series(out, "Page Views Per Visit")

    # Depth of engagement: time spent per individual visit.
    out["time_per_visit"] = time_spent / visits.replace(0, np.nan)
    # Total pages browsed across the whole relationship.
    out["total_page_views"] = pages * visits
    # Heavy right-skew on website time -> log1p to stabilise.
    out["log_time_spent"] = np.log1p(time_spent.clip(lower=0))

    # Channel diversity: how many marketing channels/interests the lead engaged with.
    present_flags = [c for c in _INTEREST_FLAG_COLUMNS if c in out.columns]
    if present_flags:
        flags = out[present_flags].apply(lambda s: s.astype(str).str.strip().str.lower() == "yes")
        out["channel_diversity"] = flags.sum(axis=1).astype(float)
    else:
        out["channel_diversity"] = 0.0

    # Missingness as signal: leads with many unfilled fields tend to be lower quality.
    modelling_cols = list(RAW_NUMERIC_FEATURES) + list(RAW_CATEGORICAL_FEATURES)
    present_modelling = [c for c in modelling_cols if c in out.columns]
    if present_modelling:
        out["missing_field_count"] = out[present_modelling].isna().sum(axis=1).astype(float)
    else:
        out["missing_field_count"] = 0.0

    # Reachability: lead opted out of email or phone.
    dne = _string_series(out, "Do Not Email", "No").str.lower()
    dnc = _string_series(out, "Do Not Call", "No").str.lower()
    out["do_not_contact"] = ((dne == "yes") | (dnc == "yes")).astype(float)

    occupation = _string_series(out, "What is your current occupation")
    out["is_working_professional"] = occupation.eq("Working Professional").astype(float)

    source = _string_series(out, "Lead Source")
    out["is_high_intent_source"] = source.isin(_HIGH_INTENT_SOURCES).astype(float)

    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless transformer wrapping :func:`add_engineered_features`."""

    def fit(self, X: pd.DataFrame, y: object | None = None) -> "FeatureEngineer":  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return add_engineered_features(X)

    def get_feature_names_out(self, input_features=None):  # pragma: no cover - sklearn API
        return np.asarray(list(input_features or []) + list(ENGINEERED_NUMERIC_FEATURES))


def model_input_columns() -> list[str]:
    """Raw columns the pipeline expects to receive (engineered ones are derived)."""
    extras = ("Do Not Email", "Do Not Call", *_INTEREST_FLAG_COLUMNS)
    cols = list(RAW_NUMERIC_FEATURES) + list(RAW_CATEGORICAL_FEATURES) + list(extras)
    # Preserve order while de-duplicating.
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def build_preprocessor() -> Pipeline:
    """Build the ``FeatureEngineer -> ColumnTransformer`` preprocessing pipeline.

    * Numeric (raw + engineered): median imputation + standard scaling.
    * Categorical: most-frequent imputation + one-hot (rare levels collapsed via
      ``min_frequency`` to keep dimensionality and serving latency sane).
    """
    numeric_features = list(RAW_NUMERIC_FEATURES) + list(ENGINEERED_NUMERIC_FEATURES)
    categorical_features = list(RAW_CATEGORICAL_FEATURES)

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=25,
                    sparse_output=False,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("columns", column_transformer),
        ]
    )
