"""Tests for leakage-aware feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lead_priority.features.engineering import (
    ENGINEERED_NUMERIC_FEATURES,
    add_engineered_features,
    build_preprocessor,
    model_input_columns,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TotalVisits": [4, 0, np.nan],
            "Total Time Spent on Website": [800, 0, 120],
            "Page Views Per Visit": [2.0, 0.0, 1.0],
            "Lead Source": ["Reference", "Google", "Olark Chat"],
            "Do Not Email": ["No", "Yes", "No"],
            "Do Not Call": ["No", "No", "No"],
            "What is your current occupation": ["Working Professional", "Student", None],
            "Search": ["Yes", "No", "Yes"],
            "Newspaper": ["No", "No", "Yes"],
        }
    )


def test_engineered_columns_present():
    out = add_engineered_features(_sample_frame())
    for col in ENGINEERED_NUMERIC_FEATURES:
        assert col in out.columns


def test_high_intent_and_reachability_flags():
    out = add_engineered_features(_sample_frame())
    assert out.loc[0, "is_high_intent_source"] == 1.0  # Reference
    assert out.loc[1, "is_high_intent_source"] == 0.0  # Google
    assert out.loc[0, "is_working_professional"] == 1.0
    # do_not_contact is only True if BOTH email and call are opted out at the engineer
    # level it tracks "any" opt-out; here row 1 opted out of email.
    assert out.loc[1, "do_not_contact"] == 1.0


def test_channel_diversity_counts_yes_flags():
    out = add_engineered_features(_sample_frame())
    assert out.loc[0, "channel_diversity"] == 1.0  # Search=Yes only
    assert out.loc[2, "channel_diversity"] == 2.0  # Search + Newspaper


def test_handles_missing_columns_gracefully():
    # API may send a partial payload; engineering must not raise.
    minimal = pd.DataFrame({"TotalVisits": [3]})
    out = add_engineered_features(minimal)
    for col in ENGINEERED_NUMERIC_FEATURES:
        assert col in out.columns


def test_preprocessor_fit_transform_runs():
    df = _sample_frame()
    # Add the remaining expected raw columns as NA so the ColumnTransformer is happy.
    for col in model_input_columns():
        if col not in df.columns:
            df[col] = pd.NA
    pre = build_preprocessor()
    matrix = pre.fit_transform(df)
    assert matrix.shape[0] == len(df)
    assert matrix.shape[1] > 0
