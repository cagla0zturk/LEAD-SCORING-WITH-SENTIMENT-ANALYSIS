"""Loading and light, model-agnostic cleaning of the raw tabular lead data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lead_priority.config import RAW_LEADS_CSV, SELECT_SENTINEL
from lead_priority.data.download import ensure_leads_csv

logger = logging.getLogger(__name__)


def load_raw_leads(path: Path = RAW_LEADS_CSV, *, download_if_missing: bool = True) -> pd.DataFrame:
    """Load the X Education ``Leads.csv`` into a tidy DataFrame.

    The only cleaning done here is *non-modelling* cleaning that is always correct:

    * download the file if it is missing (optional),
    * replace the ``"Select"`` sentinel (an un-filled dropdown) with ``NaN``,
    * strip surrounding whitespace from object columns.

    Feature engineering and imputation live in :mod:`lead_priority.features` so that
    they are fitted on the training split only (avoiding train/test leakage).
    """
    if download_if_missing:
        path = ensure_leads_csv(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Leads.csv not found at {path}. Run `python -m scripts.prepare_data` first."
        )

    df = pd.read_csv(path)
    logger.info("Loaded raw leads: %d rows x %d columns", df.shape[0], df.shape[1])

    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].str.strip()

    # "Select" is the default, never-changed dropdown value; treat it as missing.
    # Use np.nan (not pd.NA) so scikit-learn's imputers detect it correctly.
    df = df.replace(SELECT_SENTINEL, np.nan)
    return df
