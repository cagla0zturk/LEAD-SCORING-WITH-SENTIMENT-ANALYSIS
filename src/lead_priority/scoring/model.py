"""Serving-time wrapper around the persisted lead-scoring pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from lead_priority.config import SCORING_MODEL_PATH

logger = logging.getLogger(__name__)


class LeadScorer:
    """Loads the calibrated lead-scoring pipeline and predicts conversion probability.

    The wrapped object is a full scikit-learn pipeline (feature engineering +
    preprocessing + calibrated LightGBM), so callers only need to provide the *raw*
    lead fields. Missing fields are tolerated and imputed inside the pipeline.
    """

    def __init__(self, model: Any, threshold: float, feature_columns: list[str]) -> None:
        self._model = model
        self.threshold = threshold
        self.feature_columns = feature_columns

    @classmethod
    def load(cls, path: Path = SCORING_MODEL_PATH) -> "LeadScorer":
        if not path.exists():
            raise FileNotFoundError(
                f"Lead-scoring model not found at {path}. "
                "Train it first with `python -m scripts.train_all`."
            )
        artifact = joblib.load(path)
        logger.info("Loaded lead-scoring model from %s", path)
        return cls(
            model=artifact["model"],
            threshold=float(artifact["threshold"]),
            feature_columns=list(artifact["feature_columns"]),
        )

    def _to_frame(self, leads: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
        records = [leads] if isinstance(leads, dict) else list(leads)
        frame = pd.DataFrame(records)
        # Guarantee all expected raw columns exist so the pipeline never KeyErrors.
        for col in self.feature_columns:
            if col not in frame.columns:
                frame[col] = np.nan
        # Normalise all missing markers (None / pd.NA) to np.nan for sklearn imputers.
        return frame.where(frame.notna(), other=np.nan)

    def predict_proba(self, leads: dict[str, Any] | list[dict[str, Any]]) -> list[float]:
        """Return the conversion probability for one or many leads."""
        frame = self._to_frame(leads)
        probs = self._model.predict_proba(frame)[:, 1]
        return [float(p) for p in probs]

    def predict(self, leads: dict[str, Any] | list[dict[str, Any]]) -> list[int]:
        """Return the hard 0/1 prediction using the tuned operating threshold."""
        return [int(p >= self.threshold) for p in self.predict_proba(leads)]
