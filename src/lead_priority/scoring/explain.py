"""Per-lead score explanations ("bu skor neden?") via LightGBM SHAP contributions.

Uses LightGBM's native ``predict(..., pred_contrib=True)`` (exact tree SHAP values) on the
tuned, *uncalibrated* model — no extra ``shap`` dependency. One-hot columns are aggregated
back to their original feature so the explanation reads in business terms, e.g.
"Sitede geçirilen süre → dönüşümü artırıyor".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from lead_priority.config import RAW_CATEGORICAL_FEATURES, SCORING_EXPLAINER_PATH
from lead_priority.features.engineering import model_input_columns

logger = logging.getLogger(__name__)

# Human-readable Turkish labels for the model features.
_FEATURE_LABELS: dict[str, str] = {
    "TotalVisits": "Site ziyaret sayısı",
    "Total Time Spent on Website": "Sitede geçirilen süre",
    "Page Views Per Visit": "Ziyaret başına sayfa",
    "time_per_visit": "Ziyaret başına süre",
    "total_page_views": "Toplam sayfa görüntüleme",
    "log_time_spent": "Sitede geçirilen süre",
    "channel_diversity": "Kanal çeşitliliği",
    "missing_field_count": "Eksik alan sayısı",
    "do_not_contact": "İletişim engeli (opt-out)",
    "is_working_professional": "Çalışan profesyonel olması",
    "is_high_intent_source": "Yüksek niyetli kaynak",
    "Lead Origin": "Lead giriş kanalı",
    "Lead Source": "Lead kaynağı",
    "Last Activity": "Son aktivite",
    "What is your current occupation": "Meslek",
    "Specialization": "Uzmanlık alanı",
    "City": "Şehir",
    "Country": "Ülke",
    "A free copy of Mastering The Interview": "Ücretsiz kitap talebi",
    "What matters most to you in choosing a course": "Kurs seçim önceliği",
}


@dataclass(frozen=True)
class ScoreExplainer:
    """Explains a single lead's conversion score via aggregated SHAP contributions."""

    model: Any  # tuned LightGBM Pipeline(preprocess, classifier)
    feature_columns: list[str]

    @classmethod
    def load(cls, path: Path = SCORING_EXPLAINER_PATH) -> "ScoreExplainer":
        if not path.exists():
            raise FileNotFoundError(
                f"Score explainer not found at {path}. Train with `python -m scripts.train_all`."
            )
        model = joblib.load(path)
        logger.info("Loaded score explainer from %s", path)
        return cls(model=model, feature_columns=model_input_columns())

    def _to_frame(self, features: dict[str, Any]) -> pd.DataFrame:
        frame = pd.DataFrame([features])
        for col in self.feature_columns:
            if col not in frame.columns:
                frame[col] = np.nan
        return frame.where(frame.notna(), other=np.nan)

    @staticmethod
    def _to_original(name: str) -> str:
        for cat in RAW_CATEGORICAL_FEATURES:
            if name == cat or name.startswith(f"{cat}_"):
                return cat
        return name

    def explain(self, features: dict[str, Any], *, top_k: int = 4) -> list[dict[str, Any]]:
        """Return the top ``top_k`` features driving this lead's score, with direction."""
        frame = self._to_frame(features)
        preprocess = self.model.named_steps["preprocess"]
        classifier = self.model.named_steps["classifier"]

        x = preprocess.transform(frame)
        names = list(preprocess.named_steps["columns"].get_feature_names_out())
        # pred_contrib returns one row of (n_features + 1) values; last is the base value.
        contrib = classifier.booster_.predict(x, pred_contrib=True)[0]

        # Aggregate one-hot columns back to their original feature ...
        by_feature: dict[str, float] = {}
        for name, value in zip(names, contrib[:-1]):
            original = self._to_original(name)
            by_feature[original] = by_feature.get(original, 0.0) + float(value)

        # ... then merge features that share a human label (e.g. raw + log time-on-site).
        by_label: dict[str, dict[str, Any]] = {}
        for feature, value in by_feature.items():
            label = _FEATURE_LABELS.get(feature, feature)
            entry = by_label.setdefault(
                label, {"feature": feature, "label": label, "contribution": 0.0, "lead_value": None}
            )
            entry["contribution"] += value
            lead_value = _display_value(features.get(feature))
            if entry["lead_value"] is None and lead_value is not None:
                entry["lead_value"] = lead_value

        ranked = sorted(by_label.values(), key=lambda e: abs(e["contribution"]), reverse=True)
        out: list[dict[str, Any]] = []
        for entry in ranked[:top_k]:
            out.append(
                {
                    "feature": entry["feature"],
                    "label": entry["label"],
                    "lead_value": entry["lead_value"],
                    "contribution": round(float(entry["contribution"]), 4),
                    "direction": "artırıyor" if entry["contribution"] >= 0 else "azaltıyor",
                }
            )
        return out


def _display_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
