"""Behavioral segmentation via KMeans (the up-front 'segmentasyon çalışması').

Beyond the rule-based personas, we run an *unsupervised* segmentation on engagement
behaviour (time on site, pages viewed, channel diversity, depth per visit). This answers
"not every lead should be treated the same" with data-driven groups, and the clusters are
named deterministically from their engagement profile so the labels are interpretable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from lead_priority.config import (
    N_BEHAVIORAL_SEGMENTS,
    RANDOM_SEED,
    SEGMENTATION_FEATURES,
    SEGMENTATION_METRICS_PATH,
    SEGMENTATION_MODEL_PATH,
)
from lead_priority.features.engineering import add_engineered_features

logger = logging.getLogger(__name__)

# Ordered, human-readable labels by overall engagement rank (low -> high).
_ENGAGEMENT_LABELS: dict[int, str] = {
    0: "Pasif / Düşük etkileşim",
    1: "Keşif aşamasında",
    2: "İlgili / Orta etkileşim",
    3: "Yüksek etkileşimli",
}


@dataclass(frozen=True)
class BehavioralSegmenter:
    """Assign a behavioral segment to a lead using a fitted scaler + KMeans."""

    scaler: StandardScaler
    kmeans: KMeans
    features: list[str]
    cluster_to_label: dict[int, str]

    @classmethod
    def load(cls, path: Path = SEGMENTATION_MODEL_PATH) -> "BehavioralSegmenter":
        if not path.exists():
            raise FileNotFoundError(
                f"Segmentation model not found at {path}. Train with `python -m scripts.train_all`."
            )
        artifact = joblib.load(path)
        return cls(
            scaler=artifact["scaler"],
            kmeans=artifact["kmeans"],
            features=list(artifact["features"]),
            cluster_to_label={int(k): v for k, v in artifact["cluster_to_label"].items()},
        )

    def _matrix(self, leads: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame) -> np.ndarray:
        if isinstance(leads, pd.DataFrame):
            frame = leads
        else:
            records = [leads] if isinstance(leads, dict) else list(leads)
            frame = pd.DataFrame(records)
        eng = add_engineered_features(frame)
        for col in self.features:
            if col not in eng.columns:
                eng[col] = 0.0
        x = eng[self.features].astype(float).fillna(0.0).to_numpy()
        return self.scaler.transform(x)

    def assign(self, leads: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame) -> list[str]:
        """Return the behavioral segment label for one or many leads."""
        clusters = self.kmeans.predict(self._matrix(leads))
        return [self.cluster_to_label[int(c)] for c in clusters]

    def assign_one(self, features: dict[str, Any]) -> str:
        return self.assign(features)[0]


def train_segmentation_model(
    *,
    df: pd.DataFrame,
    n_clusters: int = N_BEHAVIORAL_SEGMENTS,
    model_path: Path = SEGMENTATION_MODEL_PATH,
) -> dict[str, Any]:
    """Fit StandardScaler + KMeans on engagement features and persist the artifact."""
    features = list(SEGMENTATION_FEATURES)
    eng = add_engineered_features(df)
    x = eng[features].astype(float).fillna(0.0).to_numpy()

    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10).fit(x_scaled)

    # Rank clusters by overall engagement (sum of scaled centroid coords) and name them.
    centroids = kmeans.cluster_centers_
    engagement_rank = np.argsort(centroids.sum(axis=1))  # low -> high
    cluster_to_label: dict[int, str] = {}
    for rank, cluster_id in enumerate(engagement_rank):
        label_idx = int(round(rank * (len(_ENGAGEMENT_LABELS) - 1) / max(n_clusters - 1, 1)))
        cluster_to_label[int(cluster_id)] = _ENGAGEMENT_LABELS[label_idx]

    # Cluster profiles (original-scale means) for interpretability / README.
    labels = kmeans.labels_
    profiles: list[dict[str, Any]] = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        profiles.append(
            {
                "cluster": int(cluster_id),
                "label": cluster_to_label[int(cluster_id)],
                "size": int(mask.sum()),
                "means": {f: float(eng.loc[mask, f].mean()) for f in features},
            }
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": kmeans,
            "features": features,
            "cluster_to_label": cluster_to_label,
        },
        model_path,
    )
    metrics = {
        "n_clusters": n_clusters,
        "features": features,
        "inertia": float(kmeans.inertia_),
        "profiles": profiles,
    }
    SEGMENTATION_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved behavioral segmentation model to %s (%d clusters)", model_path, n_clusters)
    return metrics
