"""Train, tune, calibrate and compare the lead-scoring models.

Pipeline:
    1. Load raw leads, drop ID + leaky columns.
    2. Stratified train/test split (class imbalance preserved).
    3. Baseline  : Logistic Regression (interpretable, ``class_weight="balanced"``).
    4. Modern    : LightGBM with a randomised hyper-parameter search, then probability
                   calibration (isotonic) so the scores are usable as real probabilities.
    5. Pick an operating threshold by maximising F1 on *out-of-fold* train predictions
       (no test leakage), then evaluate everything on the held-out test set.
    6. Persist the calibrated LightGBM pipeline as the production artifact.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from lead_priority.config import (
    LEAKY_COLUMNS,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    SCORING_METRICS_PATH,
    SCORING_MODEL_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    ID_COLUMNS,
)
from lead_priority.data.loaders import load_raw_leads
from lead_priority.features.engineering import build_preprocessor, model_input_columns
from lead_priority.scoring import evaluate as ev

logger = logging.getLogger(__name__)


@dataclass
class ScoringTrainResult:
    """Container for everything produced by :func:`train_lead_scoring_model`."""

    chosen_model: str
    threshold: float
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_path: Path | None = None


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [c for c in (*ID_COLUMNS, *LEAKY_COLUMNS) if c in df.columns]
    df = df.drop(columns=drop_cols)
    y = df[TARGET_COLUMN].astype(int)
    x = df.drop(columns=[TARGET_COLUMN])
    return x, y


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold (from a fine grid) that maximises F1; computed on train OOF preds."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _build_baseline() -> Pipeline:
    pre = build_preprocessor()
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    return Pipeline([("preprocess", pre), ("classifier", clf)])


def _build_lgbm(scale_pos_weight: float) -> Pipeline:
    pre = build_preprocessor()
    # Single-threaded per-estimator: parallelism is exploited across CV folds by the
    # search (``n_jobs=-1``). Nesting both would oversubscribe the CPUs and stall.
    clf = LGBMClassifier(
        objective="binary",
        random_state=RANDOM_SEED,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )
    return Pipeline([("preprocess", pre), ("classifier", clf)])


def _lgbm_search_space() -> dict[str, list[Any]]:
    return {
        "classifier__n_estimators": [200, 300, 400, 600],
        "classifier__learning_rate": [0.02, 0.05, 0.1],
        "classifier__num_leaves": [15, 31, 63],
        "classifier__max_depth": [-1, 4, 6, 8],
        "classifier__min_child_samples": [20, 40, 80],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
        "classifier__reg_lambda": [0.0, 1.0, 5.0],
    }


def train_lead_scoring_model(
    *,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    quick: bool = False,
    make_plots: bool = True,
) -> ScoringTrainResult:
    """Train both models, calibrate LightGBM, evaluate, and persist the best one.

    Parameters
    ----------
    quick:
        If True, use a tiny hyper-parameter search (for tests / smoke runs).
    make_plots:
        If True, write gain/calibration/confusion PNGs into ``reports_dir``.
    """
    df = load_raw_leads()
    x, y = _prepare_xy(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    pos_rate = float(y_train.mean())
    scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)
    logger.info("Train rows=%d  test rows=%d  positive_rate=%.3f", len(x_train), len(x_test), pos_rate)

    # --- Baseline: Logistic Regression ------------------------------------------------
    baseline = _build_baseline()
    baseline.fit(x_train, y_train)
    base_prob = baseline.predict_proba(x_test)[:, 1]
    base_metrics = ev.classification_metrics(y_test.to_numpy(), base_prob)

    # --- Modern: LightGBM + randomised search -----------------------------------------
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    n_iter = 4 if quick else 20
    search = RandomizedSearchCV(
        estimator=_build_lgbm(scale_pos_weight),
        param_distributions=_lgbm_search_space(),
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(x_train, y_train)
    best_lgbm: Pipeline = search.best_estimator_
    logger.info("LightGBM best params: %s", search.best_params_)

    # --- Operating threshold from an inner validation split (no test leakage) ----------
    # Fit a temporary calibrated model on an inner-train split, pick the F1-optimal
    # threshold on the inner-validation split, then discard it.
    xi_tr, xi_val, yi_tr, yi_val = train_test_split(
        x_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_SEED
    )
    threshold_model = CalibratedClassifierCV(
        estimator=clone(best_lgbm), method="isotonic", cv=3
    )
    threshold_model.fit(xi_tr, yi_tr)
    val_prob = threshold_model.predict_proba(xi_val)[:, 1]
    threshold = _best_f1_threshold(yi_val.to_numpy(), val_prob)

    # --- Final calibrated model on all of x_train (isotonic, 3-fold) -------------------
    calibrated = CalibratedClassifierCV(estimator=best_lgbm, method="isotonic", cv=3)
    calibrated.fit(x_train, y_train)
    lgbm_prob = calibrated.predict_proba(x_test)[:, 1]

    lgbm_metrics = ev.classification_metrics(y_test.to_numpy(), lgbm_prob, threshold=threshold)
    gain_table = ev.gain_lift_table(y_test.to_numpy(), lgbm_prob)
    capture_20 = ev.top_k_capture(y_test.to_numpy(), lgbm_prob, k=0.2)

    metrics: dict[str, Any] = {
        "baseline_logreg": base_metrics,
        "lightgbm_calibrated": lgbm_metrics,
        "lightgbm_best_params": {k: v for k, v in search.best_params_.items()},
        "top_20pct_capture": capture_20,
        "gain_lift_table": gain_table.to_dict(orient="records"),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "train_positive_rate": pos_rate,
        "chosen_threshold": threshold,
    }

    # --- Persist artifacts -------------------------------------------------------------
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": calibrated,
        "threshold": threshold,
        "feature_columns": model_input_columns(),
        "metadata": {
            "model_type": "lightgbm_isotonic_calibrated",
            "best_params": search.best_params_,
            "random_seed": RANDOM_SEED,
        },
    }
    joblib.dump(artifact, SCORING_MODEL_PATH)
    SCORING_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved lead-scoring model to %s", SCORING_MODEL_PATH)

    if make_plots:
        reports_dir.mkdir(parents=True, exist_ok=True)
        ev.plot_gain_chart(gain_table, reports_dir / "gain_chart.png")
        ev.plot_confusion_matrix(
            y_test.to_numpy(), lgbm_prob, reports_dir / "confusion_matrix.png", threshold=threshold
        )
        ev.plot_calibration(y_test.to_numpy(), lgbm_prob, reports_dir / "calibration_curve.png")

    return ScoringTrainResult(
        chosen_model="lightgbm_isotonic_calibrated",
        threshold=threshold,
        metrics=metrics,
        artifact_path=SCORING_MODEL_PATH,
    )
