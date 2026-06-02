"""Train both models (sentiment + lead scoring) and write artifacts + reports.

Usage:
    python -m scripts.train_all              # full run
    python -m scripts.train_all --quick      # fast hyper-parameter search (smoke test)
"""

from __future__ import annotations

import argparse
import logging

from scripts import _bootstrap  # noqa: F401  (adds src/ to sys.path when not pip-installed)

from lead_priority.api.logging_config import configure_logging
from lead_priority.scoring.train import train_lead_scoring_model
from lead_priority.sentiment.train import train_sentiment_model

logger = logging.getLogger("train_all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lead-priority models.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a tiny hyper-parameter search (for smoke tests).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating report PNGs.",
    )
    args = parser.parse_args()

    configure_logging()

    logger.info("=== Training sentiment / intent model ===")
    sentiment_result = train_sentiment_model()
    logger.info(
        "Sentiment model: accuracy=%.3f macro_f1=%.3f",
        sentiment_result.accuracy,
        sentiment_result.macro_f1,
    )

    logger.info("=== Training lead-scoring model ===")
    scoring_result = train_lead_scoring_model(quick=args.quick, make_plots=not args.no_plots)
    base = scoring_result.metrics["baseline_logreg"]
    lgbm = scoring_result.metrics["lightgbm_calibrated"]
    capture = scoring_result.metrics["top_20pct_capture"]

    logger.info("--- Lead scoring comparison (test set) ---")
    logger.info(
        "Baseline LogReg : ROC-AUC=%.3f PR-AUC=%.3f Brier=%.3f",
        base["roc_auc"],
        base["pr_auc"],
        base["brier_score"],
    )
    logger.info(
        "LightGBM (cal.) : ROC-AUC=%.3f PR-AUC=%.3f Brier=%.3f (threshold=%.2f)",
        lgbm["roc_auc"],
        lgbm["pr_auc"],
        lgbm["brier_score"],
        scoring_result.threshold,
    )
    logger.info(
        "Top 20%% of leads capture %.1f%% of all conversions (lift x%.2f).",
        capture["capture_rate"] * 100,
        capture["capture_rate"] / 0.2,
    )
    logger.info("Artifacts written to models/ and reports/.")


if __name__ == "__main__":
    main()
