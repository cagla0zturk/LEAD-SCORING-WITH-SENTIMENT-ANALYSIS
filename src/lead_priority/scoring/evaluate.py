"""Evaluation metrics tailored to lead scoring.

Beyond accuracy/AUC we care about the *business* questions:

* **Calibration** - is a predicted 0.8 really an 80% chance? (Brier score + curve)
* **Precision/Recall trade-off** - at what threshold do we burn rep time on false positives?
* **Gain / Lift** - "if I work the top 20% of leads, what share of conversions do I get?"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float = 0.5
) -> dict[str, float]:
    """Return headline classification + calibration metrics as plain floats."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "positive_rate": float(np.mean(y_true)),
    }


def gain_lift_table(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> pd.DataFrame:
    """Cumulative gain / lift by score decile (decile 1 = highest scores).

    Columns: ``decile``, ``n``, ``conversions``, ``cumulative_conversions``,
    ``cumulative_population_pct``, ``cumulative_gain_pct``, ``lift``.
    """
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = np.asarray(y_true)[order]
    n = len(y_sorted)
    total_pos = max(int(y_sorted.sum()), 1)

    bins = np.array_split(np.arange(n), n_bins)
    rows: list[dict[str, Any]] = []
    cum_pos = 0
    cum_n = 0
    for i, idx in enumerate(bins, start=1):
        seg = y_sorted[idx]
        cum_pos += int(seg.sum())
        cum_n += len(seg)
        cum_pop_pct = cum_n / n
        cum_gain_pct = cum_pos / total_pos
        rows.append(
            {
                "decile": i,
                "n": len(seg),
                "conversions": int(seg.sum()),
                "cumulative_conversions": cum_pos,
                "cumulative_population_pct": round(cum_pop_pct, 4),
                "cumulative_gain_pct": round(cum_gain_pct, 4),
                "lift": round(cum_gain_pct / cum_pop_pct, 4) if cum_pop_pct else 0.0,
            }
        )
    return pd.DataFrame(rows)


def top_k_capture(y_true: np.ndarray, y_prob: np.ndarray, *, k: float = 0.2) -> dict[str, float]:
    """Share of all conversions captured within the top ``k`` fraction of leads."""
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = np.asarray(y_true)[order]
    n_top = max(int(round(len(y_sorted) * k)), 1)
    captured = int(y_sorted[:n_top].sum())
    total_pos = max(int(y_sorted.sum()), 1)
    return {
        "k": float(k),
        "n_top": n_top,
        "captured_conversions": captured,
        "capture_rate": round(captured / total_pos, 4),
        "precision_in_top_k": round(captured / n_top, 4),
    }


# --------------------------------------------------------------------------------------
# Plotting (lazy matplotlib import; only used by the training script / notebook).
# --------------------------------------------------------------------------------------
def plot_gain_chart(table: pd.DataFrame, out_path: Path, *, title: str = "Cumulative Gain") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [0.0, *table["cumulative_population_pct"].tolist()]
    y = [0.0, *table["cumulative_gain_pct"].tolist()]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    ax.set_xlabel("Fraction of leads contacted (sorted by score)")
    ax.set_ylabel("Fraction of conversions captured")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, *, threshold: float = 0.5
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_title(f"Confusion Matrix (thr={threshold:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, *, n_bins: int = 10
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed conversion rate")
    ax.set_title("Calibration curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
