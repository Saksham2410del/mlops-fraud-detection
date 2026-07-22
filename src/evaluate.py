"""
Comprehensive Model Evaluation for Fraud Detection
===================================================

Loads a trained XGBoost model and test data, then produces:
  * Classification report and confusion matrix
  * ROC curve and Precision-Recall curve (saved as PNGs)
  * Feature importance chart
  * Business-oriented metrics (false-positive/negative rates, cost analysis)
  * Optimal probability threshold selection (maximises F1)
  * A JSON evaluation report summarising everything

Usage:
    python src/evaluate.py \
        --model models/xgboost_fraud.joblib \
        --data-dir data/processed/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")

# Apply a clean, professional plotting style.  The "seaborn-v0_8-darkgrid"
# name works on matplotlib ≥ 3.6; fall back for older versions.
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    try:
        plt.style.use("seaborn-darkgrid")
    except OSError:
        pass  # default style is fine

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


# Data & model loading


def load_model(model_path: Path) -> Any:
    """Load a joblib-serialised model from disk.

    Parameters
    ----------
    model_path : Path
        Path to the ``.joblib`` file.

    Returns
    -------
    Any
        The deserialised model object.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    logger.info("Loading model from %s", model_path)
    return joblib.load(model_path)


def load_test_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load ``X_test.csv`` and ``y_test.csv`` from *data_dir*.

    Returns
    -------
    tuple
        ``(X_test, y_test)``
    """
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")
    logger.info("Loaded test set: %d rows × %d features", *X_test.shape)
    return X_test, y_test


# Plotting


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
) -> float:
    """Plot and save the ROC curve; return the AUC.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    save_path : Path
        Destination for the PNG image.

    Returns
    -------
    float
        Area under the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved to %s (AUC=%.4f)", save_path, auc_val)
    return auc_val


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
) -> float:
    """Plot and save the Precision-Recall curve; return the average precision.

    The PR curve is more informative than ROC when the positive class is
    extremely rare (0.13 % for PaySim fraud).

    Returns
    -------
    float
        Average precision score (area under the PR curve).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.step(
        recall, precision, where="post", color="#059669", lw=2, label=f"PR curve (AP = {ap:.4f})"
    )
    # Baseline = prevalence of positive class
    prevalence = y_true.mean()
    ax.axhline(
        prevalence,
        color="#9ca3af",
        lw=1,
        linestyle="--",
        label=f"Baseline (prevalence = {prevalence:.4f})",
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved to %s (AP=%.4f)", save_path, ap)
    return ap


def plot_feature_importance(
    model: Any,
    save_path: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of the top-N most important features.

    Uses *weight* (number of times a feature appears in trees) because it is
    the most intuitive importance type for stakeholder communication.
    """
    booster = model.get_booster()
    importance = booster.get_score(importance_type="weight")
    if not importance:
        logger.warning("No feature importance scores available; skipping plot.")
        return

    sorted_imp = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names, scores = zip(*sorted_imp)

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color="#6366f1", edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (weight)", fontsize=12)
    ax.set_title(f"Top-{top_n} Feature Importance", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance chart saved to %s", save_path)


# Business metrics


def compute_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraud_cost: float = 5_000.0,
    block_cost: float = 25.0,
) -> dict[str, Any]:
    """Translate confusion-matrix counts into business-relevant numbers.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted binary labels.
    fraud_cost : float
        Average monetary loss when a fraudulent transaction goes undetected
        (false negative).
    block_cost : float
        Average cost of incorrectly blocking a legitimate transaction
        (false positive) – customer friction, support, lost revenue.

    Returns
    -------
    dict
        Human-readable business metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_fraud = tp + fn
    total_legit = tn + fp

    fpr = fp / total_legit if total_legit else 0.0  # legit txns wrongly blocked
    fnr = fn / total_fraud if total_fraud else 0.0  # fraud missed

    cost_missed_fraud = fn * fraud_cost
    cost_blocked_legit = fp * block_cost
    total_cost = cost_missed_fraud + cost_blocked_legit

    return {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "fraud_cost_per_event": fraud_cost,
        "block_cost_per_event": block_cost,
        "estimated_cost_missed_fraud": round(cost_missed_fraud, 2),
        "estimated_cost_blocked_legit": round(cost_blocked_legit, 2),
        "estimated_total_cost": round(total_cost, 2),
    }


# Optimal threshold


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Find the probability threshold that maximises the F1 score.

    We sweep thresholds from the PR curve rather than using a fixed 0.5,
    because with extreme class imbalance a lower threshold often yields a
    better precision-recall trade-off.

    Returns
    -------
    tuple[float, float]
        ``(best_threshold, best_f1)``
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision & recall arrays are one element longer than thresholds;
    # drop the last element to align.
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    logger.info("Optimal threshold: %.4f  →  F1: %.4f", best_threshold, best_f1)
    return best_threshold, best_f1


# CLI entry-point


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained fraud-detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/xgboost_fraud.joblib"),
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing X_test.csv and y_test.csv.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save evaluation plots and report.",
    )
    parser.add_argument(
        "--fraud-cost",
        type=float,
        default=5_000.0,
        help="Average cost of a missed fraud (false negative).",
    )
    parser.add_argument(
        "--block-cost",
        type=float,
        default=25.0,
        help="Average cost of blocking a legit transaction (false positive).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the full evaluation pipeline."""
    args = parse_args(argv)

    try:
        args.reports_dir.mkdir(parents=True, exist_ok=True)

        # Load
        model = load_model(args.model)
        X_test, y_test = load_test_data(args.data_dir)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Classification report
        report_text = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(report_text)

        # Confusion matrix (console)
        cm = confusion_matrix(y_test, y_pred)
        cm_pct = cm.astype(float) / cm.sum() * 100
        print("Confusion Matrix:")
        print(
            f"  TN: {cm[0, 0]:>10,}  ({cm_pct[0, 0]:6.2f}%)   "
            f"FP: {cm[0, 1]:>10,}  ({cm_pct[0, 1]:6.2f}%)"
        )
        print(
            f"  FN: {cm[1, 0]:>10,}  ({cm_pct[1, 0]:6.2f}%)   "
            f"TP: {cm[1, 1]:>10,}  ({cm_pct[1, 1]:6.2f}%)"
        )
        print()

        # Curves
        roc_auc = plot_roc_curve(y_test.values, y_prob, args.reports_dir / "roc_curve.png")
        pr_auc = plot_pr_curve(y_test.values, y_prob, args.reports_dir / "pr_curve.png")
        plot_feature_importance(model, args.reports_dir / "feature_importance.png")

        # Business metrics
        biz = compute_business_metrics(
            y_test.values,
            y_pred,
            fraud_cost=args.fraud_cost,
            block_cost=args.block_cost,
        )
        print("Business Metrics:")
        print(
            f"  False Positive Rate : {biz['false_positive_rate']:.6f}  "
            f"(legit transactions blocked)"
        )
        print(f"  False Negative Rate : {biz['false_negative_rate']:.6f}  (fraud missed)")
        print(f"  Est. cost missed fraud   : ${biz['estimated_cost_missed_fraud']:>12,.2f}")
        print(f"  Est. cost blocked legit  : ${biz['estimated_cost_blocked_legit']:>12,.2f}")
        print(f"  Est. total cost          : ${biz['estimated_total_cost']:>12,.2f}")
        print()

        # Optimal threshold
        best_thresh, best_f1 = find_optimal_threshold(y_test.values, y_prob)

        # Evaluate at the optimal threshold for comparison
        (y_prob >= best_thresh).astype(int)
        f1_default = float(f1_score(y_test, y_pred))
        print(f"Optimal Threshold : {best_thresh:.4f}")
        print(f"  F1 at 0.50 threshold : {f1_default:.4f}")
        print(f"  F1 at optimal thresh : {best_f1:.4f}")
        print("=" * 60 + "\n")

        # Save JSON report
        eval_report: dict[str, Any] = {
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=["Legitimate", "Fraud"],
                output_dict=True,
            ),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "business_metrics": biz,
            "optimal_threshold": best_thresh,
            "f1_at_default_threshold": f1_default,
            "f1_at_optimal_threshold": best_f1,
        }
        report_path = args.reports_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(eval_report, fh, indent=2, default=str)
        logger.info("Evaluation report saved to %s", report_path)

        logger.info("Evaluation complete ✓")

    except Exception:
        logger.exception("Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
