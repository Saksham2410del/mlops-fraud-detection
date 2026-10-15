import os
import json
import argparse
import logging
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_artifacts(model_path, data_dir):
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading test data from {data_dir}")
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()
    return model, X_test, y_test


def plot_roc(y_true, y_prob, roc_auc, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_pr(y_true, y_prob, avg_precision, save_path):
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, color="#059669", lw=2, label=f"PR Curve (AP = {avg_precision:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def compute_business_impact(y_true, y_pred, fraud_cost=5000.0, block_cost=25.0):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost_missed = float(fn * fraud_cost)
    cost_blocked = float(fp * block_cost)
    total_cost = cost_missed + cost_blocked

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "fraud_cases_caught": int(tp),
        "total_fraud_cases": int(tp + fn),
        "catch_rate": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "false_positive_rate": float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "cost_missed_fraud": cost_missed,
        "cost_blocked_legit": cost_blocked,
        "total_cost": total_cost
    }


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate trained fraud detection model")
    parser.add_argument("--model", default="models/xgboost_fraud.joblib", help="Path to saved model")
    parser.add_argument("--data-dir", default="data/processed", help="Path to processed test data")
    parser.add_argument("--reports-dir", default="reports", help="Directory for reports and plots")
    args = parser.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)

    model, X_test, y_test = load_artifacts(args.model, args.data_dir)

    logger.info("Generating predictions and probabilities...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    avg_precision = float(average_precision_score(y_test, y_prob))

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision
    }

    # Plot curves
    roc_path = os.path.join(args.reports_dir, "roc_curve.png")
    pr_path = os.path.join(args.reports_dir, "pr_curve.png")
    plot_roc(y_test, y_prob, roc_auc, roc_path)
    plot_pr(y_test, y_prob, avg_precision, pr_path)

    # Threshold optimization
    best_threshold, best_f1 = find_optimal_threshold(y_test, y_prob)

    # Business impact
    biz_metrics = compute_business_impact(y_test, y_pred)

    # Save JSON report
    report = {
        "metrics": metrics,
        "optimal_threshold": {
            "threshold": best_threshold,
            "best_f1": best_f1
        },
        "business_metrics": biz_metrics
    }

    report_path = os.path.join(args.reports_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved evaluation report to {report_path}")

    # Console summary
    print("\n" + "=" * 55)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1 Score:           {f1:.4f}")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"Average Precision:  {avg_precision:.4f}")
    print("-" * 55)
    print(f"Fraud Caught:       {biz_metrics['fraud_cases_caught']} / {biz_metrics['total_fraud_cases']} ({biz_metrics['catch_rate']:.2%})")
    print(f"False Positives:    {biz_metrics['false_positives']}")
    print(f"Missed Fraud:       {biz_metrics['false_negatives']}")
    print(f"Estimated Cost:     ${biz_metrics['total_cost']:,.2f}")
    print("-" * 55)
    print(f"Optimal Threshold:  {best_threshold:.4f} (Max F1: {best_f1:.4f})")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    evaluate()
