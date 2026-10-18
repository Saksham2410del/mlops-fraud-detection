# Step 3 of the pipeline: score the model on the untouched test set.
# Writes reports/metrics.json, reports/classification_report.txt and two plots.
#
# Usage:
#   python -m src.evaluate

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no GUI needed, we only save images
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    FRAUD_THRESHOLD,
    MODEL_PATH,
    PROCESSED_DIR,
    REPORTS_DIR,
    SMOTE_RATIO,
    TEST_SIZE,
)


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "fraud_caught": int(tp),
        "fraud_total": int(tp + fn),
        "false_alarms": int(fp),
        "test_rows": int(len(y_true)),
    }


def save_plots(model, y_true, y_pred, feature_names, reports_dir):
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Legitimate", "Fraud"], cmap="Blues", values_format=","
    )
    plt.title("Confusion Matrix (test set)")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png")
    plt.close()

    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    importance.plot(kind="barh", figsize=(8, 6), color="steelblue")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(reports_dir / "feature_importance.png")
    plt.close()


def log_to_mlflow(model, metrics, model_path, reports_dir):
    # Every evaluation becomes one MLflow run, so different experiments
    # (SMOTE ratio, tree depth, ...) can be compared side by side with `mlflow ui`.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraudguard")

    xgb_params = model.get_params()
    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": xgb_params["n_estimators"],
            "max_depth": xgb_params["max_depth"],
            "learning_rate": xgb_params["learning_rate"],
            "scale_pos_weight": xgb_params["scale_pos_weight"],
            "smote_ratio": SMOTE_RATIO,
            "test_size": TEST_SIZE,
            "threshold": FRAUD_THRESHOLD,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(reports_dir), artifact_path="reports")
        mlflow.log_artifact(str(model_path), artifact_path="model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=PROCESSED_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model_path)
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= FRAUD_THRESHOLD).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])
    with open(reports_dir / "classification_report.txt", "w") as f:
        f.write(report)

    save_plots(model, y_test, y_pred, X_test.columns, reports_dir)

    log_to_mlflow(model, metrics, args.model_path, reports_dir)

    print(report)
    print(f"F1-score : {metrics['f1']}")
    print(f"Recall   : {metrics['recall']:.1%}")
    print(f"Caught {metrics['fraud_caught']:,} out of {metrics['fraud_total']:,} fraud cases")
    print(f"Reports saved in {reports_dir}")


if __name__ == "__main__":
    main()
