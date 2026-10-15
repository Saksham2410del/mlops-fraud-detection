import os
import json
import argparse
import logging
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

try:
    import mlflow
    import mlflow.xgboost
except ImportError:
    mlflow = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(data_dir):
    logger.info(f"Loading data from {data_dir}")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix"
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_feature_importance(model, feature_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    importances.plot(kind="barh", ax=ax, color="#3b82f6")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def train():
    parser = argparse.ArgumentParser(description="Train XGBoost fraud detection model")
    parser.add_argument("--data-dir", default="data/processed", help="Path to processed data")
    parser.add_argument("--model-dir", default="models", help="Directory to save model")
    parser.add_argument("--reports-dir", default="reports", help="Directory to save reports and plots")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Features: {X_train.shape[1]}")

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count / max(1, pos_count))

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "random_state": 42
    }

    logger.info("Training XGBoost classifier...")
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }

    # Save plots and reports
    cm_path = os.path.join(args.reports_dir, "confusion_matrix.png")
    fi_path = os.path.join(args.reports_dir, "feature_importance.png")
    report_path = os.path.join(args.reports_dir, "classification_report.txt")

    plot_confusion_matrix(y_test, y_pred, cm_path)
    plot_feature_importance(model, X_train.columns, fi_path)

    report_text = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])
    with open(report_path, "w") as f:
        f.write(report_text)

    # Save model
    model_path = os.path.join(args.model_dir, "xgboost_fraud.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save metrics.json
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to metrics.json")

    # MLflow logging
    if mlflow:
        try:
            mlflow.set_experiment("fraud-detection")
            with mlflow.start_run(run_name="xgboost-training"):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(cm_path)
                mlflow.log_artifact(fi_path)
                mlflow.log_artifact(report_path)
                mlflow.xgboost.log_model(model, artifact_path="model")
            logger.info("Logged run to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging skipped: {e}")
    else:
        logger.info("MLflow not installed; skipping MLflow experiment logging")

    # Console summary
    print("\n" + "=" * 50)
    print("TRAINING METRICS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    train()
