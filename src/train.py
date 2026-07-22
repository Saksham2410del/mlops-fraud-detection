"""
XGBoost Training Script with MLflow Integration
================================================

Trains an XGBoost classifier on the SMOTE-augmented training set, logs every
artefact (metrics, plots, model binary) to MLflow, and registers the model
in the MLflow Model Registry.

Usage:
    python src/train.py \
        --data-dir data/processed/ \
        --model-dir models/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Use a non-interactive backend so the script works headlessly on CI servers.
matplotlib.use("Agg")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# Data loading


def load_processed_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Read the four CSVs produced by ``preprocess.py``."""
    required = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for name in required:
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"Missing {data_dir / name}")

    logger.info("Loading processed data from %s", data_dir)
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")

    logger.info(
        "Train: %d rows | Test: %d rows | Features: %d",
        len(X_train),
        len(X_test),
        X_train.shape[1],
    )
    return X_train, X_test, y_train, y_test


# Plotting helpers


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
) -> None:
    """Save a labelled confusion-matrix heatmap as a PNG."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    labels = ["Legitimate", "Fraud"]
    tick_marks = np.arange(len(labels))
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n({cm_pct[i, j]:.2f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def _plot_feature_importance(
    model: xgb.XGBClassifier,
    save_path: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of the top-N most important features."""
    importance = model.get_booster().get_score(importance_type="weight")
    sorted_imp = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names, scores = zip(*sorted_imp) if sorted_imp else ([], [])

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color="#3b82f6", edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (weight)")
    ax.set_title(f"Top-{top_n} Feature Importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", save_path)


# Training


def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    hyperparams: dict | None = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier and return the fitted model."""
    defaults: dict = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": 10,  # extra push for minority class
        "eval_metric": "aucpr",  # area under PR curve (imbalance-aware)
        "objective": "binary:logistic",
        "tree_method": "hist",  # histogram-based for speed
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 1,
    }
    if hyperparams:
        defaults.update(hyperparams)

    logger.info("Initialising XGBClassifier with %s", defaults)
    model = xgb.XGBClassifier(**defaults)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    logger.info("Training complete")
    return model


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Return a dictionary of classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost fraud-detection model with MLflow tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--experiment-name", type=str, default="fraud-detection")
    parser.add_argument("--register-model", type=str, default="fraud-detection-xgboost")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Orchestrate data loading → training → evaluation → MLflow logging."""
    args = parse_args(argv)

    try:
        # Data
        X_train, X_test, y_train, y_test = load_processed_data(args.data_dir)

        # Ensure output dirs exist
        args.model_dir.mkdir(parents=True, exist_ok=True)
        args.reports_dir.mkdir(parents=True, exist_ok=True)

        # MLflow setup
        # FORCE MLflow to connect to your live UI server on port 5000
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(args.experiment_name)

        with mlflow.start_run(run_name="xgboost-training") as run:
            logger.info("MLflow run id: %s", run.info.run_id)

            # Train
            model = train_model(X_train, X_test, y_train, y_test)

            # Log hyperparameters
            mlflow.log_params(model.get_params())

            # Evaluate
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test.values, y_pred, y_prob)

            # Log scalar metrics
            mlflow.log_metrics(metrics)
            logger.info("Metrics: %s", json.dumps(metrics, indent=2))

            # Plots & text artefacts
            cm_path = args.reports_dir / "confusion_matrix.png"
            fi_path = args.reports_dir / "feature_importance.png"
            _plot_confusion_matrix(y_test.values, y_pred, cm_path)
            _plot_feature_importance(model, fi_path)

            report_text = classification_report(
                y_test, y_pred, target_names=["Legitimate", "Fraud"]
            )
            report_path = args.reports_dir / "classification_report.txt"
            report_path.write_text(report_text, encoding="utf-8")
            logger.info("Classification report:\n%s", report_text)

            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(fi_path))
            mlflow.log_artifact(str(report_path))

            # Log & register model
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                registered_model_name=args.register_model,
            )
            logger.info(
                "Model registered as '%s' in MLflow Model Registry",
                args.register_model,
            )

            # Attempt to promote the latest version to Staging
            try:
                from mlflow.tracking import MlflowClient

                client = MlflowClient()
                latest_versions = client.get_latest_versions(args.register_model, stages=["None"])
                if latest_versions:
                    version = latest_versions[0].version
                    client.transition_model_version_stage(
                        name=args.register_model,
                        version=version,
                        stage="Staging",
                    )
                    logger.info("Promoted model version %s to 'Staging'", version)
            except Exception as exc:
                logger.warning(
                    "Could not promote model to Staging (this is non-fatal): %s",
                    exc,
                )

            # Save locally
            model_path = args.model_dir / "xgboost_fraud.joblib"
            joblib.dump(model, model_path)
            logger.info("Model saved locally to %s", model_path)

            metrics_path = Path("metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
            logger.info("Metrics written to %s", metrics_path)

            # Summary
            print("\n" + "=" * 60)
            print("TRAINING SUMMARY")
            print("=" * 60)
            for k, v in metrics.items():
                print(f"  {k:<12s}: {v:.4f}")
            print(f"  MLflow run : {run.info.run_id}")
            print(f"  Model path : {model_path}")
            print("=" * 60 + "\n")

        logger.info("Training pipeline complete ✓")

    except Exception:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
