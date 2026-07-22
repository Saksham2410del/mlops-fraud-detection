"""Unit tests for the training pipeline.

Tests cover model training, prediction outputs, probability scores,
metric validation, model serialization, and feature importance extraction.
All tests use small synthetic data and complete in under 5 seconds each.
"""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

# Fixtures


@pytest.fixture
def synthetic_train_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a small synthetic training dataset with engineered + encoded features.

    Returns (X_train, y_train) with 800 rows, ~3 % fraud, and columns that
    mirror the output of the preprocessing pipeline (numeric + one-hot encoded).
    """
    rng = np.random.RandomState(42)
    n_rows = 800
    n_fraud = 24  # ~3 %

    X = pd.DataFrame(
        {
            "step": rng.randint(1, 744, size=n_rows),
            "amount": rng.uniform(10, 500_000, size=n_rows).round(2),
            "oldbalanceOrg": rng.uniform(0, 1_000_000, size=n_rows).round(2),
            "newbalanceOrig": rng.uniform(0, 1_000_000, size=n_rows).round(2),
            "oldbalanceDest": rng.uniform(0, 1_000_000, size=n_rows).round(2),
            "newbalanceDest": rng.uniform(0, 1_000_000, size=n_rows).round(2),
            "errorBalanceOrig": rng.uniform(-50_000, 50_000, size=n_rows).round(2),
            "errorBalanceDest": rng.uniform(-50_000, 50_000, size=n_rows).round(2),
            "type_CASH_IN": rng.choice([0, 1], size=n_rows, p=[0.8, 0.2]),
            "type_CASH_OUT": rng.choice([0, 1], size=n_rows, p=[0.7, 0.3]),
            "type_DEBIT": rng.choice([0, 1], size=n_rows, p=[0.9, 0.1]),
            "type_PAYMENT": rng.choice([0, 1], size=n_rows, p=[0.6, 0.4]),
            "type_TRANSFER": rng.choice([0, 1], size=n_rows, p=[0.8, 0.2]),
        }
    )

    y = pd.Series(np.zeros(n_rows, dtype=int), name="isFraud")
    fraud_indices = rng.choice(n_rows, size=n_fraud, replace=False)
    y.iloc[fraud_indices] = 1

    return X, y


@pytest.fixture
def synthetic_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a small synthetic test dataset (200 rows, same schema)."""
    rng = np.random.RandomState(99)
    n_rows = 200
    n_fraud = 6

    X = pd.DataFrame(
        {
            "step": rng.randint(1, 744, size=n_rows),
            "amount": rng.uniform(10, 500_000, size=n_rows).round(2),
            "oldbalanceOrg": rng.uniform(0, 1_000_000, size=n_rows).round(2),
            "newbalanceOrig": rng.uniform(0, 1_000_000, size=n_rows).round(2),
            "oldbalanceDest": rng.uniform(0, 1_000_000, size=n_rows).round(2),
            "newbalanceDest": rng.uniform(0, 1_000_000, size=n_rows).round(2),
            "errorBalanceOrig": rng.uniform(-50_000, 50_000, size=n_rows).round(2),
            "errorBalanceDest": rng.uniform(-50_000, 50_000, size=n_rows).round(2),
            "type_CASH_IN": rng.choice([0, 1], size=n_rows, p=[0.8, 0.2]),
            "type_CASH_OUT": rng.choice([0, 1], size=n_rows, p=[0.7, 0.3]),
            "type_DEBIT": rng.choice([0, 1], size=n_rows, p=[0.9, 0.1]),
            "type_PAYMENT": rng.choice([0, 1], size=n_rows, p=[0.6, 0.4]),
            "type_TRANSFER": rng.choice([0, 1], size=n_rows, p=[0.8, 0.2]),
        }
    )

    y = pd.Series(np.zeros(n_rows, dtype=int), name="isFraud")
    fraud_indices = rng.choice(n_rows, size=n_fraud, replace=False)
    y.iloc[fraud_indices] = 1

    return X, y


@pytest.fixture
def trained_model(
    synthetic_train_data: tuple[pd.DataFrame, pd.Series],
) -> XGBClassifier:
    """Train and return a lightweight XGBClassifier on the synthetic data."""
    X_train, y_train = synthetic_train_data

    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.3,
        scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# Tests


class TestModelTrainsSuccessfully:
    """Verify that the model can be trained on small synthetic data."""

    def test_model_trains_successfully(self, trained_model: XGBClassifier) -> None:
        """A trained XGBClassifier must be a fitted estimator."""
        # XGBoost sets n_classes_ after fitting
        assert hasattr(trained_model, "n_classes_"), (
            "Model does not appear to be fitted (missing n_classes_ attribute)"
        )
        assert trained_model.n_classes_ == 2, (
            f"Expected binary classifier, got {trained_model.n_classes_} classes"
        )

    def test_model_has_booster(self, trained_model: XGBClassifier) -> None:
        """The underlying booster must be available after fitting."""
        booster = trained_model.get_booster()
        assert booster is not None, "Booster is None after training"


class TestModelPredictsBinary:
    """Verify predictions are binary (0 or 1)."""

    def test_model_predicts_binary(
        self,
        trained_model: XGBClassifier,
        synthetic_test_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """All predictions must be either 0 or 1."""
        X_test, _ = synthetic_test_data
        predictions = trained_model.predict(X_test)

        unique_values = set(np.unique(predictions))
        assert unique_values.issubset({0, 1}), (
            f"Predictions contain non-binary values: {unique_values}"
        )

    def test_prediction_count_matches_input(
        self,
        trained_model: XGBClassifier,
        synthetic_test_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Number of predictions must match number of input rows."""
        X_test, _ = synthetic_test_data
        predictions = trained_model.predict(X_test)
        assert len(predictions) == len(X_test), (
            f"Prediction count {len(predictions)} != input count {len(X_test)}"
        )


class TestModelPredictsProbabilities:
    """Verify predict_proba returns valid probability distributions."""

    def test_model_predicts_probabilities(
        self,
        trained_model: XGBClassifier,
        synthetic_test_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Probabilities must be in [0, 1] and each row must sum to ~1.0."""
        X_test, _ = synthetic_test_data
        probas = trained_model.predict_proba(X_test)

        # Shape check: (n_samples, 2) for binary classification
        assert probas.shape == (len(X_test), 2), (
            f"Expected shape ({len(X_test)}, 2), got {probas.shape}"
        )

        # All values in [0, 1]
        assert np.all(probas >= 0.0), "Some probabilities are negative"
        assert np.all(probas <= 1.0), "Some probabilities exceed 1.0"

        # Rows must sum to ~1.0
        row_sums = probas.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"Row sums deviate from 1.0: min={row_sums.min():.6f}, max={row_sums.max():.6f}"
        )


class TestMetricsAreValid:
    """Verify all evaluation metrics are between 0 and 1."""

    def test_metrics_are_valid(
        self,
        trained_model: XGBClassifier,
        synthetic_test_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Accuracy, precision, recall, F1, and ROC-AUC must be in [0, 1]."""
        X_test, y_test = synthetic_test_data
        y_pred = trained_model.predict(X_test)
        y_proba = trained_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"Metric '{name}' is out of range [0, 1]: {value}"

    def test_metrics_are_numeric(
        self,
        trained_model: XGBClassifier,
        synthetic_test_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """All metric values must be finite floats (no NaN or Inf)."""
        X_test, y_test = synthetic_test_data
        y_pred = trained_model.predict(X_test)
        y_proba = trained_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        for name, value in metrics.items():
            assert np.isfinite(value), f"Metric '{name}' is not finite: {value}"


class TestModelSavesAndLoads:
    """Verify model can be serialized and deserialized with joblib."""

    def test_model_saves_and_loads(
        self,
        trained_model: XGBClassifier,
        synthetic_test_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """A saved-then-loaded model must produce identical predictions."""
        X_test, _ = synthetic_test_data

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_model.joblib"

            # Save
            joblib.dump(trained_model, model_path)
            assert model_path.exists(), f"Model file not created at {model_path}"
            assert model_path.stat().st_size > 0, "Model file is empty"

            # Load
            loaded_model = joblib.load(model_path)

            # Predictions must be identical
            original_preds = trained_model.predict(X_test)
            loaded_preds = loaded_model.predict(X_test)
            np.testing.assert_array_equal(
                original_preds,
                loaded_preds,
                err_msg="Loaded model predictions differ from original",
            )

            # Probabilities must be identical
            original_probas = trained_model.predict_proba(X_test)
            loaded_probas = loaded_model.predict_proba(X_test)
            np.testing.assert_array_almost_equal(
                original_probas,
                loaded_probas,
                decimal=10,
                err_msg="Loaded model probabilities differ from original",
            )


class TestFeatureImportanceExists:
    """Verify feature importances are available after training."""

    def test_feature_importance_exists(
        self,
        trained_model: XGBClassifier,
        synthetic_train_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Feature importances must be a non-empty array matching feature count."""
        X_train, _ = synthetic_train_data
        importances = trained_model.feature_importances_

        assert importances is not None, "feature_importances_ is None"
        assert len(importances) == X_train.shape[1], (
            f"Feature importance length {len(importances)} != feature count {X_train.shape[1]}"
        )

    def test_feature_importances_non_negative(self, trained_model: XGBClassifier) -> None:
        """All feature importance values must be >= 0."""
        importances = trained_model.feature_importances_
        assert np.all(importances >= 0), (
            f"Negative feature importances found: {importances[importances < 0]}"
        )

    def test_feature_importances_sum_to_one(self, trained_model: XGBClassifier) -> None:
        """Feature importances (weight-based) should approximately sum to 1."""
        importances = trained_model.feature_importances_
        total = importances.sum()
        assert np.isclose(total, 1.0, atol=0.01), (
            f"Feature importances sum to {total:.4f}, expected ~1.0"
        )
