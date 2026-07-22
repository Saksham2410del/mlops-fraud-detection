"""
Tests for the FastAPI Fraud Detection API (``src/predict.py``)
==============================================================

All tests use a lightweight synthetic model so they run in < 2 seconds
without requiring the real PaySim-trained model.
"""

from __future__ import annotations

# We need to patch the model before importing the app
import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

# Fixtures


@pytest.fixture(scope="module")
def _tiny_model(tmp_path_factory):
    """Train a tiny XGBoost model with 16 features matching FEATURE_COLUMNS."""
    n_features = 16  # Must match len(FEATURE_COLUMNS) in predict.py
    X, y = make_classification(
        n_samples=200,
        n_features=n_features,
        n_informative=8,
        n_classes=2,
        random_state=42,
    )
    model = XGBClassifier(
        n_estimators=5,
        max_depth=2,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y)

    model_dir = tmp_path_factory.mktemp("models")
    model_path = model_dir / "xgboost_fraud.joblib"
    joblib.dump(model, model_path)
    return model, model_path


@pytest.fixture(scope="module")
def client(_tiny_model):
    """Create a TestClient with the tiny model loaded."""
    model, model_path = _tiny_model

    # Patch MODEL_PATH and pre-load the model store
    import src.predict as predict_module

    predict_module.MODEL_PATH = model_path
    predict_module.model_store["model"] = model

    return TestClient(predict_module.app)


@pytest.fixture
def legitimate_transaction():
    """A transaction that looks like a normal payment."""
    return {
        "step": 1,
        "type": "PAYMENT",
        "amount": 50.0,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 9950.0,
        "oldbalanceDest": 5000.0,
        "newbalanceDest": 5050.0,
    }


@pytest.fixture
def suspicious_transaction():
    """A transaction pattern commonly associated with fraud:
    large transfer draining the account completely."""
    return {
        "step": 1,
        "type": "TRANSFER",
        "amount": 500000.0,
        "oldbalanceOrg": 500000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }


# Health endpoint tests


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """Health endpoint should always return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client):
        """When model is loaded, status should be 'healthy'."""
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_response_structure(self, client):
        """Response should contain expected fields."""
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_path" in data


# Predict endpoint tests


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_returns_200(self, client, legitimate_transaction):
        """Predict endpoint should return 200 for valid input."""
        response = client.post("/predict", json=legitimate_transaction)
        assert response.status_code == 200

    def test_predict_response_structure(self, client, legitimate_transaction):
        """Response must contain all expected fields."""
        data = client.post("/predict", json=legitimate_transaction).json()
        assert "transaction_id" in data
        assert "decision" in data
        assert "fraud_probability" in data
        assert "threshold_used" in data
        assert "processing_time_ms" in data

    def test_predict_decision_is_valid(self, client, legitimate_transaction):
        """Decision must be either 'Approve' or 'Block'."""
        data = client.post("/predict", json=legitimate_transaction).json()
        assert data["decision"] in ("Approve", "Block")

    def test_predict_probability_range(self, client, legitimate_transaction):
        """Fraud probability must be between 0 and 1."""
        data = client.post("/predict", json=legitimate_transaction).json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_processing_time_positive(self, client, legitimate_transaction):
        """Processing time should be a positive number."""
        data = client.post("/predict", json=legitimate_transaction).json()
        assert data["processing_time_ms"] > 0

    def test_predict_transaction_id_format(self, client, legitimate_transaction):
        """Transaction ID should start with 'txn_'."""
        data = client.post("/predict", json=legitimate_transaction).json()
        assert data["transaction_id"].startswith("txn_")

    def test_predict_with_all_transaction_types(self, client):
        """All valid transaction types should be accepted."""
        for txn_type in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
            txn = {
                "step": 1,
                "type": txn_type,
                "amount": 100.0,
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 900.0,
                "oldbalanceDest": 500.0,
                "newbalanceDest": 600.0,
            }
            response = client.post("/predict", json=txn)
            assert response.status_code == 200, f"Failed for type {txn_type}"


# Validation / error tests


class TestPredictValidation:
    """Tests for input validation on POST /predict."""

    def test_predict_missing_field_returns_422(self, client):
        """Missing required fields should return 422 Unprocessable Entity."""
        incomplete = {
            "step": 1,
            "type": "TRANSFER",
            # amount is missing
        }
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_invalid_type_returns_422(self, client):
        """Invalid transaction type should return 422."""
        txn = {
            "step": 1,
            "type": "INVALID_TYPE",
            "amount": 100.0,
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 900.0,
            "oldbalanceDest": 500.0,
            "newbalanceDest": 600.0,
        }
        response = client.post("/predict", json=txn)
        assert response.status_code == 422

    def test_predict_negative_amount_returns_422(self, client):
        """Negative amounts should be rejected."""
        txn = {
            "step": 1,
            "type": "TRANSFER",
            "amount": -100.0,
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 900.0,
            "oldbalanceDest": 500.0,
            "newbalanceDest": 600.0,
        }
        response = client.post("/predict", json=txn)
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self, client):
        """Empty request body should return 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422


# Batch endpoint tests


class TestBatchPredictEndpoint:
    """Tests for POST /predict/batch."""

    def test_batch_predict_returns_200(self, client, legitimate_transaction):
        """Batch endpoint should return 200 for valid input."""
        batch = {"transactions": [legitimate_transaction, legitimate_transaction]}
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200

    def test_batch_predict_response_structure(self, client, legitimate_transaction):
        """Response should contain all expected fields."""
        batch = {"transactions": [legitimate_transaction]}
        data = client.post("/predict/batch", json=batch).json()
        assert "predictions" in data
        assert "total_transactions" in data
        assert "flagged_count" in data
        assert "processing_time_ms" in data

    def test_batch_predict_correct_count(self, client, legitimate_transaction):
        """Number of predictions should match number of input transactions."""
        n = 5
        batch = {"transactions": [legitimate_transaction] * n}
        data = client.post("/predict/batch", json=batch).json()
        assert data["total_transactions"] == n
        assert len(data["predictions"]) == n

    def test_batch_predict_empty_list_returns_422(self, client):
        """Empty transaction list should return 422."""
        response = client.post("/predict/batch", json={"transactions": []})
        assert response.status_code == 422


# Metrics endpoint tests


class TestMetricsEndpoint:
    """Tests for GET /metrics (Prometheus)."""

    def test_metrics_returns_200(self, client):
        """Prometheus metrics endpoint should return 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_custom_counters(self, client, legitimate_transaction):
        """After a prediction, custom metrics should appear."""
        # Make a prediction first to generate metrics
        client.post("/predict", json=legitimate_transaction)
        response = client.get("/metrics")
        text = response.text
        assert "fraud_predictions_total" in text
        assert "prediction_latency_seconds" in text
