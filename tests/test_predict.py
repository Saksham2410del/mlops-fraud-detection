import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.predict import app, model_store


class DummyModel:
    def __init__(self, proba):
        self.proba = proba

    def predict_proba(self, X):
        return np.array([[1.0 - self.proba, self.proba]])


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    model_store["model"] = DummyModel(0.1)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_approve(client):
    model_store["model"] = DummyModel(0.1)
    payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 50.0,
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 950.0,
        "oldbalanceDest": 200.0,
        "newbalanceDest": 250.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "Approve"
    assert data["fraud_probability"] == 0.1


def test_predict_block(client):
    model_store["model"] = DummyModel(0.9)
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 500000.0,
        "oldbalanceOrg": 500000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "Block"
    assert data["fraud_probability"] == 0.9


def test_predict_invalid_type(client):
    model_store["model"] = DummyModel(0.1)
    payload = {
        "step": 1,
        "type": "INVALID_TYPE",
        "amount": 100.0,
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 900.0,
        "oldbalanceDest": 500.0,
        "newbalanceDest": 600.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
