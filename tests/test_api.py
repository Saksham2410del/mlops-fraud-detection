import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api as api
from src.report import keep_three_sentences, template_report


class FakeModel:
    def __init__(self, prob):
        self.prob = prob

    def predict_proba(self, X):
        assert X.shape == (1, 16)
        return np.array([[1 - self.prob, self.prob]])


@pytest.fixture
def client(monkeypatch):
    # don't call a real LLM during tests
    monkeypatch.setattr(api, "generate_report", lambda txn, d, p: ("One. Two. Three.", "test", ["FP-01"]))
    return TestClient(api.app)


TXN = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181000.0,
    "oldbalanceOrg": 181000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}


def test_health(client):
    api.model = FakeModel(0.1)
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_blocks_high_probability(client):
    api.model = FakeModel(0.97)
    res = client.post("/predict", json=TXN).json()
    assert res["decision"] == "Block"
    assert res["fraud_probability"] == 0.97
    assert res["investigation_report"] == "One. Two. Three."
    assert res["policies_used"] == ["FP-01"]


def test_approves_low_probability(client):
    api.model = FakeModel(0.02)
    res = client.post("/predict", json={**TXN, "type": "PAYMENT"}).json()
    assert res["decision"] == "Approve"


def test_rejects_unknown_type(client):
    api.model = FakeModel(0.1)
    assert client.post("/predict", json={**TXN, "type": "BITCOIN"}).status_code == 422


def test_rejects_negative_amount(client):
    api.model = FakeModel(0.1)
    assert client.post("/predict", json={**TXN, "amount": -5}).status_code == 422


def test_503_when_model_missing(client):
    api.model = None
    assert client.post("/predict", json=TXN).status_code == 503


REPORT_TXN = {
    "type": "TRANSFER",
    "amount": 5000.0,
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}


def test_long_llm_output_is_cut_to_three_sentences():
    text = "First one. Second one! Third one? Fourth one. Fifth."
    assert keep_three_sentences(text) == "First one. Second one! Third one?"


def test_template_report_has_three_sentences():
    report = template_report(REPORT_TXN, "Block", 0.93)
    assert report.count(". ") + 1 == 3
    assert "high risk" in report
    assert "emptied" in report
