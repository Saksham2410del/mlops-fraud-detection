# FastAPI app that serves the trained model.
#
#   POST /predict  -> Approve / Block decision + 3-sentence investigation report
#   GET  /health   -> is the model loaded?
#
# Run locally:
#   uvicorn src.api:app --reload

import time
from contextlib import asynccontextmanager
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import FRAUD_THRESHOLD, MODEL_PATH
from src.features import transaction_to_features
from src.report import generate_report

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup instead of on every request
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"WARNING: no model at {MODEL_PATH}, run the training pipeline first")
    yield


app = FastAPI(title="FraudGuard API", lifespan=lifespan)


class Transaction(BaseModel):
    # Same fields as a row of the PaySim dataset
    step: int = Field(ge=0)
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(gt=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


class Prediction(BaseModel):
    decision: Literal["Approve", "Block"]
    fraud_probability: float
    investigation_report: str
    report_source: str
    model_latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=Prediction)
def predict(txn: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    data = txn.model_dump()
    features = transaction_to_features(data)
    probability = float(model.predict_proba(features)[0, 1])
    model_ms = (time.perf_counter() - start) * 1000

    decision = "Block" if probability >= FRAUD_THRESHOLD else "Approve"
    report, source = generate_report(data, decision, probability)

    return Prediction(
        decision=decision,
        fraud_probability=round(probability, 4),
        investigation_report=report,
        report_source=source,
        model_latency_ms=round(model_ms, 2),
    )
