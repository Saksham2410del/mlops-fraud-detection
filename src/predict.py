from contextlib import asynccontextmanager
from pathlib import Path
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.explain import router as explain_router

MODEL_PATH = Path("models/xgboost_fraud.joblib")
FRAUD_THRESHOLD = 0.5

FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "balance_diff_orig",
    "balance_diff_dest",
    "amount_ratio",
    "is_orig_balance_zero",
    "is_dest_balance_zero",
]

VALID_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}

model_store = {"model": None}

PREDICTION_COUNTER = Counter(
    "fraud_predictions_total",
    "Total number of fraud predictions",
    ["decision"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent generating a prediction",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


class TransactionRequest(BaseModel):
    step: int = Field(ge=0)
    type: str
    amount: float = Field(gt=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


class PredictionResponse(BaseModel):
    transaction_id: str
    decision: str
    fraud_probability: float
    threshold_used: float
    processing_time_ms: float


class BatchTransactionRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_transactions: int
    flagged_count: int
    processing_time_ms: float


def prepare_features(txn: TransactionRequest) -> pd.DataFrame:
    features = {
        "step": txn.step,
        "amount": txn.amount,
        "oldbalanceOrg": txn.oldbalanceOrg,
        "newbalanceOrig": txn.newbalanceOrig,
        "oldbalanceDest": txn.oldbalanceDest,
        "newbalanceDest": txn.newbalanceDest,
        "type_CASH_IN": int(txn.type.upper() == "CASH_IN"),
        "type_CASH_OUT": int(txn.type.upper() == "CASH_OUT"),
        "type_DEBIT": int(txn.type.upper() == "DEBIT"),
        "type_PAYMENT": int(txn.type.upper() == "PAYMENT"),
        "type_TRANSFER": int(txn.type.upper() == "TRANSFER"),
        "balance_diff_orig": txn.oldbalanceOrg - txn.newbalanceOrig,
        "balance_diff_dest": txn.newbalanceDest - txn.oldbalanceDest,
        "amount_ratio": txn.amount / (txn.oldbalanceOrg + 1),
        "is_orig_balance_zero": int(txn.oldbalanceOrg == 0),
        "is_dest_balance_zero": int(txn.oldbalanceDest == 0),
    }
    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        model_store["model"] = joblib.load(MODEL_PATH)
    yield
    model_store["model"] = None


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)
app.include_router(explain_router)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health")
def health():
    return {
        "status": "healthy" if model_store["model"] is not None else "degraded",
        "model_loaded": model_store["model"] is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionRequest):
    if model_store["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if txn.type.upper() not in VALID_TYPES:
        raise HTTPException(status_code=422, detail=f"Invalid transaction type: {txn.type}")

    start_time = time.perf_counter()
    features = prepare_features(txn)
    prob = float(model_store["model"].predict_proba(features)[0, 1])
    decision = "Block" if prob >= FRAUD_THRESHOLD else "Approve"
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    PREDICTION_COUNTER.labels(decision=decision).inc()
    PREDICTION_LATENCY.observe(elapsed_ms / 1000)

    return PredictionResponse(
        transaction_id=f"txn_{uuid.uuid4().hex[:12]}",
        decision=decision,
        fraud_probability=round(prob, 4),
        threshold_used=FRAUD_THRESHOLD,
        processing_time_ms=max(round(elapsed_ms, 2), 0.01),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchTransactionRequest):
    if model_store["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()
    predictions = []
    flagged = 0

    for txn in batch.transactions:
        if txn.type.upper() not in VALID_TYPES:
            raise HTTPException(status_code=422, detail=f"Invalid transaction type: {txn.type}")

        features = prepare_features(txn)
        prob = float(model_store["model"].predict_proba(features)[0, 1])
        decision = "Block" if prob >= FRAUD_THRESHOLD else "Approve"
        if decision == "Block":
            flagged += 1

        PREDICTION_COUNTER.labels(decision=decision).inc()
        predictions.append(
            PredictionResponse(
                transaction_id=f"txn_{uuid.uuid4().hex[:12]}",
                decision=decision,
                fraud_probability=round(prob, 4),
                threshold_used=FRAUD_THRESHOLD,
                processing_time_ms=0.01,
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    PREDICTION_LATENCY.observe(elapsed_ms / 1000)

    return BatchPredictionResponse(
        predictions=predictions,
        total_transactions=len(predictions),
        flagged_count=flagged,
        processing_time_ms=round(elapsed_ms, 2),
    )
