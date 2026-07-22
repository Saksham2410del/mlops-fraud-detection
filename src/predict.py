"""
FastAPI Inference Server for Fraud Detection
=============================================

Serves the trained XGBoost fraud-detection model as a REST API with:
  * ``POST /predict``  – Classify a transaction as "Approve" or "Block"
  * ``POST /explain``  – (Week 7) SHAP + LLM explanation (lazy-loaded)
  * ``GET  /health``   – Liveness / readiness probe
  * ``GET  /metrics``  – Prometheus metrics (auto-instrumented)

Usage::

    uvicorn src.predict:app --host 0.0.0.0 --port 8000 --reload

Then visit http://localhost:8000/docs for Swagger UI.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("predict")

# Configuration
MODEL_PATH = Path("models/xgboost_fraud.joblib")
FRAUD_THRESHOLD = 0.5  # Default; evaluate.py can find optimal

FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    # One-hot encoded transaction types
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    # Engineered features
    "balance_diff_orig",
    "balance_diff_dest",
    "amount_ratio",
    "is_orig_balance_zero",
    "is_dest_balance_zero",
]

# Custom Prometheus metrics
PREDICTION_COUNTER = Counter(
    "fraud_predictions_total",
    "Total number of fraud predictions",
    ["decision"],  # "Approve" or "Block"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent generating a prediction",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Pydantic models


class TransactionRequest(BaseModel):
    """Input schema for the /predict endpoint.

    Maps directly to the raw PaySim transaction columns.  Feature
    engineering (balance diffs, amount ratio, etc.) is done server-side
    so the caller doesn't need to know about internal features.
    """

    step: int = Field(..., description="Time step (1 step = 1 hour of simulation)", ge=0)
    type: str = Field(
        ...,
        description="Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER",
    )
    amount: float = Field(..., description="Transaction amount in local currency", gt=0)
    oldbalanceOrg: float = Field(..., description="Sender balance before transaction", ge=0)
    newbalanceOrig: float = Field(..., description="Sender balance after transaction", ge=0)
    oldbalanceDest: float = Field(..., description="Receiver balance before transaction", ge=0)
    newbalanceDest: float = Field(..., description="Receiver balance after transaction", ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "step": 1,
                    "type": "TRANSFER",
                    "amount": 181000.0,
                    "oldbalanceOrg": 181000.0,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for the /predict endpoint."""

    transaction_id: str = Field(..., description="Auto-generated transaction ID")
    decision: str = Field(..., description="'Approve' or 'Block'")
    fraud_probability: float = Field(
        ..., description="Probability that the transaction is fraudulent (0.0 – 1.0)"
    )
    threshold_used: float = Field(..., description="Decision threshold applied")
    processing_time_ms: float = Field(..., description="Server-side inference time in ms")


class HealthResponse(BaseModel):
    """Output schema for the /health endpoint."""

    status: str
    model_loaded: bool
    model_path: str


class BatchTransactionRequest(BaseModel):
    """Input schema for the /predict/batch endpoint."""

    transactions: list[TransactionRequest] = Field(
        ..., description="List of transactions to classify", min_length=1, max_length=1000
    )


class BatchPredictionResponse(BaseModel):
    """Output schema for the /predict/batch endpoint."""

    predictions: list[PredictionResponse]
    total_transactions: int
    flagged_count: int
    processing_time_ms: float


# Global state (populated at startup)
model_store: dict[str, Any] = {"model": None}


# Feature engineering (mirrors preprocess.py logic)


def _prepare_features(txn: TransactionRequest) -> pd.DataFrame:
    """Transform a raw transaction request into the feature vector expected
    by the model.

    This replicates the feature engineering from ``preprocess.py`` so that
    callers can send raw transaction fields without worrying about internal
    transformations.
    """
    # One-hot encode transaction type
    valid_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    type_flags = {f"type_{t}": int(txn.type.upper() == t) for t in valid_types}

    # Engineered features
    features = {
        "step": txn.step,
        "amount": txn.amount,
        "oldbalanceOrg": txn.oldbalanceOrg,
        "newbalanceOrig": txn.newbalanceOrig,
        "oldbalanceDest": txn.oldbalanceDest,
        "newbalanceDest": txn.newbalanceDest,
        "balance_diff_orig": txn.oldbalanceOrg - txn.newbalanceOrig,
        "balance_diff_dest": txn.newbalanceDest - txn.oldbalanceDest,
        "amount_ratio": txn.amount / (txn.oldbalanceOrg + 1),  # +1 avoids /0
        "is_orig_balance_zero": int(txn.oldbalanceOrg == 0),
        "is_dest_balance_zero": int(txn.oldbalanceDest == 0),
        **type_flags,
    }

    # Return a single-row DataFrame with columns in training order
    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


def _generate_txn_id() -> str:
    """Generate a simple unique transaction ID based on timestamp."""
    import uuid

    return f"txn_{uuid.uuid4().hex[:12]}"


# App lifecycle


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once at startup; clean up on shutdown."""
    if MODEL_PATH.exists():
        logger.info("Loading model from %s", MODEL_PATH)
        model_store["model"] = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully ✓")
    else:
        logger.warning(
            "Model file not found at %s — /predict will return 503 until a model is available.",
            MODEL_PATH,
        )
    yield
    logger.info("Shutting down — releasing model resources")
    model_store["model"] = None


# FastAPI app

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time fraud detection for financial transactions. "
        "Powered by XGBoost with SMOTE-balanced training, MLflow tracking, "
        "and Prometheus observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Instrument with Prometheus (auto-exposes /metrics)
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app, endpoint="/metrics")

# Mount the SHAP + LLM explainability router (Week 7)
try:
    from src.explain import router as explain_router

    app.include_router(explain_router)
    logger.info("Explainability endpoint (/explain) mounted ✓")
except ImportError:
    logger.warning("explain module not available — /explain endpoint disabled")


# Endpoints


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    return HealthResponse(
        status="healthy" if model_store["model"] is not None else "degraded",
        model_loaded=model_store["model"] is not None,
        model_path=str(MODEL_PATH),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(transaction: TransactionRequest):
    """Classify a single transaction as 'Approve' or 'Block'.

    The endpoint applies the same feature engineering as the training pipeline
    (balance diffs, amount ratio, one-hot encoding) so callers only need to
    provide the raw transaction fields.

    Returns the fraud probability and binary decision using the configured
    threshold (default 0.5; adjust via the ``FRAUD_THRESHOLD`` env var or
    the optimal threshold from ``evaluate.py``).
    """
    if model_store["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure models/xgboost_fraud.joblib exists.",
        )

    # Validate transaction type
    valid_types = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    if transaction.type.upper() not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid transaction type '{transaction.type}'. Must be one of: {valid_types}",
        )

    start = time.perf_counter()

    # Prepare features and predict
    features_df = _prepare_features(transaction)
    proba = model_store["model"].predict_proba(features_df)[0, 1]
    decision = "Block" if proba >= FRAUD_THRESHOLD else "Approve"

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Update Prometheus counters
    PREDICTION_COUNTER.labels(decision=decision).inc()
    PREDICTION_LATENCY.observe(elapsed_ms / 1000)  # histogram expects seconds

    txn_id = _generate_txn_id()

    logger.info(
        "Prediction: %s | prob=%.4f | decision=%s | %.1fms",
        txn_id,
        proba,
        decision,
        elapsed_ms,
    )

    return PredictionResponse(
        transaction_id=txn_id,
        decision=decision,
        fraud_probability=round(float(proba), 6),
        threshold_used=FRAUD_THRESHOLD,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(batch: BatchTransactionRequest):
    """Classify a batch of transactions (up to 1000 at a time).

    Useful for bulk processing or back-testing against historical data.
    """
    if model_store["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure models/xgboost_fraud.joblib exists.",
        )

    start = time.perf_counter()
    predictions = []
    flagged = 0

    for txn in batch.transactions:
        features_df = _prepare_features(txn)
        proba = model_store["model"].predict_proba(features_df)[0, 1]
        decision = "Block" if proba >= FRAUD_THRESHOLD else "Approve"
        if decision == "Block":
            flagged += 1

        PREDICTION_COUNTER.labels(decision=decision).inc()

        predictions.append(
            PredictionResponse(
                transaction_id=_generate_txn_id(),
                decision=decision,
                fraud_probability=round(float(proba), 6),
                threshold_used=FRAUD_THRESHOLD,
                processing_time_ms=0,  # individual times not tracked in batch
            )
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    PREDICTION_LATENCY.observe(elapsed_ms / 1000)

    logger.info(
        "Batch prediction: %d transactions | %d flagged | %.1fms",
        len(predictions),
        flagged,
        elapsed_ms,
    )

    return BatchPredictionResponse(
        predictions=predictions,
        total_transactions=len(predictions),
        flagged_count=flagged,
        processing_time_ms=round(elapsed_ms, 2),
    )


# Run with: uvicorn src.predict:app --host 0.0.0.0 --port 8000
