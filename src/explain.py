"""
SHAP + LLM Explainability Endpoint
====================================

The "2026 X-Factor" — combines traditional ML (XGBoost) predictions with
GenAI (local Ollama LLM) to produce human-readable fraud investigation
reports.

Pipeline:
  1. Run transaction through XGBoost → get prediction
  2. Calculate SHAP values → identify top contributing features
  3. Send SHAP context + transaction to Ollama LLM
  4. LLM generates a 3-sentence "Fraud Investigation Report"

Usage:
    This module adds the ``/explain`` endpoint to the FastAPI app.
    Import and mount it in ``predict.py``, or run standalone for testing.

Dependencies:
    - shap (SHAP values for XGBoost)
    - ollama (local LLM inference via Ollama)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import APIRouter, HTTPException
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("explain")

# Router (mounted on the main FastAPI app)
router = APIRouter(tags=["Explainability"])

# Custom Prometheus metrics for LLM observability (2026 upgrade)
LLM_REQUEST_COUNTER = Counter(
    "llm_explanation_requests_total",
    "Total number of LLM explanation requests",
    ["status"],  # "success", "fallback", "error"
)

LLM_LATENCY = Histogram(
    "llm_explanation_latency_seconds",
    "Time spent generating LLM explanations",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

LLM_TOKEN_COUNTER = Counter(
    "llm_tokens_total",
    "Total tokens consumed by LLM explanations",
    ["direction"],  # "prompt" or "completion"
)


# Pydantic models


class ExplainRequest(BaseModel):
    """Input for the /explain endpoint — same fields as /predict."""

    step: int = Field(..., ge=0)
    type: str = Field(...)
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)

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


class ShapFeature(BaseModel):
    """A single SHAP feature contribution."""

    feature: str
    value: float
    shap_impact: float
    direction: str  # "increases fraud risk" or "decreases fraud risk"


class ExplainResponse(BaseModel):
    """Output from the /explain endpoint."""

    transaction_id: str
    decision: str
    fraud_probability: float
    top_features: list[ShapFeature]
    investigation_report: str = Field(..., description="LLM-generated natural language explanation")
    llm_model_used: str
    processing_time_ms: float


# Configuration
MODEL_PATH = Path("models/xgboost_fraud.joblib")
FRAUD_THRESHOLD = 0.5
OLLAMA_MODEL = "llama3.2"  # Lightweight, fast for inference
OLLAMA_FALLBACK_MODEL = "mistral"  # Fallback if llama3.2 not available

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


# Global state
explain_store: dict[str, Any] = {
    "model": None,
    "explainer": None,
}


def _load_model_and_explainer() -> None:
    """Lazily load the model and SHAP explainer on first /explain call."""
    if explain_store["model"] is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    logger.info("Loading model for SHAP explainer...")
    model = joblib.load(MODEL_PATH)
    explain_store["model"] = model

    # TreeExplainer is O(TLD) — very fast for tree-based models like XGBoost
    logger.info("Initialising SHAP TreeExplainer...")
    explain_store["explainer"] = shap.TreeExplainer(model)
    logger.info("SHAP explainer ready ✓")


# Feature engineering (same as predict.py)


def _prepare_features(txn: ExplainRequest) -> pd.DataFrame:
    """Transform raw transaction into the model's feature vector."""
    valid_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    type_flags = {f"type_{t}": int(txn.type.upper() == t) for t in valid_types}

    features = {
        "step": txn.step,
        "amount": txn.amount,
        "oldbalanceOrg": txn.oldbalanceOrg,
        "newbalanceOrig": txn.newbalanceOrig,
        "oldbalanceDest": txn.oldbalanceDest,
        "newbalanceDest": txn.newbalanceDest,
        "balance_diff_orig": txn.oldbalanceOrg - txn.newbalanceOrig,
        "balance_diff_dest": txn.newbalanceDest - txn.oldbalanceDest,
        "amount_ratio": txn.amount / (txn.oldbalanceOrg + 1),
        "is_orig_balance_zero": int(txn.oldbalanceOrg == 0),
        "is_dest_balance_zero": int(txn.oldbalanceDest == 0),
        **type_flags,
    }

    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


# SHAP analysis


def _compute_shap_values(features_df: pd.DataFrame) -> list[ShapFeature]:
    """Compute SHAP values and return the top contributing features.

    Returns features sorted by absolute SHAP impact (most important first).
    """
    shap_values = explain_store["explainer"](features_df)

    # For binary classification, shap_values.values may be 2D or 3D
    # depending on the XGBoost version. Handle both cases.
    sv = shap_values.values[0]
    if sv.ndim > 1:
        sv = sv[:, 1]  # Take the positive class SHAP values

    feature_names = features_df.columns.tolist()
    feature_values = features_df.values[0]

    # Sort by absolute SHAP value (most impactful first)
    sorted_indices = np.argsort(np.abs(sv))[::-1]

    top_features = []
    for idx in sorted_indices[:7]:  # Top 7 features
        impact = float(sv[idx])
        top_features.append(
            ShapFeature(
                feature=feature_names[idx],
                value=round(float(feature_values[idx]), 4),
                shap_impact=round(impact, 6),
                direction="increases fraud risk" if impact > 0 else "decreases fraud risk",
            )
        )

    return top_features


# LLM explanation


def _build_llm_prompt(
    txn: ExplainRequest,
    fraud_prob: float,
    decision: str,
    top_features: list[ShapFeature],
) -> str:
    """Construct a structured prompt for the LLM.

    The prompt instructs the LLM to base its explanation ONLY on the SHAP
    data provided — this prevents hallucination and keeps the explanation
    grounded in the model's actual reasoning.
    """
    feature_lines = "\n".join(
        f"  - {f.feature}: value={f.value}, SHAP impact={f.shap_impact:+.4f} ({f.direction})"
        for f in top_features
    )

    return f"""You are a fraud detection expert writing a brief investigation report
for a bank's security analyst.

TRANSACTION DETAILS:
  - Type: {txn.type}
  - Amount: ${txn.amount:,.2f}
  - Sender balance: ${txn.oldbalanceOrg:,.2f} → ${txn.newbalanceOrig:,.2f}
  - Receiver balance: ${txn.oldbalanceDest:,.2f} → ${txn.newbalanceDest:,.2f}

MODEL PREDICTION:
  - Decision: {decision}
  - Fraud probability: {fraud_prob:.1%}

TOP CONTRIBUTING FACTORS (from SHAP explainability analysis):
{feature_lines}

INSTRUCTIONS:
Write exactly 3 sentences:
1. State whether this transaction is likely fraudulent and the overall risk level.
2. Explain the 2-3 most important factors driving this prediction (use the SHAP data above).
3. Recommend a specific action for the security analyst.

Be concise, professional, and base your explanation ONLY on the data provided above."""


def _call_ollama(prompt: str) -> tuple[str, str]:
    """Call the local Ollama LLM and return (response_text, model_name).

    Tries the primary model first, falls back to alternative, and finally
    returns a template-based explanation if Ollama is unavailable.
    """
    try:
        import ollama

        # Try primary model
        for model_name in [OLLAMA_MODEL, OLLAMA_FALLBACK_MODEL]:
            try:
                logger.info("Calling Ollama model: %s", model_name)
                response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": 0.3,  # Low temp for consistent, factual output
                        "num_predict": 200,  # Limit response length
                    },
                )

                text = response["message"]["content"].strip()

                # Track token usage for LLM observability
                if "prompt_eval_count" in response:
                    LLM_TOKEN_COUNTER.labels(direction="prompt").inc(response["prompt_eval_count"])
                if "eval_count" in response:
                    LLM_TOKEN_COUNTER.labels(direction="completion").inc(response["eval_count"])

                LLM_REQUEST_COUNTER.labels(status="success").inc()
                return text, model_name

            except Exception as e:
                logger.warning("Model %s failed: %s — trying fallback", model_name, e)
                continue

    except ImportError:
        logger.warning("ollama package not installed")
    except Exception as e:
        logger.warning("Ollama connection failed: %s", e)

    # Fallback: template-based explanation (no LLM needed)
    LLM_REQUEST_COUNTER.labels(status="fallback").inc()
    return "", "fallback-template"


def _generate_fallback_explanation(
    txn: ExplainRequest,
    fraud_prob: float,
    decision: str,
    top_features: list[ShapFeature],
) -> str:
    """Generate a template-based explanation when Ollama is unavailable.

    This ensures the /explain endpoint always returns useful information,
    even without an LLM.
    """
    risk_level = "HIGH" if fraud_prob > 0.7 else "MEDIUM" if fraud_prob > 0.4 else "LOW"

    top_2 = top_features[:2]
    factor_text = " and ".join(f"{f.feature} (value={f.value}, {f.direction})" for f in top_2)

    action = (
        "Immediately escalate for manual review and temporarily block the account."
        if decision == "Block"
        else "No action required; transaction appears legitimate."
    )

    return (
        f"This {txn.type} transaction of ${txn.amount:,.2f} has been assessed as "
        f"{risk_level} risk with a {fraud_prob:.1%} fraud probability. "
        f"The primary factors driving this prediction are {factor_text}. "
        f"{action}"
    )


# Endpoint


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(transaction: ExplainRequest):
    """Generate an AI-powered fraud investigation report.

    Combines XGBoost prediction → SHAP explainability → LLM narration
    into a single response. If Ollama is not running, falls back to a
    template-based explanation.
    """
    start = time.perf_counter()

    try:
        _load_model_and_explainer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Validate transaction type
    valid_types = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    if transaction.type.upper() not in valid_types:
        raise HTTPException(status_code=422, detail=f"Invalid type. Must be one of: {valid_types}")

    # Step 1: Predict
    features_df = _prepare_features(transaction)
    fraud_prob = float(explain_store["model"].predict_proba(features_df)[0, 1])
    decision = "Block" if fraud_prob >= FRAUD_THRESHOLD else "Approve"

    # Step 2: SHAP values
    top_features = _compute_shap_values(features_df)

    # Step 3: LLM explanation
    prompt = _build_llm_prompt(transaction, fraud_prob, decision, top_features)

    llm_start = time.perf_counter()
    llm_text, model_used = _call_ollama(prompt)
    llm_elapsed = time.perf_counter() - llm_start
    LLM_LATENCY.observe(llm_elapsed)

    # If LLM returned empty (fallback), generate template explanation
    if not llm_text:
        llm_text = _generate_fallback_explanation(transaction, fraud_prob, decision, top_features)
        model_used = "fallback-template"

    elapsed_ms = (time.perf_counter() - start) * 1000

    import uuid

    txn_id = f"txn_{uuid.uuid4().hex[:12]}"

    logger.info(
        "Explanation: %s | prob=%.4f | decision=%s | llm=%s | %.1fms",
        txn_id,
        fraud_prob,
        decision,
        model_used,
        elapsed_ms,
    )

    return ExplainResponse(
        transaction_id=txn_id,
        decision=decision,
        fraud_probability=round(fraud_prob, 6),
        top_features=top_features,
        investigation_report=llm_text,
        llm_model_used=model_used,
        processing_time_ms=round(elapsed_ms, 2),
    )
