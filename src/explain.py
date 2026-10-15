from pathlib import Path
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram

router = APIRouter()

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

LLM_REQUESTS = Counter(
    "fraud_llm_requests_total",
    "Total LLM explanation calls",
    ["status"],
)
LLM_LATENCY = Histogram(
    "fraud_llm_latency_seconds",
    "LLM explanation latency in seconds",
)

explain_store = {"model": None, "explainer": None}


class ExplainRequest(BaseModel):
    step: int = Field(ge=0)
    type: str
    amount: float = Field(gt=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


class ShapFeature(BaseModel):
    feature: str
    value: float
    shap_impact: float
    direction: str


class ExplainResponse(BaseModel):
    transaction_id: str
    decision: str
    fraud_probability: float
    top_features: list[ShapFeature]
    investigation_report: str
    llm_model_used: str
    processing_time_ms: float


def load_model_and_explainer():
    if explain_store["explainer"] is None:
        import shap

        model = explain_store.get("model")
        if model is None:
            try:
                from src.predict import model_store
                model = model_store.get("model")
            except Exception:
                model = None

        if model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            model = joblib.load(MODEL_PATH)

        explain_store["model"] = model
        explain_store["explainer"] = shap.TreeExplainer(model)
    return explain_store["model"], explain_store["explainer"]


def prepare_features(txn: ExplainRequest) -> pd.DataFrame:
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


def compute_top_features(explainer, features_df: pd.DataFrame) -> list[ShapFeature]:
    shap_output = explainer(features_df)
    shap_vals = shap_output.values[0] if hasattr(shap_output, "values") else shap_output[0]
    if shap_vals.ndim > 1:
        shap_vals = shap_vals[:, 1]

    indices = np.argsort(np.abs(shap_vals))[::-1][:5]
    top = []
    for idx in indices:
        impact = float(shap_vals[idx])
        top.append(
            ShapFeature(
                feature=FEATURE_COLUMNS[idx],
                value=round(float(features_df.iloc[0, idx]), 4),
                shap_impact=round(impact, 4),
                direction="increases fraud risk" if impact > 0 else "decreases fraud risk",
            )
        )
    return top


def fallback_report(txn: ExplainRequest, decision: str, prob: float, top_features: list[ShapFeature]) -> str:
    risk = "high" if prob >= 0.7 else "moderate" if prob >= 0.4 else "low"
    drivers = " and ".join(f"{f.feature} ({f.direction})" for f in top_features[:2])
    action = (
        "Immediate hold recommended pending customer verification."
        if decision == "Block"
        else "Transaction appears safe to process."
    )
    return (
        f"This {txn.type} transaction of ${txn.amount:,.2f} presents a {risk} risk with {prob:.1%} fraud probability. "
        f"The primary factors influencing the score are {drivers}. "
        f"{action}"
    )


def generate_llm_report(
    txn: ExplainRequest,
    decision: str,
    prob: float,
    top_features: list[ShapFeature],
) -> tuple[str, str]:
    factors = ", ".join(f"{f.feature} ({f.direction}, SHAP impact: {f.shap_impact:+.3f})" for f in top_features)

    prompt = (
        f"You are a fraud prevention analyst. Write an investigation report of exactly 3 sentences analyzing this transaction.\n\n"
        f"Details:\n"
        f"- Type: {txn.type}\n"
        f"- Amount: ${txn.amount:,.2f}\n"
        f"- Sender balance: ${txn.oldbalanceOrg:,.2f} -> ${txn.newbalanceOrig:,.2f}\n"
        f"- Receiver balance: ${txn.oldbalanceDest:,.2f} -> ${txn.newbalanceDest:,.2f}\n"
        f"- Prediction: {decision} ({prob:.1%} fraud probability)\n"
        f"- Key SHAP features: {factors}\n\n"
        f"Requirements:\n"
        f"1. Sentence 1: Risk level and legitimacy evaluation.\n"
        f"2. Sentence 2: Key contributing factors driving the score.\n"
        f"3. Sentence 3: Recommended action for the analyst.\n"
        f"Output exactly 3 sentences, nothing else."
    )

    start_llm = time.perf_counter()
    try:
        import ollama

        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
        )
        report = response["message"]["content"].strip()
        LLM_REQUESTS.labels(status="success").inc()
        LLM_LATENCY.observe(time.perf_counter() - start_llm)
        return report, "llama3.2"
    except Exception:
        LLM_REQUESTS.labels(status="fallback").inc()
        LLM_LATENCY.observe(time.perf_counter() - start_llm)
        return fallback_report(txn, decision, prob, top_features), "fallback-template"


@router.post("/explain", response_model=ExplainResponse)
def explain(txn: ExplainRequest):
    if txn.type.upper() not in VALID_TYPES:
        raise HTTPException(status_code=422, detail=f"Invalid transaction type: {txn.type}")

    start_time = time.perf_counter()

    try:
        model, explainer = load_model_and_explainer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    features = prepare_features(txn)
    prob = float(model.predict_proba(features)[0, 1])
    decision = "Block" if prob >= FRAUD_THRESHOLD else "Approve"

    top_features = compute_top_features(explainer, features)
    report, model_used = generate_llm_report(txn, decision, prob, top_features)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return ExplainResponse(
        transaction_id=f"txn_{uuid.uuid4().hex[:12]}",
        decision=decision,
        fraud_probability=round(prob, 4),
        top_features=top_features,
        investigation_report=report,
        llm_model_used=model_used,
        processing_time_ms=round(elapsed_ms, 2),
    )
