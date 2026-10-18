# Writes a short investigation report for a transaction using Llama 3.2.
# The model runs locally through Ollama (https://ollama.com), so no data
# leaves the machine and no API key is needed.
#
# If Ollama isn't running, we fall back to a simple template so the API
# still returns a decision instead of crashing.

import os
import re

import ollama

from src.config import LLM_MODEL

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def risk_level(probability):
    if probability >= 0.8:
        return "high"
    if probability >= 0.5:
        return "medium"
    return "low"


def build_prompt(txn, decision, probability):
    # Small models are bad at arithmetic, so we work out the facts here
    # and only ask the LLM to put them into words.
    if txn["oldbalanceOrg"] > 0:
        share = f"{txn['amount'] / txn['oldbalanceOrg']:.0%} of the sender's balance"
    else:
        share = "more than the sender's recorded balance (it was 0)"
    emptied = "yes" if txn["oldbalanceOrg"] > 0 and txn["newbalanceOrig"] == 0 else "no"

    return f"""You are a fraud analyst at a bank. Write an investigation report
of exactly 3 sentences for the transaction below. Use only these facts and
do not change the risk level or the decision.

Transaction type: {txn['type']}
Amount: {txn['amount']:,.2f}
Amount moved: {share}
Sender account emptied: {emptied}
Sender balance: {txn['oldbalanceOrg']:,.2f} -> {txn['newbalanceOrig']:,.2f}
Receiver balance: {txn['oldbalanceDest']:,.2f} -> {txn['newbalanceDest']:,.2f}
Model decision: {decision}
Fraud probability: {probability:.1%}
Risk level: {risk_level(probability)}

Sentence 1: state the risk level and the decision.
Sentence 2: explain which facts above support it.
Sentence 3: what the analyst should do next.
Reply with the 3 sentences only, as one plain paragraph with no labels or headings."""


def keep_three_sentences(text):
    # LLMs don't always follow length instructions, so cut it down ourselves
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:3])


def template_report(txn, decision, probability):
    risk = risk_level(probability)
    emptied = txn["oldbalanceOrg"] > 0 and txn["newbalanceOrig"] == 0
    detail = (
        "the sender's account was fully emptied by this transaction"
        if emptied
        else "the balance changes look consistent with the amount sent"
    )
    action = (
        "Hold the transaction and contact the customer to verify it."
        if decision == "Block"
        else "No action needed, the transaction can be processed."
    )
    return (
        f"This {txn['type']} of {txn['amount']:,.2f} is {risk} risk with a "
        f"{probability:.1%} fraud probability. The main signal is that {detail}. {action}"
    )


def generate_report(txn, decision, probability):
    """Returns (report_text, source) where source is 'llama3.2' or 'template'."""
    try:
        client = ollama.Client(host=OLLAMA_HOST, timeout=30)
        response = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": build_prompt(txn, decision, probability)}],
            options={"temperature": 0.2},  # keep the wording factual, not creative
        )
        return keep_three_sentences(response["message"]["content"]), LLM_MODEL
    except Exception as e:
        print(f"Ollama not available, using template report ({e})")
        return template_report(txn, decision, probability), "template"
