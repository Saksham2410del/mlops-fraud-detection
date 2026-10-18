# Writes a short investigation report for a transaction using Llama 3.2.
# The model runs locally through Ollama (https://ollama.com), so no data
# leaves the machine and no API key is needed.
#
# The report is grounded in the bank's policy documents (RAG): the most
# relevant rules are found by policy_search.py and pasted into the prompt,
# so the recommended next step comes from policy, not from the LLM's memory.
#
# If Ollama isn't running, we fall back to a simple template so the API
# still returns a decision instead of crashing.

import os
import re

import ollama

from src.config import LLM_MODEL
from src.policy_search import search

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def risk_level(probability):
    if probability >= 0.8:
        return "high"
    if probability >= 0.5:
        return "medium"
    return "low"


def amount_size(amount):
    if amount > 200000:
        return "very large (over 200,000)"
    if amount > 100000:
        return "large (100,000 to 200,000)"
    if amount > 10000:
        return "medium (10,000 to 100,000)"
    return "small (under 10,000)"


def describe_transaction(txn, decision):
    # Plain-English summary used as the search query for the policy rules.
    # Embedding models compare meaning, not numbers, so a sentence like
    # "the account was emptied" matches the right rule much better than
    # a list of balances.
    amount = txn["amount"]
    sentences = [f"A {amount_size(amount)} {txn['type']}."]

    if txn["oldbalanceOrg"] > 0 and txn["newbalanceOrig"] == 0:
        sentences.append("It moved the entire balance out of the sender's account, leaving it at zero.")
    elif txn["oldbalanceOrg"] > 0 and amount < txn["oldbalanceOrg"] / 2:
        sentences.append("The sender kept most of their balance after it.")
    else:
        sentences.append("The sender kept part of their balance after it.")

    # PaySim doesn't track merchant balances, so this check only makes sense for transfers
    if txn["type"] in ("TRANSFER", "CASH_OUT"):
        received = txn["newbalanceDest"] - txn["oldbalanceDest"]
        if abs(received - amount) >= 1:
            sentences.append("The receiver's balance did not go up by the amount sent.")
        elif txn["oldbalanceDest"] == 0:
            sentences.append("The money went to a receiver account that was empty before.")

    action = "blocked" if decision == "Block" else "approved"
    sentences.append(f"The fraud model {action} it.")
    return " ".join(sentences)


def transaction_facts(txn, decision, probability):
    # Small models are bad at arithmetic, so we work out the facts here
    # and only ask the LLM to put them into words.
    if txn["oldbalanceOrg"] > 0:
        share = f"{txn['amount'] / txn['oldbalanceOrg']:.0%} of the sender's balance"
    else:
        share = "more than the sender's recorded balance (it was 0)"
    emptied = "yes" if txn["oldbalanceOrg"] > 0 and txn["newbalanceOrig"] == 0 else "no"

    if txn["type"] in ("TRANSFER", "CASH_OUT"):
        received = txn["newbalanceDest"] - txn["oldbalanceDest"]
        receiver_ok = "yes" if abs(received - txn["amount"]) < 1 else "no"
        receiver_empty = "yes" if txn["oldbalanceDest"] == 0 else "no"
    else:
        receiver_ok = receiver_empty = "not tracked for this type"

    return f"""- Transaction type: {txn['type']}
- Amount: {txn['amount']:,.2f}, which is {amount_size(txn['amount'])}
- Amount moved: {share}
- Sender account emptied: {emptied}
- Receiver balance went up by the amount sent: {receiver_ok}
- Receiver account was empty before: {receiver_empty}
- Model decision: {decision}
- Fraud probability: {probability:.1%}
- Risk level: {risk_level(probability)}"""


def build_prompt(facts, policies):
    rules = "\n".join(f"[{p['title']}] {p['text']}" for p in policies)
    return f"""You are a fraud analyst at a bank. Write an investigation report
of exactly 3 sentences for one transaction.

FACTS ABOUT THIS TRANSACTION (the only facts you may use):
{facts}

POLICY RULES (general bank rules, not facts about this transaction):
{rules}

Sentence 1: the risk level and the decision, exactly as given in the facts.
Sentence 2: which facts support the decision.
Sentence 3: choose the ONE policy rule that best fits the facts and state the
action it requires, starting with "Per FP-xx,".
Never mention anything that is not in the facts. Reply with the 3 sentences
only, as one plain paragraph with no labels or headings."""


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
    """Returns (report_text, source, policy_ids). source is 'llama3.2' or 'template'."""
    try:
        facts = transaction_facts(txn, decision, probability)
        policies = search(describe_transaction(txn, decision))

        client = ollama.Client(host=OLLAMA_HOST, timeout=30)
        response = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": build_prompt(facts, policies)}],
            options={"temperature": 0.2},  # keep the wording factual, not creative
        )
        report = keep_three_sentences(response["message"]["content"])
        return report, LLM_MODEL, [p["id"] for p in policies]
    except Exception as e:
        print(f"Ollama not available, using template report ({e})")
        return template_report(txn, decision, probability), "template", []
