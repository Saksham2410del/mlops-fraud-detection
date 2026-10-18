# Quick check that the policy search picks the right rule.
# Each case is a transaction where I know which policy should apply.
# Needs Ollama running with nomic-embed-text pulled.
#
# Usage:
#   python -m src.check_retrieval

from src.config import TOP_K_POLICIES
from src.policy_search import search
from src.report import describe_transaction


def txn(type_, amount, old_org, new_org, old_dest, new_dest):
    return {
        "type": type_, "amount": amount,
        "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
    }


# (transaction, decision, fraud probability, policy that should be found)
CASES = [
    (txn("TRANSFER", 181000, 181000, 0, 0, 0), "Block", 0.99, "FP-01"),
    (txn("CASH_OUT", 95000, 95000, 0, 12000, 107000), "Block", 0.97, "FP-01"),
    (txn("TRANSFER", 60000, 60000, 0, 5000, 65000), "Block", 0.92, "FP-01"),
    (txn("PAYMENT", 1500, 20000, 18500, 0, 0), "Approve", 0.01, "FP-07"),
    (txn("CASH_IN", 8000, 12000, 20000, 30000, 22000), "Approve", 0.0, "FP-07"),
    (txn("DEBIT", 700, 9000, 8300, 4000, 4700), "Approve", 0.0, "FP-07"),
    (txn("TRANSFER", 3000, 90000, 87000, 15000, 18000), "Approve", 0.02, "FP-07"),
    (txn("TRANSFER", 350000, 1200000, 850000, 40000, 390000), "Approve", 0.04, "FP-03"),
    (txn("TRANSFER", 450000, 2000000, 1550000, 100000, 550000), "Block", 0.71, "FP-03"),
    (txn("CASH_OUT", 150000, 400000, 250000, 20000, 170000), "Approve", 0.12, "FP-04"),
    (txn("TRANSFER", 20000, 80000, 60000, 5000, 5000), "Approve", 0.3, "FP-05"),
    (txn("TRANSFER", 90000, 300000, 210000, 0, 90000), "Block", 0.65, "FP-06"),
]


def main():
    top1 = topk = 0
    for t, decision, prob, expected in CASES:
        found = [r["id"] for r in search(describe_transaction(t, decision))]
        top1 += found[0] == expected
        topk += expected in found
        mark = "ok  " if expected in found else "MISS"
        print(f"{mark} {t['type']:<9} {t['amount']:>10,.0f}  expected {expected}, got {found}")

    print(f"\nCorrect policy ranked 1st: {top1}/{len(CASES)}")
    print(f"Correct policy in top {TOP_K_POLICIES}:   {topk}/{len(CASES)}")


if __name__ == "__main__":
    main()
