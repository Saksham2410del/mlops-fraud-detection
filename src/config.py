# Settings shared by the training scripts and the API.
# Keeping them in one place means the API always builds features
# exactly the same way the model saw them during training.

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_PATH = ROOT_DIR / "models" / "xgboost_fraud.joblib"
REPORTS_DIR = ROOT_DIR / "reports"
POLICIES_DIR = ROOT_DIR / "policies"

TARGET = "isFraud"

# nameOrig / nameDest are just account IDs and isFlaggedFraud is a
# rule-based flag from the simulator, so none of them are useful inputs.
DROP_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud"]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# 6 original numeric columns + 5 one-hot type columns + 5 engineered = 16
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

TEST_SIZE = 0.2
# After SMOTE, fraud cases become 10% of the non-fraud count in the training set
SMOTE_RATIO = 0.1
RANDOM_STATE = 42

# Probability above which a transaction gets blocked
FRAUD_THRESHOLD = 0.5

LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

# How many policy rules to give the LLM for each transaction
TOP_K_POLICIES = 3
