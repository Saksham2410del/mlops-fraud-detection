# Feature engineering used by both preprocess.py (training) and api.py (serving).

import pandas as pd

from src.config import DROP_COLUMNS, FEATURE_COLUMNS, TRANSACTION_TYPES


def clean_data(df):
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # One-hot encode the transaction type. Done by hand instead of
    # pd.get_dummies so a single API request still gets all 5 columns.
    for t in TRANSACTION_TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(int)

    return df.drop(columns=["type"])


def add_features(df):
    df = df.copy()

    # How much money actually left the sender / reached the receiver
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Amount compared to what the sender had. Fraudsters usually empty the
    # account, so this ratio tends to be close to 1 for fraud. +1 avoids
    # dividing by zero.
    df["amount_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # Sender left with nothing / receiver ends with a zero balance
    df["is_orig_balance_zero"] = (df["newbalanceOrig"] == 0).astype(int)
    df["is_dest_balance_zero"] = (df["newbalanceDest"] == 0).astype(int)

    return df


def build_features(df):
    """Raw PaySim rows in, the 16 model inputs out (in the right order)."""
    df = clean_data(df)
    df = add_features(df)
    return df[FEATURE_COLUMNS]


def transaction_to_features(txn):
    """Turn a single transaction (like the API receives) into a 1-row DataFrame."""
    return build_features(pd.DataFrame([txn]))
