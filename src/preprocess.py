import os
import json
import argparse
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
COLUMNS_TO_DROP = ["nameOrig", "nameDest", "isFlaggedFraud"]
FEATURE_COLS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    "balance_diff_orig", "balance_diff_dest", "amount_ratio", "is_orig_balance_zero", "is_dest_balance_zero"
]


def load_data(filepath):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    logger.info(f"Loading raw data from {path}")
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()
    logger.info("Dropping identifier columns")
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns], errors="ignore")

    logger.info("One-hot encoding transaction type")
    for t in TRANSACTION_TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(int)
    df = df.drop(columns=["type"], errors="ignore")
    return df


def engineer_features(df):
    df = df.copy()
    logger.info("Creating engineered features")
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["is_orig_balance_zero"] = (df["newbalanceOrig"] == 0).astype(int)
    df["is_dest_balance_zero"] = (df["newbalanceDest"] == 0).astype(int)
    return df


def split_and_resample(df, test_size=0.2, smote_ratio=0.5, random_state=42):
    target = "isFraud"
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols] if cols else df.drop(columns=[target], errors="ignore")
    y = df[target]

    logger.info(f"Splitting data 80/20 stratified on {target}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_ref = X_train.copy()

    logger.info(f"Applying SMOTE to training data only (pre-SMOTE size: {len(X_train)})")
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)
    y_train_smote = pd.Series(y_train_smote, name=target)

    logger.info(f"Post-SMOTE train size: {len(X_train_smote)}, test size: {len(X_test)}")
    return X_train_smote, X_test, y_train_smote, y_test, X_train_ref


def save_data(output_dir, X_train, X_test, y_train, y_test, X_train_ref, total_rows, raw_fraud_rate):
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving processed files to {output_dir}")
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    X_train_ref.to_csv(os.path.join(output_dir, "X_train_ref.csv"), index=False)

    stats = {
        "total_rows": int(total_rows),
        "raw_fraud_rate": float(raw_fraud_rate),
        "train_rows_raw": int(len(X_train_ref)),
        "train_rows_smote": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "features": list(X_train.columns),
        "train_fraud_rate_smote": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean())
    }

    stats_path = os.path.join(output_dir, "preprocessing_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved preprocessing stats to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess PaySim dataset for fraud detection")
    parser.add_argument("--input", default="data/raw/PS_20174392719_1491204439457_log.csv", help="Input CSV path")
    parser.add_argument("--output", default="data/processed", help="Output directory path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    df = load_data(args.input)
    total_rows = len(df)
    raw_fraud_rate = df["isFraud"].mean() if "isFraud" in df.columns else 0.0

    logger.info(f"Raw dataset: {total_rows:,} rows, fraud rate: {raw_fraud_rate:.4%}")

    df_clean = clean_data(df)
    df_featured = engineer_features(df_clean)
    X_train, X_test, y_train, y_test, X_train_ref = split_and_resample(df_featured)

    save_data(args.output, X_train, X_test, y_train, y_test, X_train_ref, total_rows, raw_fraud_rate)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
