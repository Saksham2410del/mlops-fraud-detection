"""
Data Preprocessing Pipeline for Fraud Detection
================================================

Loads the raw PaySim CSV, engineers features, applies SMOTE to the training
set, and writes processed train/test splits to disk.

Usage:
    python src/preprocess.py \
        --input data/raw/PS_20174392719_1491204439457_log.csv \
        --output data/processed/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess")

# Constants
# Columns that carry no predictive signal or leak the target
COLUMNS_TO_DROP = ["nameOrig", "nameDest", "isFlaggedFraud"]

# Transaction types present in PaySim (used for one-hot encoding)
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


# Core functions


def load_data(filepath: Path) -> pd.DataFrame:
    """Load the raw PaySim CSV into a DataFrame.

    Parameters
    ----------
    filepath : Path
        Absolute or relative path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset as-is from the CSV.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    logger.info("Loading raw data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant columns and one-hot encode the transaction type.

    * ``nameOrig`` / ``nameDest`` are anonymised identifiers → no signal.
    * ``isFlaggedFraud`` is a naïve rule-based flag present in the dataset;
      keeping it would leak information about the target.
    * ``type`` is a low-cardinality categorical → one-hot encode it so
      tree-based models can split on each type independently.

    Parameters
    ----------
    df : pd.DataFrame
        Raw PaySim DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with dummy columns for each transaction type.
    """
    logger.info("Dropping columns: %s", COLUMNS_TO_DROP)
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    # One-hot encode 'type'.  We use pd.get_dummies and reindex to guarantee
    # a fixed column order even if a type is absent in a particular data slice.
    logger.info("One-hot encoding 'type' column")
    type_dummies = pd.get_dummies(df["type"], prefix="type").reindex(
        columns=[f"type_{t}" for t in TRANSACTION_TYPES], fill_value=0
    )
    df = pd.concat([df.drop(columns=["type"]), type_dummies], axis=1)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific features that capture fraud patterns.

    Rationale for each feature:
    * ``balance_diff_orig``  – Fraudulent transfers often drain the sender's
      account completely; the difference exposes that signal.
    * ``balance_diff_dest``  – Legitimate transfers should increase the
      destination balance by the transfer amount; deviations hint at fraud.
    * ``amount_ratio``       – A transfer that is huge relative to the
      sender's balance is suspicious.  The +1 avoids division-by-zero when
      the origin balance is 0.
    * ``is_orig_balance_zero`` / ``is_dest_balance_zero`` – Zero-balance
      accounts are disproportionately involved in fraud (mule accounts).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (after ``clean_data``).

    Returns
    -------
    pd.DataFrame
        DataFrame with new engineered columns appended.
    """
    logger.info("Engineering features")

    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # +1 in the denominator prevents inf when oldbalanceOrg == 0
    df["amount_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    df["is_orig_balance_zero"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["is_dest_balance_zero"] = (df["oldbalanceDest"] == 0).astype(int)

    return df


def split_and_resample(
    df: pd.DataFrame,
    test_size: float = 0.2,
    smote_ratio: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Separate features/target, split, and apply SMOTE to the training set.

    SMOTE is applied **only** to the training fold to avoid data leakage.
    A pre-SMOTE copy of ``X_train`` is also returned so that it can be saved
    as a reference distribution for downstream drift detection.

    Parameters
    ----------
    df : pd.DataFrame
        Fully preprocessed DataFrame including the ``isFraud`` target.
    test_size : float
        Fraction of data reserved for testing.
    smote_ratio : float
        ``sampling_strategy`` passed to SMOTE.  0.5 means the minority class
        will be oversampled to 50 % of the majority class size.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        ``(X_train_smote, X_test, y_train_smote, y_test, X_train_ref)``
        where ``X_train_ref`` is the pre-SMOTE training features.
    """
    target = "isFraud"
    X = df.drop(columns=[target])
    y = df[target]

    logger.info("Splitting data (test_size=%.2f, random_state=%d)", test_size, random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Keep a pre-SMOTE snapshot for drift detection reference
    X_train_ref = X_train.copy()

    _log_class_distribution("BEFORE SMOTE", y_train)

    logger.info("Applying SMOTE (sampling_strategy=%.2f)", smote_ratio)
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Convert back to pandas types for consistency
    X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)
    y_train_smote = pd.Series(y_train_smote, name=target)

    _log_class_distribution("AFTER SMOTE", y_train_smote)

    return X_train_smote, X_test, y_train_smote, y_test, X_train_ref


def save_artifacts(
    output_dir: Path,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_ref: pd.DataFrame,
) -> None:
    """Persist processed datasets and a summary JSON to *output_dir*.

    Files written:
        * ``X_train.csv``       – SMOTE-augmented training features
        * ``X_test.csv``        – Test features
        * ``y_train.csv``       – SMOTE-augmented training labels
        * ``y_test.csv``        – Test labels
        * ``X_train_ref.csv``   – Pre-SMOTE training features (drift baseline)
        * ``data_summary.json`` – Descriptive statistics for bookkeeping

    Parameters
    ----------
    output_dir : Path
        Directory where all artefacts are saved (created if absent).
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_test : pd.Series
        Label vectors.
    X_train_ref : pd.DataFrame
        Pre-SMOTE training features for drift reference.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving processed datasets to %s", output_dir)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)
    X_train_ref.to_csv(output_dir / "X_train_ref.csv", index=False)

    # Build a summary for quick inspection and downstream tooling (e.g. DVC)
    summary = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_ref_rows": len(X_train_ref),
        "num_features": X_train.shape[1],
        "feature_names": list(X_train.columns),
        "train_class_distribution": y_train.value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict(),
    }

    summary_path = output_dir / "data_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        # default=str handles numpy int64 keys from value_counts
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Data summary written to %s", summary_path)


# Helpers


def _log_class_distribution(label: str, y: pd.Series) -> None:
    """Pretty-print the fraud / non-fraud class counts."""
    counts = y.value_counts().sort_index()
    total = len(y)
    parts = " | ".join(f"class {cls}: {cnt:,} ({cnt / total:.4%})" for cls, cnt in counts.items())
    logger.info("%s — %s", label, parts)


# CLI entry-point


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Preprocess PaySim data for fraud detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/PS_20174392719_1491204439457_log.csv"),
        help="Path to the raw PaySim CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed files are saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to reserve for the test set.",
    )
    parser.add_argument(
        "--smote-ratio",
        type=float,
        default=0.5,
        help="SMOTE sampling_strategy (minority / majority ratio).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the full preprocessing pipeline end-to-end."""
    args = parse_args(argv)

    try:
        df = load_data(args.input)
        df = clean_data(df)
        df = engineer_features(df)
        X_train, X_test, y_train, y_test, X_train_ref = split_and_resample(
            df,
            test_size=args.test_size,
            smote_ratio=args.smote_ratio,
            random_state=args.random_state,
        )
        save_artifacts(args.output, X_train, X_test, y_train, y_test, X_train_ref)
        logger.info("Preprocessing complete ✓")
    except Exception:
        logger.exception("Preprocessing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
