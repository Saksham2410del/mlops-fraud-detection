"""Unit tests for the preprocessing pipeline.

Tests cover data loading, feature engineering, encoding, splitting,
SMOTE resampling, and data leakage prevention. All tests use a small
synthetic DataFrame that mimics PaySim structure so the actual dataset
is never required.
"""

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Fixtures


@pytest.fixture
def sample_paysim_df() -> pd.DataFrame:
    """Create a small (1000-row) synthetic PaySim-like DataFrame.

    The DataFrame mirrors the real PaySim schema:
        step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
        nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

    Approximately 3 % of rows are labelled as fraud (TRANSFER or CASH_OUT)
    to keep the minority class small yet large enough for SMOTE to work.
    """
    rng = np.random.RandomState(42)
    n_rows = 1000
    n_fraud = 30  # ~3 % fraud rate

    transaction_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    fraud_eligible_types = ["TRANSFER", "CASH_OUT"]

    # Generate base columns
    steps = rng.randint(1, 744, size=n_rows)
    types = rng.choice(transaction_types, size=n_rows)
    amounts = rng.uniform(10, 500_000, size=n_rows).round(2)
    old_balance_org = rng.uniform(0, 1_000_000, size=n_rows).round(2)
    new_balance_orig = (old_balance_org - amounts).clip(min=0).round(2)
    old_balance_dest = rng.uniform(0, 1_000_000, size=n_rows).round(2)
    new_balance_dest = (old_balance_dest + amounts).round(2)

    name_orig = [f"C{rng.randint(1_000_000, 9_999_999)}" for _ in range(n_rows)]
    name_dest = [f"M{rng.randint(1_000_000, 9_999_999)}" for _ in range(n_rows)]

    # Label fraud – only on eligible transaction types
    is_fraud = np.zeros(n_rows, dtype=int)
    eligible_indices = np.where(np.isin(types, fraud_eligible_types))[0]
    if len(eligible_indices) >= n_fraud:
        fraud_indices = rng.choice(eligible_indices, size=n_fraud, replace=False)
    else:
        fraud_indices = eligible_indices
    is_fraud[fraud_indices] = 1

    is_flagged_fraud = np.zeros(n_rows, dtype=int)
    # Flag a small subset of the fraudulent transactions
    flagged_count = min(5, len(fraud_indices))
    is_flagged_fraud[fraud_indices[:flagged_count]] = 1

    df = pd.DataFrame(
        {
            "step": steps,
            "type": types,
            "amount": amounts,
            "nameOrig": name_orig,
            "oldbalanceOrg": old_balance_org,
            "newbalanceOrig": new_balance_orig,
            "nameDest": name_dest,
            "oldbalanceDest": old_balance_dest,
            "newbalanceDest": new_balance_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged_fraud,
        }
    )
    return df


# Helper functions (mirrors src/preprocess.py logic for unit-test isolation)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not useful for modelling."""
    cols_to_drop = ["nameOrig", "nameDest", "isFlaggedFraud"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw PaySim columns."""
    df = df.copy()
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df


def _one_hot_encode_type(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the ``type`` column."""
    return pd.get_dummies(df, columns=["type"], prefix="type", dtype=int)


# Tests


class TestLoadAndCleanData:
    """Verify that unwanted columns are dropped correctly."""

    def test_load_and_clean_data(self, sample_paysim_df: pd.DataFrame) -> None:
        """After cleaning, nameOrig, nameDest, and isFlaggedFraud must be gone."""
        cleaned = _clean_columns(sample_paysim_df)

        assert "nameOrig" not in cleaned.columns
        assert "nameDest" not in cleaned.columns
        assert "isFlaggedFraud" not in cleaned.columns
        # Core columns must still exist
        assert "amount" in cleaned.columns
        assert "isFraud" in cleaned.columns
        assert "oldbalanceOrg" in cleaned.columns

    def test_row_count_unchanged(self, sample_paysim_df: pd.DataFrame) -> None:
        """Cleaning should not drop any rows."""
        cleaned = _clean_columns(sample_paysim_df)
        assert len(cleaned) == len(sample_paysim_df)


class TestFeatureEngineering:
    """Verify engineered features are calculated correctly."""

    def test_feature_engineering(self, sample_paysim_df: pd.DataFrame) -> None:
        """errorBalanceOrig and errorBalanceDest must equal the expected formula."""
        cleaned = _clean_columns(sample_paysim_df)
        engineered = _engineer_features(cleaned)

        assert "errorBalanceOrig" in engineered.columns
        assert "errorBalanceDest" in engineered.columns

        # Spot-check first row
        row = engineered.iloc[0]
        expected_orig = row["newbalanceOrig"] + row["amount"] - row["oldbalanceOrg"]
        assert np.isclose(row["errorBalanceOrig"], expected_orig), (
            f"errorBalanceOrig mismatch: {row['errorBalanceOrig']} != {expected_orig}"
        )

        expected_dest = row["oldbalanceDest"] + row["amount"] - row["newbalanceDest"]
        assert np.isclose(row["errorBalanceDest"], expected_dest), (
            f"errorBalanceDest mismatch: {row['errorBalanceDest']} != {expected_dest}"
        )

    def test_no_nans_in_engineered_features(self, sample_paysim_df: pd.DataFrame) -> None:
        """Engineered features must not contain NaN values."""
        engineered = _engineer_features(_clean_columns(sample_paysim_df))
        assert engineered["errorBalanceOrig"].isna().sum() == 0
        assert engineered["errorBalanceDest"].isna().sum() == 0


class TestOneHotEncoding:
    """Verify that the ``type`` column is one-hot encoded properly."""

    def test_one_hot_encoding(self, sample_paysim_df: pd.DataFrame) -> None:
        """After encoding, 'type' column is removed and dummies are created."""
        cleaned = _clean_columns(sample_paysim_df)
        encoded = _one_hot_encode_type(cleaned)

        # Original column must be gone
        assert "type" not in encoded.columns

        # At least one dummy column must exist with the expected prefix
        type_cols = [c for c in encoded.columns if c.startswith("type_")]
        assert len(type_cols) > 0, "No one-hot encoded type columns found"

        # Each dummy column should contain only 0s and 1s
        for col in type_cols:
            assert set(encoded[col].unique()).issubset({0, 1}), (
                f"Column {col} has values other than 0 and 1"
            )

    def test_one_hot_row_sums(self, sample_paysim_df: pd.DataFrame) -> None:
        """Each row must have exactly one type flag set to 1."""
        cleaned = _clean_columns(sample_paysim_df)
        encoded = _one_hot_encode_type(cleaned)
        type_cols = [c for c in encoded.columns if c.startswith("type_")]
        row_sums = encoded[type_cols].sum(axis=1)
        assert (row_sums == 1).all(), "Some rows have != 1 active type column"


class TestTrainTestSplitStratified:
    """Verify split ratios and stratification."""

    def test_train_test_split_stratified(self, sample_paysim_df: pd.DataFrame) -> None:
        """Split should honour the 80/20 ratio and preserve class proportions."""
        cleaned = _clean_columns(sample_paysim_df)
        engineered = _engineer_features(cleaned)
        encoded = _one_hot_encode_type(engineered)

        X = encoded.drop(columns=["isFraud"])
        y = encoded["isFraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Size checks (allow ±1 due to rounding)
        assert abs(len(X_train) - 0.8 * len(X)) <= 1
        assert abs(len(X_test) - 0.2 * len(X)) <= 1

        # Stratification: fraud ratio in train ≈ fraud ratio in full dataset
        full_fraud_ratio = y.mean()
        train_fraud_ratio = y_train.mean()
        assert abs(train_fraud_ratio - full_fraud_ratio) < 0.02, (
            f"Stratification broken: train={train_fraud_ratio:.4f} vs full={full_fraud_ratio:.4f}"
        )


class TestSMOTEAppliedOnlyToTrain:
    """Verify SMOTE increases the minority class in train data only."""

    def test_smote_applied_only_to_train(self, sample_paysim_df: pd.DataFrame) -> None:
        """After SMOTE, train minority count must increase; test must be untouched."""
        cleaned = _clean_columns(sample_paysim_df)
        engineered = _engineer_features(cleaned)
        encoded = _one_hot_encode_type(engineered)

        X = encoded.drop(columns=["isFraud"])
        y = encoded["isFraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        original_train_fraud_count = int(y_train.sum())
        original_test_fraud_count = int(y_test.sum())
        original_test_size = len(y_test)

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Train minority class must have grown
        resampled_fraud_count = int(y_train_res.sum())
        assert resampled_fraud_count > original_train_fraud_count, (
            "SMOTE did not increase the minority class in train data"
        )

        # Test set must remain unchanged
        assert len(y_test) == original_test_size
        assert int(y_test.sum()) == original_test_fraud_count

    def test_smote_balances_classes(self, sample_paysim_df: pd.DataFrame) -> None:
        """After SMOTE the two classes should be equally sized."""
        cleaned = _clean_columns(sample_paysim_df)
        engineered = _engineer_features(cleaned)
        encoded = _one_hot_encode_type(engineered)

        X = encoded.drop(columns=["isFraud"])
        y = encoded["isFraud"]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        smote = SMOTE(random_state=42)
        _, y_train_res = smote.fit_resample(X_train, y_train)

        class_counts = pd.Series(y_train_res).value_counts()
        assert class_counts[0] == class_counts[1], (
            f"Classes not balanced after SMOTE: {class_counts.to_dict()}"
        )


class TestNoDataLeakage:
    """Verify test set indices don't appear in the train set."""

    def test_no_data_leakage(self, sample_paysim_df: pd.DataFrame) -> None:
        """Train and test indices must be completely disjoint."""
        cleaned = _clean_columns(sample_paysim_df)
        engineered = _engineer_features(cleaned)
        encoded = _one_hot_encode_type(engineered)

        X = encoded.drop(columns=["isFraud"])
        y = encoded["isFraud"]

        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        overlap = train_indices & test_indices

        assert len(overlap) == 0, (
            f"Data leakage detected! {len(overlap)} indices appear in both splits: "
            f"{list(overlap)[:10]}"
        )

    def test_all_indices_accounted_for(self, sample_paysim_df: pd.DataFrame) -> None:
        """Union of train + test indices must equal the original index set."""
        cleaned = _clean_columns(sample_paysim_df)
        engineered = _engineer_features(cleaned)
        encoded = _one_hot_encode_type(engineered)

        X = encoded.drop(columns=["isFraud"])
        y = encoded["isFraud"]

        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        all_indices = set(X.index)
        split_indices = set(X_train.index) | set(X_test.index)

        assert all_indices == split_indices, (
            f"Missing indices after split: {all_indices - split_indices}"
        )
