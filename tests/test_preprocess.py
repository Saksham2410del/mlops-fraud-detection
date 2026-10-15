from pathlib import Path
import pandas as pd
import pytest

from src.preprocess import clean_data, engineer_features, load_data, split_and_resample


def test_load_data_file_not_found(tmp_path):
    missing_file = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_data(missing_file)


def test_clean_data():
    raw_df = pd.DataFrame({
        "step": [1, 2],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": [100.0, 500.0],
        "nameOrig": ["C123", "C456"],
        "oldbalanceOrg": [1000.0, 500.0],
        "newbalanceOrig": [900.0, 0.0],
        "nameDest": ["M123", "C789"],
        "oldbalanceDest": [0.0, 100.0],
        "newbalanceDest": [100.0, 600.0],
        "isFraud": [0, 1],
        "isFlaggedFraud": [0, 0],
    })

    cleaned = clean_data(raw_df)

    for col in ["nameOrig", "nameDest", "isFlaggedFraud", "type"]:
        assert col not in cleaned.columns

    expected_types = [
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ]
    for col in expected_types:
        assert col in cleaned.columns

    assert cleaned.loc[0, "type_PAYMENT"] == 1
    assert cleaned.loc[1, "type_TRANSFER"] == 1


def test_engineer_features():
    df = pd.DataFrame({
        "step": [1],
        "amount": [150.0],
        "oldbalanceOrg": [500.0],
        "newbalanceOrig": [350.0],
        "oldbalanceDest": [100.0],
        "newbalanceDest": [250.0],
        "isFraud": [0],
    })

    featured = engineer_features(df)

    assert "balance_diff_orig" in featured.columns
    assert "balance_diff_dest" in featured.columns
    assert "amount_ratio" in featured.columns
    assert "is_orig_balance_zero" in featured.columns
    assert "is_dest_balance_zero" in featured.columns

    assert featured["balance_diff_orig"].iloc[0] == 150.0
    assert featured["balance_diff_dest"].iloc[0] == 150.0
    assert featured["amount_ratio"].iloc[0] == 150.0 / (500.0 + 1)
    assert featured["is_orig_balance_zero"].iloc[0] == 0
    assert featured["is_dest_balance_zero"].iloc[0] == 0


def test_smote_only_on_train():
    n_majority = 80
    n_minority = 20
    total = n_majority + n_minority

    data = {
        "step": [1] * total,
        "amount": [100.0] * total,
        "oldbalanceOrg": [500.0] * total,
        "newbalanceOrig": [400.0] * total,
        "oldbalanceDest": [200.0] * total,
        "newbalanceDest": [300.0] * total,
        "balance_diff_orig": [100.0] * total,
        "balance_diff_dest": [100.0] * total,
        "amount_ratio": [0.2] * total,
        "is_orig_balance_zero": [0] * total,
        "is_dest_balance_zero": [0] * total,
        "type_CASH_IN": [0] * total,
        "type_CASH_OUT": [0] * total,
        "type_DEBIT": [0] * total,
        "type_PAYMENT": [1] * total,
        "type_TRANSFER": [0] * total,
        "isFraud": [0] * n_majority + [1] * n_minority,
    }
    df = pd.DataFrame(data)

    X_train_smote, X_test, y_train_smote, y_test, X_train_ref = split_and_resample(
        df, test_size=0.2, smote_ratio=0.5, random_state=42
    )

    # Test set must remain untouched
    assert len(X_test) == 20
    assert len(y_test) == 20
    assert y_test.sum() == 4

    # Train set should be augmented by SMOTE
    assert len(X_train_smote) > len(X_train_ref)
    assert y_train_smote.sum() > (n_minority - 4)
