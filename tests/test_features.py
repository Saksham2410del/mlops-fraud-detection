import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.features import build_features, transaction_to_features
from src.preprocess import load_data, split_and_resample


def sample_rows():
    return pd.DataFrame({
        "step": [1, 1],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": [100.0, 500.0],
        "nameOrig": ["C1", "C2"],
        "oldbalanceOrg": [1000.0, 500.0],
        "newbalanceOrig": [900.0, 0.0],
        "nameDest": ["M1", "C3"],
        "oldbalanceDest": [0.0, 100.0],
        "newbalanceDest": [0.0, 600.0],
        "isFraud": [0, 1],
        "isFlaggedFraud": [0, 0],
    })


def test_gives_16_columns_in_order():
    X = build_features(sample_rows())
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape == (2, 16)


def test_id_columns_and_label_are_dropped():
    X = build_features(sample_rows())
    for col in ["nameOrig", "nameDest", "isFlaggedFraud", "isFraud", "type"]:
        assert col not in X.columns


def test_engineered_values():
    X = build_features(sample_rows())
    fraud_row = X.iloc[1]

    assert fraud_row["type_TRANSFER"] == 1
    assert fraud_row["type_PAYMENT"] == 0
    assert fraud_row["balance_diff_orig"] == 500.0
    assert fraud_row["balance_diff_dest"] == 500.0
    assert fraud_row["amount_ratio"] == 500.0 / 501.0
    assert fraud_row["is_orig_balance_zero"] == 1
    assert fraud_row["is_dest_balance_zero"] == 0


def test_single_api_transaction_matches_training_features():
    # The API path must produce exactly what training produced for the same row
    row = sample_rows().iloc[[1]].drop(columns=["nameOrig", "nameDest", "isFraud", "isFlaggedFraud"])
    from_api = transaction_to_features(row.iloc[0].to_dict())
    from_training = build_features(sample_rows()).iloc[[1]].reset_index(drop=True)

    pd.testing.assert_frame_equal(from_api, from_training, check_dtype=False)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv")


def test_smote_only_touches_training_data():
    # 180 normal rows and 20 fraud rows with some spread so SMOTE has neighbours
    X = pd.DataFrame({"a": range(200), "b": [i * 2.0 for i in range(200)]})
    y = pd.Series([0] * 180 + [1] * 20)

    X_train, X_test, y_train, y_test = split_and_resample(X, y, test_size=0.2, smote_ratio=0.5)

    # test set keeps the original size and fraud count
    assert len(X_test) == 40
    assert y_test.sum() == 4

    # training set gets extra synthetic fraud rows: 144 normal -> 72 fraud
    assert (y_train == 0).sum() == 144
    assert (y_train == 1).sum() == 72
