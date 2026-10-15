import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

try:
    from src.train import load_processed_data
except ImportError:
    from src.train import load_data as load_processed_data


def test_load_processed_data(tmp_path):
    X_train = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    X_test = pd.DataFrame({"f1": [5, 6], "f2": [7, 8]})
    y_train = pd.Series([0, 1], name="isFraud")
    y_test = pd.Series([0, 1], name="isFraud")

    X_train.to_csv(tmp_path / "X_train.csv", index=False)
    X_test.to_csv(tmp_path / "X_test.csv", index=False)
    y_train.to_csv(tmp_path / "y_train.csv", index=False)
    y_test.to_csv(tmp_path / "y_test.csv", index=False)

    loaded_X_train, loaded_X_test, loaded_y_train, loaded_y_test = load_processed_data(tmp_path)

    assert loaded_X_train.shape == (2, 2)
    assert loaded_X_test.shape == (2, 2)
    assert len(loaded_y_train) == 2
    assert len(loaded_y_test) == 2


def test_model_training():
    X_train = pd.DataFrame({
        "f1": [1.0, 1.5, 2.0, 8.0, 9.0, 8.5] * 5,
        "f2": [0.5, 0.8, 1.0, 4.0, 5.0, 4.2] * 5,
    })
    y_train = pd.Series([0, 0, 0, 1, 1, 1] * 5)

    X_test = pd.DataFrame({
        "f1": [1.2, 8.8],
        "f2": [0.6, 4.5],
    })
    y_test = pd.Series([0, 1])

    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    assert len(preds) == 2
    assert acc >= 0.5
    assert roc >= 0.5
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_model_save_load(tmp_path):
    X = pd.DataFrame({"f1": [1.0, 2.0, 8.0, 9.0], "f2": [0.5, 1.0, 4.0, 5.0]})
    y = pd.Series([0, 0, 1, 1])

    model = XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "xgboost_fraud.joblib"
    joblib.dump(model, model_path)
    assert model_path.exists()

    loaded_model = joblib.load(model_path)
    preds_orig = model.predict(X)
    preds_loaded = loaded_model.predict(X)

    np.testing.assert_array_equal(preds_orig, preds_loaded)
