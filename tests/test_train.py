import pandas as pd

from src.train import build_model, load_train_data


def test_scale_pos_weight_follows_class_ratio():
    y = pd.Series([0] * 100 + [1] * 10)
    model = build_model(y)
    assert model.get_params()["scale_pos_weight"] == 10


def test_model_trains_and_predicts(tmp_path):
    X = pd.DataFrame({"a": [1, 2, 3, 4, 50, 60] * 5, "b": [0, 0, 1, 0, 9, 8] * 5})
    y = pd.Series([0, 0, 0, 0, 1, 1] * 5)
    X.to_csv(tmp_path / "X_train.csv", index=False)
    y.to_csv(tmp_path / "y_train.csv", index=False)

    X_loaded, y_loaded = load_train_data(tmp_path)
    model = build_model(y_loaded)
    model.set_params(n_estimators=10)
    model.fit(X_loaded, y_loaded)

    probs = model.predict_proba(pd.DataFrame({"a": [2, 55], "b": [0, 9]}))[:, 1]
    assert probs[0] < 0.5 < probs[1]
