# Step 2 of the pipeline: train the XGBoost classifier on the SMOTE-balanced data.
#
# Usage:
#   python -m src.train

import argparse
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.config import MODEL_PATH, PROCESSED_DIR, RANDOM_STATE


def load_train_data(data_dir):
    data_dir = Path(data_dir)
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    return X_train, y_train


def build_model(y_train):
    # Even after SMOTE there are ~10x more normal rows than fraud rows,
    # so we tell XGBoost to weigh each fraud row more.
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = round(neg / max(pos, 1))

    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=PROCESSED_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    args = parser.parse_args()

    X_train, y_train = load_train_data(args.data_dir)
    print(f"Training on {len(X_train):,} rows, {X_train.shape[1]} features")

    model = build_model(y_train)
    print(f"scale_pos_weight = {model.get_params()['scale_pos_weight']}")
    model.fit(X_train, y_train)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
