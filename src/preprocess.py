# Step 1 of the pipeline: load PaySim, build features, split, apply SMOTE.
#
# Usage:
#   python -m src.preprocess
#   python -m src.preprocess --input path/to/paysim.csv

import argparse
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SMOTE_RATIO,
    TARGET,
    TEST_SIZE,
)
from src.features import build_features


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download PaySim from Kaggle "
            "and put the CSV inside data/raw/"
        )
    print(f"Loading {path} ...")
    return pd.read_csv(path)


def split_and_resample(X, y, test_size=TEST_SIZE, smote_ratio=SMOTE_RATIO):
    # Split FIRST, then oversample only the training part.
    # If SMOTE ran before the split, synthetic fraud rows built from test
    # rows would leak into training and the test score would be fake.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    smote = SMOTE(sampling_strategy=smote_ratio, random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=RAW_DATA_PATH)
    parser.add_argument("--output", default=PROCESSED_DIR)
    args = parser.parse_args()

    df = load_data(args.input)
    print(f"Rows: {len(df):,} | fraud rate: {df[TARGET].mean():.2%}")

    X = build_features(df)
    y = df[TARGET]
    print(f"Model inputs: {X.shape[1]} features")

    X_train, X_test, y_train, y_test = split_and_resample(X, y)

    print(f"Train after SMOTE: {len(X_train):,} rows ({int(y_train.sum()):,} fraud)")
    print(f"Test (untouched):  {len(X_test):,} rows ({int(y_test.sum()):,} fraud)")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out / "X_train.csv", index=False)
    X_test.to_csv(out / "X_test.csv", index=False)
    y_train.to_csv(out / "y_train.csv", index=False)
    y_test.to_csv(out / "y_test.csv", index=False)

    print(f"Saved processed data to {out}")


if __name__ == "__main__":
    main()
