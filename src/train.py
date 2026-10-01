import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error

from src.features import (
    load_feature_columns,
    prepare_features,
    prepare_target,
)


ROOT = "../data/"

TRAIN_PATH = ROOT + "early_warning_train.csv"
VALIDATION_PATH = ROOT + "early_warning_validation.csv"
FEATURE_COLUMNS_PATH = ROOT + "feature_columns.txt"

MODEL_PATH = "../models/early_warning_model.joblib"

TARGET_COLUMN = "target_bloom_next_7d"


def main():
    # Load data
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    # Load feature columns
    feature_columns = load_feature_columns(
        FEATURE_COLUMNS_PATH
    )

    # Make sure target isn't used as a feature
    assert TARGET_COLUMN not in feature_columns

    # Prepare training data
    X_train = prepare_features(
        train,
        feature_columns,
    )
    y_train = prepare_target(train)

    # Prepare validation data
    X_val = prepare_features(
        validation,
        feature_columns,
    )
    y_val = prepare_target(validation)

    # Train baseline
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run():

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        rmse = root_mean_squared_error(
            y_val,
            probabilities,
        )

        print(f"Validation RMSE: {rmse:.4f}")

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier",
        )
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("random_state", 42)
        mlflow.log_param(
            "n_features",
            len(feature_columns),
        )

        mlflow.log_metric(
            "validation_rmse",
            rmse,
        )

        mlflow.sklearn.log_model(
            model,
            "model",
        )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
