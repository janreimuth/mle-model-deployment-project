from pathlib import Path
import sys

from fastapi import FastAPI
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT / "src"))

from features import (
    load_feature_columns,
    prepare_features,
)

from model import load_model


DATA_PATH = ROOT / "data"
MODEL_PATH = ROOT / "models" / "early_warning_model.joblib"

FEATURE_COLUMNS_PATH = DATA_PATH / "feature_columns.txt"


app = FastAPI(
    title="HAB Early Warning API",
    description="Predict bloom probability for the next 7 days.",
    version="1.0.0",
)


feature_columns = load_feature_columns(
    FEATURE_COLUMNS_PATH
)

model = load_model(
    MODEL_PATH
)


@app.get("/")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(payload: dict):

    df = pd.DataFrame([payload])

    X = prepare_features(
        df,
        feature_columns,
    )

    probability = model.predict_proba(X)[0, 1]

    return {
        "bloom_probability": float(probability)
    }
