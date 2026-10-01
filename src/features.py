import pandas as pd


TARGET_COLUMN = "target_bloom_next_7d"


def load_feature_columns(path="../data/feature_columns.txt"):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def prepare_features(df, feature_columns):
    return df[feature_columns].copy()


def prepare_target(df):
    return df[TARGET_COLUMN].copy()
