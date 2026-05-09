import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["total_charges"] = pd.to_numeric(cleaned["total_charges"], errors="coerce")
    cleaned["monthly_charges"] = pd.to_numeric(cleaned["monthly_charges"], errors="coerce")
    cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce")
    cleaned["senior_citizen"] = pd.to_numeric(cleaned["senior_citizen"], errors="coerce").fillna(0).astype(int)
    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].fillna("Unknown").astype(str)
    return cleaned


def split_features_target(df: pd.DataFrame):
    cleaned = clean_customer_data(df)
    x = cleaned[FEATURE_COLUMNS]
    y = cleaned[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)
    return x, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
