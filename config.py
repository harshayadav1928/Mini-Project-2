from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
MONITORING_DIR = ROOT_DIR / "monitoring"

DATABASE_PATH = DATA_DIR / "mlopsflow.db"
SAMPLE_DATA_PATH = DATA_DIR / "sample_customers.csv"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
PREDICTION_LOG_PATH = MONITORING_DIR / "predictions.jsonl"
MLFLOW_TRACKING_URI = f"file:{ROOT_DIR / 'mlruns'}"
MLFLOW_EXPERIMENT_NAME = "MLOpsFlow Customer Churn"

TARGET_COLUMN = "churn"

CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

NUMERIC_FEATURES = [
    "senior_citizen",
    "tenure",
    "monthly_charges",
    "total_charges",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
