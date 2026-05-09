import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import METRICS_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_DIR, MODEL_PATH
from src.database import load_customers
from src.preprocessing import build_preprocessor, split_features_target


def _candidate_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=220,
            max_depth=9,
            min_samples_leaf=4,
            random_state=42,
            class_weight="balanced",
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.055,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    except Exception:
        pass
    return models


def _evaluate(model: Pipeline, x_test, y_test) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


def train_model() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_customers()
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.22,
        random_state=42,
        stratify=y,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best = None
    for name, estimator in _candidate_models().items():
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])
        with mlflow.start_run(run_name=name):
            pipeline.fit(x_train, y_train)
            metrics = _evaluate(pipeline, x_test, y_test)
            mlflow.log_params({"model_type": name, "train_rows": len(x_train), "test_rows": len(x_test)})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            candidate = {"name": name, "pipeline": pipeline, "metrics": metrics}
            if best is None or candidate["metrics"]["roc_auc"] > best["metrics"]["roc_auc"]:
                best = candidate

    payload = {
        "model_name": best["name"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
        **best["metrics"],
    }
    joblib.dump(best["pipeline"], MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(train_model(), indent=2))
