import json
from datetime import datetime, timezone

from src.config import PREDICTION_LOG_PATH
from src.database import load_customers


def log_prediction(payload: dict, churn_probability: float, prediction: str, latency_ms: float) -> None:
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "customer_payload": payload,
        "churn_probability": churn_probability,
        "prediction": prediction,
        "latency_ms": latency_ms,
    }
    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def drift_summary() -> dict:
    df = load_customers()
    baseline_churn_rate = round(float((df["churn"] == "Yes").mean()), 4)
    prediction_count = 0
    prediction_churn_rate = 0.0
    avg_latency_ms = 0.0

    if PREDICTION_LOG_PATH.exists():
        lines = [json.loads(line) for line in PREDICTION_LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        prediction_count = len(lines)
        if lines:
            prediction_churn_rate = round(sum(item["prediction"] == "Yes" for item in lines) / len(lines), 4)
            avg_latency_ms = round(sum(float(item["latency_ms"]) for item in lines) / len(lines), 2)

    drift_score = round(abs(prediction_churn_rate - baseline_churn_rate), 4) if prediction_count else 0.0
    return {
        "baseline_churn_rate": baseline_churn_rate,
        "prediction_churn_rate": prediction_churn_rate,
        "prediction_count": prediction_count,
        "avg_latency_ms": avg_latency_ms,
        "drift_score": drift_score,
        "status": "review" if drift_score > 0.12 and prediction_count >= 25 else "stable",
    }
