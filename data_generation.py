import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATABASE_PATH, SAMPLE_DATA_PATH


def _yes_no(rng: np.random.Generator, p_yes: float, size: int) -> np.ndarray:
    return rng.choice(["Yes", "No"], p=[p_yes, 1 - p_yes], size=size)


def generate_synthetic_churn_data(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, rows)
    senior = rng.choice([0, 1], p=[0.83, 0.17], size=rows)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.24, 0.21], size=rows)
    internet = rng.choice(["DSL", "Fiber optic", "No"], p=[0.38, 0.45, 0.17], size=rows)
    monthly = np.round(
        rng.normal(62, 18, rows)
        + (internet == "Fiber optic") * 22
        - (internet == "No") * 30
        + senior * 5,
        2,
    )
    monthly = np.clip(monthly, 18, 120)
    total = np.round(monthly * tenure + rng.normal(0, 90, rows), 2)
    total = np.clip(total, monthly, None)

    paperless = _yes_no(rng, 0.6, rows)
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        p=[0.36, 0.21, 0.22, 0.21],
        size=rows,
    )
    tech_support = np.where(internet == "No", "No internet service", _yes_no(rng, 0.32, rows))
    online_security = np.where(internet == "No", "No internet service", _yes_no(rng, 0.35, rows))
    risk_score = (
        -2.3
        + (contract == "Month-to-month") * 1.45
        + (internet == "Fiber optic") * 0.75
        + (payment == "Electronic check") * 0.58
        + (paperless == "Yes") * 0.25
        + (tech_support == "No") * 0.35
        + (online_security == "No") * 0.35
        + senior * 0.28
        - tenure * 0.028
        + (monthly > 85) * 0.22
    )
    probability = 1 / (1 + np.exp(-risk_score))
    churn = np.where(rng.random(rows) < probability, "Yes", "No")

    df = pd.DataFrame(
        {
            "customer_id": [f"S{i:05d}" for i in range(1, rows + 1)],
            "gender": rng.choice(["Female", "Male"], size=rows),
            "senior_citizen": senior,
            "partner": _yes_no(rng, 0.48, rows),
            "dependents": _yes_no(rng, 0.31, rows),
            "tenure": tenure,
            "phone_service": _yes_no(rng, 0.9, rows),
            "multiple_lines": rng.choice(["Yes", "No", "No phone service"], p=[0.42, 0.48, 0.10], size=rows),
            "internet_service": internet,
            "online_security": online_security,
            "online_backup": np.where(internet == "No", "No internet service", _yes_no(rng, 0.44, rows)),
            "device_protection": np.where(internet == "No", "No internet service", _yes_no(rng, 0.43, rows)),
            "tech_support": tech_support,
            "streaming_tv": np.where(internet == "No", "No internet service", _yes_no(rng, 0.49, rows)),
            "streaming_movies": np.where(internet == "No", "No internet service", _yes_no(rng, 0.49, rows)),
            "contract": contract,
            "paperless_billing": paperless,
            "payment_method": payment,
            "monthly_charges": monthly,
            "total_charges": total,
            "churn": churn,
        }
    )
    return df


def save_to_sqlite(df: pd.DataFrame, database_path: Path = DATABASE_PATH) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as conn:
        df.to_sql("customers", conn, if_exists="replace", index=False)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                customer_payload TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                prediction TEXT NOT NULL,
                latency_ms REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_customers_churn ON customers(churn)")
        conn.commit()


def build_database(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    seed_df = pd.read_csv(SAMPLE_DATA_PATH) if SAMPLE_DATA_PATH.exists() else pd.DataFrame()
    synthetic_df = generate_synthetic_churn_data(rows=rows, seed=seed)
    df = pd.concat([seed_df, synthetic_df], ignore_index=True)
    save_to_sqlite(df)
    return df


if __name__ == "__main__":
    data = build_database()
    print(f"Created {DATABASE_PATH} with {len(data)} customer rows.")
