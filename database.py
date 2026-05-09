import sqlite3

import pandas as pd

from src.config import DATABASE_PATH
from src.data_generation import build_database


def get_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DATABASE_PATH)


def load_customers() -> pd.DataFrame:
    if not DATABASE_PATH.exists():
        build_database()
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM customers", conn)


def sample_customers(limit: int = 8) -> list[dict]:
    with get_connection() as conn:
        rows = pd.read_sql_query(
            "SELECT * FROM customers ORDER BY RANDOM() LIMIT ?",
            conn,
            params=(limit,),
        )
    return rows.to_dict(orient="records")
