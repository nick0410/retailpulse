"""Warehouse loader: writes the star schema into a local SQLite database.

SQLite keeps the project dependency-free while still being a *real* SQL
warehouse - the dashboard and every analytics module read through SQL, not
through pickled DataFrames. Indexes are created explicitly so the query plans
in `sql/` behave the way a reviewer would expect on a columnar warehouse.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from ..config import WAREHOUSE_DB

INDEXES: dict[str, list[str]] = {
    "fact_sales": [
        "CREATE INDEX IF NOT EXISTS ix_fact_date ON fact_sales(date_key)",
        "CREATE INDEX IF NOT EXISTS ix_fact_customer ON fact_sales(customer_id)",
        "CREATE INDEX IF NOT EXISTS ix_fact_store ON fact_sales(store_id)",
        "CREATE INDEX IF NOT EXISTS ix_fact_product ON fact_sales(product_id)",
        "CREATE INDEX IF NOT EXISTS ix_fact_txn ON fact_sales(transaction_id)",
    ],
    "dim_customer": ["CREATE UNIQUE INDEX IF NOT EXISTS ix_dim_customer ON dim_customer(customer_id)"],
    "dim_product": ["CREATE UNIQUE INDEX IF NOT EXISTS ix_dim_product ON dim_product(product_id)"],
    "dim_store": ["CREATE UNIQUE INDEX IF NOT EXISTS ix_dim_store ON dim_store(store_id)"],
    "dim_date": ["CREATE UNIQUE INDEX IF NOT EXISTS ix_dim_date ON dim_date(date_key)"],
    "mart_daily_store": ["CREATE INDEX IF NOT EXISTS ix_mart_store_date ON mart_daily_store(date, store_id)"],
    "mart_daily_total": ["CREATE INDEX IF NOT EXISTS ix_mart_total_date ON mart_daily_total(date)"],
}


@contextmanager
def connect(db_path: Path | str = WAREHOUSE_DB):
    """Open a tuned SQLite connection."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def _serialise_dates(df: pd.DataFrame) -> pd.DataFrame:
    """SQLite has no date type - store ISO strings so BETWEEN still works."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif out[col].dtype == "bool":
            out[col] = out[col].astype(int)
    return out


def load_warehouse(tables: dict[str, pd.DataFrame], db_path: Path | str = WAREHOUSE_DB) -> dict[str, int]:
    """Replace the warehouse contents with ``tables`` and rebuild indexes."""
    written: dict[str, int] = {}
    with connect(db_path) as conn:
        for name, df in tables.items():
            if df is None or df.empty and name not in INDEXES:
                continue
            _serialise_dates(df).to_sql(name, conn, if_exists="replace", index=False)
            written[name] = len(df)
            for stmt in INDEXES.get(name, []):
                conn.execute(stmt)
        conn.execute("ANALYZE")
    return written


def query(sql: str, db_path: Path | str = WAREHOUSE_DB, params: tuple = ()) -> pd.DataFrame:
    """Run a SQL query against the warehouse and return a DataFrame."""
    with connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def read_table(name: str, db_path: Path | str = WAREHOUSE_DB,
               parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Convenience reader that restores date dtypes."""
    df = query(f"SELECT * FROM {name}", db_path)
    for col in parse_dates or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def table_sizes(db_path: Path | str = WAREHOUSE_DB) -> pd.DataFrame:
    """List every table in the warehouse with its row count."""
    with connect(db_path) as conn:
        names = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
        )["name"].tolist()
        rows = [
            {"table": n,
             "rows": pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {n}", conn)["n"].iat[0]}
            for n in names
        ]
    return pd.DataFrame(rows)
