"""The star schema must reconcile: nothing invented, nothing lost."""

from __future__ import annotations

import pandas as pd
import pytest

from retailpulse.etl import query, read_table, table_sizes


def test_fact_has_no_orphan_keys(star):
    fact = star["fact_sales"]
    assert set(fact["product_id"]) <= set(star["dim_product"]["product_id"])
    assert set(fact["store_id"]) <= set(star["dim_store"]["store_id"])
    assert set(fact["date_key"]) <= set(star["dim_date"]["date_key"])
    identified = fact["customer_id"].dropna()
    assert set(identified) <= set(star["dim_customer"]["customer_id"])


def test_line_id_is_the_grain(star):
    assert star["fact_sales"]["line_id"].is_unique


def test_marts_reconcile_to_the_fact_table(star):
    """A pre-aggregate that disagrees with its source is worse than no aggregate."""
    fact = star["fact_sales"]
    total = fact["line_amount"].sum()
    assert star["mart_daily_store"]["revenue"].sum() == pytest.approx(total, rel=1e-9)
    assert star["mart_daily_total"]["revenue"].sum() == pytest.approx(total, rel=1e-9)
    assert star["mart_daily_category"]["revenue"].sum() == pytest.approx(total, rel=1e-9)

    units = fact["quantity"].sum()
    assert star["mart_daily_store"]["units"].sum() == units
    assert star["mart_daily_total"]["units"].sum() == units


def test_margin_is_revenue_minus_cost(star):
    fact = star["fact_sales"]
    diff = (fact["gross_margin"] - (fact["line_amount"] - fact["line_cost"])).abs()
    assert diff.max() < 0.01


def test_date_dimension_is_complete_and_unique(star):
    dim = star["dim_date"]
    assert dim["date_key"].is_unique
    dates = pd.to_datetime(dim["date"]).sort_values()
    # A gap in the date dimension silently drops days from every report.
    assert (dates.diff().dropna() == pd.Timedelta(days=1)).all()


def test_warehouse_round_trips_through_sql(warehouse, star):
    sizes = table_sizes(warehouse).set_index("table")["rows"]
    for name, df in star.items():
        if df.empty:
            continue
        assert sizes[name] == len(df), f"{name} lost rows on the way into SQLite"

    total = query("SELECT SUM(line_amount) AS revenue FROM fact_sales",
                  db_path=warehouse)["revenue"].iat[0]
    assert abs(total - star["fact_sales"]["line_amount"].sum()) < 1.0


def test_dates_survive_serialisation(warehouse):
    fact = read_table("fact_sales", db_path=warehouse, parse_dates=["date"])
    assert pd.api.types.is_datetime64_any_dtype(fact["date"])
    assert fact["date"].notna().all()


def test_indexes_exist(warehouse):
    idx = query("SELECT name FROM sqlite_master WHERE type='index'", db_path=warehouse)
    names = set(idx["name"])
    assert "ix_fact_date" in names
    assert "ix_fact_customer" in names
