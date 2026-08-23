"""Raw -> warehouse transformation: cleaning, conforming, and star-schema build.

The layout follows the medallion pattern:

    bronze  data/raw/*.csv                (exactly as produced upstream)
    silver  cleaned + quarantined frames  (this module, `clean_layer`)
    gold    dim_* / fact_* star schema    (this module, `build_star_schema`)

Rows that fail a *critical* quality check are moved to a quarantine table
rather than deleted, so the pipeline is auditable: every row is either in the
warehouse or in quarantine with a reason attached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RAW_DIR
from .quality import run_quality_suite

RAW_TABLES = [
    "customers", "products", "stores", "promotions",
    "transactions", "transaction_items", "calendar",
    "anomaly_ground_truth", "customer_ground_truth",
]

DATE_COLUMNS = {
    "customers": ["signup_date"],
    "stores": ["opened_date"],
    "promotions": ["start_date", "end_date"],
    "transactions": ["date"],
    "transaction_items": ["date"],
    "calendar": ["date"],
    "anomaly_ground_truth": ["start_date", "end_date"],
}


def load_raw() -> dict[str, pd.DataFrame]:
    """Read the bronze layer off disk with correct dtypes."""
    out: dict[str, pd.DataFrame] = {}
    for name in RAW_TABLES:
        path = RAW_DIR / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for col in DATE_COLUMNS.get(name, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        out[name] = df
    return out


def clean_layer(raw: dict[str, pd.DataFrame], start_date: str,
                end_date: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Run the quality suite, quarantine critical failures, return silver tables."""
    report, quarantine, results = run_quality_suite(raw, start_date, end_date)

    silver: dict[str, pd.DataFrame] = {k: v.copy() for k, v in raw.items()}
    precise = []

    for table, bad_index in quarantine.items():
        if len(bad_index) == 0:
            continue
        silver[table] = raw[table].drop(index=bad_index)

        # Reasons come straight from the original evaluation. Re-running the
        # checks on just the quarantined rows would be wrong: a uniqueness
        # check sees only one copy of each duplicate in that subset and would
        # declare it clean.
        subset = raw[table].loc[bad_index]
        row_reason: dict[object, list[str]] = {i: [] for i in bad_index}
        for res in results:
            if res.table != table or res.severity != "critical":
                continue
            for i in res.failed_index:
                row_reason[i].append(res.check)
        precise.append(
            pd.DataFrame(
                {
                    "table": table,
                    "row_key": subset.iloc[:, 0].astype(str).to_numpy(),
                    "reason": [", ".join(row_reason[i]) or "unspecified" for i in subset.index],
                }
            )
        )

    quarantine_df = (pd.concat(precise, ignore_index=True) if precise
                     else pd.DataFrame(columns=["table", "row_key", "reason"]))

    # Cascade: line items whose parent transaction was quarantined must go too.
    surviving_txns = set(silver["transactions"]["transaction_id"])
    items = silver["transaction_items"]
    orphaned = items[~items["transaction_id"].isin(surviving_txns)]
    if len(orphaned):
        quarantine_df = pd.concat(
            [
                quarantine_df,
                pd.DataFrame({"table": "transaction_items",
                              "row_key": orphaned["line_id"].astype(str).to_numpy(),
                              "reason": "parent_transaction_quarantined"}),
            ],
            ignore_index=True,
        )
        silver["transaction_items"] = items[items["transaction_id"].isin(surviving_txns)]

    return silver, report, quarantine_df


# --------------------------------------------------------------------------
# Star schema
# --------------------------------------------------------------------------
def _date_key(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s)
    return (d.dt.year * 10_000 + d.dt.month * 100 + d.dt.day).astype("int64")


def build_dim_date(calendar: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(calendar["date"])
    dim = pd.DataFrame(
        {
            "date_key": _date_key(d),
            "date": d,
            "year": d.dt.year,
            "quarter": d.dt.quarter,
            "month": d.dt.month,
            "month_name": d.dt.strftime("%b"),
            "day": d.dt.day,
            "day_of_week": d.dt.dayofweek,
            "day_name": d.dt.strftime("%a"),
            "iso_week": d.dt.isocalendar().week.astype(int),
            "year_month": d.dt.strftime("%Y-%m"),
            "is_weekend": d.dt.dayofweek >= 5,
        }
    )
    for col in ("is_festival_window", "demand_factor", "trend_factor",
                "dow_factor", "season_factor", "festival_factor", "payday_factor"):
        if col in calendar.columns:
            dim[col] = calendar[col].to_numpy()
    return dim


def build_star_schema(silver: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Assemble conformed dimensions and the sales fact at line-item grain."""
    customers = silver["customers"]
    products = silver["products"]
    stores = silver["stores"]
    txns = silver["transactions"]
    items = silver["transaction_items"]

    dim_date = build_dim_date(silver["calendar"])

    dim_customer = customers.rename(columns={"city": "customer_city",
                                             "region": "customer_region"}).copy()
    dim_customer["customer_city"] = dim_customer["customer_city"].fillna("Unknown")
    dim_customer["signup_date_key"] = _date_key(dim_customer["signup_date"])
    dim_customer["signup_cohort"] = pd.to_datetime(dim_customer["signup_date"]).dt.strftime("%Y-%m")

    dim_product = products.copy()
    dim_product["price_band"] = pd.cut(
        dim_product["base_price"],
        bins=[0, 100, 300, 800, 2000, np.inf],
        labels=["<100", "100-300", "300-800", "800-2000", "2000+"],
    ).astype(str)
    dim_product["margin_pct"] = np.round(
        (dim_product["base_price"] - dim_product["unit_cost"]) / dim_product["base_price"], 4
    )

    dim_store = stores.rename(columns={"city": "store_city", "region": "store_region"}).copy()

    # ---- fact table --------------------------------------------------------
    fact = items.merge(
        txns[["transaction_id", "customer_id", "store_id", "channel", "payment_method"]],
        on="transaction_id",
        how="inner",
        validate="many_to_one",
    )
    fact["date"] = pd.to_datetime(fact["date"])
    fact["date_key"] = _date_key(fact["date"])
    fact["is_promo_line"] = fact["discount_pct"] > 0
    fact["is_identified"] = fact["customer_id"].notna()
    # unit_price = base_price * (1 - d), so the rupees given away on a line are
    # line_amount * d / (1 - d).
    d = fact["discount_pct"].clip(0, 0.95)
    fact["discount_amount"] = np.round(fact["line_amount"] * d / (1 - d), 2)

    fact = fact[
        ["line_id", "transaction_id", "date_key", "date", "customer_id", "store_id",
         "product_id", "category", "channel", "payment_method", "quantity", "unit_price",
         "discount_pct", "discount_amount", "line_amount", "line_cost", "gross_margin",
         "is_promo_line", "is_identified"]
    ]

    # ---- pre-aggregated marts ---------------------------------------------
    daily_store = (
        fact.groupby(["date", "store_id"], as_index=False)
        .agg(
            revenue=("line_amount", "sum"),
            units=("quantity", "sum"),
            margin=("gross_margin", "sum"),
            transactions=("transaction_id", "nunique"),
            lines=("line_id", "count"),
        )
        .sort_values(["store_id", "date"])
        .reset_index(drop=True)
    )
    daily_store["avg_basket_value"] = np.round(
        daily_store["revenue"] / daily_store["transactions"].replace(0, np.nan), 2
    )

    daily_total = (
        daily_store.groupby("date", as_index=False)
        .agg(revenue=("revenue", "sum"), units=("units", "sum"),
             margin=("margin", "sum"), transactions=("transactions", "sum"))
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily_category = (
        fact.groupby(["date", "category"], as_index=False)
        .agg(revenue=("line_amount", "sum"), units=("quantity", "sum"),
             margin=("gross_margin", "sum"))
        .sort_values(["category", "date"])
        .reset_index(drop=True)
    )

    return {
        "dim_date": dim_date,
        "dim_customer": dim_customer,
        "dim_product": dim_product,
        "dim_store": dim_store,
        "fact_sales": fact,
        "mart_daily_store": daily_store,
        "mart_daily_total": daily_total,
        "mart_daily_category": daily_category,
        "dim_promotion": silver.get("promotions", pd.DataFrame()),
    }
