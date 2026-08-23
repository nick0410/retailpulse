"""Cohort retention and revenue-per-cohort analysis.

Group every customer by the month they joined, then follow each group forward:
month 0 is their sign-up month, month 1 the next month, and so on. The result
is the triangular matrix every growth team recognises - one row per cohort,
one column per month of age - which answers "is the product getting stickier
over time, or are we just buying more customers?"

Two matrices are produced:

* **retention** - share of the cohort that transacted in that month;
* **cumulative revenue per customer** - how much the average member of the
  cohort has spent by that age, which is the empirical counterpart to the
  model-based CLV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _month_index(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return d.dt.year * 12 + d.dt.month


def build_cohort_table(fact_sales: pd.DataFrame, dim_customer: pd.DataFrame) -> pd.DataFrame:
    """One row per (cohort, month_index) with active customers and revenue."""
    df = fact_sales[fact_sales["customer_id"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])

    cust = dim_customer[["customer_id", "signup_date"]].copy()
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])
    cust["cohort"] = cust["signup_date"].dt.strftime("%Y-%m")
    cust["cohort_month_idx"] = _month_index(cust["signup_date"])

    df = df.merge(cust, on="customer_id", how="inner")
    df["month_idx"] = _month_index(df["date"])
    df["months_since_signup"] = df["month_idx"] - df["cohort_month_idx"]
    # Purchases dated before sign-up cannot happen; guard against clock skew.
    df = df[df["months_since_signup"] >= 0]

    grid = (df.groupby(["cohort", "months_since_signup"], as_index=False)
            .agg(active_customers=("customer_id", "nunique"),
                 revenue=("line_amount", "sum"),
                 transactions=("transaction_id", "nunique")))

    cohort_size = cust.groupby("cohort")["customer_id"].nunique().rename("cohort_size")
    grid = grid.merge(cohort_size, on="cohort", how="left")
    grid["retention_rate"] = np.round(grid["active_customers"] / grid["cohort_size"], 4)
    grid["revenue_per_customer"] = np.round(grid["revenue"] / grid["cohort_size"], 2)
    return grid.sort_values(["cohort", "months_since_signup"]).reset_index(drop=True)


def retention_matrix(cohort_table: pd.DataFrame, max_months: int = 18) -> pd.DataFrame:
    """Pivot the cohort table into the classic triangular retention heatmap."""
    sub = cohort_table[cohort_table["months_since_signup"] <= max_months]
    mat = sub.pivot(index="cohort", columns="months_since_signup", values="retention_rate")
    return mat.sort_index()


def cumulative_revenue_matrix(cohort_table: pd.DataFrame, max_months: int = 18) -> pd.DataFrame:
    """Cumulative revenue per acquired customer, by cohort age."""
    sub = cohort_table[cohort_table["months_since_signup"] <= max_months]
    mat = sub.pivot(index="cohort", columns="months_since_signup", values="revenue_per_customer")
    return mat.sort_index().cumsum(axis=1)


def retention_curve(cohort_table: pd.DataFrame, min_cohort_size: int = 50) -> pd.DataFrame:
    """Average retention by age across all sufficiently large cohorts.

    Cohorts are weighted by size, and each age is only averaged over cohorts
    old enough to have been observed at that age - otherwise young cohorts
    would drag the tail of the curve to zero for purely mechanical reasons.
    """
    big = cohort_table[cohort_table["cohort_size"] >= min_cohort_size]
    if big.empty:
        return pd.DataFrame(columns=["months_since_signup", "retention_rate", "cohorts_observed"])

    max_age_per_cohort = big.groupby("cohort")["months_since_signup"].max()
    rows = []
    for age in range(0, int(big["months_since_signup"].max()) + 1):
        eligible = max_age_per_cohort[max_age_per_cohort >= age].index
        if len(eligible) == 0:
            continue
        sub = big[(big["cohort"].isin(eligible)) & (big["months_since_signup"] == age)]
        observed_sizes = (big[big["cohort"].isin(eligible)]
                          .drop_duplicates("cohort")["cohort_size"].sum())
        rows.append(
            {
                "months_since_signup": age,
                "retention_rate": round(float(sub["active_customers"].sum() / observed_sizes), 4),
                "cohorts_observed": int(len(eligible)),
            }
        )
    return pd.DataFrame(rows)


def cohort_quality_trend(cohort_table: pd.DataFrame, at_month: int = 3) -> pd.DataFrame:
    """Is customer quality improving? Retention at a fixed age, cohort by cohort."""
    sub = cohort_table[cohort_table["months_since_signup"] == at_month]
    out = sub[["cohort", "cohort_size", "retention_rate", "revenue_per_customer"]].copy()
    out = out.rename(columns={"retention_rate": f"retention_m{at_month}",
                              "revenue_per_customer": f"revenue_per_customer_m{at_month}"})
    return out.sort_values("cohort").reset_index(drop=True)
